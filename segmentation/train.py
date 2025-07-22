from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import SamProcessor, SamModel
import numpy as np
from tqdm import tqdm
from dataset import COVID19SegmentationDataset
from loss import BCEDiceLoss
import torch
from pprint import pprint
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, Affine
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import os
import shutil
import cv2
from utils import *

def get_args():
    parser = ArgumentParser(description="COVID SEGMENTATION")

    parser.add_argument("--root", '-r', type=str, default="../data", help="root path")
    parser.add_argument("--batch_size", "-b",type=int, default=2, help="batch size")
    parser.add_argument("--epochs", "-e",type=int, default=10, help="number of epochs")
    parser.add_argument("--learning_rate", "-lr",type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight_decay", "-wd",type=float, default=0.0, help="weight decay")
    parser.add_argument("--num_workers", "-nw", type=int, default=6, help="number of workers")
    parser.add_argument("--image_size", '-i', type=int, default=512, help="image size")
    parser.add_argument("--logging", '-l', type=str, default="tensorboard", help="logging path")
    parser.add_argument("--checkpoint", '-c', type=str, default=None, help="checkpoint path")
    parser.add_argument("--save_models", "-sv", type=str, default="SAM_models", help="save path")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = Compose([
        Resize(args.image_size, args.image_size),
        HorizontalFlip(p=0.5),
        Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        RandomBrightnessContrast(p=0.3),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = Compose([
        Resize(args.image_size, args.image_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    processor = SamProcessor.from_pretrained('facebook/sam-vit-base')

    train_dataset = COVID19SegmentationDataset(root=args.root, train=True, processor=processor, transform=train_transform)
    val_dataset = COVID19SegmentationDataset(root=args.root, train=False, processor=processor, transform=val_transform)

    # train_dataset = COVID19SegmentationDataset(root=args.root, train=True, processor=processor)
    # val_dataset = COVID19SegmentationDataset(root=args.root, train=False, processor=processor)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    model = SamModel.from_pretrained('facebook/sam-vit-base')

    for name, param in model.named_parameters():
        if name.startswith('vision_encoder') or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = BCEDiceLoss()
    best_val_loss = float('inf')
    num_iterations = len(train_dataloader)
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    os.makedirs(args.logging)

    if not os.path.isdir(args.save_models):
        os.makedirs(args.save_models)

    writer = SummaryWriter(log_dir=args.logging)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        # TRAINING
        model.train()
        train_losses, val_losses = [], []
        train_acc, train_dice, train_iou = [], [], []
        train_progress_bar = tqdm(train_dataloader, colour="green")
        for iter, batch in enumerate(train_progress_bar):
            images = batch["pixel_values"].to(device)
            masks = batch["ground_truth_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images, input_boxes=batch["input_boxes"].to(device), multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = masks.unsqueeze(1)

            if ground_truth_masks.ndim == 5:
                ground_truth_masks = ground_truth_masks.squeeze(2)

            ground_truth_masks = F.interpolate(ground_truth_masks, size=predicted_masks.shape[-2:], mode='nearest')
            ground_truth_masks = ground_truth_masks.float()
            loss = criterion(predicted_masks, ground_truth_masks)

            train_progress_bar.set_description("Epoch {}/{}. Iteration: {}/ {}. Loss {:.3f}".format(epoch + 1, num_epochs, iter + 1, num_iterations, loss))

            # Backward
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            accuracy, dice, iou = calculate_metrics(predicted_masks, ground_truth_masks)

            train_acc.append(accuracy)
            train_dice.append(dice)
            train_iou.append(iou)

            writer.add_scalar("Train/Accuracy", accuracy, epoch * num_iterations + iter)
            writer.add_scalar("Train/Dice", dice, epoch * num_iterations + iter)
            writer.add_scalar("Train/IoU", iou, epoch * num_iterations + iter)
            writer.add_scalar("Train/Loss", loss, epoch * num_iterations + iter)


        # VALIDATION
        model.eval()
        val_acc, val_dice, val_iou = [], [], []
        save_dir = os.path.join(args.logging, "logs", f"epoch_{epoch + 1}")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, colour="cyan")
            for val_batch in val_progress_bar:
                images = val_batch["pixel_values"].to(device)
                masks = val_batch["ground_truth_mask"].to(device)
                outputs = model(pixel_values=images, input_boxes=val_batch["input_boxes"].to(device), multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = masks.unsqueeze(1)

                if ground_truth_masks.ndim == 5:
                    ground_truth_masks = ground_truth_masks.squeeze(2)
                ground_truth_masks = F.interpolate(ground_truth_masks, size=predicted_masks.shape[-2:], mode="nearest")
                ground_truth_masks = ground_truth_masks.float()
                val_loss = criterion(predicted_masks, ground_truth_masks)
                val_losses.append(val_loss.item())
                accuracy, dice, iou = calculate_metrics(predicted_masks, ground_truth_masks)

                val_acc.append(accuracy)
                val_dice.append(dice)
                val_iou.append(iou)

                writer.add_scalar("Val/Accuracy", accuracy, epoch)
                writer.add_scalar("Val/Dice", dice, epoch)
                writer.add_scalar("Val/IoU", iou, epoch)
                writer.add_scalar("Val/Loss", val_loss, epoch)

                for i in range(images.size(0)):
                    img = images[i]
                    gt_mask = masks[i]
                    pred = predicted_masks[i]
                    image_name = val_batch["image_name"][i]
                    base_name = os.path.splitext(image_name)[0]
                    save_path = os.path.join(save_dir, f"{base_name}.png")

                    save_visualization(img, gt_mask, pred, save_path)

        # Calculate mean stats
        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)

        mean_train_acc = np.mean(train_acc)
        mean_val_acc = np.mean(val_acc)

        mean_train_iou = np.mean(train_iou)
        mean_val_iou = np.mean(val_iou)

        mean_train_dice = np.mean(train_dice)
        mean_val_dice = np.mean(val_dice)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {mean_train_loss}, Train Acc: {mean_train_acc}, Train Dice: {mean_train_dice}, Train IoU: {mean_train_iou}')
        print(f'Val Loss: {mean_val_loss}, Val Acc: {mean_val_acc}, Val Dice: {mean_val_dice}, Val IoU: {mean_val_iou}')

        writer.add_scalar("Train/Mean_Loss", mean_train_loss, epoch)
        writer.add_scalar("Train/Mean_Acc", mean_train_acc, epoch)
        writer.add_scalar("Train/Mean_Dice", mean_train_dice, epoch)
        writer.add_scalar("Train/Mean_Iou", mean_train_iou, epoch)

        writer.add_scalar("Val/Mean_Loss", mean_val_loss, epoch)
        writer.add_scalar("Val/Mean_Acc", mean_val_acc, epoch)
        writer.add_scalar("Val/Mean_Dice", mean_val_dice, epoch)
        writer.add_scalar("Val/Mean_IoU", mean_val_iou, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": mean_val_loss,
        }
        torch.save(checkpoint, "{}/last_model.pt".format(args.save_models))

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_val_loss,
            }
            torch.save(checkpoint, "{}/best_model.pt".format(args.save_models))
            print("Best model saved with loss: ", best_val_loss)
