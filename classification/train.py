import torch
import cv2
from dataset import COVID19_Classification_Dataset
from argparse import ArgumentParser
from torchvision.transforms import ToTensor, Compose, Resize, RandomAffine, ColorJitter, Normalize, RandomHorizontalFlip
from torch.utils.data import DataLoader
from torchvision.models import ResNet152_Weights, resnet152, ResNet50_Weights, resnet50, densenet121, DenseNet121_Weights, EfficientNet_B3_Weights, efficientnet_b3, efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from torch.optim import SGD, AdamW, Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from torchsummary import summary



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_args():
    parser = ArgumentParser(description="COVID CLASSIFICATION")

    parser.add_argument("--root", '-r', type=str, default="../data", help="root path")
    parser.add_argument("--batch_size", "-b",type=int, default=32, help="batch size")
    parser.add_argument("--epochs", "-e",type=int, default=100, help="number of epochs")
    parser.add_argument("--learning_rate", "-lr",type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", "-wd",type=float, default=0.01, help="weight decay")
    parser.add_argument("--momentum", "-m",type=float, default=0.9, help="momentum")
    parser.add_argument("--num_workers", "-nw", type=int, default=4, help="number of workers")
    parser.add_argument("--image_size", '-i', type=int, default=224, help="image size")
    parser.add_argument("--early_stopping", "-p", type=int, default=10, help="early stopping patience")
    parser.add_argument("--logging", '-l', type=str, default="tensorboard", help="logging path")
    parser.add_argument("--checkpoint", '-c', type=str, default=None, help="checkpoint path")
    parser.add_argument("--save_models", "-sv", type=str, default="ViT_b16_models", help="save path")
    parser.add_argument("--report", '-rp', type=str, default="classification_reports", help="classification_report path")

    args = parser.parse_args()
    return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train(args):
    num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = Compose([
        RandomHorizontalFlip(),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = COVID19_Classification_Dataset(root=args.root, train=True, transform=train_transform)
    val_dataset = COVID19_Classification_Dataset(root=args.root, train=False, transform=val_transform)

    num_classes = len(train_dataset.categories)

    train_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }

    train_dataloader = DataLoader(train_dataset, **train_params)
    val_dataloader = DataLoader(val_dataset, **val_params)

    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features

    model.heads.head = nn.Linear(in_features, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, threshold=1e-4, min_lr=1e-10 )
    num_iters = len(train_dataloader)

    patience = args.early_stopping
    no_improve_epochs = 0
    best_accuracy = -1

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    os.makedirs(args.logging)

    if not os.path.isdir(args.save_models):
        os.makedirs(args.save_models)
    if not os.path.isdir(args.report):
        os.makedirs(args.report)

    writer = SummaryWriter(log_dir=args.logging)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0



    for epoch in range(start_epoch, num_epochs):
        # Train
        model.train()
        train_progress_bar = tqdm(train_dataloader, colour="green")
        for iter, (images, labels) in enumerate(train_progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # Forward
            output = model(images)
            loss = criterion(output, labels)
            train_progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch + 1, num_epochs, iter + 1, num_iters, loss))

            writer.add_scalar("Train/Loss", loss, epoch * num_iters + iter)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Val
        val_losses = []
        model.eval()
        all_predictions = []
        all_labels = []
        val_progress_bar = tqdm(val_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(val_progress_bar):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss = criterion(predictions, labels)
            val_losses.append(loss.item())


        all_labels = [label.item() for label in all_labels]
        all_predictions = [pred.item() for pred in all_predictions]

        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=train_dataset.categories, epoch=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))

        val_loss_avg = np.mean(val_losses)
        scheduler.step(val_loss_avg)
        writer.add_scalar("Validation/Loss", val_loss_avg, epoch)
        writer.add_scalar("Validation/Accuracy", accuracy, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, "{}/last_model.pt".format(args.save_models))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improve_epochs = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
                "scheduler": scheduler.state_dict(),
            }
            torch.save(checkpoint, "{}/best_model.pt".format(args.save_models))

            report_text = classification_report(all_labels, all_predictions, target_names=train_dataset.categories)
            print(report_text)

            with open(os.path.join(args.report, "report.txt"), "w") as f:
                f.write(report_text)


        else:
            no_improve_epochs += 1
            print(f"Accuracy did not improve ({no_improve_epochs}/{patience})")
            if no_improve_epochs >= patience:
                print("Stop training at epoch {} because no improvement".format(epoch + 1))
                break
        print("Best Accuracy: {}".format(best_accuracy))

if __name__ == '__main__':
    args = get_args()
    train(args)