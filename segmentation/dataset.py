from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
import torch
from transformers import SamProcessor
import torch.nn.functional as F
from utils import *

# Custom SAM Dataset
class COVID19SegmentationDataset(Dataset):
    def __init__(self, root, train=True, transform=None, processor=None):
        self.processor = processor
        self.root = root
        self.transform = transform
        self.images = []
        self.masks = []
        self.categories = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
        # self.categories = ["COVID"]
        if train:
            mode = "train"
        else:
            mode = "val"

        data = [os.path.join(self.root, cate) for cate in self.categories]
        self.images = []
        self.masks = []
        self.images_paths = [os.path.join(cate, mode + "_images") for cate in data]
        self.masks_paths = [os.path.join(cate, mode + "_masks") for cate in data]

        for images_path, masks_path in zip(self.images_paths, self.masks_paths):
            image_files = sorted(os.listdir(images_path))
            mask_files = sorted(os.listdir(masks_path))
            for image, mask in zip(image_files, mask_files):
                self.images.append(os.path.join(images_path, image))
                self.masks.append(os.path.join(masks_path, mask))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        mask_file = self.masks[idx]
        image_orig = cv2.imread(str(image_file))
        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if image_orig.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image_orig.shape[1], image_orig.shape[0]))
        mask = (mask > 0).astype(np.uint8)

        # augmentation
        if self.transform:
            augmented = self.transform(image=image_orig, mask=mask)
            image_aug = augmented["image"]
            mask_aug = augmented["mask"]
            image_for_processor = (image_aug.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_np = mask_aug.numpy()
        else:
            image_aug = torch.from_numpy(image_orig).permute(2, 0, 1).float() / 255.0
            mask_aug = torch.from_numpy(mask).float()
            image_for_processor = image_orig
            mask_np = mask

        bbox = get_bounding_box(mask_np)

        # sam processor
        inputs = self.processor(images=image_for_processor, input_boxes=[[bbox]], return_tensors="pt")
        input = {k: v.squeeze(0) for k, v in inputs.items()}

        # resize mask
        mask_tensor = F.interpolate(mask_aug.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode="nearest")
        mask_tensor = mask_tensor.squeeze(0).squeeze(0)

        input["ground_truth_mask"] = mask_tensor
        input["augmented_image"] = image_aug
        input["image_name"] = os.path.basename(image_file)
        return input

if __name__ == '__main__':
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_transform = Compose([
        Resize(1024, 1024),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    dataset = COVID19SegmentationDataset(root="../data", train=True, processor=processor, transform=train_transform)
    for idx in [5, 20, 45, 67, 100]:
        item = dataset[idx]
        aug_img = item["augmented_image"]
        gt_mask = item["ground_truth_mask"]

        gt_mask_resized = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0), size=aug_img.shape[1:], mode="nearest").squeeze()

        draw_bounding_box(aug_img, gt_mask_resized)

    #
    #
    # print(get_bounding_box(mask))
    #
    # image = image.numpy()
    # image = np.transpose(image, (1, 2, 0))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mask = mask.numpy()
    # cv2.imshow("image", image)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
