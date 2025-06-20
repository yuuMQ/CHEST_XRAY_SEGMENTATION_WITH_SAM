from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

class COVID19_Segmentation_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.masks = []
        self.categories = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
        if train:
            mode = "train"
        else:
            mode = "valid"
        data = [os.path.join(self.root, cate) for cate in self.categories]
        self.images = []
        self.masks = []
        self.labels = []
        self.images_paths = [os.path.join(cate, mode + "_images") for cate in data]
        self.masks_paths = [os.path.join(cate, mode + "_masks") for cate in data]

        for i,(images_path, masks_path) in enumerate(zip(self.images_paths, self.masks_paths)):
            image_files = sorted(os.listdir(images_path))
            mask_files = sorted(os.listdir(masks_path))
            for image, mask in zip(image_files, mask_files):
                self.labels.append(i)
                self.images.append(os.path.join(images_path, image))
                self.masks.append(os.path.join(masks_path, mask))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        label = self.labels[idx]
        mask_file = self.masks[idx]

        image_file, mask_file = str(image_file), str(mask_file)

        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))


        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

if __name__ == '__main__':
    train_transform = Compose([
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    dataset = COVID19_Segmentation_Dataset(root="../data", train=True, transform=train_transform)
    image, mask = dataset.__getitem__(100)


    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = mask.numpy()
    cv2.imshow("image", image)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
