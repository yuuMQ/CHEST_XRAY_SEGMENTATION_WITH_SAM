import torch
import cv2
from torch.utils.data import Dataset
import os
from torchvision.transforms import ToTensor, PILToTensor, Compose, ColorJitter, Resize, RandomAffine, Normalize, ToPILImage
import numpy as np
from PIL import Image

class COVID19_Classification_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.categories = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
        if train:
            mode = "train_images"
        else:
            mode = "valid_images"

        data = [os.path.join(self.root, cate, mode) for cate in self.categories]
        self.images = []
        self.labels = []
        for i, images_path in enumerate(data):
            label = i
            for image in os.listdir(images_path):
                self.images.append(os.path.join(images_path, image))
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        label = self.labels[idx]
        image = cv2.imread(image_file)
        if self.transform:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    transform = Compose([
        Resize((416, 416)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = COVID19_Classification_Dataset(root="../data", train=True, transform=transform)
    '''
        📊 Class distribution:
          COVID           : 2531 samples
          Lung_Opacity    : 4208 samples
          Normal          : 7135 samples
          Viral_Pneumonia : 941 samples
        
        ⚠️ Imbalance ratio (max/min): 7.58

    '''
    # image, label = dataset.__getitem__(100)
    # image = image.numpy()
    # image = np.transpose(image, (1, 2, 0))
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

