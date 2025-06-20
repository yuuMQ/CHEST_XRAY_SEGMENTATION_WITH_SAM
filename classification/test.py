import numpy as np
import cv2
import torch
from argparse import ArgumentParser
import torch.nn as nn
from torchsummary import summary
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import os
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", '-p', type=str, default="test_images", help='image path')
    parser.add_argument("--image_size", '-i', type=int, default=224, help='image size')
    parser.add_argument("--checkpoint", "-c", type=str, default="ViT_b16_models/best_model.pt")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    categories = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 4)
    # summary(model, (3, args.image_size, args.image_size))
    model.to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found!")
        exit(0)

    model.eval()
    data_path = args.data_path
    for image_file in os.listdir(data_path):
        image_path = os.path.join(data_path, image_file)
        org_image = cv2.imread(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.image_size, args.image_size))
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = image[None, :, :, :]  # 1 x 3 x 224 x 224
        image = torch.from_numpy(image).to(device).float()
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            output = model(image)
            probs = softmax(output)

        max_idx = torch.argmax(probs)
        predicted_class = categories[max_idx]
        print("The test image is about {} with confident score of {} - The actual image is about {}".format(predicted_class, probs[0, max_idx], image_file))
        cv2.imshow("{}:{:.2f}%".format(predicted_class, probs[0, max_idx] * 100), org_image)
        cv2.waitKey(0)


