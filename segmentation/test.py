from transformers import SamModel, SamProcessor
from argparse import ArgumentParser
import os
from utils import *
from tqdm import tqdm
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--images_path", '-ip', type=str, default="test_images", help='image path')
    parser.add_argument("--masks_path", '-mp', type=str, default="test_masks", help='mask path')
    parser.add_argument("--image_size", '-i', type=int, default=512, help='image size')
    parser.add_argument("--checkpoint", "-c", type=str, default="SAM_models/best_model.pt")
    parser.add_argument("--output", '-o', type=str, default="masks_predicted", help='output')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    data = "../data"
    categories = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained('facebook/sam-vit-base')
    processor = SamProcessor.from_pretrained('facebook/sam-vit-base')

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found!")
        exit(0)

    model.to(device)

    model.eval()
    images_path = args.images_path
    masks_path = args.masks_path

    output_path = args.output
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_accuracy, all_dice, all_iou = [], [], []
    for category in categories:
        print("Processing: {}".format(category))
        images_data_path = os.path.join(data, category, images_path)
        masks_data_path = os.path.join(data, category, masks_path)

        images_folder = sorted(os.listdir(images_data_path))
        masks_folder = sorted(os.listdir(masks_data_path))
        for image_file, mask_file in tqdm(zip(images_folder, masks_folder), total=len(images_folder)):
            image = os.path.join(images_data_path, image_file)
            mask = os.path.join(masks_data_path, mask_file)

            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.image_size, args.image_size))

            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
            if image.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            bbox = get_bounding_box(mask)

            inputs = processor(images=image, input_boxes=[[bbox]], return_tensors="pt")
            '''
            input:
            {'pixel_values': tensor([[[[ 0.1768,  0.1768,  0.1939,  ..., -2.1179, -2.1179, -2.1179],
            [ 0.1768,  0.1768,  0.1939,  ..., -2.1179, -2.1179, -2.1179],
            [ 0.1597,  0.1597,  0.1939,  ..., -2.1179, -2.1179, -2.1179],
            ...,
            [-1.7522, -1.7522, -1.7522,  ..., -1.8044, -1.8044, -1.8044],
            [-1.7522, -1.7522, -1.7522,  ..., -1.8044, -1.8044, -1.8044],
            [-1.7522, -1.7522, -1.7522,  ..., -1.8044, -1.8044, -1.8044]]]]),
            'original_sizes': tensor([[299, 299]]),
            'reshaped_input_sizes': tensor([[1024, 1024]]),
            'input_boxes': tensor([[[109.5920,   0.0000, 952.0803, 835.6388]]],
             dtype=torch.float64)}
             pixel_values tensor([[[[ 0.1768,  0.1768,  0.1939,  ..., -2.1179, -2.1179, -2.1179],
            =========================================================================================            
            Example os k and v:
                    k                        v
                original_sizes          tensor([[299, 299]])
                reshaped_input_sizes    tensor([[1024, 1024]])
                input_boxes             tensor([[[109.5920,   0.0000, 952.0803, 835.6388]]], dtype=torch.float64)
            '''
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(pixel_values=inputs["pixel_values"], input_boxes=inputs["input_boxes"], multimask_output=False)
                pred_mask = outputs.pred_masks.squeeze(1)

            pred_mask_tensor = outputs.pred_masks.squeeze()  # shape = [H, W]
            pred_mask_np = (pred_mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255
            os.makedirs(os.path.join(output_path, category), exist_ok=True)
            save_path = os.path.join(output_path, category, f"{os.path.splitext(image_file)[0]}.png")
            cv2.imwrite(save_path, pred_mask_np)
