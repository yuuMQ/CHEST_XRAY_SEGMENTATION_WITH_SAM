import numpy as np
import torch
import cv2


def get_bounding_box(mask):
    y_indices, x_indices = np.where(mask)
    if len(x_indices)==0 or len(y_indices)==0:
        return [0, 0, 1, 1]
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = mask.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    return [x_min, y_min, x_max, y_max]

def draw_bounding_box(image, mask):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8)

    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    box = get_bounding_box(mask)
    x_min, y_min, x_max, y_max = box

    # Overlay
    overlay = image.copy()
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[:, :, 2] = mask
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1, mask_rgb, alpha, 0)

    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Bounding Box Overlay", overlay)
    cv2.waitKey(0)

def calculate_metrics(outputs, targets, smooth=1e-8):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > 0.5).float()
    intersection = (preds * targets).sum()
    accuracy = (preds == targets).sum() / torch.numel(preds)
    dice = (2. * intersection) / (preds.sum() + targets.sum() + smooth)
    iou = intersection / (preds.sum() + targets.sum() - intersection + smooth)
    return accuracy.item(), dice.item(), iou.item()


# CHATBOT AI GENERATED THIS CODE !!!
def save_visualization(image, gt_mask_tensor, pred_mask_tensor, save_path):
    image_np = image
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    gt_mask = gt_mask_tensor.squeeze().cpu().numpy()
    gt_mask = cv2.resize(gt_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_mask_bin = (gt_mask > 0.5).astype(np.uint8)
    gt_mask_gray = (gt_mask_bin * 255).astype(np.uint8)

    pred_mask = pred_mask_tensor.squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
    pred_mask_gray = (pred_mask_bin * 255).astype(np.uint8)


    gt_mask_bgr   = cv2.cvtColor(gt_mask_gray, cv2.COLOR_GRAY2BGR)
    pred_mask_bgr = cv2.cvtColor(pred_mask_gray, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([gt_mask_bgr, pred_mask_bgr])


    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    w = image_np.shape[1]
    cv2.putText(combined, "GT Mask",     (w + 10, 25),     font, scale, color, thickness)
    cv2.putText(combined, "Predicted",   (2 * w + 10, 25), font, scale, color, thickness)
    cv2.imwrite(save_path, combined)

