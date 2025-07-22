import numpy as np
import cv2
import matplotlib.pyplot as plt
def mask_to_box(mask):

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    y_indices, x_indices = np.where(mask)
    if len(x_indices)==0 or len(y_indices)==0:
        return None
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)
    return [x_min, y_min, x_max, y_max]

if __name__ == '__main__':
    mask_file = '../data/COVID/train_masks/COVID-4.png'
    mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    mask = mask_image.astype(float) / 255.0 # Chuan hoa ve 0 - 1
    box = mask_to_box(mask_image)
    x_min, y_min, x_max, y_max = box # Bounding box
    cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Box", mask)
    cv2.waitKey(0)
    exit()
    image_file = '../data/COVID/train_images/COVID-4.png'
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    overlay = image.copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    # BRG -> Giam G, tang R
    overlay[:, :, 1] = overlay[:, :, 1] * (1 - mask)
    overlay[:, :, 2] = overlay[:, :, 2] + (mask * 255)

    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)