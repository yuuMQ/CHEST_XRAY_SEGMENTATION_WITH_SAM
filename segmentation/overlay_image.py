from utils import *
import os

images_data = "../data"
masks_data = "masks_predicted"
categories = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]

for category in categories:
    images_folder = sorted(os.listdir(os.path.join(images_data, category, "test_images")))
    masks_folder = sorted(os.listdir(os.path.join(masks_data, category)))
    for image_file, mask_file in zip(images_folder, masks_folder):
        image_path = os.path.join(images_data, category, "test_images", image_file)
        mask_path = os.path.join(masks_data, category, mask_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        draw_bounding_box(image, mask)

