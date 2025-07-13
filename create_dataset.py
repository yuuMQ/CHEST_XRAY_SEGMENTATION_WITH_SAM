import os
import shutil
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_dataset(data_path, output_path, category, all_image_output):
    category = str(category)
    all_images_path = os.path.join(output_path, all_image_output)
    path = os.path.join(data_path, category)
    output_category_path = os.path.join(output_path, category)

    if os.path.exists(output_category_path):
        shutil.rmtree(output_category_path)

    if not os.path.exists(all_images_path):
        os.makedirs(all_images_path)
        os.makedirs(os.path.join(all_images_path, "train_images"))
        os.makedirs(os.path.join(all_images_path, "train_masks"))
        os.makedirs(os.path.join(all_images_path, "val_images"))
        os.makedirs(os.path.join(all_images_path, "val_masks"))
        os.makedirs(os.path.join(all_images_path, "test_images"))
        os.makedirs(os.path.join(all_images_path, "test_masks"))

    os.makedirs(os.path.join(output_category_path, "train_images"))
    os.makedirs(os.path.join(output_category_path, "train_masks"))
    os.makedirs(os.path.join(output_category_path, "val_images"))
    os.makedirs(os.path.join(output_category_path, "val_masks"))
    os.makedirs(os.path.join(output_category_path, "test_images"))
    os.makedirs(os.path.join(output_category_path, "test_masks"))



    folders = sorted(os.listdir(path))
    images_path = os.path.join(path, folders[0])
    masks_path = os.path.join(path, folders[1])

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    # Chia train/val/test: 80% train, 10% val, 10% test
    image_train_valid, image_test, mask_train_valid, mask_test = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=42
    )
    image_train, image_val, mask_train, mask_val = train_test_split(
        image_train_valid, mask_train_valid, test_size=0.1111, random_state=42
    )

    # Train
    for image_file, mask_file in tqdm(zip(image_train, mask_train), total=len(image_train), desc=f"Copying train {category}", colour='green'):
        img_src = os.path.join(images_path, image_file)
        mask_src = os.path.join(masks_path, mask_file)
        img_dst = os.path.join(output_category_path, "train_images", image_file)
        mask_dst = os.path.join(output_category_path, "train_masks", mask_file)

        all_img_dst = os.path.join(all_images_path, "train_images", image_file)
        all_mask_dst = os.path.join(all_images_path, "train_masks", mask_file)

        img = cv2.imread(img_src)
        mask = cv2.imread(mask_src)
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        cv2.imwrite(img_dst, img)
        cv2.imwrite(mask_dst, mask)
        cv2.imwrite(all_img_dst, img)
        cv2.imwrite(all_mask_dst, mask)
    # Val
    for image_file, mask_file in tqdm(zip(image_val, mask_val), total=len(image_val), desc=f"Copying val {category}", colour='green'):
        img_src = os.path.join(images_path, image_file)
        mask_src = os.path.join(masks_path, mask_file)
        img_dst = os.path.join(output_category_path, "val_images", image_file)
        mask_dst = os.path.join(output_category_path, "val_masks", mask_file)

        all_img_dst = os.path.join(all_images_path, "val_images", image_file)
        all_mask_dst = os.path.join(all_images_path, "val_masks", mask_file)

        img = cv2.imread(img_src)
        mask = cv2.imread(mask_src)
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        cv2.imwrite(img_dst, img)
        cv2.imwrite(mask_dst, mask)
        cv2.imwrite(all_img_dst, img)
        cv2.imwrite(all_mask_dst, mask)

    # Test
    for image_file, mask_file in tqdm(zip(image_test, mask_test), total=len(image_test), desc=f"Copying test {category}", colour='green'):
        img_src = os.path.join(images_path, image_file)
        mask_src = os.path.join(masks_path, mask_file)
        img_dst = os.path.join(output_category_path, "test_images", image_file)
        mask_dst = os.path.join(output_category_path, "test_masks", mask_file)

        all_img_dst = os.path.join(all_images_path, "test_images", image_file)
        all_mask_dst = os.path.join(all_images_path, "test_masks", mask_file)

        img = cv2.imread(img_src)
        mask = cv2.imread(mask_src)
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        cv2.imwrite(img_dst, img)
        cv2.imwrite(mask_dst, mask)
        cv2.imwrite(all_img_dst, img)
        cv2.imwrite(all_mask_dst, mask)

if __name__ == '__main__':
    if not os.path.exists("data"):
        os.mkdir("data")
    all_image_output = "lung_xray"
    data_path = "./dataset/COVID-19_Radiography_Dataset"
    output_path = "./data"
    categories = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]
    for category in categories:
        create_dataset(data_path, output_path, category, all_image_output)
