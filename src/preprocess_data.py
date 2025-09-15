import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Define paths
DATA_DIR = "data/train/"
OUTPUT_DIR = "data/preprocessed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preprocessing parameters
IMG_SIZE = (64, 64)

# Load and preprocess images
def preprocess_images():
    X = []
    y = []

    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.abspath(os.path.join(DATA_DIR, class_name))
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            img_path = os.path.abspath(os.path.join(class_dir, img_name))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            try:
                img = cv2.resize(img, IMG_SIZE)
                X.append(img)
                y.append(class_name)
            except cv2.error as e:
                print(f"Error resizing image {img_path}: {e}")

    X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0  # Normalize
    y = np.array(y)

    return X, y

# Split data into train, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Save preprocessed data
def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# Augment images manually using OpenCV
def augment_images():
    augmented_dir = os.path.abspath(os.path.join(DATA_DIR, "augmented"))
    os.makedirs(augmented_dir, exist_ok=True)

    for class_name in os.listdir(DATA_DIR):
        class_dir = os.path.abspath(os.path.join(DATA_DIR, class_name))
        if not os.path.isdir(class_dir):
            continue

        # Check if the directory contains any images
        image_files = [fname for fname in os.listdir(class_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"Skipping empty directory: {class_dir}")
            continue

        print(f"Found {len(image_files)} images in {class_dir}. Initializing augmentation...")

        class_augmented_dir = os.path.abspath(os.path.join(augmented_dir, class_name))
        os.makedirs(class_augmented_dir, exist_ok=True)

        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            # Apply augmentations
            augmented_images = []

            # Flip
            if random.random() < 0.5:
                augmented_images.append(cv2.flip(img, 1))

            # Rotate
            if random.random() < 0.7:
                angle = random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                augmented_images.append(cv2.warpAffine(img, M, (w, h)))

            # Zoom
            if random.random() < 0.5:
                scale = random.uniform(0.8, 1.0)
                h, w = img.shape[:2]
                nh, nw = int(h * scale), int(w * scale)
                if nh > 0 and nw > 0:
                    resized = cv2.resize(img, (nw, nh))
                    dh, dw = (h - nh) // 2, (w - nw) // 2
                    zoomed = cv2.copyMakeBorder(resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    augmented_images.append(zoomed)

            # Brightness
            if random.random() < 0.5:
                factor = random.uniform(0.7, 1.3)
                bright = cv2.convertScaleAbs(img, alpha=factor, beta=0)
                augmented_images.append(bright)

            # Contrast
            if random.random() < 0.5:
                factor = random.uniform(0.7, 1.3)
                contrast = cv2.convertScaleAbs(img, alpha=factor, beta=0)
                augmented_images.append(contrast)

            # Save augmented images
            for i, aug_img in enumerate(augmented_images):
                aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                aug_img_path = os.path.join(class_augmented_dir, aug_img_name)
                cv2.imwrite(aug_img_path, aug_img)

        print(f"Augmentation complete for class: {class_name}")

# Main function
def main():
    print("Augmenting images...")
    augment_images()
    print("Preprocessing images...")
    X, y = preprocess_images()
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print("Saving preprocessed data...")
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()