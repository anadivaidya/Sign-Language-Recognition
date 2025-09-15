import os
import numpy as np
import cv2

# Define paths
DATA_DIR = "data/preprocessed/"
AUGMENTED_DIR = "data/augmented/"
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# Load data
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    return X_train, y_train

# Augment data using OpenCV
def augment_data(X, y):
    augmented_images = []
    augmented_labels = []

    for i in range(len(X)):
        img = X[i]
        label = y[i]

        # Ensure the image is in the correct format
        img = img.astype(np.uint8)
        original_shape = img.shape[:2]

        # Helper function to resize and ensure consistent type
        def process_image(image):
            resized = cv2.resize(image, (original_shape[1], original_shape[0]))
            return resized.astype(np.uint8)

        # Original image
        augmented_images.append(process_image(img))
        augmented_labels.append(label)

        # Rotate image
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 20, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        augmented_images.append(process_image(rotated))
        augmented_labels.append(label)

        # Flip image horizontally
        flipped = cv2.flip(img, 1)
        augmented_images.append(process_image(flipped))
        augmented_labels.append(label)

        # Add brightness adjustment
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        augmented_images.append(process_image(bright))
        augmented_labels.append(label)

        # Add Gaussian blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        augmented_images.append(process_image(blurred))
        augmented_labels.append(label)

        # Add scaling
        scaled = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
        scaled = scaled[:rows, :cols]  # Crop to original size
        augmented_images.append(process_image(scaled))
        augmented_labels.append(label)

        # Add random noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        noisy = cv2.add(img, noise)
        augmented_images.append(process_image(noisy))
        augmented_labels.append(label)

        # Adjust contrast
        contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        augmented_images.append(process_image(contrast))
        augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train = load_data()

    print("Augmenting data...")
    X_augmented, y_augmented = augment_data(X_train, y_train)

    print("Saving augmented data...")
    np.save(os.path.join(AUGMENTED_DIR, "X_train_augmented.npy"), X_augmented)
    np.save(os.path.join(AUGMENTED_DIR, "y_train_augmented.npy"), y_augmented)
    print("Augmented data saved to", AUGMENTED_DIR)