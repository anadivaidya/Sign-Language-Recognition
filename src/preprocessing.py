import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import Augmentor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def load_data(csv_path):
    """Load data from CSV file."""
    data = pd.read_csv(csv_path)
    labels = data['label']
    images = data.drop('label', axis=1).values
    images = images.reshape(-1, 28, 28, 1)  # Reshape for grayscale images
    return images, labels

def normalize_images(images):
    """Normalize pixel values to range [0, 1]."""
    return images / 255.0

def augment_data(images, labels):
    """Apply data augmentation using Augmentor."""
    p = Augmentor.Pipeline()
    p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.flip_left_right(probability=0.5)

    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        p.set_seed(42)
        p.sample(1)
        augmented_images.append(image)
        augmented_labels.append(label)

    return augmented_images, augmented_labels

def split_data(images, labels):
    """Split data into training, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save preprocessed data to files."""
    np.save(os.path.join(Config.DATA_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(Config.DATA_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(Config.DATA_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(Config.DATA_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(Config.DATA_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(Config.DATA_PATH, 'y_test.npy'), y_test)

def main():
    """Main preprocessing pipeline."""
    # Load data
    train_images, train_labels = load_data(Config.TRAIN_CSV)
    test_images, test_labels = load_data(Config.TEST_CSV)

    # Normalize data
    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)

    # Augment data
    train_images, train_labels = augment_data(train_images, train_labels)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(train_images, train_labels)

    # Save preprocessed data
    save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()