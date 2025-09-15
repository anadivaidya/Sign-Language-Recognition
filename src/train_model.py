import os
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Update data directory to use augmented data
DATA_DIR = "data/augmented/"

# Load preprocessed data
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train_augmented.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train_augmented.npy"))

    # Ensure the dataset has three channels for RGB
    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension if missing
    X_train = np.repeat(X_train, 3, axis=-1)  # Repeat grayscale to RGB

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = to_categorical(label_encoder.fit_transform(y_train))

    return X_train, y_train, label_encoder

# Build the model using MobileNetV2
def build_model(input_shape, num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

    # Unfreeze more layers for fine-tuning
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Train the model
def train_model():
    print("Loading data...")
    X_train, y_train, label_encoder = load_data()

    print("Shape of X_train:", X_train.shape)  # Debugging line to check the shape of X_train

    print("Building model...")
    model = build_model(X_train.shape[1:], y_train.shape[1])

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=30,  # Increased epochs for better convergence
        batch_size=16  # Reduced batch size for finer updates
    )

    print("Saving model...")
    model.save("model/sign_language_model_mobilenet_tuned.h5")
    print("Model saved to model/sign_language_model_mobilenet_tuned.h5")

    return history, model, label_encoder

# Analyze misclassifications
def analyze_misclassifications(model, X_train, y_train, label_encoder):
    print("Analyzing misclassifications...")

    # Make predictions
    y_pred = model.predict(X_train)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_train, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_labels = label_encoder.classes_

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=cm_labels))

if __name__ == "__main__":
    # Load the saved model
    model = load_model("model/sign_language_model_mobilenet_tuned.h5")
    print("Loaded saved model.")

    # Analyze misclassifications
    X_train, y_train, label_encoder = load_data()
    analyze_misclassifications(model, X_train, y_train, label_encoder)