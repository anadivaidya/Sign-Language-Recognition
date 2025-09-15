import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from collections import deque
from statistics import mode

# Load the trained model
MODEL_PATH = "model/sign_language_model_mobilenet_tuned.h5"
model = load_model(MODEL_PATH)

# Load the label encoder
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
label_encoder = LabelEncoder()
label_encoder.fit(LABELS)

# Define the video capture
cap = cv2.VideoCapture(0)

# Define preprocessing function
def preprocess_frame(frame):
    # Resize to 64x64 (model input size)
    resized = cv2.resize(frame, (64, 64))
    # Convert to RGB
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    normalized = rgb_frame / 255.0
    # Expand dimensions to match model input
    return np.expand_dims(normalized, axis=0)

# Define a fixed ROI
ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = 100, 100, 300, 300

# Initialize a deque to store predictions for smoothing
SMOOTHING_WINDOW = 10
predictions_queue = deque(maxlen=SMOOTHING_WINDOW)

# Update the main loop to include enhanced smoothing
print("Starting real-world prediction with enhanced smoothing. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Extract the fixed ROI
    hand_roi = frame[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]

    # Preprocess the ROI
    preprocessed_frame = preprocess_frame(hand_roi)

    # Make prediction
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions)
    predictions_queue.append(predicted_class)

    # Apply enhanced smoothing using mode
    if len(predictions_queue) == SMOOTHING_WINDOW:
        smoothed_prediction = mode(predictions_queue)
        predicted_label = label_encoder.inverse_transform([smoothed_prediction])[0]

        # Display the prediction on the frame
        cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X+ROI_WIDTH, ROI_Y+ROI_HEIGHT), (255, 0, 0), 2)
        cv2.putText(frame, f"Prediction: {predicted_label}", (ROI_X, ROI_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-World Prediction", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()