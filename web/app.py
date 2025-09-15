from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model("c:/Users/asus/Desktop/Hackathon/SLR/model/sign_language_model_mobilenet_tuned.h5")

# Map class numbers to alphabets
CLASS_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Store the latest prediction globally
latest_prediction = "Waiting for prediction..."

# Define a function to preprocess frames for prediction
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize to model input size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame / 255.0  # Normalize

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_prediction
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess and predict
            preprocessed = preprocess_frame(frame)
            prediction = model.predict(preprocessed)
            predicted_class = np.argmax(prediction)
            predicted_class = CLASS_LABELS.get(predicted_class, "Unknown")

            # Update the latest prediction
            latest_prediction = predicted_class

            # Display prediction on the frame
            cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"predicted_class": latest_prediction})

if __name__ == '__main__':
    app.run(debug=True)