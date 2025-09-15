import cv2
import os
import time

def capture_images(class_name, save_dir, num_images=100):
    """Capture images for a specific class, focusing on the palm region."""
    cap = cv2.VideoCapture(0)
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    print("Camera is opening. Place your hand inside the green rectangle.")

    # Add a delay to ensure the camera feed opens properly
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Define ROI for palm (adjust coordinates as needed)
        height, width, _ = frame.shape
        x1, y1, x2, y2 = int(width * 0.3), int(height * 0.3), int(width * 0.7), int(height * 0.7)

        # Draw the green rectangle on the camera feed
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to start capturing", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the camera feed
        cv2.imshow("Camera Feed", frame)

        # Start capturing images when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Starting image capture...")
            break

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Define ROI for palm
        roi = frame[y1:y2, x1:x2]

        # Save the ROI image
        file_path = os.path.join(save_dir, f"{class_name}_{count}.jpg")
        cv2.imwrite(file_path, roi)
        count += 1
        print(f"Captured {count}/{num_images} images")

        # Display the camera feed with ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Camera Feed", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    class_name = input("Enter the class name: ")
    save_dir = f"data/train/{class_name}"
    num_images = int(input("Enter the number of images to capture: "))
    capture_images(class_name, save_dir, num_images)