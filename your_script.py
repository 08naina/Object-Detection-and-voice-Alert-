

import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

# Load YOLOv5 model
model = YOLO('yolov5s.pt')

# Function to generate voice alert
def voice_alert(text):
    engine.say(text)
    engine.runAndWait()

# Function to estimate distance
def estimate_distance(bbox, frame_width):
    object_width_in_pixels = bbox[2] - bbox[0]
    focal_length = 700  # Example focal length in pixels
    real_object_width = 0.5  # Default real object width in meters (adjust per object type if needed)
    distance = (focal_length * real_object_width) / object_width_in_pixels
    return distance

# Function to detect objects and alert
def detect_and_alert(frame):
    results = model.predict(frame, verbose=False)
    # Objects of interest
    target_objects = ["person", "chair", "table", "glass", "bottle", "cell phone"]
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            object_name = model.names[int(class_id)]
            if confidence > 0.5 and object_name in target_objects:
                distance = estimate_distance((x1, y1, x2, y2), frame.shape[1])
                alert_message = (f"Detected {object_name} with confidence {confidence:.2f}. "
                                 f"Estimated distance: {distance:.2f} meters.")
                print(alert_message)
                voice_alert(f"Warning! {object_name} ahead at {distance:.2f} meters.")

# Start video capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for performance
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect and alert
    detect_and_alert(frame_resized)

    # Display the frame
    cv2.imshow("Object Detection", frame_resized)

    # Break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()