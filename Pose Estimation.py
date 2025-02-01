import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Function to detect keypoints
def detect_pose(frame):
    img = cv2.resize(frame, (192, 192))  # Resize for MoveNet
    img = np.expand_dims(img, axis=0).astype(np.int32)
    
    # Run inference
    outputs = movenet(tf.constant(img))
    keypoints = outputs['output_0'].numpy().reshape(17, 3)  # 17 keypoints
    return keypoints

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect keypoints
    keypoints = detect_pose(frame)
    
    # Draw keypoints
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > 0.3:
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
    
    # Display output
    cv2.imshow('MoveNet Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
