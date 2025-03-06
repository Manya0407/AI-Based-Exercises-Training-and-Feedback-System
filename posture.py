import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Define keypoint pairs for drawing skeleton
KEYPOINT_PAIRS = [
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 6), (5, 11), (6, 12),  # Torso
    (11, 12), (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)  # Right leg
]

# Function to detect keypoints
def detect_pose(frame):
    img = cv2.resize(frame, (192, 192))
    img = np.expand_dims(img, axis=0).astype(np.int32)
    
    outputs = movenet(tf.constant(img))
    keypoints = outputs['output_0'].numpy().reshape(17, 3)
    return keypoints

# Function to provide posture feedback
def analyze_posture(keypoints):
    feedback = "Good posture!"
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_ear, right_ear = keypoints[3], keypoints[4]
    
    # Approximate head position using ears
    head_x = (left_ear[1] + right_ear[1]) / 2
    shoulder_avg_x = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_avg_x = (left_hip[1] + right_hip[1]) / 2
    
    # Detect slouching (side view: head, shoulders, and hips should be aligned)
    if abs(head_x - shoulder_avg_x) > 0.05 or abs(shoulder_avg_x - hip_avg_x) > 0.05:
        return "Straighten your back! You're slouching."
    
    # Detect uneven shoulders
    if abs(left_shoulder[0] - right_shoulder[0]) > 0.05:
        return "Your shoulders are uneven!"
    
    # Detect if torso is not straight (hips should be aligned)
    if abs(left_hip[0] - right_hip[0]) > 0.05:
        return "Keep your waist straight! Your torso is uneven."
    
    return feedback

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    keypoints = detect_pose(frame)
    feedback = analyze_posture(keypoints)
    
    # Draw keypoints and skeleton
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > 0.3:
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
    
    for p1, p2 in KEYPOINT_PAIRS:
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > 0.3 and c2 > 0.3:
            pt1 = (int(x1 * frame.shape[1]), int(y1 * frame.shape[0]))
            pt2 = (int(x2 * frame.shape[1]), int(y2 * frame.shape[0]))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    
    # Display feedback
    cv2.putText(frame, feedback, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show output
    cv2.imshow('Real-Time Pose Estimation & Feedback', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
