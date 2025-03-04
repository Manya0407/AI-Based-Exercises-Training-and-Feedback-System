import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Keypoint indices
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10

# Function to detect keypoints
def detect_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192)
    input_img = tf.cast(input_img, dtype=tf.int32)
    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy().reshape((17, 3))
    return keypoints

# Calculate angle between three points
def calculate_angle(a, b, c):
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
    except:
        return None

# Bicep Curl Counter for both arms
class BicepCurlCounter:
    def __init__(self, arm):
        self.count = 0
        self.stage = None
        self.last_angle = None
        self.feedback = "Waiting..."
        self.arm = arm
    
    def process(self, shoulder, elbow, wrist):
        angle = calculate_angle(shoulder, elbow, wrist)
        if angle is None:
            self.feedback = "Cannot calculate angle"
            return
        
        self.last_angle = angle
        
        if angle > 140:
            self.stage = "down"
            self.feedback = "Ready for curl"
        elif angle < 60 and self.stage == "down":
            self.stage = "up"
            self.count += 1
            self.feedback = "Good curl!"
        elif 60 <= angle <= 140:
            self.feedback = "Control your arm" if self.stage == "up" else "Keep curling"

# Initialize counters for both arms
right_arm_counter = BicepCurlCounter("right")
left_arm_counter = BicepCurlCounter("left")

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Bicep Curl Tracker', cv2.WINDOW_NORMAL)
font = cv2.FONT_HERSHEY_SIMPLEX
small_font_scale = 0.5
large_font_scale = 0.7
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    keypoints = detect_pose(frame)
    height, width, _ = frame.shape
    keypoint_coords = {}
    
    for idx, keypoint in enumerate(keypoints):
        y, x, confidence = keypoint
        if confidence > 0.3:
            px, py = int(x * width), int(y * height)
            keypoint_coords[idx] = (px, py)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
    
    connections = [
        (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST)
    ]
    
    for connection in connections:
        if connection[0] in keypoint_coords and connection[1] in keypoint_coords:
            cv2.line(frame, keypoint_coords[connection[0]], 
                     keypoint_coords[connection[1]], (0, 255, 255), 2)

    # Right arm processing
    if RIGHT_SHOULDER in keypoint_coords and RIGHT_ELBOW in keypoint_coords and RIGHT_WRIST in keypoint_coords:
        right_arm_counter.process(
            keypoint_coords[RIGHT_SHOULDER],
            keypoint_coords[RIGHT_ELBOW],
            keypoint_coords[RIGHT_WRIST]
        )
        cv2.putText(frame, f"{int(right_arm_counter.last_angle)}°", 
                    keypoint_coords[RIGHT_ELBOW], font, small_font_scale, (255, 255, 0), 2)
    
    # Left arm processing
    if LEFT_SHOULDER in keypoint_coords and LEFT_ELBOW in keypoint_coords and LEFT_WRIST in keypoint_coords:
        left_arm_counter.process(
            keypoint_coords[LEFT_SHOULDER],
            keypoint_coords[LEFT_ELBOW],
            keypoint_coords[LEFT_WRIST]
        )
        cv2.putText(frame, f"{int(left_arm_counter.last_angle)}°", 
                    keypoint_coords[LEFT_ELBOW], font, small_font_scale, (255, 255, 0), 2)

    # UI Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 100), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"Right Arm Curls: {right_arm_counter.count}", (10, 30), font, large_font_scale, (0, 255, 255), 2)
    cv2.putText(frame, f"{right_arm_counter.feedback}", (10, 60), font, small_font_scale, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Left Arm Curls: {left_arm_counter.count}", (10, 90), font, large_font_scale, (0, 255, 255), 2)
    cv2.putText(frame, f"{left_arm_counter.feedback}", (10, 120), font, small_font_scale, (0, 255, 0), 2)
    
    elapsed = time.time() - start_time
    cv2.putText(frame, f"Time: {elapsed:.1f}s", (width - 200, 30), font, large_font_scale, (0, 255, 255), 2)
    
    cv2.imshow('Bicep Curl Tracker', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Generate Workout Summary Text File
summary_text = f"""
Workout Summary:
Duration: {elapsed:.1f} seconds
Right Arm Curls: {right_arm_counter.count} reps
Left Arm Curls: {left_arm_counter.count} reps
Right Arm Feedback: {right_arm_counter.feedback}
Left Arm Feedback: {left_arm_counter.feedback}
"""

with open("workout_summary.txt", "w") as file:
    file.write(summary_text)

print(summary_text)

cap.release()
cv2.destroyAllWindows()
