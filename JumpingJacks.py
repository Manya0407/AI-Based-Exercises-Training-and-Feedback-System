import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import os

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Define keypoint indices
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Detect keypoints
def detect_pose(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192)
    input_img = tf.cast(input_img, dtype=tf.int32)
    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy().reshape((17, 3))
    return keypoints

# Calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Jumping Jacks Tracker Class
class JumpingJacksTracker:
    def __init__(self):
        self.count = 0
        self.stage = "ready"
        self.feedback = "Start jumping jacks!"
        self.summary = []
        self.last_state_change = time.time()
        self.debounce_time = 0.5

    def process(self, keypoint_coords):
        current_time = time.time()

        # Check if required points are visible
        required_points = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]
        if not all(point in keypoint_coords for point in required_points):
            self.feedback = "Ensure full body is visible"
            return

        # Get coordinates
        left_shoulder = keypoint_coords[LEFT_SHOULDER]
        right_shoulder = keypoint_coords[RIGHT_SHOULDER]
        left_wrist = keypoint_coords[LEFT_WRIST]
        right_wrist = keypoint_coords[RIGHT_WRIST]
        left_ankle = keypoint_coords[LEFT_ANKLE]
        right_ankle = keypoint_coords[RIGHT_ANKLE]

        # Calculate angles
        arm_angle = calculate_angle(left_wrist, left_shoulder, right_wrist)
        leg_distance = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))

        # Provide detailed feedback
        if arm_angle > 150:
            self.feedback = "Arms fully extended upwards"
        elif arm_angle < 30:
            self.feedback = "Arms fully down"
        else:
            self.feedback = "Keep arms moving symmetrically"

        # Leg stance feedback
        if leg_distance < 100:
            self.feedback += " | Widen your legs"
        elif leg_distance > 300:
            self.feedback += " | Bring legs closer together"

        # State machine for counting repetitions
        if self.stage == "ready" and arm_angle > 150 and leg_distance > 200:
            self.stage = "jumping"
            self.feedback = "Jump up and bring arms down!"

        if (self.stage == "jumping" and 
            arm_angle < 30 and 
            leg_distance < 150 and 
            current_time - self.last_state_change > self.debounce_time):
            self.count += 1
            self.stage = "ready"
            self.last_state_change = current_time
            self.feedback = f"Rep completed! Total: {self.count}"

        # Store summary data
        self.summary.append({
            'rep': self.count,
            'arm_angle': arm_angle,
            'leg_distance': leg_distance,
            'feedback': self.feedback
        })

        return arm_angle, leg_distance

    def save_summary(self, duration):
        # Define save directory and file path
        save_dir = "Records"
        file_path = os.path.join(save_dir, "jumpingjacks.txt")

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save summary in the specified path
        with open(file_path, "w") as f:
            f.write(f"Jumping Jacks Summary\n")
            f.write(f"Duration: {duration:.1f} seconds\n")
            f.write(f"Total Reps: {self.count}\n\n")
            f.write("Detailed Rep Analysis:\n")
            for rep in self.summary:
                f.write(f"Rep {rep['rep']}: Arm Angle {rep['arm_angle']:.1f}° | "
                        f"Leg Distance {rep['leg_distance']:.1f}px\n")
                f.write(f"Feedback: {rep['feedback']}\n")

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Jumping Jacks Tracker', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Jumping Jacks Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    tracker = JumpingJacksTracker()
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
            if confidence > 0.2:
                px, py = int(x * width), int(y * height)
                keypoint_coords[idx] = (px, py)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

        result = tracker.process(keypoint_coords)

        cv2.putText(frame, f"Jumping Jacks", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Count: {tracker.count}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, tracker.feedback, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        elapsed = time.time() - start_time
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Jumping Jacks Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    tracker.save_summary(elapsed)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
