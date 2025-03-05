import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Define keypoint indices for easier reference
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

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
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Lunge Tracker Class
class LungeTracker:
    def __init__(self):
        self.count = 0
        self.stage = "ready"
        self.last_angle = None
        self.feedback = "Get ready for lunges"
        self.lunge_leg = "right"  # Can be "right" or "left"
        self.summary = []
        self.last_state_change = time.time()
        self.debounce_time = 0.5  # Prevent rapid counting
        
    def process(self, keypoint_coords):
        current_time = time.time()
        
        # Determine which points to use based on current lunge leg
        if self.lunge_leg == "right":
            hip_idx = RIGHT_HIP
            knee_idx = RIGHT_KNEE
            ankle_idx = RIGHT_ANKLE
            opposite_ankle_idx = LEFT_ANKLE
        else:
            hip_idx = LEFT_HIP
            knee_idx = LEFT_KNEE
            ankle_idx = LEFT_ANKLE
            opposite_ankle_idx = RIGHT_ANKLE
        
        # Check if all required points are visible
        required_points = [hip_idx, knee_idx, ankle_idx, opposite_ankle_idx]
        if not all(point in keypoint_coords for point in required_points):
            self.feedback = f"Need full body visibility for {self.lunge_leg} leg lunge"
            return
        
        # Get coordinates
        hip = keypoint_coords[hip_idx]
        knee = keypoint_coords[knee_idx]
        ankle = keypoint_coords[ankle_idx]
        opposite_ankle = keypoint_coords[opposite_ankle_idx]
        
        # Calculate angles and distances
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(knee, hip, (hip[0], hip[1] - 100))
        
        # Calculate distance between ankles to ensure proper lunge stance
        ankle_distance = np.linalg.norm(np.array(ankle) - np.array(opposite_ankle))
        
        # Provide detailed feedback
        if knee_angle > 160:
            self.feedback = "Straighten your back leg more"
        elif knee_angle < 80:
            self.feedback = "Bend your front knee less"
        else:
            self.feedback = "Good knee angle"
        
        # Add stance feedback
        if ankle_distance < 100:
            self.feedback += " | Widen your stance"
        elif ankle_distance > 300:
            self.feedback += " | Bring legs closer together"
        
        # State machine for counting lunges
        if self.stage == "ready" and knee_angle < 120:
            # Entered lunge position
            self.stage = "lunging"
            self.feedback = "Good lunge position"
        
        if (self.stage == "lunging" and 
            knee_angle > 150 and 
            current_time - self.last_state_change > self.debounce_time):
            # Returned to standing
            self.count += 1
            self.stage = "ready"
            self.last_state_change = current_time
            self.feedback = f"Lunge completed! Total: {self.count}"
        
        # Store summary data
        self.summary.append({
            'rep': self.count,
            'knee_angle': knee_angle,
            'ankle_distance': ankle_distance,
            'feedback': self.feedback
        })
        
        return knee_angle, ankle_distance

    def switch_leg(self):
        self.lunge_leg = "left" if self.lunge_leg == "right" else "right"
        self.count = 0
        self.stage = "ready"
        self.feedback = f"Switching to {self.lunge_leg} leg lunges"

    def reset(self):
        self.count = 0
        self.stage = "ready"
        self.summary.clear()
        self.feedback = "Counter reset. Ready to start!"

    def save_summary(self, duration):
        with open("lunge_summary.txt", "w") as f:
            f.write(f"Lunge Workout Summary ({self.lunge_leg} leg)\n")
            f.write(f"Duration: {duration:.1f} seconds\n")
            f.write(f"Total Lunges: {self.count} reps\n\n")
            f.write("Detailed Rep Analysis:\n")
            for rep in self.summary:
                f.write(f"Rep {rep['rep']}: Knee Angle {rep['knee_angle']:.1f}° | "
                        f"Ankle Distance {rep['ankle_distance']:.1f}px\n")
                f.write(f"Feedback: {rep['feedback']}\n")

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Create fullscreen window
    cv2.namedWindow('Lunge Tracker', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Lunge Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize tracker
    tracker = LungeTracker()
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Detect pose
        keypoints = detect_pose(frame)
        
        # Convert keypoints to pixel coordinates
        height, width, _ = frame.shape
        keypoint_coords = {}
        
        # Draw skeleton and store coordinates
        for idx, keypoint in enumerate(keypoints):
            y, x, confidence = keypoint
            if confidence > 0.2:  # Lowered confidence threshold
                px, py = int(x * width), int(y * height)
                keypoint_coords[idx] = (px, py)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
        
        # Draw body connections for visualization
        connections = [
            (LEFT_HIP, LEFT_KNEE),
            (LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE),
            (RIGHT_KNEE, RIGHT_ANKLE)
        ]
        
        for connection in connections:
            if connection[0] in keypoint_coords and connection[1] in keypoint_coords:
                cv2.line(frame, keypoint_coords[connection[0]], 
                         keypoint_coords[connection[1]], (0, 165, 255), 2)
        
        # Process pose for lunges
        result = tracker.process(keypoint_coords)
        
        # Display information
        cv2.putText(frame, f"Lunges ({tracker.lunge_leg} leg)", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Count: {tracker.count}", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, tracker.feedback, (30, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show elapsed time
        elapsed = time.time() - start_time
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (30, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw progress indicator if pose tracking successful
        if result:
            knee_angle, ankle_distance = result
            progress_bar_width = 300
            progress_bar_height = 30
            progress_bar_x = 30
            progress_bar_y = height - 100
            
            # Background bar
            cv2.rectangle(frame, (progress_bar_x, progress_bar_y), 
                         (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height), 
                         (100, 100, 100), -1)
            
            # Knee angle progress (lower angle = deeper lunge)
            max_angle = 160
            min_angle = 80
            knee_progress = max(0, min(1, (max_angle - knee_angle) / (max_angle - min_angle)))
            knee_progress_width = int(progress_bar_width * knee_progress)
            
            cv2.rectangle(frame, (progress_bar_x, progress_bar_y), 
                         (progress_bar_x + knee_progress_width, progress_bar_y + progress_bar_height), 
                         (0, 255, 0), -1)
            cv2.putText(frame, "Knee Angle", (progress_bar_x, progress_bar_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display instructions
        instructions = "Press 's' to switch legs | 'r' to reset | 'q' to quit"
        cv2.putText(frame, instructions, (30, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Lunge Tracker', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            tracker.switch_leg()
        elif key == ord('r'):
            tracker.reset()
        elif key == 27:  # ESC key to exit fullscreen
            cv2.setWindowProperty('Lunge Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    # Save workout summary
    tracker.save_summary(time.time() - start_time)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()