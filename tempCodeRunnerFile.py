import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time

# Load MoveNet model for pose detection
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Define keypoint indices for relevant body parts
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Function to detect pose from camera frame
def detect_pose(frame):
    # Convert to RGB and resize for the model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192)
    input_img = tf.cast(input_img, dtype=tf.int32)
    
    # Get keypoints from model
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

# Class to track squats
class SquatTracker:
    def __init__(self):
        self.count = 0
        self.stage = "up"  # Start in standing position
        self.feedback = "Stand straight, face the camera"
        self.knee_angle = 0
        self.hip_angle = 0
        self.last_state_change = time.time()
        self.debounce_time = 0.5  # Debounce time in seconds
        
    def process(self, keypoint_coords):
        current_time = time.time()
        
        # Need hips, knees, and ankles to track squats
        if not all(k in keypoint_coords for k in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
            self.feedback = "Position your full body in the frame (side view is best)"
            return
            
        # Calculate knee angles (average of left and right)
        left_knee_angle = calculate_angle(
            keypoint_coords[LEFT_HIP],
            keypoint_coords[LEFT_KNEE],
            keypoint_coords[LEFT_ANKLE]
        )
        
        right_knee_angle = calculate_angle(
            keypoint_coords[RIGHT_HIP],
            keypoint_coords[RIGHT_KNEE],
            keypoint_coords[RIGHT_ANKLE]
        )
        
        # Average knee angle
        self.knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # State machine for counting squats
        if self.knee_angle < 120 and self.stage == "up" and current_time - self.last_state_change > self.debounce_time:
            self.stage = "down"
            self.feedback = "Good squat depth! Now stand back up"
            self.last_state_change = current_time
            
        if self.knee_angle > 160 and self.stage == "down" and current_time - self.last_state_change > self.debounce_time:
            self.stage = "up"
            self.count += 1
            self.feedback = f"Great squat! Count: {self.count}"
            self.last_state_change = current_time
            
        # Form feedback based on knee angle
        if self.stage == "up" and 140 < self.knee_angle < 170:
            self.feedback = "Stand up straight to start"
        elif self.stage == "down" and self.knee_angle > 130:
            self.feedback = "Go lower for a full squat"
        elif self.stage == "down" and self.knee_angle < 80:
            self.feedback = "Be careful not to go too low"
    
    def reset(self):
        self.count = 0
        self.stage = "up"
        self.feedback = "Counter reset. Ready to start!"

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Create fullscreen window
    cv2.namedWindow('Squat Counter', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Squat Counter', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    tracker = SquatTracker()
    
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
        
        # Draw skeleton overlay on frame
        for idx, keypoint in enumerate(keypoints):
            y, x, confidence = keypoint
            if confidence > 0.2:  # Lower confidence threshold
                px, py = int(x * width), int(y * height)
                keypoint_coords[idx] = (px, py)
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
        
        # Process keypoints to track squats
        tracker.process(keypoint_coords)
        
        # Draw connections between keypoints for better visualization
        leg_connections = [
            (LEFT_HIP, LEFT_KNEE),
            (LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE),
            (RIGHT_KNEE, RIGHT_ANKLE),
            (LEFT_HIP, RIGHT_HIP)
        ]
        
        for connection in leg_connections:
            if connection[0] in keypoint_coords and connection[1] in keypoint_coords:
                cv2.line(frame, keypoint_coords[connection[0]], 
                         keypoint_coords[connection[1]], (0, 165, 255), 2)
        
        # Display information on frame with larger text for fullscreen
        cv2.putText(frame, f"Squats: {tracker.count}", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                   
        cv2.putText(frame, f"Knee Angle: {int(tracker.knee_angle)}°", (30, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                   
        cv2.putText(frame, tracker.feedback, (30, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Progress bar for squat depth
        bar_x = 30
        bar_y = height - 150
        bar_width = 300
        bar_height = 30
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Dynamic bar based on knee angle (180° = standing straight, 90° = full squat)
        # Convert angle to percentage of the bar (inverted since smaller angle = deeper squat)
        angle_range = 90  # From 180° to 90°
        progress = min(1.0, max(0.0, (180 - tracker.knee_angle) / angle_range))
        progress_width = int(bar_width * progress)
        
        # Color changes from yellow to green based on squat depth
        if progress < 0.3:
            color = (0, 165, 255)  # Orange - not deep enough
        elif progress > 0.8:
            color = (0, 165, 255)  # Orange - too deep
        else:
            color = (0, 255, 0)  # Green - good depth
            
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     color, -1)
        
        cv2.putText(frame, "Squat Depth", (bar_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add instructions at the bottom
        instructions = "Press 'r' to reset counter | Press 'q' to quit | Stand sideways for best results"
        cv2.putText(frame, instructions, (30, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Squat Counter', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
        elif key == 27:  # ESC key to exit fullscreen
            cv2.setWindowProperty('Squat Counter', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()