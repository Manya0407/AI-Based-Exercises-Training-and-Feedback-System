import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import traceback
import os

# Model loading with better error handling and fallback mechanism
def load_model():
    try:
        # Try loading lightning model first
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        print("Lightning model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading lightning model: {e}")
        try:
            # Fallback to thunder model
            print("Attempting to load fallback model...")
            model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            print("Thunder model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            return None

# Load MoveNet model
model = load_model()
if model is None:
    print("Could not load any model. Exiting.")
    exit()

movenet = model.signatures['serving_default']

# Keypoint indices
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

# Function to detect pose keypoints with improved error handling
def detect_pose(frame):
    try:
        # Ensure frame is valid
        if frame is None or frame.size == 0:
            print("Invalid frame received")
            return None
            
        # Resize image to prevent memory issues
        if frame.shape[0] > 720 or frame.shape[1] > 1280:
            frame = cv2.resize(frame, (min(1280, frame.shape[1]), min(720, frame.shape[0])))
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize image to expected format
        input_img = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192)
        input_img = tf.cast(input_img, dtype=tf.int32)
        
        # Run inference with timeout protection
        outputs = movenet(input_img)
        keypoints = outputs['output_0'].numpy().reshape((17, 3))
        return keypoints
    except tf.errors.ResourceExhaustedError:
        print("GPU memory exhausted. Try lowering resolution.")
        return None
    except Exception as e:
        print(f"Error in pose detection: {e}")
        return None

# Improved angle calculation function
def calculate_angle(a, b, c):
    try:
        if a is None or b is None or c is None:
            return None
            
        a, b, c = np.array(a), np.array(b), np.array(c)
        
        # Check for zero vectors that would cause division by zero
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return None
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
    except Exception as e:
        print(f"Error in angle calculation: {e}")
        return None

# Enhanced tracker class for wrist rotations
class WristRotationTracker:
    def __init__(self, wrist_name="Right"):
        self.wrist_name = wrist_name
        self.last_position = None
        self.positions_history = []  # Store recent positions for smoothing
        self.clockwise_count = 0
        self.anticlockwise_count = 0
        self.feedback = ""
        self.angle = None
        self.last_update_time = time.time()
        self.min_update_interval = 0.1  # Minimum time between updates in seconds
        self.movement_threshold = 5  # Minimum movement distance to count as movement
    
    def get_smoothed_position(self, position):
        # Add new position to history
        self.positions_history.append(position)
        
        # Keep only the most recent positions
        if len(self.positions_history) > 5:
            self.positions_history.pop(0)
            
        # Calculate smoothed position (average of recent positions)
        if len(self.positions_history) > 0:
            smoothed_x = sum(p[0] for p in self.positions_history) / len(self.positions_history)
            smoothed_y = sum(p[1] for p in self.positions_history) / len(self.positions_history)
            return (int(smoothed_x), int(smoothed_y))
        else:
            return position
    
    def process(self, wrist_position, elbow_position, shoulder_position, frame):
        current_time = time.time()
        
        # Apply smoothing to reduce jitter
        smoothed_position = self.get_smoothed_position(wrist_position)
        
        # Only process if enough time has passed (limits processing rate)
        if current_time - self.last_update_time < self.min_update_interval:
            return
            
        self.last_update_time = current_time
        
        # Calculate angle between shoulder, elbow and wrist
        limb_angle = None
        if shoulder_position is not None and elbow_position is not None:
            limb_angle = calculate_angle(shoulder_position, elbow_position, wrist_position)
            if limb_angle is not None:
                angle_text = f"Arm angle: {int(limb_angle)}°"
                cv2.putText(frame, angle_text, 
                           (elbow_position[0] - 10, elbow_position[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if self.last_position is not None:
            x1, y1 = self.last_position
            x2, y2 = smoothed_position
            
            # Check if movement exceeds threshold
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < self.movement_threshold:
                return
                
            # Calculate angle of movement
            movement_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Determine rotation direction based on the change in angle
            if x2 > x1 and y2 > y1:
                self.feedback = f"{self.wrist_name} wrist rotating clockwise"
                self.clockwise_count += 1
            elif x2 < x1 and y2 < y1:
                self.feedback = f"{self.wrist_name} wrist rotating anticlockwise"
                self.anticlockwise_count += 1
            else:
                self.feedback = f"Maintain smooth {self.wrist_name} wrist movement"
            
            # Display movement angle near the wrist joint
            angle_text = f"{int(movement_angle)}°"
            cv2.putText(frame, angle_text, (x2, y2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            self.feedback = f"Start rotating your {self.wrist_name} wrist"
        
        self.last_position = smoothed_position
    
    def reset(self):
        self.clockwise_count = 0
        self.anticlockwise_count = 0
        self.feedback = f"{self.wrist_name} wrist rotation counters reset"
        self.positions_history = []

# Function to safely initialize webcam
def initialize_webcam():
    try:
        # Try to open the default camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Try alternative indices if default fails
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Opened camera at index {i}")
                    break
        
        if not cap.isOpened():
            print("Could not open any webcam.")
            return None
            
        # Set lower resolution to improve performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        return None

# Initialize webcam
cap = initialize_webcam()
if cap is None:
    print("Exiting due to webcam initialization failure.")
    exit()

# Create window but don't force fullscreen (which can cause issues)
cv2.namedWindow('Wrist Rotation Tracker', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Wrist Rotation Tracker', 800, 600)

# Create trackers for both wrists
right_wrist_tracker = WristRotationTracker("Right")
left_wrist_tracker = WristRotationTracker("Left")
start_time = time.time()
frame_count = 0
last_fps_update = time.time()
fps = 0

try:
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame. Trying to reinitialize camera...")
                cap.release()
                cap = initialize_webcam()
                if cap is None:
                    print("Could not reinitialize camera. Exiting.")
                    break
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Calculate FPS every second
            if current_time - last_fps_update >= 1.0:
                fps = frame_count / (current_time - last_fps_update)
                frame_count = 0
                last_fps_update = current_time
            
            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect pose keypoints
            keypoints = detect_pose(frame)
            if keypoints is None:
                # If detection failed, still show the frame with a message
                cv2.putText(frame, "Pose detection failed - adjust lighting or position", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Wrist Rotation Tracker', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            height, width, _ = frame.shape
            keypoint_coords = {}
            
            # Process keypoints with confidence filtering
            for idx, keypoint in enumerate(keypoints):
                y, x, confidence = keypoint
                if confidence > 0.3:  # Only use keypoints with sufficient confidence
                    px, py = int(x * width), int(y * height)
                    keypoint_coords[idx] = (px, py)
                    cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            
            # Process right wrist if detected
            if RIGHT_WRIST in keypoint_coords:
                cv2.circle(frame, keypoint_coords[RIGHT_WRIST], 10, (255, 0, 0), -1)
                right_elbow = keypoint_coords.get(RIGHT_ELBOW)
                right_shoulder = keypoint_coords.get(RIGHT_SHOULDER)
                right_wrist_tracker.process(
                    keypoint_coords[RIGHT_WRIST], right_elbow, right_shoulder, frame
                )
            
            # Process left wrist if detected
            if LEFT_WRIST in keypoint_coords:
                cv2.circle(frame, keypoint_coords[LEFT_WRIST], 10, (0, 0, 255), -1)
                left_elbow = keypoint_coords.get(LEFT_ELBOW)
                left_shoulder = keypoint_coords.get(LEFT_SHOULDER)
                left_wrist_tracker.process(
                    keypoint_coords[LEFT_WRIST], left_elbow, left_shoulder, frame
                )
            
            # Display feedback and counters with status information
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display rotation counters
            cv2.putText(frame, f"Right CW: {right_wrist_tracker.clockwise_count}", 
                      (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Right CCW: {right_wrist_tracker.anticlockwise_count}", 
                      (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Left CW: {left_wrist_tracker.clockwise_count}", 
                      (width - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Left CCW: {left_wrist_tracker.anticlockwise_count}", 
                      (width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Display feedback
            cv2.putText(frame, right_wrist_tracker.feedback, (10, height - 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, left_wrist_tracker.feedback, (10, height - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 'r' to reset counters", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
            # Show the frame
            cv2.imshow('Wrist Rotation Tracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                right_wrist_tracker.reset()
                left_wrist_tracker.reset()
    
        except Exception as e:
            print(f"Error in main loop: {e}")
            print(traceback.format_exc())
            # Continue running despite errors
            time.sleep(0.1)  # Small delay to prevent rapid error loops
            
except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    # Ensure resources are properly released
    elapsed_time = time.time() - start_time
    
    # Generate a summary text file
    try:
        with open("wrist_rotation_summary.txt", "w") as f:
            f.write("Wrist Rotation Summary\n")
            f.write("=====================\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {elapsed_time:.1f} seconds\n\n")
            f.write("Right Wrist:\n")
            f.write(f"  Clockwise Rotations: {right_wrist_tracker.clockwise_count}\n")
            f.write(f"  Anticlockwise Rotations: {right_wrist_tracker.anticlockwise_count}\n\n")
            f.write("Left Wrist:\n")
            f.write(f"  Clockwise Rotations: {left_wrist_tracker.clockwise_count}\n")
            f.write(f"  Anticlockwise Rotations: {left_wrist_tracker.anticlockwise_count}\n")
        print(f"Summary saved to wrist_rotation_summary.txt")
    except Exception as e:
        print(f"Error saving summary: {e}")
    
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Program terminated")