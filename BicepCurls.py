import cv2
import numpy as np
import time
import traceback
import os
from model_handler import (
    KEYPOINTS, detect_pose, calculate_angle, 
    create_fullscreen_window, initialize_webcam
)

# Define keypoint indices for relevant body parts
LEFT_SHOULDER = KEYPOINTS['LEFT_SHOULDER']
LEFT_ELBOW = KEYPOINTS['LEFT_ELBOW']
LEFT_WRIST = KEYPOINTS['LEFT_WRIST']
RIGHT_SHOULDER = KEYPOINTS['RIGHT_SHOULDER']
RIGHT_ELBOW = KEYPOINTS['RIGHT_ELBOW']
RIGHT_WRIST = KEYPOINTS['RIGHT_WRIST']

# Class to track bicep curls
class BicepCurlTracker:
    def __init__(self):
        self.left_count = 0
        self.right_count = 0
        self.left_stage = "down"  # Start in down position
        self.right_stage = "down"  # Start in down position
        self.feedback = "Start with arms down"
        self.left_angle = 0
        self.right_angle = 0
        self.last_left_change = time.time()
        self.last_right_change = time.time()
        self.debounce_time = 0.5  # Debounce time in seconds
        self.form_feedback = []
        
    def process(self, keypoint_coords):
        current_time = time.time()
        
        # Need shoulders, elbows, and wrists to track curls
        if not all(k in keypoint_coords for k in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]):
            self.feedback = "Position your full body in the frame"
            return
            
        # Calculate angles for both arms
        self.left_angle = calculate_angle(
            keypoint_coords[LEFT_SHOULDER],
            keypoint_coords[LEFT_ELBOW],
            keypoint_coords[LEFT_WRIST]
        )
        
        self.right_angle = calculate_angle(
            keypoint_coords[RIGHT_SHOULDER],
            keypoint_coords[RIGHT_ELBOW],
            keypoint_coords[RIGHT_WRIST]
        )
        
        # State machine for left arm
        if self.left_angle > 160 and self.left_stage == "up" and current_time - self.last_left_change > self.debounce_time:
            self.left_stage = "down"
            self.last_left_change = current_time
            
        if self.left_angle < 30 and self.left_stage == "down" and current_time - self.last_left_change > self.debounce_time:
            self.left_stage = "up"
            self.left_count += 1
            self.last_left_change = current_time
            
        # State machine for right arm
        if self.right_angle > 160 and self.right_stage == "up" and current_time - self.last_right_change > self.debounce_time:
            self.right_stage = "down"
            self.last_right_change = current_time
            
        if self.right_angle < 30 and self.right_stage == "down" and current_time - self.last_right_change > self.debounce_time:
            self.right_stage = "up"
            self.right_count += 1
            self.last_right_change = current_time
            
        # Form feedback
        if self.left_stage == "down" and self.left_angle < 160:
            self.feedback = "Straighten your left arm completely"
        elif self.left_stage == "up" and self.left_angle > 30:
            self.feedback = "Lift your left arm higher"
        elif self.right_stage == "down" and self.right_angle < 160:
            self.feedback = "Straighten your right arm completely"
        elif self.right_stage == "up" and self.right_angle > 30:
            self.feedback = "Lift your right arm higher"
        elif abs(self.left_angle - self.right_angle) > 15:
            self.feedback = "Keep both arms at the same height"
        else:
            self.feedback = "Good form!"
            
        # Record form feedback for summary
        if self.feedback not in self.form_feedback:
            self.form_feedback.append(self.feedback)
    
    def reset(self):
        self.left_count = 0
        self.right_count = 0
        self.left_stage = "down"
        self.right_stage = "down"
        self.feedback = "Counter reset. Ready to start!"
        self.form_feedback = []

def run_bicep_curls():
    # Initialize webcam
    cap = initialize_webcam()
    if cap is None:
        print("Exiting due to webcam initialization failure.")
        return False
    
    # Create fullscreen window
    if not create_fullscreen_window('Bicep Curls Counter'):
        print("Failed to create fullscreen window")
        return False
    
    tracker = BicepCurlTracker()
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
                
                # Detect pose
                keypoints = detect_pose(frame)
                if keypoints is None:
                    # If detection failed, still show the frame with a message
                    cv2.putText(frame, "Pose detection failed - adjust lighting or position", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Bicep Curls Counter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Convert keypoints to pixel coordinates
                height, width, _ = frame.shape
                keypoint_coords = {}
                
                # Process keypoints with confidence filtering
                for idx, keypoint in enumerate(keypoints):
                    y, x, confidence = keypoint
                    if confidence > 0.3:  # Only use keypoints with sufficient confidence
                        px, py = int(x * width), int(y * height)
                        keypoint_coords[idx] = (px, py)
                        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                
                # Process keypoints to track curls
                tracker.process(keypoint_coords)
                
                # Draw connections between keypoints for better visualization
                arm_connections = [
                    (LEFT_SHOULDER, LEFT_ELBOW),
                    (LEFT_ELBOW, LEFT_WRIST),
                    (RIGHT_SHOULDER, RIGHT_ELBOW),
                    (RIGHT_ELBOW, RIGHT_WRIST)
                ]
                
                for connection in arm_connections:
                    if connection[0] in keypoint_coords and connection[1] in keypoint_coords:
                        cv2.line(frame, keypoint_coords[connection[0]], 
                                 keypoint_coords[connection[1]], (0, 165, 255), 2)
                
                # Display information on frame
                cv2.putText(frame, f"Left Arm: {tracker.left_count}", (30, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(frame, f"Right Arm: {tracker.right_count}", (30, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(frame, f"Left Elbow Angle: {tracker.left_angle:.1f}°", (30, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"Right Elbow Angle: {tracker.right_angle:.1f}°", (30, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, tracker.feedback, (30, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(frame, f"FPS: {fps:.1f}", (30, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                
                # Add instructions at the bottom
                instructions = "Press 'q' to quit | Keep elbows close to body"
                cv2.putText(frame, instructions, (30, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Bicep Curls Counter', frame)
                
                # Handle keyboard input
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
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
        
        # Ensure the "Records" folder exists
        records_folder = "Records"
        if not os.path.exists(records_folder):
            os.makedirs(records_folder)
        
        # Generate a summary text file inside "Records" folder
        summary_path = os.path.join(records_folder, "bicep_curls.txt")
        
        # Generate a summary text file
        try:
            # Read existing content if file exists
            existing_content = ""
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    existing_content = f.read()
            
            with open(summary_path, "w") as f:
                # Write separator and new session header
                f.write("\n" + "="*50 + "\n")
                f.write(f"Session: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                
                # Write session details
                f.write("Bicep Curls Exercise Summary\n")
                f.write("---------------------------\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {elapsed_time:.1f} seconds\n")
                f.write(f"Left Arm Curls: {tracker.left_count}\n")
                f.write(f"Right Arm Curls: {tracker.right_count}\n")
                f.write(f"Left Elbow Angle Range: {tracker.left_angle:.1f}°\n")
                f.write(f"Right Elbow Angle Range: {tracker.right_angle:.1f}°\n\n")
                
                # Write form feedback
                f.write("Form Feedback:\n")
                f.write("-------------\n")
                for feedback in tracker.form_feedback:
                    f.write(f"- {feedback}\n")
                f.write("\n")
                
                # Write existing content
                f.write(existing_content)
                
            print(f"Summary saved to {summary_path}")
        except Exception as e:
            print(f"Error saving summary: {e}")
        
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Program terminated")
        return True

if __name__ == "__main__":
    run_bicep_curls()
