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
LEFT_HIP = KEYPOINTS['LEFT_HIP']
RIGHT_HIP = KEYPOINTS['RIGHT_HIP']
LEFT_KNEE = KEYPOINTS['LEFT_KNEE']
RIGHT_KNEE = KEYPOINTS['RIGHT_KNEE']
LEFT_ANKLE = KEYPOINTS['LEFT_ANKLE']
RIGHT_ANKLE = KEYPOINTS['RIGHT_ANKLE']

# Class to track squats
class SquatTracker:
    def __init__(self):
        self.count = 0
        self.stage = "up"  # Start in standing position
        self.feedback = "Stand straight, face the camera"
        self.left_knee_angle = 0
        self.right_knee_angle = 0
        self.left_hip_angle = 0
        self.right_hip_angle = 0
        self.last_state_change = time.time()
        self.debounce_time = 0.5  # Debounce time in seconds
        self.form_feedback = []
        
    def process(self, keypoint_coords):
        current_time = time.time()
        
        # Need hips, knees, and ankles to track squats
        if not all(k in keypoint_coords for k in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
            self.feedback = "Position your full body in the frame (side view is best)"
            return
            
        # Calculate knee angles
        self.left_knee_angle = calculate_angle(
            keypoint_coords[LEFT_HIP],
            keypoint_coords[LEFT_KNEE],
            keypoint_coords[LEFT_ANKLE]
        )
        
        self.right_knee_angle = calculate_angle(
            keypoint_coords[RIGHT_HIP],
            keypoint_coords[RIGHT_KNEE],
            keypoint_coords[RIGHT_ANKLE]
        )
        
        # Calculate hip angles
        self.left_hip_angle = calculate_angle(
            keypoint_coords[LEFT_KNEE],
            keypoint_coords[LEFT_HIP],
            keypoint_coords[RIGHT_HIP]
        )
        
        self.right_hip_angle = calculate_angle(
            keypoint_coords[RIGHT_KNEE],
            keypoint_coords[RIGHT_HIP],
            keypoint_coords[LEFT_HIP]
        )
        
        # Average knee angle for counting
        avg_knee_angle = (self.left_knee_angle + self.right_knee_angle) / 2
        
        # State machine for counting squats
        if avg_knee_angle < 120 and self.stage == "up" and current_time - self.last_state_change > self.debounce_time:
            self.stage = "down"
            self.feedback = "Good squat depth! Now stand back up"
            self.last_state_change = current_time
            
        if avg_knee_angle > 160 and self.stage == "down" and current_time - self.last_state_change > self.debounce_time:
            self.stage = "up"
            self.count += 1
            self.feedback = f"Great squat! Count: {self.count}"
            self.last_state_change = current_time
            
        # Form feedback based on angles
        if self.stage == "up" and 140 < avg_knee_angle < 170:
            self.feedback = "Stand up straight to start"
        elif self.stage == "down" and avg_knee_angle > 130:
            self.feedback = "Go lower for a full squat"
        elif self.stage == "down" and avg_knee_angle < 80:
            self.feedback = "Be careful not to go too low"
        elif abs(self.left_knee_angle - self.right_knee_angle) > 15:
            self.feedback = "Keep both knees at the same angle"
        elif abs(self.left_hip_angle - self.right_hip_angle) > 15:
            self.feedback = "Keep hips level"
        else:
            self.feedback = "Good form!"
            
        # Record form feedback for summary
        if self.feedback not in self.form_feedback:
            self.form_feedback.append(self.feedback)
    
    def reset(self):
        self.count = 0
        self.stage = "up"
        self.feedback = "Counter reset. Ready to start!"
        self.form_feedback = []

def run_squat():
    # Initialize webcam
    cap = initialize_webcam()
    if cap is None:
        print("Exiting due to webcam initialization failure.")
        return False
    
    # Create fullscreen window
    if not create_fullscreen_window('Squat Counter'):
        print("Failed to create fullscreen window")
        return False
    
    tracker = SquatTracker()
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
                
                # Flip the frame horizontally for a more intuitive mirror view
                frame = cv2.flip(frame, 1)
                
                # Detect pose
                keypoints = detect_pose(frame)
                if keypoints is None:
                    # If detection failed, still show the frame with a message
                    cv2.putText(frame, "Pose detection failed - adjust lighting or position", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Squat Counter', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
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
                cv2.putText(frame, f"Left Knee: {tracker.left_knee_angle:.1f}°", (30, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"Right Knee: {tracker.right_knee_angle:.1f}°", (30, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"Left Hip: {tracker.left_hip_angle:.1f}°", (30, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"Right Hip: {tracker.right_hip_angle:.1f}°", (30, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, tracker.feedback, (30, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(frame, f"FPS: {fps:.1f}", (30, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                
                # Add instructions at the bottom
                instructions = "Press 'q' to quit | Keep back straight, knees over toes"
                cv2.putText(frame, instructions, (30, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow('Squat Counter', frame)
                
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
        summary_path = os.path.join(records_folder, "squats.txt")
        
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
                f.write("Squat Exercise Summary\n")
                f.write("---------------------\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {elapsed_time:.1f} seconds\n")
                f.write(f"Total Squats: {tracker.count}\n")
                f.write(f"Left Knee Angle Range: {tracker.left_knee_angle:.1f}°\n")
                f.write(f"Right Knee Angle Range: {tracker.right_knee_angle:.1f}°\n")
                f.write(f"Left Hip Angle Range: {tracker.left_hip_angle:.1f}°\n")
                f.write(f"Right Hip Angle Range: {tracker.right_hip_angle:.1f}°\n\n")
                
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
    run_squat()