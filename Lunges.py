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
LEFT_SHOULDER = KEYPOINTS['LEFT_SHOULDER']
RIGHT_SHOULDER = KEYPOINTS['RIGHT_SHOULDER']

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
        self.left_knee_angle = 0
        self.right_knee_angle = 0
        self.left_hip_angle = 0
        self.right_hip_angle = 0
        self.form_feedback = []
        
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
        
        # Calculate angles for both legs
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
        
        # Calculate distance between ankles to ensure proper lunge stance
        ankle_distance = np.linalg.norm(np.array(ankle) - np.array(opposite_ankle))
        
        # Use the appropriate knee angle based on the lunge leg
        knee_angle = self.right_knee_angle if self.lunge_leg == "right" else self.left_knee_angle
        
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
        
        # Record form feedback for summary
        if self.feedback not in self.form_feedback:
            self.form_feedback.append(self.feedback)
        
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
        self.form_feedback = []
        self.feedback = "Counter reset. Ready to start!"

def run_lunges():
    # Initialize webcam
    cap = initialize_webcam()
    if cap is None:
        print("Exiting due to webcam initialization failure.")
        return False
    
    # Create fullscreen window
    if not create_fullscreen_window('Lunge Tracker'):
        print("Failed to create fullscreen window")
        return False
    
    tracker = LungeTracker()
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
                    cv2.imshow('Lunge Tracker', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
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
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(frame, f"Count: {tracker.count}", (30, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(frame, f"Left Knee: {tracker.left_knee_angle:.1f}°", (30, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"Right Knee: {tracker.right_knee_angle:.1f}°", (30, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"Left Hip: {tracker.left_hip_angle:.1f}°", (30, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"Right Hip: {tracker.right_hip_angle:.1f}°", (30, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, tracker.feedback, (30, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(frame, f"FPS: {fps:.1f}", (30, 340), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                
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
                
                # Add instructions at the bottom
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
        summary_path = os.path.join(records_folder, "lunges.txt")
        
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
                f.write("Lunge Exercise Summary\n")
                f.write("---------------------\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {elapsed_time:.1f} seconds\n")
                f.write(f"Total Lunges: {tracker.count}\n")
                f.write(f"Lunge Leg: {tracker.lunge_leg}\n")
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
    run_lunges()