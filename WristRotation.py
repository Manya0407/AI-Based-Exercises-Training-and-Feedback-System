import cv2
import numpy as np
import time
import traceback
import os
from model_handler import (
    KEYPOINTS, detect_pose, calculate_angle, 
    create_fullscreen_window, initialize_webcam
)

# Enhanced tracker class for wrist rotations
class WristRotationTracker:
    def __init__(self, wrist_name="Right"):
        self.wrist_name = wrist_name
        self.last_position = None
        self.positions_history = []
        self.clockwise_count = 0
        self.anticlockwise_count = 0
        self.feedback = ""
        self.angle = None
        self.last_update_time = time.time()
        self.min_update_interval = 0.1
        self.rotation_threshold = 120  # Reduced threshold for easier detection
        self.current_rotation = 0
        self.last_angle = 0
        self.stage = "ready"
        self.rotation_start_angle = None
        self.rotation_direction = None
        self.angle_history = []
        self.min_angle_history = 2  # Reduced for more responsive detection
        self.angle_changes = []
        self.min_angle_change = 5  # Minimum angle change to consider as movement
        self.max_angle_change = 30  # Maximum angle change per frame to prevent noise
        self.continuous_rotation = 0  # Track continuous rotation in same direction

    def get_smoothed_position(self, position):
        self.positions_history.append(position)
        if len(self.positions_history) > 3:
            self.positions_history.pop(0)
        if len(self.positions_history) > 0:
            smoothed_x = sum(p[0] for p in self.positions_history) / len(self.positions_history)
            smoothed_y = sum(p[1] for p in self.positions_history) / len(self.positions_history)
            return (int(smoothed_x), int(smoothed_y))
        return position

    def calculate_angle(self, wrist_pos, elbow_pos):
        if wrist_pos is None or elbow_pos is None:
            return None
        try:
            dx = wrist_pos[0] - elbow_pos[0]
            dy = wrist_pos[1] - elbow_pos[1]
            angle = np.degrees(np.arctan2(dx, -dy))
            return angle
        except:
            return None

    def determine_rotation_direction(self):
        if len(self.angle_changes) < 2:
            return None
            
        # Calculate average change in angle
        avg_change = sum(self.angle_changes) / len(self.angle_changes)
        
        # If the change is too small, ignore it
        if abs(avg_change) < self.min_angle_change:
            return None
            
        return "clockwise" if avg_change > 0 else "anticlockwise"

    def process(self, wrist_position, elbow_position, shoulder_position, frame):
        current_time = time.time()
        if current_time - self.last_update_time < self.min_update_interval:
            return

        self.last_update_time = current_time
        smoothed_position = self.get_smoothed_position(wrist_position)
        
        # Calculate current angle
        current_angle = self.calculate_angle(smoothed_position, elbow_position)
        if current_angle is None:
            self.feedback = f"Keep your {self.wrist_name} wrist visible"
            return

        # Calculate angle change
        if self.last_angle is not None:
            angle_diff = current_angle - self.last_angle
            
            # Normalize angle difference to handle angle wrap-around
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
                
            # Filter out noise and large jumps
            if abs(angle_diff) < self.max_angle_change:
                self.angle_changes.append(angle_diff)
                if len(self.angle_changes) > 5:
                    self.angle_changes.pop(0)
                
                # Update continuous rotation
                if len(self.angle_changes) >= 2:
                    if all(change > 0 for change in self.angle_changes[-2:]):
                        self.continuous_rotation += angle_diff
                    elif all(change < 0 for change in self.angle_changes[-2:]):
                        self.continuous_rotation += angle_diff
                    else:
                        self.continuous_rotation = 0

        # State machine for rotation detection
        if self.stage == "ready":
            if self.last_position is not None:
                self.rotation_start_angle = current_angle
                self.rotation_direction = None
                self.current_rotation = 0
                self.stage = "rotating"
                self.feedback = f"Start rotating your {self.wrist_name} wrist"
            else:
                self.feedback = f"Start rotating your {self.wrist_name} wrist"

        elif self.stage == "rotating":
            # Determine rotation direction if not set
            if self.rotation_direction is None:
                self.rotation_direction = self.determine_rotation_direction()
            
            # Check if rotation is complete
            if abs(self.continuous_rotation) >= self.rotation_threshold:
                if self.rotation_direction == "clockwise":
                    self.clockwise_count += 1
                    self.feedback = f"{self.wrist_name} wrist clockwise rotation completed!"
                else:
                    self.anticlockwise_count += 1
                    self.feedback = f"{self.wrist_name} wrist anticlockwise rotation completed!"
                
                # Reset for next rotation
                self.stage = "ready"
                self.rotation_start_angle = None
                self.rotation_direction = None
                self.current_rotation = 0
                self.continuous_rotation = 0
                self.angle_changes.clear()
            else:
                # Provide feedback during rotation
                if self.rotation_direction:
                    progress = min(1.0, abs(self.continuous_rotation) / self.rotation_threshold)
                    self.feedback = f"{self.wrist_name} wrist rotating {self.rotation_direction} ({int(progress * 100)}%)"
                else:
                    self.feedback = f"Start rotating your {self.wrist_name} wrist"

        # Draw rotation progress
        if self.stage == "rotating" and self.rotation_direction:
            progress = min(1.0, abs(self.continuous_rotation) / self.rotation_threshold)
            radius = 20
            center = (smoothed_position[0], smoothed_position[1] - 30)
            cv2.circle(frame, center, radius, (255, 255, 255), 2)
            end_angle = int(360 * progress)
            cv2.ellipse(frame, center, (radius, radius), 0, 0, end_angle, (0, 255, 0), 2)

        self.last_position = smoothed_position
        self.last_angle = current_angle

    def reset(self):
        self.clockwise_count = 0
        self.anticlockwise_count = 0
        self.feedback = f"{self.wrist_name} wrist rotation counters reset"
        self.positions_history = []
        self.angle_history = []
        self.angle_changes = []
        self.stage = "ready"
        self.current_rotation = 0
        self.continuous_rotation = 0
        self.rotation_start_angle = None
        self.rotation_direction = None

def run_wrist_rotation():
    # Initialize webcam
    cap = initialize_webcam()
    if cap is None:
        print("Exiting due to webcam initialization failure.")
        return False
    
    # Create fullscreen window
    if not create_fullscreen_window('Wrist Rotation Tracker'):
        print("Failed to create fullscreen window")
        return False
    
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
                    if confidence > 0.2:  # Lowered confidence threshold
                        px, py = int(x * width), int(y * height)
                        keypoint_coords[idx] = (px, py)
                        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                
                # Process right wrist if detected
                if KEYPOINTS['RIGHT_WRIST'] in keypoint_coords:
                    cv2.circle(frame, keypoint_coords[KEYPOINTS['RIGHT_WRIST']], 10, (255, 0, 0), -1)
                    right_elbow = keypoint_coords.get(KEYPOINTS['RIGHT_ELBOW'])
                    right_shoulder = keypoint_coords.get(KEYPOINTS['RIGHT_SHOULDER'])
                    right_wrist_tracker.process(
                        keypoint_coords[KEYPOINTS['RIGHT_WRIST']], right_elbow, right_shoulder, frame
                    )
                
                # Process left wrist if detected
                if KEYPOINTS['LEFT_WRIST'] in keypoint_coords:
                    cv2.circle(frame, keypoint_coords[KEYPOINTS['LEFT_WRIST']], 10, (0, 0, 255), -1)
                    left_elbow = keypoint_coords.get(KEYPOINTS['LEFT_ELBOW'])
                    left_shoulder = keypoint_coords.get(KEYPOINTS['LEFT_SHOULDER'])
                    left_wrist_tracker.process(
                        keypoint_coords[KEYPOINTS['LEFT_WRIST']], left_elbow, left_shoulder, frame
                    )
                
                # Display information
                cv2.putText(frame, "Wrist Rotation Tracker", (30, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                # Right wrist info
                cv2.putText(frame, f"Right Wrist:", (30, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f"Clockwise: {right_wrist_tracker.clockwise_count}", (30, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f"Anticlockwise: {right_wrist_tracker.anticlockwise_count}", (30, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, right_wrist_tracker.feedback, (30, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Left wrist info
                cv2.putText(frame, f"Left Wrist:", (30, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f"Clockwise: {left_wrist_tracker.clockwise_count}", (30, 340), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f"Anticlockwise: {left_wrist_tracker.anticlockwise_count}", (30, 380), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, left_wrist_tracker.feedback, (30, 420), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # FPS counter
                cv2.putText(frame, f"FPS: {fps:.1f}", (30, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Instructions
                instructions = "Press 'r' to reset counters | 'q' to quit"
                cv2.putText(frame, instructions, (width - 400, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow('Wrist Rotation Tracker', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    right_wrist_tracker.reset()
                    left_wrist_tracker.reset()
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                print(traceback.format_exc())
                time.sleep(0.1)
                
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
        summary_path = os.path.join(records_folder, "wrist_rotation.txt")
        
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
                f.write("Wrist Rotation Exercise Summary\n")
                f.write("------------------------------\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {elapsed_time:.1f} seconds\n")
                f.write(f"Right Wrist Clockwise: {right_wrist_tracker.clockwise_count}\n")
                f.write(f"Right Wrist Anticlockwise: {right_wrist_tracker.anticlockwise_count}\n")
                f.write(f"Left Wrist Clockwise: {left_wrist_tracker.clockwise_count}\n")
                f.write(f"Left Wrist Anticlockwise: {left_wrist_tracker.anticlockwise_count}\n\n")
                
                # Write form feedback
                f.write("Form Feedback:\n")
                f.write("-------------\n")
                f.write(f"Right Wrist: {right_wrist_tracker.feedback}\n")
                f.write(f"Left Wrist: {left_wrist_tracker.feedback}\n\n")
                
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
    run_wrist_rotation()