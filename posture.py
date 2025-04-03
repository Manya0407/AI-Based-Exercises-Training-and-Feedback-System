import cv2
import numpy as np
import time
import traceback
import os
from model_handler import (
    KEYPOINTS, detect_pose, calculate_angle, 
    create_fullscreen_window, initialize_webcam
)

# Define keypoint pairs for drawing skeleton
KEYPOINT_PAIRS = [
    (KEYPOINTS['LEFT_SHOULDER'], KEYPOINTS['LEFT_ELBOW']), (KEYPOINTS['LEFT_ELBOW'], KEYPOINTS['LEFT_WRIST']),  # Left arm
    (KEYPOINTS['RIGHT_SHOULDER'], KEYPOINTS['RIGHT_ELBOW']), (KEYPOINTS['RIGHT_ELBOW'], KEYPOINTS['RIGHT_WRIST']),  # Right arm
    (KEYPOINTS['LEFT_SHOULDER'], KEYPOINTS['RIGHT_SHOULDER']), (KEYPOINTS['LEFT_SHOULDER'], KEYPOINTS['LEFT_HIP']), (KEYPOINTS['RIGHT_SHOULDER'], KEYPOINTS['RIGHT_HIP']),  # Torso
    (KEYPOINTS['LEFT_HIP'], KEYPOINTS['RIGHT_HIP']), (KEYPOINTS['LEFT_HIP'], KEYPOINTS['LEFT_KNEE']), (KEYPOINTS['LEFT_KNEE'], KEYPOINTS['LEFT_ANKLE']),  # Left leg
    (KEYPOINTS['RIGHT_HIP'], KEYPOINTS['RIGHT_KNEE']), (KEYPOINTS['RIGHT_KNEE'], KEYPOINTS['RIGHT_ANKLE'])  # Right leg
]

# Function to provide posture feedback
def analyze_posture(keypoints):
    feedback = "Good posture!"
    
    # Get required keypoints
    left_shoulder = keypoints[KEYPOINTS['LEFT_SHOULDER']]
    right_shoulder = keypoints[KEYPOINTS['RIGHT_SHOULDER']]
    left_hip = keypoints[KEYPOINTS['LEFT_HIP']]
    right_hip = keypoints[KEYPOINTS['RIGHT_HIP']]
    left_ear = keypoints[KEYPOINTS['LEFT_EAR']]
    right_ear = keypoints[KEYPOINTS['RIGHT_EAR']]
    
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

def run_posture():
    # Initialize webcam
    cap = initialize_webcam()
    if cap is None:
        print("Exiting due to webcam initialization failure.")
        return False
    
    # Create fullscreen window
    if not create_fullscreen_window('Posture Detection'):
        print("Failed to create fullscreen window")
        return False
    
    start_time = time.time()
    frame_count = 0
    last_fps_update = time.time()
    fps = 0
    good_posture_time = 0
    total_time = 0
    
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
                    cv2.imshow('Posture Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Get feedback
                feedback = analyze_posture(keypoints)
                
                # Update good posture time
                if feedback == "Good posture!":
                    good_posture_time += 1/fps
                total_time += 1/fps
                
                # Draw keypoints and skeleton
                for idx, keypoint in enumerate(keypoints):
                    y, x, confidence = keypoint
                    if confidence > 0.3:
                        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
                
                for p1, p2 in KEYPOINT_PAIRS:
                    y1, x1, c1 = keypoints[p1]
                    y2, x2, c2 = keypoints[p2]
                    if c1 > 0.3 and c2 > 0.3:
                        pt1 = (int(x1 * frame.shape[1]), int(y1 * frame.shape[0]))
                        pt2 = (int(x2 * frame.shape[1]), int(y2 * frame.shape[0]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                
                # Display information on frame
                cv2.putText(frame, feedback, (30, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (30, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f"Good Posture: {good_posture_time:.1f}s", (30, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Add instructions at the bottom
                instructions = "Press 'q' to quit | Stand sideways for best results"
                cv2.putText(frame, instructions, (30, frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Posture Detection', frame)
                
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
        summary_path = os.path.join(records_folder, "posture.txt")
        
        # Generate a summary text file
        try:
            with open(summary_path, "w") as f:
                f.write("Posture Analysis Summary\n")
                f.write("====================\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {elapsed_time:.1f} seconds\n")
                f.write(f"Time with Good Posture: {good_posture_time:.1f} seconds\n")
                f.write(f"Percentage of Good Posture: {(good_posture_time/total_time*100):.1f}%\n")
            print(f"Summary saved to {summary_path}")
        except Exception as e:
            print(f"Error saving summary: {e}")
        
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Program terminated")
        return True

if __name__ == "__main__":
    run_posture()
