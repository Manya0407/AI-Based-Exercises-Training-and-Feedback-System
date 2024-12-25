import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to analyze posture for table tennis
def analyze_posture(landmarks):
    feedback = []

    # Right arm analysis
    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    right_elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
    right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)

    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Feedback for right arm (Forehand/Backhand)
    if 70 <= right_arm_angle <= 110:
        feedback.append(f"Right arm angle: {right_arm_angle:.2f}° – Good alignment for a forehand stroke!")
    elif 30 <= right_arm_angle <= 60:
        feedback.append(f"Right arm angle: {right_arm_angle:.2f}° – Prepare for a backhand stroke!")
    else:
        feedback.append(f"Right arm angle: {right_arm_angle:.2f}° – Adjust arm posture for better control.")

    # Left arm analysis
    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
    left_elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
    left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # Feedback for left arm (Forehand/Backhand)
    if 70 <= left_arm_angle <= 110:
        feedback.append(f"Left arm angle: {left_arm_angle:.2f}° – Good alignment for a forehand stroke!")
    elif 30 <= left_arm_angle <= 60:
        feedback.append(f"Left arm angle: {left_arm_angle:.2f}° – Prepare for a backhand stroke!")
    else:
        feedback.append(f"Left arm angle: {left_arm_angle:.2f}° – Adjust arm posture for better control.")

    # Wrist angle analysis (for control)
    right_wrist_angle = calculate_angle(right_elbow, right_wrist, (right_wrist[0] + 0.1, right_wrist[1] + 0.1))
    left_wrist_angle = calculate_angle(left_elbow, left_wrist, (left_wrist[0] + 0.1, left_wrist[1] + 0.1))

    if right_wrist_angle < 30:
        feedback.append("Right wrist angle: Good wrist angle for a strong spin!")
    elif right_wrist_angle > 150:
        feedback.append("Right wrist angle: Too much wrist angle – may affect spin and control.")
    else:
        feedback.append(f"Right wrist angle: {right_wrist_angle:.2f}° – Adjust wrist angle for better spin control.")

    if left_wrist_angle < 30:
        feedback.append("Left wrist angle: Good wrist angle for control.")
    elif left_wrist_angle > 150:
        feedback.append("Left wrist angle: Too much wrist angle – may affect control.")
    else:
        feedback.append(f"Left wrist angle: {left_wrist_angle:.2f}° – Adjust wrist angle for better control.")

    # Torso analysis for balance
    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
    right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
    torso_angle = calculate_angle(left_shoulder, right_shoulder, right_hip)

    if 80 <= torso_angle <= 100:
        feedback.append("Torso angle: Good upright posture for balance!")
    else:
        feedback.append(f"Torso angle: {torso_angle:.2f}° – Adjust torso posture for better balance.")

    return feedback

# Open video capture
cap = cv2.VideoCapture(0)  # Replace 0 with a video file path if using a recorded video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks and analyze the posture
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Analyze posture
        landmarks = results.pose_landmarks.landmark
        feedback = analyze_posture(landmarks)

        # Display feedback on the frame
        for i, message in enumerate(feedback):
            cv2.putText(frame, message, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display the frame
    cv2.imshow("Enhanced Table Tennis Pose Analysis", frame)

    # Exit on 'q' key or close button
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Enhanced Table Tennis Pose Analysis", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
