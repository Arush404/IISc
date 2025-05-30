import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Bicep Curl Parameters
ANGLE_THRESHOLD = (35, 160)  
FORM_THRESHOLD = 25  

# Counters and states
counters = {"left": 0, "right": 0}
stages = {"left": None, "right": None}
form_warnings = {"left": False, "right": False}

def calculate_angle(a, b, c):
    """Calculates angle at point b (between points a-b-c)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

def get_landmark_coords(landmarks, landmark_type):
    """Returns (x, y) coordinates of specified landmark"""
    return [
        landmarks[landmark_type.value].x,
        landmarks[landmark_type.value].y
    ]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame with MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        for side in ["left", "right"]:
            # Get key landmarks
            shoulder = get_landmark_coords(landmarks, 
                mp_pose.PoseLandmark[f"{side.upper()}_SHOULDER"])
            elbow = get_landmark_coords(landmarks,
                mp_pose.PoseLandmark[f"{side.upper()}_ELBOW"])
            wrist = get_landmark_coords(landmarks,
                mp_pose.PoseLandmark[f"{side.upper()}_WRIST"])
            hip = get_landmark_coords(landmarks,
                mp_pose.PoseLandmark[f"{side.upper()}_HIP"])

            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            upper_arm_angle = calculate_angle(hip, shoulder, elbow)

            # Form check (upper arm should stay relatively vertical)
            if upper_arm_angle > FORM_THRESHOLD:
                form_warnings[side] = True
            else:
                form_warnings[side] = False

            # Rep counting logic
            if elbow_angle > ANGLE_THRESHOLD[1]:  # Arm extended
                stages[side] = "down"
            if elbow_angle < ANGLE_THRESHOLD[0] and stages[side] == "down":
                stages[side] = "up"
                counters[side] += 1

            # Visual feedback
            color = (0, 255, 0) if not form_warnings[side] else (0, 0, 255)
            cv2.putText(image, f"{side.upper()}: {counters[side]}", 
                (10 if side == "left" else 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw angles
            elbow_pos = tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int))
            cv2.putText(image, f"{elbow_angle:.0f}°", elbow_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Draw form warnings
        for side in ["left", "right"]:
            if form_warnings[side]:
                cv2.putText(image, f"BAD FORM {side.upper()}!",
                    (10 if side == "left" else 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Bicep Curl Counter', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
