import cv2
import numpy as np
import mediapipe as mp

def calculate_angle(a, b, c):
    """Calculate the angle at point b given three points a, b, c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

counter = 0
stage = None

# Angle thresholds
ELBOW_DOWN_MIN = 70
ELBOW_DOWN_MAX = 100
ELBOW_UP = 160
HIP_MIN = 160
HIP_MAX = 200

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        feedback = ""
        feedback_color = (0, 255, 0)
        nice_one = False

        try:
            landmarks = results.pose_landmarks.landmark

            def get_point(idx):
                lm = landmarks[idx]
                return [int(lm.x * w), int(lm.y * h)]

            # Left and right joints
            l_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            l_elbow    = get_point(mp_pose.PoseLandmark.LEFT_ELBOW.value)
            l_wrist    = get_point(mp_pose.PoseLandmark.LEFT_WRIST.value)
            l_hip      = get_point(mp_pose.PoseLandmark.LEFT_HIP.value)
            l_ankle    = get_point(mp_pose.PoseLandmark.LEFT_ANKLE.value)

            r_shoulder = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            r_elbow    = get_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            r_wrist    = get_point(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            r_hip      = get_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
            r_ankle    = get_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

            # Calculate angles
            left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

            left_hip_angle = calculate_angle(l_ankle, l_hip, l_shoulder)
            right_hip_angle = calculate_angle(r_ankle, r_hip, r_shoulder)
            hip_angle = (left_hip_angle + right_hip_angle) / 2

            # Draw lines and circles for left side (for clarity, you can add right side too)
            points = [l_ankle, l_hip, l_shoulder, l_elbow, l_wrist]
            for pt in points:
                cv2.circle(image, tuple(pt), 10, (0, 255, 0), -1)
            cv2.line(image, tuple(l_ankle), tuple(l_hip), (255, 255, 255), 3)
            cv2.line(image, tuple(l_hip), tuple(l_shoulder), (255, 255, 255), 3)
            cv2.line(image, tuple(l_shoulder), tuple(l_elbow), (255, 255, 255), 3)
            cv2.line(image, tuple(l_elbow), tuple(l_wrist), (255, 255, 255), 3)

            # Draw angle values
            cv2.putText(image, f'{int(hip_angle)} degrees', (l_hip[0]-50, l_hip[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(image, f'{int(elbow_angle)} degrees', (l_elbow[0]-50, l_elbow[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # Push-up logic: only count if both elbow and hip angles are in correct range
            if ELBOW_DOWN_MIN <= elbow_angle <= ELBOW_DOWN_MAX and HIP_MIN <= hip_angle <= HIP_MAX:
                stage = "down"
            if elbow_angle > ELBOW_UP and stage == "down" and HIP_MIN <= hip_angle <= HIP_MAX:
                stage = "up"
                counter += 1

            # Visual feedback
            if HIP_MIN <= hip_angle <= HIP_MAX:
                feedback = "Perfect Long Line Body"
                feedback_color = (0, 255, 0)
                nice_one = elbow_angle > ELBOW_UP
            else:
                feedback = "Attention! Not perfect Long Line Body"
                feedback_color = (0, 0, 255)
                nice_one = False

            # Overlay rectangles and text
            # Top left: Push up label
            cv2.rectangle(image, (0, 0), (300, 120), (0, 0, 255), -1)
            cv2.putText(image, 'Push up', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

            # Top center: feedback
            feedback_box_w = 650
            cv2.rectangle(image, (w//2 - feedback_box_w//2, 20), (w//2 + feedback_box_w//2, 90), feedback_color, -1)
            cv2.putText(image, feedback, (w//2 - feedback_box_w//2 + 20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

            # Right side: progress bar and "Nice One"
            bar_x = w - 100
            bar_y = 100
            bar_height = h - 200
            bar_width = 40
            bar_min = ELBOW_DOWN_MIN
            bar_max = ELBOW_UP
            # Map elbow angle to bar fill
            bar_fill = int(np.interp(elbow_angle, [bar_min, bar_max], [bar_y + bar_height, bar_y]))
            bar_color = (0,255,0) if nice_one else (0,0,255)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0,255,0), 2)
            cv2.rectangle(image, (bar_x, bar_fill), (bar_x + bar_width, bar_y + bar_height), bar_color, -1)
            if nice_one:
                cv2.putText(image, "Nice One", (bar_x - 60, bar_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

            # Counter at bottom left
            cv2.putText(image, f'Push-Ups: {counter}', (30, h - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            pass

        cv2.imshow('Push-Up Counter', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
