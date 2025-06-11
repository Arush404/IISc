import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# --- Pose Detector Classes ---

class TrikonasanaPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_start_time = None
        self.pose_hold_duration = 5
        self.stability_frames = []
        self.stability_threshold = 0.7
        self.max_stability_frames = 10
        self.angle_tolerances = {
            'leg_straight': 15,
            'torso_bend': 20,
            'arm_alignment': 25
        }

    def calculate_angle(self, point1, point2, point3):
        try:
            a = np.array(point1)
            b = np.array(point2)
            c = np.array(point3)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except:
            return 0

    def get_landmark_coords(self, landmarks, landmark_id):
        if landmarks and len(landmarks.landmark) > landmark_id:
            landmark = landmarks.landmark[landmark_id]
            return [landmark.x, landmark.y]
        return [0, 0]

    def analyze_pose(self, landmarks):
        if not landmarks:
            return False, []
        feedback = []
        correct_elements = 0
        total_elements = 5
        left_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        left_hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        left_ankle = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = self.get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        # 1. Front leg straightness
        front_leg_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        if abs(front_leg_angle - 180) <= self.angle_tolerances['leg_straight']:
            correct_elements += 1
            feedback.append("✓ Front leg is straight")
        else:
            feedback.append("⚠ Straighten your front leg more")
        # 2. Back leg straightness
        back_leg_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        if abs(back_leg_angle - 180) <= self.angle_tolerances['leg_straight']:
            correct_elements += 1
            feedback.append("✓ Back leg is straight")
        else:
            feedback.append("⚠ Straighten your back leg")
        # 3. Torso side bend
        torso_angle = self.calculate_angle(left_shoulder, left_hip, [left_hip[0], left_hip[1] - 0.1])
        if 30 <= abs(torso_angle - 90) <= 60:
            correct_elements += 1
            feedback.append("✓ Good torso side bend")
        else:
            feedback.append("⚠ Adjust your side bend - reach further")
        # 4. Top arm extension
        top_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        if abs(top_arm_angle - 180) <= self.angle_tolerances['arm_alignment']:
            correct_elements += 1
            feedback.append("✓ Top arm is extended")
        else:
            feedback.append("⚠ Extend your top arm straight")
        # 5. Arms alignment
        arm_line_angle = self.calculate_angle(left_wrist, left_shoulder, right_wrist)
        if abs(arm_line_angle - 180) <= self.angle_tolerances['arm_alignment']:
            correct_elements += 1
            feedback.append("✓ Arms are aligned")
        else:
            feedback.append("⚠ Align your arms in a straight line")
        is_correct = correct_elements >= 4
        return is_correct, feedback

    def draw_landmarks_and_feedback(self, image, landmarks, feedback, pose_correct, pose_name):
        if landmarks:
            self.mp_drawing.draw_landmarks(
                image, landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        y_offset = 30
        for i, text in enumerate(feedback):
            color = (0, 255, 0) if text.startswith("✓") else (0, 165, 255)
            cv2.putText(image, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        status_text = f"{pose_name} Status: "
        if pose_correct:
            status_text += "CORRECT - Hold the pose!"
            status_color = (0, 255, 0)
        else:
            status_text += "Adjust your posture"
            status_color = (0, 165, 255)
        cv2.putText(image, status_text, (10, image.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        if self.pose_start_time and pose_correct:
            elapsed_time = time.time() - self.pose_start_time
            remaining_time = max(0, self.pose_hold_duration - elapsed_time)
            if remaining_time > 0:
                timer_text = f"Hold for: {remaining_time:.1f}s"
                cv2.putText(image, timer_text, (10, image.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            else:
                success_text = "GREAT! Pose completed successfully!"
                cv2.putText(image, success_text, (10, image.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        instruction_text = f"Instructions: {pose_name} - Stand with feet wide, reach down, extend arm up"
        cv2.putText(image, instruction_text, (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    def update_pose_timer(self, pose_correct):
        if pose_correct:
            if self.pose_start_time is None:
                self.pose_start_time = time.time()
                print("✓ Correct pose detected! Hold for 5 seconds...")
            elif time.time() - self.pose_start_time >= self.pose_hold_duration:
                return True  # Pose held long enough
        else:
            if self.pose_start_time is not None:
                print("⚠ Pose alignment lost. Adjust your posture to restart timer.")
            self.pose_start_time = None
        return False

    def check_pose_stability(self, pose_correct):
        self.stability_frames.append(pose_correct)
        if len(self.stability_frames) > self.max_stability_frames:
            self.stability_frames.pop(0)
        if len(self.stability_frames) >= 5:
            stability_ratio = sum(self.stability_frames) / len(self.stability_frames)
            return stability_ratio >= self.stability_threshold
        return False

# --- Pose Queue Sequencer ---

class YogaPoseSequencer:
    def __init__(self, pose_queue):
        self.pose_queue = deque(pose_queue)
        self.detectors = {
            'Trikonasana': TrikonasanaPoseDetector(),
            # Add other pose detectors here, e.g. 'WarriorII': WarriorIIPoseDetector()
        }

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("🧘 Yoga Pose Sequencer Started!")
        print("Press 'q' to quit at any time.\n")

        while self.pose_queue:
            current_pose = self.pose_queue[0]
            detector = self.detectors[current_pose]
            print(f"\n➡️  Please perform: {current_pose}")
            detector.pose_start_time = None
            detector.stability_frames = []

            pose_completed = False
            while not pose_completed:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.pose.process(rgb_frame)
                pose_correct, feedback = detector.analyze_pose(results.pose_landmarks)
                stable_pose = detector.check_pose_stability(pose_correct)
                pose_held = detector.update_pose_timer(stable_pose)
                annotated_frame = detector.draw_landmarks_and_feedback(
                    frame, results.pose_landmarks, feedback, stable_pose, current_pose
                )
                cv2.imshow('Yoga Pose Sequencer', annotated_frame)
                if pose_held:
                    print(f"🎉 {current_pose} completed! Moving to next pose.")
                    time.sleep(2)
                    pose_completed = True
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("👋 Session ended. Namaste!")
                    return
            self.pose_queue.popleft()

        cap.release()
        cv2.destroyAllWindows()
        print("✅ All poses completed! Well done.")

# --- Main ---

if __name__ == "__main__":
    pose_sequence = ['Trikonasana']  # Add more pose names as you implement them
    sequencer = YogaPoseSequencer(pose_sequence)
    sequencer.run(camera_index=0)
