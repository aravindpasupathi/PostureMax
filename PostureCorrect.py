import cv2
import mediapipe as mp
import math
import time

def calculate_angle(a, b):
    radians = math.atan2(b.y - a.y, b.x - a.x)
    return abs(math.degrees(radians))

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Use 0 if 1 doesn't work

baseline_shoulder_width = None
bad_posture_streak = 0
bad_posture_required_streak = 30  # ~30 frames (~1 second)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lm = results.pose_landmarks.landmark
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]

        # Extract posture features
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_diff = abs(left_hip.y - right_hip.y)
        back_angle = calculate_angle(left_shoulder, left_hip)
        neck_angle = calculate_angle(left_ear, left_shoulder)
        avg_shoulder_z = (left_shoulder.z + right_shoulder.z) / 2

        # Calibrate baseline shoulder width on first frame
        if baseline_shoulder_width is None:
            baseline_shoulder_width = shoulder_width
            print(f"✅ Calibrated baseline shoulder width: {baseline_shoulder_width:.2f}")

        width_ratio = shoulder_width / baseline_shoulder_width
        slouching = width_ratio < 0.85
        leaning = shoulder_diff > 0.12 or hip_diff > 0.12
        neck_forward = neck_angle > 90
        leaning_forward = avg_shoulder_z < -0.25

        is_bad_posture = slouching or leaning or neck_forward or leaning_forward

        # Buffer streak to avoid flicker
        if is_bad_posture:
            bad_posture_streak += 1
        else:
            bad_posture_streak = 0

        if bad_posture_streak >= bad_posture_required_streak:
            cv2.putText(image, "Bad Posture Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Good Posture", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Posture Corrector", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
