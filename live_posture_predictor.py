import cv2
import mediapipe as mp
import joblib
import math
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def calculate_angle(a, b, c):
    """Calculate the angle at point b given 3 points a-b-c."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Load your trained model
model = joblib.load("posture_model_v3.pkl")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 if 1 doesn't work

print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmark features
        features = []
        for landmark in results.pose_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])

        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        mid_spine = Point(
            x=(left_hip.x + right_hip.x) / 2,
            y=(left_hip.y + right_hip.y) / 2
        )

        back_angle = calculate_angle(
            Point(left_shoulder.x, left_shoulder.y),
            mid_spine,
            Point(left_hip.x, left_hip.y)
        )

        # Optionally print or display it
        cv2.putText(image, f'Back Angle: {int(back_angle)}', (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Predict posture
        prediction = model.predict([features])[0]

        # Display result
        color = (0, 255, 0) if prediction == "good" else (0, 0, 255)
        label = f"Posture: {prediction.upper()}"
        cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show webcam feed
    cv2.imshow("Posture AI Predictor", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
