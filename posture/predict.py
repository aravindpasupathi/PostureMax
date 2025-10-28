import cv2
import mediapipe as mp
import joblib
from .geometry import Point, calculate_angle_three_points


def run_live_predictor(model_path: str = "models/posture_model_v3.pkl") -> None:
    model = joblib.load(model_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
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

            features = []
            for landmark in results.pose_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])

            lm = results.pose_landmarks.landmark
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            mid_spine = Point(x=(left_hip.x + right_hip.x) / 2, y=(left_hip.y + right_hip.y) / 2)

            back_angle = calculate_angle_three_points(
                Point(left_shoulder.x, left_shoulder.y),
                mid_spine,
                Point(left_hip.x, left_hip.y),
            )

            cv2.putText(image, f"Back Angle: {int(back_angle)}", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            prediction = model.predict([features])[0]
            color = (0, 255, 0) if prediction == "good" else (0, 0, 255)
            label = f"Posture: {prediction.upper()}"
            cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Posture AI Predictor", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
