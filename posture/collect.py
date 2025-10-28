import os
import csv
import cv2
import mediapipe as mp
from .geometry import Point, calculate_angle_three_points


def run_data_collector(csv_path: str = "data/posture_dataset.csv") -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    file_exists = os.path.isfile(csv_path)
    csv_file = open(csv_path, mode='a', newline='')
    csv_writer = csv.writer(csv_file)

    if not file_exists:
        header = []
        for landmark in mp_pose.PoseLandmark:
            header += [f"{landmark.name}_x", f"{landmark.name}_y", f"{landmark.name}_z"]
        header.append("label")
        csv_writer.writerow(header)

    print("🎥 Posture Data Collector Running")
    print("Press 'g' to label Good posture, 'b' for Bad posture, 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            mid_spine = Point(x=(left_hip.x + right_hip.x) / 2, y=(left_hip.y + right_hip.y) / 2)
            back_angle = calculate_angle_three_points(
                Point(left_shoulder.x, left_shoulder.y),
                mid_spine,
                Point(left_hip.x, left_hip.y),
            )

            cv2.putText(image, f"Back Angle: {int(back_angle)}", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Posture Data Collector", image)
        key = cv2.waitKey(10) & 0xFF

        if key in (ord('g'), ord('b')):
            if results.pose_landmarks:
                row = []
                for landmark in results.pose_landmarks.landmark:
                    row += [landmark.x, landmark.y, landmark.z]
                row.append('good' if key == ord('g') else 'bad')
                csv_writer.writerow(row)
                print(f"✅ Saved label: {'good' if key == ord('g') else 'bad'}")

        if key == ord('q'):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
