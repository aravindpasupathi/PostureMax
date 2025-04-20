import cv2
import mediapipe as mp
import csv
import os
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def calculate_angle(a, b, c):
    """Calculate angle at point b given points a-b-c"""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 0 if needed

# Setup CSV file
csv_filename = "posture_dataset.csv"
file_exists = os.path.isfile(csv_filename)
csv_file = open(csv_filename, mode='a', newline='')
csv_writer = csv.writer(csv_file)

# Write headers only if file doesn't exist
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

    # Detect pose
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate midpoint of hips
        mid_spine = Point(
            x=(left_hip.x + right_hip.x) / 2,
            y=(left_hip.y + right_hip.y) / 2
        )

        # Calculate arc angle at spine
        back_angle = calculate_angle(
            Point(left_shoulder.x, left_shoulder.y),
            mid_spine,
            Point(left_hip.x, left_hip.y)
        )

        # Display back angle on the video feed
        cv2.putText(image, f"Back Angle: {int(back_angle)}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display video
    cv2.imshow("Posture Data Collector", image)
    key = cv2.waitKey(10) & 0xFF

    # Record labeled data
    if key == ord('g') or key == ord('b'):
        if results.pose_landmarks:
            row = []
            for landmark in results.pose_landmarks.landmark:
                row += [landmark.x, landmark.y, landmark.z]
            row.append('good' if key == ord('g') else 'bad')
            csv_writer.writerow(row)
            print(f"✅ Saved label: {'good' if key == ord('g') else 'bad'}")

    # Quit
    if key == ord('q'):
        break

# Cleanup
cap.release()
csv_file.close()
cv2.destroyAllWindows()
