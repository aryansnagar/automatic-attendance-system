import cv2
import face_recognition
import os
import csv
from datetime import datetime, timedelta

# Folder with reference images
KNOWN_PEOPLE_DIR = "people_images"

# Load known faces
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_PEOPLE_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(KNOWN_PEOPLE_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

print(f"[INFO] Loaded {len(known_encodings)} known faces.")

# CSV file for logging
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# Dictionary to track last log time for each person
last_logged = {}

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

COOLDOWN_SECONDS = 5  # Minimum time between logs for the same person

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Match face
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Scale coordinates
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Check cooldown before logging
            now = datetime.now()
            if name not in last_logged or (now - last_logged[name]).total_seconds() >= COOLDOWN_SECONDS:
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, now.date(), now.strftime("%H:%M:%S")])
                last_logged[name] = now  # update last logged time

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
