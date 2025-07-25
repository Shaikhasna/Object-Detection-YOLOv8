import cv2
from ultralytics import YOLO
from utils import get_valid_classes
import csv
from datetime import datetime
import simpleaudio as sa  # ✅ Replaced playsound with simpleaudio

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Get list of all YOLO class names
all_classes = list(model.names.values())

# Show available classes to the user
print("Available YOLO classes:")
print(", ".join(all_classes))

# Ask user to enter object names
user_input = input("Enter object names to detect (comma-separated, e.g., person, car): ").strip()
target_classes = get_valid_classes(user_input, all_classes)

if not target_classes:
    print("❌ No valid classes entered. Exiting.")
    exit()

print("📹 Starting detection. Press 'q' to quit.")

# Open webcam
cap = cv2.VideoCapture(0)

# Setup CSV logging
csv_file = open("detections.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Class", "Confidence"])

alert_played = False
wave_obj = None

try:
    # Load WAV file
    try:
        wave_obj = sa.WaveObject.from_wave_file("alert.wav")
    except Exception as e:
        print("⚠️ Could not load alert.wav:", e)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO prediction
        results = model(frame)[0]

        # Process detections
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            class_name = model.names[int(class_id)]

            if class_name in target_classes:
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} {score:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Log detection
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([timestamp, class_name, f"{score:.2f}"])

                # 🔊 Play alert only once
                if not alert_played and wave_obj:
                    wave_obj.play()
                    alert_played = True  # Set to False if you want it to repeat later

        cv2.imshow("Agentic Object Detector", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
