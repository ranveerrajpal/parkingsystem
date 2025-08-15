from fastapi import FastAPI
import cv2
import time
import requests
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use your custom model if trained on two-wheelers

# Max parking capacity
MAX_CAPACITY = 10

# Receiver endpoint (your Render web app)
RECEIVER_URL = "https://twowheeler-lig5.onrender.com/update"

# Labels to count
TWO_WHEELER_LABELS = ["motorcycle", "bicycle"]

def count_and_draw(frame):
    results = model.predict(frame, imgsz=640, conf=0.5)
    count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in TWO_WHEELER_LABELS:
                count += 1

                # Get box coordinates and convert to int
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return count, frame

def send_data(count):
    status = "Full" if count >= MAX_CAPACITY else "Available"
    data = {
        "count": count,
        "capacity": MAX_CAPACITY,
        "status": status
    }
    try:
        res = requests.post(RECEIVER_URL, json=data, timeout=5)
        print(f"[✔] Sent {data} -> {res.status_code}")
    except Exception as e:
        print(f"[✖] Failed to send data: {e}")

@app.get("/")
def start_detection():
    cap = cv2.VideoCapture(0)  # Use webcam or replace with video file path
    last_sent = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count, annotated = count_and_draw(frame)

        cv2.putText(annotated, f"Detected: {count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.imshow("Two-Wheeler Detection", annotated)

        if time.time() - last_sent > 3:
            send_data(count)
            last_sent = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"status": "Stopped"}

# 🏃 Local run
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
