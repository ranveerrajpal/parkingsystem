import cv2
from ultralytics import YOLO

# Load YOLOv8 model (use yolov8n.pt or a custom one)
model = YOLO("yolov8n.pt")
TWO_WHEELER_CLASSES = [3]  # COCO class 3 = motorcycle

# Define parking capacity
MAX_CAPACITY = 10

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 for webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        detections = results[0].boxes

        count = 0
        for box in detections:
            cls = int(box.cls.item())
            if cls in TWO_WHEELER_CLASSES:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Two-Wheeler", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        status = "Full" if count >= MAX_CAPACITY else "Available"
        color = (0, 0, 255) if status == "Full" else (0, 255, 0)

        cv2.putText(frame, f"Two-Wheelers: {count}/{MAX_CAPACITY}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Status: {status}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Streaming format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
