import cv2
import torch
import numpy as np
from flask import Flask, Response, render_template
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("runs/detect/train10/weights/best.pt")

app = Flask(__name__)


def generate_frames():
    cap = cv2.VideoCapture(0)  # Capture from webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform detection
        results = model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            if conf > 0.5:  # Confidence threshold
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Weapon ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
