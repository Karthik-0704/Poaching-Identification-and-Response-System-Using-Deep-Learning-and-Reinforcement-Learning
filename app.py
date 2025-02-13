import os
import base64
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Flask setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
MODEL_PATH = "runs/detect/train7/weights/best.pt"
model = YOLO(MODEL_PATH)

# Serve UI
@app.route("/")
def index():
    return render_template("index.html")

# Handle file upload and detection
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Convert image to OpenCV format
    image = Image.open(file).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Send progress updates
    socketio.emit("progress", {"message": "Processing started...", "progress": 10})

    # Run YOLO detection
    results = model(image_cv, conf=0.3)

    # Annotate image
    detected_image = image_cv.copy()
    count = 0
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].cpu().numpy()
            if conf < 0.3:
                continue
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detected_image, f"Elephant {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    # Convert image to base64
    _, buffer = cv2.imencode(".jpg", detected_image)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    socketio.emit("progress", {"message": "Processing completed!", "progress": 100})

    return jsonify({"count": count, "annotated_image": encoded_image})

# Run Flask
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)