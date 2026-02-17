import base64
import random
import os
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# --------------------
# App setup
# --------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --------------------
# Load YOLO model
# --------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# --------------------
# Config
# --------------------
class_names = ["line0", "line1", "line2", "line3"]
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]

predictions = {
    "line0": [
        "Life line: vitality and health.",
        "Life line: resilience and stamina.",
        "Life line: inner strength."
    ],
    "line1": [
        "Heart line: emotions and love.",
        "Heart line: empathy and compassion.",
        "Heart line: emotional balance."
    ],
    "line2": [
        "Head line: intelligence and mindset.",
        "Head line: analytical thinking.",
        "Head line: creativity."
    ],
    "line3": [
        "Fate line: career and destiny.",
        "Fate line: ambition.",
        "Fate line: long-term vision."
    ]
}

log_data = []

# --------------------
# Routes
# --------------------

@app.route("/", methods=["GET"])
def home():
    """Health check route (prevents 404)"""
    return jsonify({
        "status": "OK",
        "message": "Palmistry YOLO API is running"
    })


def analyze_image(img):
    results = model.predict(source=img, imgsz=256, conf=0.3, verbose=False)
    interpretations = {}

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        for box, label in zip(boxes, labels):
            class_text = class_names[int(label)]

            if class_text not in interpretations:
                chosen = random.sample(predictions[class_text], 2)
                interpretations[class_text] = chosen

                for interp in chosen:
                    log_data.append({
                        "timestamp": datetime.now().isoformat(),
                        "line_name": class_text,
                        "interpretation": interp
                    })

    return results, interpretations


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    results, interpretations = analyze_image(img)

    annotated = img.copy()

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            color = colors[int(label)]
            class_text = class_names[int(label)]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                class_text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    _, buffer = cv2.imencode(".jpg", annotated)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "predictions": interpretations,
        "image": f"data:image/jpeg;base64,{img_b64}"
    })


@app.route("/download-log", methods=["GET"])
def download_log():
    return jsonify(log_data)


# --------------------
# Run
# --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
