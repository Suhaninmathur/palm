import io
import base64
import random
import json
import os
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model once
# When packaging the entire project inside `backend/`, the model folder
# will live alongside this file.  BASE_DIR points to the same directory
# so the service is self-contained.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. \n"
        "Please place your weights in backend/model/best.pt"
    )
model = YOLO(MODEL_PATH)

class_names = ["line0", "line1", "line2", "line3"]
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]

# interpretations as in original script
predictions = {
    "line0": [
        "Life line: vitality and health.",
        "Life line: resilience and stamina.",
        "Life line: adventurous spirit.",
        "Life line: balanced lifestyle.",
        "Life line: strong energy flow.",
        "Life line: inner strength.",
        "Life line: ability to recover quickly.",
        "Life line: grounded personality.",
        "Life line: harmony with nature.",
        "Life line: steady growth and maturity."
    ],
    "line1": [
        "Heart line: emotions and love.",
        "Heart line: empathy and compassion.",
        "Heart line: romantic nature.",
        "Heart line: deep social bonds.",
        "Heart line: sensitive and caring.",
        "Heart line: openness to relationships.",
        "Heart line: passion and warmth.",
        "Heart line: emotional resilience.",
        "Heart line: nurturing instincts.",
        "Heart line: balanced affection."
    ],
    "line2": [
        "Head line: intelligence and mindset.",
        "Head line: creativity and imagination.",
        "Head line: logical thinking.",
        "Head line: quick decision making.",
        "Head line: analytical approach.",
        "Head line: curiosity and learning.",
        "Head line: innovative ideas.",
        "Head line: sharp focus.",
        "Head line: problem-solving skills.",
        "Head line: visionary thinking."
    ],
    "line3": [
        "Fate line: career and destiny.",
        "Fate line: adaptability in life path.",
        "Fate line: ambition and independence.",
        "Fate line: strong career focus.",
        "Fate line: flexible destiny.",
        "Fate line: perseverance in challenges.",
        "Fate line: leadership qualities.",
        "Fate line: long-term vision.",
        "Fate line: professional growth.",
        "Fate line: determination and drive."
    ]
}

# logging
log_data = []


def draw_side_panel(frame, interpretations):
    panel_width = 320
    panel_x = frame.shape[1] - panel_width
    cv2.rectangle(frame, (panel_x, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
    cv2.putText(frame, "Palmistry Insights", (panel_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_offset = 60
    if not interpretations:
        cv2.putText(frame, "No lines detected", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        for class_text, interps in interpretations.items():
            idx = class_names.index(class_text)
            color = colors[idx]
            for interp in interps:
                cv2.putText(frame, interp, (panel_x + 10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 30


def analyze_image(img):
    # run YOLO detection
    results = model.predict(source=img, imgsz=256, conf=0.3, show=False)
    interpretations = {}
    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        for box, label in zip(boxes, labels):
            class_text = class_names[int(label)]
            if class_text not in interpretations:
                chosen = random.sample(predictions[class_text], 3)
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
        return jsonify({"error": "No image file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # read into opencv image
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    results, interpretations = analyze_image(img)

    # draw boxes and panel on copy of original
    annotated = img.copy()
    for result in results:
        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            class_text = class_names[int(label)]
            color = colors[int(label)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, class_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    draw_side_panel(annotated, interpretations)

    # encode annotated image as base64
    _, buffer = cv2.imencode('.jpg', annotated)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    img_b64 = f"data:image/jpeg;base64,{jpg_as_text}"

    return jsonify({
        "predictions": interpretations,
        "image": img_b64

    })


@app.route("/download-log", methods=["GET"])
def download_log():
    # return the JSON log
    return jsonify(log_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
