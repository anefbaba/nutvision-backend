from flask import Flask, request, jsonify
from ultralytics import YOLO
from collections import Counter
import base64
import cv2
import os

app = Flask(__name__)

# MODEL
model = YOLO("yeni_model.pt")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "NutVision backend çalışıyor"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:
            return jsonify({
                "detections": [],
                "image": None,
                "error": "Dosya bulunamadı"
            }), 400

        file = request.files["file"]

        file_path = "temp.jpg"
        file.save(file_path)

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return jsonify({
                "detections": [],
                "image": None,
                "error": "Fotoğraf kaydedilemedi veya boş geldi"
            }), 400

        results = model.predict(
            source=file_path,
            conf=0.50,
            iou=0.5
        )

        if not results or len(results) == 0:
            return jsonify({
                "detections": [],
                "image": None
            })

        names = model.names
        classes = []

        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]
                classes.append(class_name)

        counts = Counter(classes)
        total = sum(counts.values())

        detections = []

        if total > 0:
            for name, count in counts.items():
                percentage = round((count / total) * 100)

                detections.append({
                    "name": name,
                    "count": count,
                    "percentage": percentage
                })

        plotted = results[0].plot()

        success, buffer = cv2.imencode(".jpg", plotted)

        encoded_string = None

        if success:
            encoded_string = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "detections": detections,
            "image": encoded_string
        })

    except Exception as e:

        print("SERVER HATASI:", e)

        return jsonify({
            "detections": [],
            "image": None,
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )