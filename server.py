from flask import Flask, request, jsonify
from ultralytics import YOLO
from collections import Counter
import base64
import cv2
import os

app = Flask(__name__)

# MODEL
model = YOLO("yeni_model.pt")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # DOSYA VAR MI
        if "file" not in request.files:

            return jsonify({
                "error": "Dosya bulunamadı"
            }), 400

        file = request.files["file"]

        # GEÇİCİ DOSYA
        file_path = "temp.jpg"

        # KAYDET
        file.save(file_path)

        # YOLO
        results = model.predict(
            source=file_path,
            conf=0.50,
            iou=0.5
        )

        names = model.names

        classes = []

        # TESPİTLER
        for box in results[0].boxes:

            class_id = int(
                box.cls[0]
            )

            class_name = names[
                class_id
            ]

            classes.append(
                class_name
            )

        counts = Counter(classes)

        total = sum(
            counts.values()
        )

        detections = []

        # SONUÇLAR
        if total > 0:

            for name, count in counts.items():

                percentage = round(
                    (count / total) * 100
                )

                detections.append({
                    "name": name,
                    "count": count,
                    "percentage": percentage
                })

        # KUTULU GÖRSEL
        plotted = results[0].plot()

        # JPG KAYDET
        cv2.imwrite(
            "result.jpg",
            plotted
        )

        # BASE64
        with open(
            "result.jpg",
            "rb"
        ) as image_file:

            encoded_string = (
                base64.b64encode(
                    image_file.read()
                ).decode("utf-8")
            )

        return jsonify({
            "detections": detections,
            "image": encoded_string
        })

    except Exception as e:

        print(
            "SERVER HATASI:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )