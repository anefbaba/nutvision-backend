from flask import Flask, request, jsonify
from ultralytics import YOLO

from collections import Counter, defaultdict

import torch
import torch.nn as nn

from torchvision import models, transforms

from PIL import Image

import base64
import cv2
import os
import uuid
import math
import numpy as np

app = Flask(__name__)

# =========================
# MODEL YOLLARI
# =========================

YOLO_MODEL_PATH = "models/yeni_model.pt"
CNN_MODEL_PATH = "models/best.pth"

# =========================
# SINIFLAR
# =========================

CLASS_NAMES = [
    "antep_fistigi",
    "badem",
    "cekirdek",
    "findik",
    "fistik",
    "kabak_cekirdegi",
    "kaju",
    "leblebi"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================
# YOLO MODEL
# =========================

yolo_model = YOLO(YOLO_MODEL_PATH)

# =========================
# CNN MODEL
# =========================

cnn_model = models.mobilenet_v2(weights=None)

cnn_model.classifier[1] = nn.Linear(
    cnn_model.last_channel,
    len(CLASS_NAMES)
)

cnn_model.load_state_dict(
    torch.load(
        CNN_MODEL_PATH,
        map_location=DEVICE
    )
)

cnn_model = cnn_model.to(DEVICE)
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# AYARLAR
# =========================

YOLO_CONF = 0.55
YOLO_IOU = 0.30

GROUP_DISTANCE_FACTOR = 2.2
MIN_GROUP_SIZE = 2

CROP_PADDING = 40

YOLO_MAJORITY_RATIO = 0.60
CNN_MIN_CONF = 0.80

# =========================
# YARDIMCI FONKSİYONLAR
# =========================

def center_of_box(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )

def box_size(box):

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    return (w + h) / 2

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

def find_groups(boxes):

    n = len(boxes)

    visited = [False] * n

    groups = []

    centers = [
        center_of_box(b)
        for b in boxes
    ]

    sizes = [
        box_size(b)
        for b in boxes
    ]

    avg_size = (
        sum(sizes) / len(sizes)
        if sizes else 0
    )

    threshold = (
        avg_size * GROUP_DISTANCE_FACTOR
    )

    graph = defaultdict(list)

    for i in range(n):

        for j in range(i + 1, n):

            d = distance(
                centers[i],
                centers[j]
            )

            if d < threshold:

                graph[i].append(j)
                graph[j].append(i)

    for i in range(n):

        if not visited[i]:

            stack = [i]

            component = []

            while stack:

                node = stack.pop()

                if visited[node]:
                    continue

                visited[node] = True

                component.append(node)

                for neighbor in graph[node]:

                    if not visited[neighbor]:

                        stack.append(neighbor)

            groups.append(component)

    return groups

def crop_group(image, boxes, indexes, padding=CROP_PADDING):

    h, w = image.shape[:2]

    x1 = min([
        boxes[i][0]
        for i in indexes
    ])

    y1 = min([
        boxes[i][1]
        for i in indexes
    ])

    x2 = max([
        boxes[i][2]
        for i in indexes
    ])

    y2 = max([
        boxes[i][3]
        for i in indexes
    ])

    x1 = max(0, int(x1 - padding))
    y1 = max(0, int(y1 - padding))

    x2 = min(w, int(x2 + padding))
    y2 = min(h, int(y2 + padding))

    crop = image[y1:y2, x1:x2]

    return crop, (x1, y1, x2, y2)

def predict_cnn(crop):

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_img = Image.fromarray(crop_rgb)

    img_tensor = cnn_transform(
        pil_img
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = cnn_model(img_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        conf, pred = torch.max(
            probs,
            1
        )

    class_name = CLASS_NAMES[
        pred.item()
    ]

    confidence = conf.item()

    return class_name, confidence

# =========================
# ANA SAYFA
# =========================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "NutVision backend çalışıyor"
    })

# =========================
# PREDICT
# =========================

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

        file_path = (
            f"temp_{uuid.uuid4()}.jpg"
        )

        file.save(file_path)

        image = cv2.imread(file_path)

        if image is None:

            return jsonify({
                "detections": [],
                "image": None,
                "error": "Görüntü okunamadı"
            }), 400

        # =========================
        # YOLO
        # =========================

        results = yolo_model(
            image,
            conf=YOLO_CONF,
            iou=YOLO_IOU
        )

        boxes = []

        yolo_classes = []

        for box in results[0].boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            cls_id = int(box.cls[0])

            cls_name = yolo_model.names[
                cls_id
            ]

            boxes.append([
                x1,
                y1,
                x2,
                y2
            ])

            yolo_classes.append(
                cls_name
            )

        if len(boxes) == 0:

            return jsonify({
                "detections": [],
                "image": None
            })

        groups = find_groups(boxes)

        final_counts = Counter()

        used_indexes = set()

        # =========================
        # GRUPLAR
        # =========================

        for group in groups:

            if len(group) >= MIN_GROUP_SIZE:

                crop, group_box = crop_group(
                    image,
                    boxes,
                    group
                )

                cnn_class, cnn_conf = predict_cnn(
                    crop
                )

                group_yolo_classes = [
                    yolo_classes[i]
                    for i in group
                ]

                most_common_class, most_common_count = Counter(
                    group_yolo_classes
                ).most_common(1)[0]

                yolo_ratio = (
                    most_common_count / len(group)
                )

                if yolo_ratio >= YOLO_MAJORITY_RATIO:

                    final_class = most_common_class

                elif cnn_conf >= CNN_MIN_CONF:

                    final_class = cnn_class

                else:

                    final_class = most_common_class

                adet = len(group)

                final_counts[final_class] += adet

                used_indexes.update(group)

                gx1, gy1, gx2, gy2 = group_box

                cv2.rectangle(
                    image,
                    (gx1, gy1),
                    (gx2, gy2),
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    image,
                    f"{final_class} ({adet})",
                    (
                        gx1,
                        max(gy1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

        # =========================
        # TEKİL NESNELER
        # =========================

        for i, cls_name in enumerate(
            yolo_classes
        ):

            if i not in used_indexes:

                final_counts[cls_name] += 1

                x1, y1, x2, y2 = map(
                    int,
                    boxes[i]
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    cls_name,
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        total = sum(
            final_counts.values()
        )

        detections = []

        for name, count in final_counts.items():

            percentage = round(
                (count / total) * 100
            )

            detections.append({
                "name": name,
                "count": count,
                "percentage": percentage
            })

        # =========================
        # BASE64 GÖRSEL
        # =========================

        success, buffer = cv2.imencode(
            ".jpg",
            image
        )

        encoded_string = None

        if success:

            encoded_string = (
                base64.b64encode(buffer)
                .decode("utf-8")
            )

        # TEMP SİL

        if os.path.exists(file_path):
            os.remove(file_path)

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

# =========================
# ÇALIŞTIR
# =========================

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