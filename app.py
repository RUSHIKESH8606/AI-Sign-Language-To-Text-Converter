
import os

import cv2
import torch
import numpy as np
import mediapipe as mp

from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import models, transforms


# ---------------- FLASK CONFIGURATION ----------------

import os
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model_mobilenet.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASSES = [
    "FOOD",
    "HELLO",
    "I",
    "NO",
    "SORRY",
    "THANKS",
    "WANT",
    "WASHROOM",
    "WATER"
]


# ---------------- LOAD TRAINED MODEL ----------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

classes = checkpoint.get("classes", DEFAULT_CLASSES)
num_classes = checkpoint.get("num_classes", len(classes))
image_size = checkpoint.get("image_size", 224)
mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
std = checkpoint.get("std", [0.229, 0.224, 0.225])
architecture = checkpoint.get("architecture", "mobilenet_v3_small")

if isinstance(classes, dict):
    classes = list(classes.keys())

classes = list(classes)

if architecture == "mobilenet_v3_large":
    model = models.mobilenet_v3_large(weights=None)
elif architecture == "mobilenet_v3_small":
    model = models.mobilenet_v3_small(weights=None)
else:
    raise ValueError(
        f"Unsupported architecture: {architecture}. "
        "Check the architecture used during training."
    )

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

model.load_state_dict(checkpoint["state_dict"], strict=True)
model = model.to(DEVICE)
model.eval()


# ---------------- IMAGE PREPROCESSING ----------------

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands

hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


# ---------------- PREDICTION FUNCTION ----------------

def predict_sign(image):
    if image is None:
        return {"error": "No image received."}, 400

    # Browser images arrive as RGB.
    rgb_image = np.asarray(image.convert("RGB"))
    height, width, _ = rgb_image.shape

    result = hands_detector.process(rgb_image)

    if not result.multi_hand_landmarks:
        return {
            "prediction": "No hand detected",
            "confidence": 0
        }, 200

    landmarks = result.multi_hand_landmarks[0].landmark

    x_coords = [point.x * width for point in landmarks]
    y_coords = [point.y * height for point in landmarks]

    x1 = max(0, int(min(x_coords)) - 30)
    y1 = max(0, int(min(y_coords)) - 30)
    x2 = min(width, int(max(x_coords)) + 30)
    y2 = min(height, int(max(y_coords)) + 30)

    hand_crop = rgb_image[y1:y2, x1:x2]

    if hand_crop.size == 0:
        return {"error": "Unable to crop hand."}, 400

    hand_image = Image.fromarray(hand_crop)
    input_tensor = preprocess(hand_image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

    index = predicted_index.item()

    return {
        "prediction": classes[index],
        "confidence": round(confidence.item() * 100, 2)
    }, 200


# ---------------- WEB ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
        result, status_code = predict_sign(image)
        return jsonify(result), status_code

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(error)}), 500


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
