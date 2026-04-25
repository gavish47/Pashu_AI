
import os
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# =========================
# CONFIG — must match train_breed_classifier.py
# =========================
BACKBONE       = "efficientnet_b0"          # or "mobilenet_v2"
MODEL_PATH     = "models/best_finetuned.keras"
CLASS_MAP_PATH = "models/class_indices.json"
IMAGE_SIZE     = (224, 224)
UPLOAD_FOLDER  = "static/uploads"
ALLOWED_EXT    = {"png", "jpg", "jpeg", "bmp", "webp"}
LOW_CONF_THRESHOLD = 60.0                   # warn user below this %

# Preprocessing must match the training backbone
if BACKBONE == "efficientnet_b0":
    from tensorflow.keras.applications.efficientnet import preprocess_input
elif BACKBONE == "mobilenet_v2":
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
else:
    raise ValueError(f"Unknown backbone: {BACKBONE}")

# =========================
# APP SETUP
# =========================
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print(f"Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)

with open(CLASS_MAP_PATH, "r") as f:
    class_indices = json.load(f)
# JSON keys come back as strings; values are ints. Reverse-map for lookups.
index_to_class = {int(v): k for k, v in class_indices.items()}

breed_info = {
    "Sahiwal": {
        "origin": "Punjab (India/Pakistan)",
        "description": "High milk yield and heat tolerant breed",
    },
    "Lakhimi": {
        "origin": "Assam (India)",
        "description": "Indigenous breed adapted to humid climate",
    },
    "Siri": {
        "origin": "Himalayan region",
        "description": "Strong breed used for draft and milk",
    },
    "Umblachery": {
        "origin": "Tamil Nadu (India)",
        "description": "Used for draught work",
    },
}

# =========================
# HELPERS
# =========================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_and_preprocess(filepath: str) -> np.ndarray:
    """Load an image from disk and prepare it exactly like the training pipeline.

    Returns a (1, H, W, 3) float32 batch ready for model.predict.
    Raises ValueError on unreadable files.
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("Could not read image (corrupt or unsupported format)")

    # CRITICAL: cv2 loads BGR, but the model was trained on RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size. INTER_AREA is best for downscaling.
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # preprocess_input expects float and applies the backbone's specific
    # normalization (e.g. MobileNetV2 -> [-1, 1]; EfficientNet -> identity).
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


def predict_breed(filepath: str) -> dict:
    batch = load_and_preprocess(filepath)
    preds = model.predict(batch, verbose=0)[0]

    top_index = int(np.argmax(preds))
    confidence = float(preds[top_index] * 100)
    breed = index_to_class[top_index]

    # Top-3, excluding the winner
    top_indices = preds.argsort()[-3:][::-1]
    others = [
        {
            "breed": index_to_class[int(i)],
            "conf": round(float(preds[int(i)] * 100), 2),
        }
        for i in top_indices[1:]
    ]

    info = breed_info.get(breed, {"origin": "Unknown", "description": "N/A"})

    result = {
        "breed": breed,
        "confidence": round(confidence, 2),
        "origin": info["origin"],
        "description": info["description"],
        "others": others,
    }
    if confidence < LOW_CONF_THRESHOLD:
        result["warning"] = (
            f"Low confidence ({confidence:.1f}%). The image may not clearly "
            f"show a known breed, or it may be a breed not in the training set."
        )
    return result


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXT))}"
        }), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = predict_breed(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Don't leak internals to the client, but log them server-side.
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. See server logs."}), 500

    # Make the uploaded image URL available for the front-end to display
    result["image_url"] = f"/{UPLOAD_FOLDER}/{filename}"
    return jsonify(result)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
