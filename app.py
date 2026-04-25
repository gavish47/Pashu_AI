import gradio as gr
import numpy as np
import cv2
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/best_finetuned.keras"   # ✅ using .keras
CLASS_MAP_PATH = "models/class_indices.json"
IMAGE_SIZE = (224, 224)

# =========================
# LOAD MODEL (SAFE WAY)
# =========================
print("🚀 Loading model...")

try:
    model = load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False   # 🔥 fixes normalization issue
    )
    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Model load failed:", e)
    model = None

# =========================
# LOAD CLASS MAP
# =========================
with open(CLASS_MAP_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {int(v): k for k, v in class_indices.items()}

# =========================
# BREED INFO
# =========================
breed_info = {
    "Sahiwal": "Punjab (India/Pakistan) - High milk yield",
    "Lakhimi": "Assam - Adapted to humid climate",
    "Siri": "Himalayan region - Strong breed",
    "Umblachery": "Tamil Nadu - Draught breed",
}

# =========================
# PREPROCESS
# =========================
def preprocess_image(image):
    img = cv2.resize(image, IMAGE_SIZE)
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

# =========================
# PREDICT
# =========================
def predict(image):
    if model is None:
        return "❌ Model failed to load."

    try:
        img = preprocess_image(image)
        preds = model.predict(img)[0]

        top_index = int(np.argmax(preds))
        confidence = float(preds[top_index] * 100)
        breed = index_to_class[top_index]

        # Top 3
        top_indices = preds.argsort()[-3:][::-1]
        top_results = [
            f"{index_to_class[int(i)]}: {preds[int(i)]*100:.2f}%"
            for i in top_indices
        ]

        info = breed_info.get(breed, "Unknown breed")

        return f"""
🐄 Breed: {breed}
📊 Confidence: {confidence:.2f}%

📍 Info: {info}

🔝 Top Predictions:
{chr(10).join(top_results)}
""".strip()

    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

# =========================
# UI
# =========================
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="🐄 Pashu AI - Cow Breed Predictor",
    description="Upload a cow image to predict its breed using AI"
)

# =========================
# RUN
# =========================
interface.launch()
