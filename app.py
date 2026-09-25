"""
Brain Tumor MRI — Flask Backend API
Endpoints:
  POST /predict  — upload MRI image → tumor type + confidence scores
  POST /explain  — upload MRI image → Grad-CAM heatmap overlay (base64 PNG)
  GET  /health   — liveness check
"""

import os
import io
import base64
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from preprocessing import preprocess_image, IMG_SIZE
from gradcam import explain, get_last_conv_layer

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = "brain_tumor_model.keras"

# Class descriptions shown in the UI
CLASS_INFO = {
    "glioma":      "Glioma is a tumor that originates in the glial cells of the brain or spine.",
    "meningioma":  "Meningioma arises from the meninges, the membranes surrounding the brain and spinal cord.",
    "notumor":     "No tumor detected. The MRI scan appears normal.",
    "pituitary":   "Pituitary tumor forms in the pituitary gland at the base of the brain.",
}

# ─── Load model once at startup ───────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
LAST_CONV = get_last_conv_layer(model)
print(f"Model loaded. Last Conv2D layer: {LAST_CONV}")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def read_image_from_request():
    """
    Read uploaded image from Flask request (multipart/form-data, key='file').
    Returns:
        original_img    : np.ndarray (H, W, 3) uint8
        preprocessed    : np.ndarray (1, 224, 224, 3) float32
    """
    if "file" not in request.files:
        return None, None

    file       = request.files["file"]
    img_bytes  = file.read()
    pil_img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    original   = np.array(pil_img)                         # (H, W, 3) uint8

    preprocessed = preprocess_image(original.copy())       # (224, 224, 3) float32
    preprocessed = np.expand_dims(preprocessed, axis=0)    # (1, 224, 224, 3)
    return original, preprocessed


def ndarray_to_base64_png(img_array):
    """Convert a uint8 np.ndarray (H, W, 3) to a base64-encoded PNG string."""
    img_bgr   = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    _, buffer  = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory(".", "index.html")
  
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart/form-data  { file: <image> }
    Returns JSON:
    {
      "predicted_class": "glioma",
      "confidence": 0.943,
      "scores": { "glioma": 0.943, "meningioma": 0.032, ... },
      "description": "...",
      "original_image": "<base64 PNG>"
    }
    """
    original, preprocessed = read_image_from_request()
    if original is None:
        return jsonify({"error": "No file provided. Use key 'file'."}), 400

    preds         = model.predict(preprocessed, verbose=0)[0]       # (4,)
    class_idx     = int(np.argmax(preds))
    predicted     = CLASSES[class_idx]
    confidence    = float(preds[class_idx])
    scores        = {cls: float(preds[i]) for i, cls in enumerate(CLASSES)}

    # Resize original to 224×224 for display consistency
    display_img   = cv2.resize(original, IMG_SIZE)
    original_b64  = ndarray_to_base64_png(display_img)

    return jsonify({
        "predicted_class": predicted,
        "confidence":      round(confidence, 4),
        "scores":          {k: round(v, 4) for k, v in scores.items()},
        "description":     CLASS_INFO[predicted],
        "original_image":  original_b64,
    })


@app.route("/explain", methods=["POST"])
def explain_route():
    """
    Accepts: multipart/form-data  { file: <image> }
    Returns JSON:
    {
      "predicted_class": "glioma",
      "confidence": 0.943,
      "scores": { ... },
      "description": "...",
      "original_image":  "<base64 PNG>",
      "gradcam_image":   "<base64 PNG>",
      "explanation":     "Grad-CAM highlights the region..."
    }
    """
    original, preprocessed = read_image_from_request()
    if original is None:
        return jsonify({"error": "No file provided. Use key 'file'."}), 400

    # Prediction
    preds         = model.predict(preprocessed, verbose=0)[0]
    class_idx     = int(np.argmax(preds))
    predicted     = CLASSES[class_idx]
    confidence    = float(preds[class_idx])
    scores        = {cls: float(preds[i]) for i, cls in enumerate(CLASSES)}

    # Grad-CAM
    display_img   = cv2.resize(original, IMG_SIZE)
    overlay, _, _ = explain(model, preprocessed, display_img, layer_name=LAST_CONV)

    original_b64  = ndarray_to_base64_png(display_img)
    gradcam_b64   = ndarray_to_base64_png(overlay)

    explanation = (
        f"The highlighted regions (warm colours — red/yellow) show the areas "
        f"the model focused on to classify this scan as '{predicted}' "
        f"with {confidence*100:.1f}% confidence. "
        f"Cool colours (blue/green) indicate regions with lower influence."
    )

    return jsonify({
        "predicted_class": predicted,
        "confidence":      round(confidence, 4),
        "scores":          {k: round(v, 4) for k, v in scores.items()},
        "description":     CLASS_INFO[predicted],
        "original_image":  original_b64,
        "gradcam_image":   gradcam_b64,
        "explanation":     explanation,
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
