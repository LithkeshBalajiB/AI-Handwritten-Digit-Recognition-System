"""
MNIST Handwritten Digit Recognizer — Flask Backend
===================================================

Serves a small web UI where users can either draw a digit on a canvas or
upload an image file, then sends the image to the model and returns the
predicted digit, confidence score, and full per-class probability vector.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import base64
import io
import os
from typing import Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "mnist_model.h5")

app = Flask(__name__)

# Lazy-loaded model (loaded on first request to keep startup fast).
_model = None


def get_model():
    """Load and cache the trained Keras model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run `python train_model.py` first."
            )
        _model = load_model(MODEL_PATH)
    return _model


# ----------------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------------

def _decode_image(payload: dict, files) -> Image.Image:
    """Accept either a base64 data URL (canvas) or an uploaded image file."""
    if "image" in payload and payload["image"]:
        data_url: str = payload["image"]
        # Strip the "data:image/png;base64," prefix if present.
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        return Image.open(io.BytesIO(raw))

    if "file" in files:
        return Image.open(files["file"].stream)

    raise ValueError("No image provided. Send `image` (base64) or `file` (upload).")


def preprocess(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an arbitrary input image into a (1, 28, 28) float32 array
    matching the MNIST training distribution: white digit on black background,
    cropped to its bounding box, resized so the longer side fits in 20px, then
    pasted into a 28x28 canvas and shifted so the digit's center of mass sits
    at the image center. This mirrors the original LeCun MNIST preprocessing.
    """
    # 1. Grayscale.
    img = image.convert("L")
    arr = np.array(img)

    # 2. Invert if background is light (canvas / typical photo).
    if arr.mean() > 127:
        arr = 255 - arr

    # 3. Bounding box of "ink" pixels.
    mask = arr > 50
    if not mask.any():
        empty = np.zeros((28, 28), dtype="float32")
        return empty.reshape(1, 28, 28), empty

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape
    # 4. Resize longest side to 20px (MNIST keeps a 4px margin around each digit).
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    resized = Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS)

    # 5. Paste centered into a 28x28 black canvas.
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))
    canvas_arr = np.array(canvas).astype("float32")

    # 6. Shift so the digit's center of mass is at (14, 14) — what MNIST does.
    total = canvas_arr.sum()
    if total > 0:
        ys2, xs2 = np.indices(canvas_arr.shape)
        cy = (ys2 * canvas_arr).sum() / total
        cx = (xs2 * canvas_arr).sum() / total
        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))
        canvas_arr = np.roll(canvas_arr, shift=(shift_y, shift_x), axis=(0, 1))

    canvas_arr /= 255.0
    return canvas_arr.reshape(1, 28, 28), canvas_arr


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        image = _decode_image(payload, request.files)
        x, _ = preprocess(image)

        model = get_model()
        probs = model.predict(x, verbose=0)[0]
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])

        return jsonify(
            {
                "digit": digit,
                "confidence": confidence,
                "probabilities": [float(p) for p in probs],
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001 — return any error to the client
        return jsonify({"error": f"Prediction failed: {exc}"}), 400


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Bind to localhost only by default for safety.
    app.run(host="127.0.0.1", port=5000, debug=True)
