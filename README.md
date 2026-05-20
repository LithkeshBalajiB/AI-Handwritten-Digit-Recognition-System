# 🔢 MNIST Digit Recognizer — AI-Powered Web Application

> Draw a digit or upload an image and watch a neural network identify it in real time.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)


---

## Overview

This project is a full-stack machine learning web application that recognizes handwritten digits (0–9) using a neural network trained on the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Users can either **draw a digit directly on an in-browser canvas** or **upload an image file**, and the model instantly returns the predicted digit, its confidence score, and the full probability distribution across all ten classes.

The preprocessing pipeline faithfully replicates the original LeCun MNIST methodology — bounding-box cropping, proportional resizing to 20 px, center-of-mass shifting — so the model generalizes well to real user input beyond the training set.

---

## Screenshots

| Draw Mode | Prediction Result |
|---|---|
|<img width="1918" height="955" alt="image" src="https://github.com/user-attachments/assets/cf9d7076-6d17-4316-8fc6-ebc3afae3442" />|<img width="1918" height="953" alt="image" src="https://github.com/user-attachments/assets/99e4102c-a23f-4a03-8051-1fe0433780ff" />|

---

## Features

- **Interactive canvas** — draw freehand in the browser, no external tools needed
- **Image upload** — accepts any common image format (PNG, JPG, etc.)
- **Real-time inference** — sub-second prediction via a REST `/predict` endpoint
- **Confidence scores** — returns the predicted digit, top confidence, and all 10 class probabilities
- **Faithful MNIST preprocessing** — bounding-box crop, 20 px resize with 4 px margin, center-of-mass shift
- **Lazy model loading** — the Keras model is loaded on first request, keeping startup instant
- **~98% test accuracy** — achieved in 10 training epochs on CPU (≈ 2–5 minutes)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Web Framework | Flask 3 |
| Image Processing | Pillow, NumPy |
| Frontend | HTML5 Canvas, CSS3, Vanilla JavaScript |
| Model Format | Keras `.h5` |

---

## Project Structure

```
AI-Handwritten-Digit-Recognition-System/
├── app.py              # Flask application & /predict REST endpoint
├── train_model.py      # Model definition, training loop, and evaluation
├── requirements.txt    # Python dependencies
├── model/
│   └── mnist_model.h5  # Saved Keras model (generated after training)
├── static/             # CSS and JavaScript assets
├── templates/
│   └── index.html      # Main UI (canvas + upload)
├── uploads/            # Temporary storage for uploaded images
└── screenshots/        # UI screenshots for README (add your own here)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/LithkeshBalajiB/AI-Handwritten-Digit-Recognition-System.git
cd AI-Handwritten-Digit-Recognition-System

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the Model

Run this once to download MNIST and train the neural network. The trained weights are saved to `model/mnist_model.h5`.

```bash
python train_model.py
```

Expected output:
```
Test loss:     0.0712
Test accuracy: 0.9812
Saved trained model to: model/mnist_model.h5
```

Training takes approximately **2–5 minutes on CPU**.

### Run the Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## How It Works

### Model Architecture

The classifier is a compact feedforward network trained with the Adam optimizer and sparse categorical cross-entropy loss:

```
Input (28×28)
    └─ Flatten
    └─ Dense(256, ReLU) → Dropout(0.2)
    └─ Dense(128, ReLU) → Dropout(0.2)
    └─ Dense(10, Softmax)
```

### Preprocessing Pipeline

Raw input images go through the following normalization steps before inference, matching the original MNIST distribution:

1. Convert to grayscale
2. Invert if the background is light (white paper / canvas background)
3. Crop to the bounding box of non-zero (ink) pixels
4. Resize proportionally so the longer side fits within 20 px
5. Paste centered into a blank 28×28 canvas (4 px margin on each side)
6. Shift so the digit's center of mass aligns with pixel (14, 14)
7. Normalize pixel values to `[0, 1]`

### API Endpoint

**`POST /predict`**

Accepts either a base64-encoded data URL (from the canvas) or a multipart file upload.

```json
// Response
{
  "digit": 7,
  "confidence": 0.9983,
  "probabilities": [0.0001, 0.0002, 0.0003, 0.0001, 0.0002, 0.0001, 0.0002, 0.9983, 0.0003, 0.0002]
}
```

---

## Dependencies

```
Flask>=3.0.0
tensorflow>=2.15.0,<2.22.0
numpy>=1.24.0
Pillow>=10.0.0
```

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for bug fixes, improvements, or new features (e.g. multi-digit recognition, CNN upgrade, Docker support).

---


## Acknowledgements

- [Yann LeCun et al.](http://yann.lecun.com/exdb/mnist/) for the MNIST dataset
- [TensorFlow / Keras](https://www.tensorflow.org/) for the deep learning framework
- [Flask](https://flask.palletsprojects.com/) for the lightweight web server
