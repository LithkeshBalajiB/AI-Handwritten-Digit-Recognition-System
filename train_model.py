"""
MNIST Handwritten Digit Recognizer — Model Training
====================================================

Trains a small but accurate Sequential neural network on the MNIST dataset
(60,000 training images, 10,000 test images of 28x28 grayscale handwritten
digits) and saves the trained weights to ``model/mnist_model.h5``.

Run from the project root:

    python train_model.py

Expected test accuracy: ~98% in 10 epochs on CPU (~2-5 minutes).
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Make runs reproducible
tf.random.set_seed(42)
np.random.seed(42)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_model.h5")


def load_data():
    """Load MNIST and normalize pixel values to the [0, 1] range."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_model() -> tf.keras.Model:
    """Build a simple Sequential network: Flatten -> Dense -> Dense -> Softmax."""
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28), name="input"),
            layers.Flatten(name="flatten"),
            layers.Dense(256, activation="relu", name="dense_1"),
            layers.Dropout(0.2, name="dropout_1"),
            layers.Dense(128, activation="relu", name="dense_2"),
            layers.Dropout(0.2, name="dropout_2"),
            layers.Dense(10, activation="softmax", name="output"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    model = build_model()
    model.summary()

    print("\nTraining...")
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=2,
    )

    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss:     {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    model.save(MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
