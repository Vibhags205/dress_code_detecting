import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import numpy as np
import tensorflow as tf
import threading
import time

import pyttsx3


VOICE_COOLDOWN_SECONDS = 8
last_voice_time = 0.0


def _init_voice_engine() -> pyttsx3.Engine:
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)
    return engine


voice_engine = _init_voice_engine()
voice_lock = threading.Lock()


def speak_async(message: str) -> None:
    def task() -> None:
        with voice_lock:
            voice_engine.say(message)
            voice_engine.runAndWait()

    threading.Thread(target=task, daemon=True).start()


def load_labels(labels_path: str) -> list[str]:
    with open(labels_path, "r", encoding="utf-8") as labels_file:
        lines = [line.strip() for line in labels_file if line.strip()]
    labels = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels.append(parts[1])
        else:
            labels.append(line)
    return labels


def _is_hdf5_signature(header: bytes) -> bool:
    return header.startswith(b"\x89HDF\r\n\x1a\n")


def _is_zip_signature(header: bytes) -> bool:
    return header.startswith(b"PK\x03\x04")


def load_model_from_path(model_path: str) -> tf.keras.Model:
    if os.path.isdir(model_path):
        return tf.keras.models.load_model(model_path, compile=False)

    with open(model_path, "rb") as model_file:
        header = model_file.read(8)

    if _is_hdf5_signature(header) or _is_zip_signature(header):
        return tf.keras.models.load_model(model_path, compile=False)

    raise ValueError(
        "Model file does not look like a valid HDF5 or .keras archive. "
        "Re-export the model as HDF5 (.h5) or Keras (.keras) and update the path."
    )


def resolve_model_path() -> str:
    explicit_path = os.environ.get("MODEL_PATH")
    if explicit_path:
        return explicit_path

    candidates = [
        "model/keras_model.h5",
        "model/keras_model.keras",
        "model/saved_model",
        "model/model.savedmodel",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No model file found. Set MODEL_PATH or place a model at model/keras_model.h5, "
        "model/keras_model.keras, model/saved_model, or model/model.savedmodel."
    )


# 1. Load the Keras model
model = load_model_from_path(resolve_model_path())
input_shape = model.input_shape
height = int(input_shape[1])
width = int(input_shape[2])

# 2. Load labels
labels = load_labels("model/labels.txt")

# 3. Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert BGR → RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    image_resized = cv2.resize(image_rgb, (width, height))

    # Normalize
    input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)
    input_data = (input_data / 127.5) - 1.0

    # Run Inference
    predictions = model.predict(input_data, verbose=0)
    results = np.squeeze(predictions)

    # Show top result
    top_index = int(np.argmax(results))
    confidence = float(results[top_index]) * 100.0
    label = labels[top_index] if top_index < len(labels) else f"Class {top_index}"
    text = f"{label}: {confidence:.1f}%"
    label_lower = label.lower()
    is_non_compliant = "non" in label_lower and ("complaince" in label_lower or "compliance" in label_lower)
    text_color = (0, 0, 255) if is_non_compliant else (0, 255, 0)

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    current_time = time.time()
    if current_time - last_voice_time >= VOICE_COOLDOWN_SECONDS:
        last_voice_time = current_time
        if is_non_compliant:
            speak_async("Please follow proper dress code")
        else:
            speak_async("Thank you, you may enter")

    cv2.imshow('Dress Code Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()