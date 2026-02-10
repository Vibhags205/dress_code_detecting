import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import numpy as np
import tensorflow as tf


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


# 1. Load the Keras model
model = tf.keras.models.load_model("model/keras_model.h5", compile=False)
input_shape = model.input_shape
height = int(input_shape[1])
width = int(input_shape[2])

# 2. Load labels
labels = load_labels("model/labels.txt")

# 3. Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Resize and normalize for the model
    image_resized = cv2.resize(frame, (width, height))
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
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Dress Code Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()