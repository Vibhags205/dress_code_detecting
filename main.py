import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import numpy as np
import tensorflow as tf
import threading
import time

import pyttsx3

import cv2
import numpy as np
import tensorflow as tf
import requests
import time
from datetime import datetime

# ================= TELEGRAM CONFIG =================
TOKEN = "8300038302:AAFmWafTVDGpHgm5coo3YzmI3VuyMhaMBTY"
CHANNEL_ID = -1003871574876   # your channel id
ALERT_COOLDOWN = 20
last_alert_time = 0
# ==================================================

# -------- TELEGRAM TEST FUNCTION --------
def test_telegram_connection():
    """Test if bot token and channel ID are valid"""
    print("\n" + "="*50)
    print("TESTING TELEGRAM CONNECTION")
    print("="*50)
    
    # Test 1: Verify bot token
    print(f"\n1. Testing bot token...")
    print(f"   Token format: {TOKEN[:20]}...{TOKEN[-10:]}")
    url = f"https://api.telegram.org/bot{TOKEN}/getMe"
    response = requests.get(url)
    print(f"   Bot API Response: {response.json()}")
    
    if not response.json().get('ok'):
        print("   ❌ Bot token is INVALID!")
        print("   ⚠️  Token should be in format: 123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw")
        return False
    else:
        bot_info = response.json()['result']
        print(f"   ✅ Bot verified: @{bot_info['username']}")
    
    # Test 2: Verify channel access
    print(f"\n2. Testing channel access...")
    print(f"   Channel ID: {CHANNEL_ID}")
    url = f"https://api.telegram.org/bot{TOKEN}/getChat"
    response = requests.get(url, params={"chat_id": CHANNEL_ID})
    print(f"   Channel API Response: {response.json()}")
    
    if not response.json().get('ok'):
        print("   ❌ Cannot access channel!")
        print("   Make sure bot is admin in the channel")
        return False
    else:
        chat_info = response.json()['result']
        print(f"   ✅ Channel found: {chat_info.get('title', 'Unknown')}")
    
    print("\n" + "="*50)
    print("TELEGRAM CONNECTION TEST COMPLETE")
    print("="*50 + "\n")
    return True

# -------- TELEGRAM SEND FUNCTION --------
def send_telegram_photo(frame, caption):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

    _, buffer = cv2.imencode(".jpg", frame)
    files = {"photo": buffer.tobytes()}
    data = {
        "chat_id": CHANNEL_ID,
        "caption": caption
    }

    response = requests.post(url, data=data, files=files)

    print("Telegram response:", response.text)
    return response.json().get('ok', False)


# -------- CONVERT SAVED MODEL TO TFLITE IF NEEDED --------
if not os.path.exists("model.tflite"):
    print("Converting saved model to TFLite...")
    model = tf.keras.models.load_model("model/model.savedmodel")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Conversion complete!")

# -------- LOAD MODEL --------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# -------- LOAD LABELS --------
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("Labels:", labels)

# -------- VOICE SETUP --------
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

# 2.5. Test Telegram Connection
print("Testing Telegram configuration...")
telegram_working = test_telegram_connection()
if not telegram_working:
    print("⚠️  WARNING: Telegram alerts will not work!")
    print("⚠️  Please check your bot token and channel ID")
    input("Press Enter to continue anyway, or Ctrl+C to exit...")

# 3. Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
    
    # Check if non-compliant
    is_non_compliant = "non" in label.lower()
    text_color = (0, 0, 255) if is_non_compliant else (0, 255, 0)

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    current_time = time.time()
    
    # Voice Output with cooldown
    if current_time - last_voice_time >= VOICE_COOLDOWN_SECONDS:
        last_voice_time = current_time
        if is_non_compliant:
            speak_async("Please follow proper dress code")
        else:
            speak_async("Thank you, you may enter")
    
    # Telegram Alert for non-compliance with high confidence
    if is_non_compliant and confidence > 80:
        if current_time - last_alert_time > ALERT_COOLDOWN:
            last_alert_time = current_time

            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

            caption = (
                "🚨 DRESS CODE VIOLATION DETECTED\n\n"
                f"🕒 Time: {timestamp}\n"
                "📍 Location: Library Entry Gate\n"
                f"📊 Confidence: {confidence:.1f}%"
            )

            print(f"Sending telegram alert for {label} with {confidence:.1f}% confidence")
            send_telegram_photo(frame.copy(), caption)

    cv2.imshow('Dress Code Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()