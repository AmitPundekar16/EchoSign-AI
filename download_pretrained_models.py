"""
download_pretrained_models.py
Downloads both:
1. Google's Official MediaPipe Gesture Recognizer (.task)
2. Open-source 26-Letter ASL Alphabet TFLite Classifier (.tflite) + label map
"""

import os
import urllib.request
import ssl

# Bypass SSL verify issues if any corporate/antivirus proxy exists
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

MODELS_DIR = os.path.abspath("mp_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Google Official Gesture Recognizer
url_gesture = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
path_gesture = os.path.join(MODELS_DIR, "gesture_recognizer.task")

print(f"Downloading Google MediaPipe Gesture Recognizer from {url_gesture}...")
try:
    req = urllib.request.Request(url_gesture, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl_ctx) as resp, open(path_gesture, "wb") as out_file:
        out_file.write(resp.read())
    print(f"SUCCESS: gesture_recognizer.task ({os.path.getsize(path_gesture):,} bytes)")
except Exception as e:
    print(f"Failed to download gesture recognizer: {e}")

# 2. 26-Letter ASL TFLite Model
url_asl_tflite = "https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/model/keypoint_classifier/keypoint_classifier.tflite"
path_asl_tflite = os.path.join(MODELS_DIR, "asl_alphabet_26.tflite")

print(f"\nDownloading ASL 26-Letter Alphabet TFLite model from {url_asl_tflite}...")
try:
    req = urllib.request.Request(url_asl_tflite, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl_ctx) as resp, open(path_asl_tflite, "wb") as out_file:
        out_file.write(resp.read())
    print(f"SUCCESS: asl_alphabet_26.tflite ({os.path.getsize(path_asl_tflite):,} bytes)")
except Exception as e:
    print(f"Failed to download ASL tflite model: {e}")

# 3. ASL Label Map
url_labels = "https://raw.githubusercontent.com/Muhib-Mehdi/ASL-Recognition-System/main/model/keypoint_classifier/keypoint_classifier_label.csv"
path_labels = os.path.join(MODELS_DIR, "asl_labels.csv")

print(f"\nDownloading ASL label map from {url_labels}...")
try:
    req = urllib.request.Request(url_labels, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl_ctx) as resp, open(path_labels, "wb") as out_file:
        out_file.write(resp.read())
    print(f"SUCCESS: asl_labels.csv ({os.path.getsize(path_labels):,} bytes)")
except Exception as e:
    print(f"Failed to download ASL labels: {e}")

print("\nModel download process completed!")
