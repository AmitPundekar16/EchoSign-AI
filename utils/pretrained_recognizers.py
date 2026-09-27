"""
utils/pretrained_recognizers.py
===============================
Unified Ensemble Inference Engine combining:
1. Google's Official MediaPipe Gesture Recognizer (.task)
2. Pre-trained 26-Letter ASL Alphabet Neural Network (.tflite)
3. Deterministic Hand Kinematics & Diagnostic Feedback Engine

Provides zero-latency, multi-model consensus sign language detection.
"""

import copy
import csv
import itertools
import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from . import config
from .finger_geometry import classify_static_sign

# Google Gesture -> Standard Sign Mapping
GOOGLE_GESTURE_MAP = {
    "Pointing_Up": "1",
    "Victory": "2",
    "Open_Palm": "5",
    "Closed_Fist": "A",
    "ILoveYou": "I_LOVE_YOU",
    "Thumb_Up": "THUMBS_UP",
    "Thumb_Down": "THUMBS_DOWN"
}


def preprocess_landmarks_for_tflite(landmarks_21):
    """
    Normalizes 21 MediaPipe hand landmarks into 42-length relative vector
    [x0, y0, x1, y1, ... x20, y20] centered at wrist and scaled by max distance.
    """
    if landmarks_21 is None or len(landmarks_21) < 21:
        return None

    # Deep copy 2D coords
    coords = [[lm[0], lm[1]] for lm in landmarks_21]

    # Translate relative to wrist (index 0)
    base_x, base_y = coords[0][0], coords[0][1]
    for i in range(len(coords)):
        coords[i][0] -= base_x
        coords[i][1] -= base_y

    # Flatten to 1D
    flat = list(itertools.chain.from_iterable(coords))

    # Scale normalization
    max_val = max(map(abs, flat))
    if max_val < 1e-6:
        max_val = 1.0

    normalized = [val / max_val for val in flat]
    return np.array([normalized], dtype=np.float32)


class PretrainedGestureEngine:
    """Ensemble wrapper for Google's Gesture Recognizer & ASL Alphabet TFLite."""

    def __init__(self):
        models_dir = os.path.join(config.PROJECT_ROOT, "mp_models")

        # 1. Initialize Google Gesture Recognizer
        task_path = os.path.join(models_dir, "gesture_recognizer.task")
        self.google_recognizer = None
        if os.path.exists(task_path):
            try:
                options = mp_vision.GestureRecognizerOptions(
                    base_options=BaseOptions(model_asset_path=task_path),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.google_recognizer = mp_vision.GestureRecognizer.create_from_options(options)
            except Exception as e:
                print(f"[Warning] Could not initialize Google Gesture Recognizer: {e}")

        # 2. Initialize 26-Letter ASL TFLite Classifier
        tflite_path = os.path.join(models_dir, "asl_alphabet_26.tflite")
        labels_path = os.path.join(models_dir, "asl_labels.csv")
        self.tflite_interpreter = None
        self.asl_labels = []

        if os.path.exists(tflite_path):
            try:
                import tensorflow as tf
                self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
                self.tflite_interpreter.allocate_tensors()
                self.input_details = self.tflite_interpreter.get_input_details()
                self.output_details = self.tflite_interpreter.get_output_details()

                if os.path.exists(labels_path):
                    with open(labels_path, encoding="utf-8-sig") as f:
                        reader = csv.reader(f)
                        self.asl_labels = [row[0].strip().upper() for row in reader if row]
                else:
                    self.asl_labels = [chr(ord('A') + i) for i in range(26)]
            except Exception as e:
                print(f"[Warning] Could not initialize ASL TFLite model: {e}")

    def predict_asl_alphabet(self, landmarks_21):
        """Runs TFLite ASL Alphabet classifier on 21 hand landmarks."""
        if self.tflite_interpreter is None or landmarks_21 is None:
            return None, 0.0

        vec = preprocess_landmarks_for_tflite(landmarks_21)
        if vec is None:
            return None, 0.0

        self.tflite_interpreter.set_tensor(self.input_details[0]["index"], vec)
        self.tflite_interpreter.invoke()
        output = self.tflite_interpreter.get_tensor(self.output_details[0]["index"])[0]

        top_idx = int(np.argmax(output))
        top_conf = float(output[top_idx])

        if top_idx < len(self.asl_labels):
            letter = self.asl_labels[top_idx]
            return letter, top_conf

        return None, 0.0

    def predict_google_gesture(self, frame_bgr, timestamp_ms):
        """Runs Google MediaPipe Gesture Recognizer on video frame."""
        if self.google_recognizer is None:
            return None, 0.0

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.google_recognizer.recognize_for_video(mp_image, timestamp_ms)
            if result.gestures and len(result.gestures) > 0:
                top_gesture = result.gestures[0][0]
                raw_name = top_gesture.category_name
                conf = float(top_gesture.score)

                mapped = GOOGLE_GESTURE_MAP.get(raw_name, raw_name)
                return mapped, conf
        except Exception:
            pass

        return None, 0.0

    def evaluate(self, frame_bgr, detection, timestamp_ms, target_sign=None):
        """
        Runs the full ensemble:
        - Geometric rule engine (kinematics + feedback)
        - Google Gesture Recognizer
        - ASL 26-Letter TFLite classifier

        Returns:
            dict: {
                "consensus_sign": str or None,
                "consensus_conf": float,
                "feedback": str,
                "google": (label, conf),
                "asl_nn": (label, conf),
                "geometric": (label, conf)
            }
        """
        # 1. Geometric evaluation & feedback
        geom_sign, geom_conf, feedback = classify_static_sign(detection, target_sign=target_sign)

        # Extract dominant hand landmarks
        hands = detection.get("hands", {})
        dom_lms = None
        for side in ("Right", "Left"):
            if hands.get(side):
                dom_lms = hands[side]
                break

        # 2. ASL 26-Letter TFLite classification
        asl_letter, asl_conf = self.predict_asl_alphabet(dom_lms)

        # 3. Google Gesture Recognizer
        google_gesture, google_conf = self.predict_google_gesture(frame_bgr, timestamp_ms)

        # Consensus Resolution:
        # Check target sign agreement across engines
        consensus_sign = None
        consensus_conf = 0.0

        if target_sign:
            t_upper = target_sign.upper()
            scores = []
            if geom_sign == t_upper:
                scores.append(geom_conf)
            if asl_letter == t_upper:
                scores.append(asl_conf)
            if google_gesture == t_upper:
                scores.append(google_conf)

            if scores:
                consensus_sign = t_upper
                # Boost confidence if multiple independent models agree
                consensus_conf = max(scores) if len(scores) == 1 else min(0.99, max(scores) + 0.05 * (len(scores) - 1))
        else:
            # Free mode: pick highest confidence prediction
            candidates = {}
            if geom_sign and geom_conf >= 0.70:
                candidates[geom_sign] = candidates.get(geom_sign, 0) + geom_conf
            if asl_letter and asl_conf >= 0.70:
                candidates[asl_letter] = candidates.get(asl_letter, 0) + asl_conf
            if google_gesture and google_conf >= 0.70:
                candidates[google_gesture] = candidates.get(google_gesture, 0) + google_conf

            if candidates:
                best_label, total_score = max(candidates.items(), key=lambda item: item[1])
                consensus_sign = best_label
                consensus_conf = min(0.99, total_score / (2 if total_score > 1.0 else 1.0))

        return {
            "consensus_sign": consensus_sign,
            "consensus_conf": consensus_conf,
            "feedback": feedback,
            "google": (google_gesture, google_conf),
            "asl_nn": (asl_letter, asl_conf),
            "geometric": (geom_sign, geom_conf)
        }

    def close(self):
        if self.google_recognizer:
            try:
                self.google_recognizer.close()
            except Exception:
                pass
