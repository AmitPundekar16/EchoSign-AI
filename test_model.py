"""
test_model.py
==============
Real-time sign-language recognition. Run with no arguments:

    python test_model.py

It immediately:
    1. Loads the trained model + label encoder + scaler + config.json
       produced by train_model.py.
    2. Opens the webcam.
    3. Detects both hands and the face every frame using exactly the same
       feature pipeline used during training (utils/features.py).
    4. Maintains a rolling buffer of recent frames, resamples it to the
       model's fixed sequence length, and runs the model on it.
    5. Smooths predictions over time (voting + confidence threshold) so
       the displayed sign doesn't flicker frame to frame.
"""

import collections
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

from utils import config
from utils.features import build_feature_vector, FEATURE_NAMES
from utils.landmarks import LandmarkDetector, draw_landmarks, hand_bounding_box
from utils.preprocessing import resample_sequence


class SignModel:
    """Wraps either a Keras LSTM model or a scikit-learn RandomForest
    behind one common `predict_proba(sequence)` interface."""

    def __init__(self, run_config):
        self.run_config = run_config
        self.model_type = run_config["model_type"]

        if self.model_type == "lstm":
            from tensorflow import keras
            self.model = keras.models.load_model(run_config["model_path"])
        else:
            with open(run_config["model_path"], "rb") as f:
                self.model = pickle.load(f)

    def predict_proba(self, fixed_sequence):
        """fixed_sequence: (sequence_len, num_features) scaled array."""
        if self.model_type == "lstm":
            batch = fixed_sequence[np.newaxis, ...]
            probs = self.model.predict(batch, verbose=0)[0]
            return probs
        else:
            mean = fixed_sequence.mean(axis=0)
            std = fixed_sequence.std(axis=0)
            mn = fixed_sequence.min(axis=0)
            mx = fixed_sequence.max(axis=0)
            agg = np.concatenate([mean, std, mn, mx])[np.newaxis, :]
            probs = self.model.predict_proba(agg)[0]
            return probs


def load_artifacts():
    for path in (config.CONFIG_FILE, config.LABEL_ENCODER_FILE, config.SCALER_FILE):
        if not os.path.exists(path):
            print(f"Missing {path}. Run train_model.py first.")
            sys.exit(1)

    with open(config.CONFIG_FILE) as f:
        run_config = json.load(f)

    with open(config.LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)

    with open(config.SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    if run_config["feature_length"] != len(FEATURE_NAMES):
        print(
            "WARNING: the loaded model was trained with a different feature "
            "vector length than the current utils/features.py. Predictions "
            "will likely be wrong. Retrain the model."
        )

    model = SignModel(run_config)
    return model, label_encoder, scaler, run_config


def split_class_name(class_name, separator):
    if separator in class_name:
        language, label = class_name.split(separator, 1)
        return language, label
    return "", class_name


def draw_prediction_panel(frame, language, label, confidence):
    text_sign = f"SIGN: {label if label else 'UNKNOWN'}"
    text_conf = f"CONFIDENCE: {confidence * 100:.1f}%" if label else "CONFIDENCE: --"
    cv2.rectangle(frame, (0, 0), (330, 70), (0, 0, 0), -1)
    cv2.putText(frame, text_sign, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0) if label else (0, 0, 255), 2)
    cv2.putText(frame, text_conf, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0) if label else (0, 0, 255), 2)
    if language:
        cv2.putText(frame, f"Language: {language}", (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def draw_hand_prediction_boxes(frame, detection, image_w, image_h, label, confidence):
    for hand_name, color in (("Left", (255, 0, 0)), ("Right", (0, 255, 0))):
        pts = detection["hands"][hand_name]
        if pts is not None:
            x1, y1, x2, y2 = hand_bounding_box(pts, image_w, image_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(frame, f"{label} {confidence * 100:.0f}%",
                            (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)


def main():
    model, label_encoder, scaler, run_config = load_artifacts()
    sequence_len = run_config["sequence_len"]
    separator = run_config.get("class_separator", "::")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    detector = LandmarkDetector()
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    frame_buffer = collections.deque(maxlen=config.MAX_SEQUENCE_LEN)
    prediction_buffer = collections.deque(maxlen=config.PREDICTION_BUFFER_SIZE)

    min_frame_interval = 1.0 / config.TARGET_CAPTURE_FPS
    last_capture = time.time()

    displayed_language, displayed_label, displayed_conf = "", "", 0.0

    print("Camera opened. Show a sign to the camera.")
    print(f"Classes known to the model: {list(label_encoder.classes_)}\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = cv2.flip(frame, 1)
            detection = detector.process(frame)

            now = time.time()
            if now - last_capture >= min_frame_interval:
                vec = build_feature_vector(detection, image_w, image_h)
                frame_buffer.append(vec)
                last_capture = now

            if len(frame_buffer) >= config.MIN_SEQUENCE_LEN:
                raw_seq = np.stack(list(frame_buffer), axis=0)
                fixed_seq = resample_sequence(raw_seq, target_len=sequence_len)
                scaled_seq = scaler.transform(fixed_seq)

                probs = model.predict_proba(scaled_seq)
                top_idx = int(np.argmax(probs))
                top_conf = float(probs[top_idx])
                top_class = label_encoder.classes_[top_idx]

                if top_conf >= config.MIN_CONFIDENCE_THRESHOLD:
                    prediction_buffer.append((top_class, top_conf))
                else:
                    prediction_buffer.append((None, top_conf))

                # Majority vote over the recent prediction buffer.
                votes = collections.Counter(
                    c for c, _ in prediction_buffer if c is not None
                )
                if votes:
                    best_class, count = votes.most_common(1)[0]
                    if count >= config.MIN_CONSECUTIVE_AGREEMENT:
                        confs = [c for cls, c in prediction_buffer if cls == best_class]
                        displayed_language, displayed_label = split_class_name(
                            best_class, separator
                        )
                        displayed_conf = float(np.mean(confs))
                    else:
                        displayed_language, displayed_label, displayed_conf = "", "", 0.0
                else:
                    displayed_language, displayed_label, displayed_conf = "", "", 0.0

            draw_landmarks(frame, detection)
            draw_hand_prediction_boxes(frame, detection, image_w, image_h,
                                        displayed_label, displayed_conf)
            draw_prediction_panel(frame, displayed_language, displayed_label,
                                   displayed_conf)

            cv2.imshow("Sign Language Recognition", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
