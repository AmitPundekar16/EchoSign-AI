"""
tutor_app.py
============
EchoSign: Interactive Sign Language Learning Companion for the Non-Verbal.

Features:
- Structured interactive lessons for practicing vital signs.
- Real-time AI landmark detection (hands + facial relation).
- Immediate evaluation with visual cues and celebratory audio chimes.
- Gamified scoring, streaks, and confidence thresholds.
- Dynamic dual mode: Interactive Tutor Mode & Free Practice Mode.
"""

import collections
import json
import os
import pickle
import random
import sys
import time

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

import cv2
import numpy as np

from utils import config
from utils.features import build_feature_vector, FEATURE_NAMES
from utils.landmarks import LandmarkDetector, draw_landmarks, hand_bounding_box
from utils.preprocessing import resample_sequence
from utils.ui_renderer import render_studio_frame


# ---------------------------------------------------------------------------
# Pre-defined Lesson Curriculum & Visual Instructions
# ---------------------------------------------------------------------------
LESSON_GUIDES = {
    "FATHER": {
        "title": "Father (Family)",
        "guide": "Open hand with 5 fingers extended. Touch thumb tip to FOREHEAD.",
        "tip": "Keep your hand near your forehead so the facial-distance AI detects it."
    },
    "MOTHER": {
        "title": "Mother (Family)",
        "guide": "Open hand with 5 fingers extended. Touch thumb tip to CHIN.",
        "tip": "Rest your thumb gently on your chin."
    },
    "NO": {
        "title": "No (Essentials)",
        "guide": "Extend index & middle finger, then snap them down to the thumb.",
        "tip": "Perform a decisive closing motion toward your thumb."
    }
}


class SignModel:
    """Wraps either a Keras LSTM model or a scikit-learn RandomForest."""

    def __init__(self, run_config):
        self.run_config = run_config
        self.model_type = run_config["model_type"]

        model_path = run_config.get("model_path", "")
        if not os.path.isabs(model_path) or not os.path.exists(model_path):
            fallback = config.MODEL_FILE_KERAS if self.model_type == "lstm" else config.MODEL_FILE_SKLEARN
            if os.path.exists(fallback):
                model_path = fallback
            elif os.path.exists(os.path.join(config.PROJECT_ROOT, model_path)):
                model_path = os.path.join(config.PROJECT_ROOT, model_path)

        if self.model_type == "lstm":
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
        else:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

    def predict_proba(self, fixed_sequence):
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
            print(f"Missing {path}. Train the model first.")
            sys.exit(1)

    with open(config.CONFIG_FILE) as f:
        run_config = json.load(f)

    with open(config.LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)

    with open(config.SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    model = SignModel(run_config)
    return model, label_encoder, scaler, run_config


def play_chime(success=True):
    if HAS_WINSOUND:
        try:
            if success:
                winsound.MessageBeep(winsound.MB_OK)
            else:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass


def extract_positive_signs(label_encoder, separator="::"):
    """Filter out NOT_* negative classes to get actionable learning signs."""
    positive_signs = []
    for cls_name in label_encoder.classes_:
        raw_label = cls_name.split(separator)[-1] if separator in cls_name else cls_name
        if not raw_label.startswith("NOT_"):
            positive_signs.append(raw_label)
    return list(dict.fromkeys(positive_signs))



def main():
    model, label_encoder, scaler, run_config = load_artifacts()
    sequence_len = run_config["sequence_len"]
    separator = run_config.get("class_separator", "::")

    positive_signs = extract_positive_signs(label_encoder, separator)
    if not positive_signs:
        positive_signs = ["FATHER", "MOTHER", "NO"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    window_name = "EchoSign AI - Interactive Tutor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    detector = LandmarkDetector()
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    frame_buffer = collections.deque(maxlen=config.MAX_SEQUENCE_LEN)
    prediction_buffer = collections.deque(maxlen=config.PREDICTION_BUFFER_SIZE)

    min_frame_interval = 1.0 / config.TARGET_CAPTURE_FPS
    last_capture = time.time()

    # Tutor State
    state = {
        "mode": "tutor",             # 'tutor' or 'free'
        "current_target": positive_signs[0],
        "target_idx": 0,
        "score": 0,
        "streak": 0,
        "status": "active",          # 'active', 'success', 'timeout'
        "step_start_time": time.time(),
        "time_limit": 10.0,          # seconds per challenge
        "last_pred_label": "",
        "last_pred_conf": 0.0,
        "cooldown_until": 0.0
    }

    print("\n" + "=" * 60)
    print("  EchoSign AI - Interactive Sign Language Companion")
    print(f"  Available lessons: {', '.join(positive_signs)}")
    print("=" * 60 + "\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            detection = detector.process(frame)

            now = time.time()
            if now - last_capture >= min_frame_interval:
                vec = build_feature_vector(detection, image_w, image_h)
                frame_buffer.append(vec)
                last_capture = now

            # AI Inference
            pred_label = ""
            pred_conf = 0.0
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

                votes = collections.Counter(c for c, _ in prediction_buffer if c is not None)
                if votes:
                    best_class, count = votes.most_common(1)[0]
                    if count >= config.MIN_CONSECUTIVE_AGREEMENT:
                        confs = [c for cls, c in prediction_buffer if cls == best_class]
                        raw_label = best_class.split(separator)[-1] if separator in best_class else best_class
                        if not raw_label.startswith("NOT_"):
                            pred_label = raw_label
                            pred_conf = float(np.mean(confs))

            state['last_pred_label'] = pred_label
            state['last_pred_conf'] = pred_conf

            # Tutor Logic Loop
            if state['mode'] == "tutor":
                if state['status'] == "active":
                    # Check for correct sign execution
                    if pred_label == state['current_target'] and pred_conf >= 0.70:
                        state['status'] = "success"
                        state['score'] += 100
                        state['streak'] += 1
                        state['cooldown_until'] = now + 2.2
                        play_chime(success=True)

                    # Check for timeout
                    elif (now - state['step_start_time']) >= state['time_limit']:
                        state['status'] = "timeout"
                        state['streak'] = 0
                        state['cooldown_until'] = now + 2.0
                        play_chime(success=False)

                elif state['status'] in ("success", "timeout"):
                    if now >= state['cooldown_until']:
                        # Advance to next lesson
                        state['target_idx'] = (state['target_idx'] + 1) % len(positive_signs)
                        state['current_target'] = positive_signs[state['target_idx']]
                        state['status'] = "active"
                        state['step_start_time'] = now
                        frame_buffer.clear()
                        prediction_buffer.clear()

            # Render landmarks on camera feed
            draw_landmarks(frame, detection)

            # Build full modern Studio Canvas
            studio_canvas = render_studio_frame(frame, detection, state, positive_signs)

            cv2.imshow(window_name, studio_canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n'):  # Skip / Next sign
                state['target_idx'] = (state['target_idx'] + 1) % len(positive_signs)
                state['current_target'] = positive_signs[state['target_idx']]
                state['status'] = "active"
                state['step_start_time'] = time.time()
                frame_buffer.clear()
                prediction_buffer.clear()
            elif key == ord('m'):  # Toggle mode
                state['mode'] = "free" if state['mode'] == "tutor" else "tutor"
                state['status'] = "active"
                state['step_start_time'] = time.time()
            elif key == ord('r'):  # Reset score
                state['score'] = 0
                state['streak'] = 0

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
