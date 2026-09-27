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


def draw_hud(frame, state):
    """Draws a modern, polished tutor overlay on the video feed."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Header Bar
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "EchoSign AI", (20, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 215, 255), 2)
    cv2.putText(frame, "| Interactive Tutor for Non-Verbal Learning", (195, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Mode indicator
    mode_text = f"Mode: {state['mode'].upper()}"
    cv2.putText(frame, mode_text, (w - 240, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # 2. Score & Streak Box (Top Right)
    score_overlay = frame.copy()
    cv2.rectangle(score_overlay, (w - 230, 65), (w - 15, 135), (25, 25, 30), -1)
    cv2.addWeighted(score_overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (w - 230, 65), (w - 15, 135), (80, 80, 90), 1)

    cv2.putText(frame, f"SCORE: {state['score']}", (w - 215, 95),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    streak_col = (0, 165, 255) if state['streak'] > 0 else (180, 180, 180)
    cv2.putText(frame, f"STREAK: {state['streak']}x", (w - 215, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, streak_col, 2)

    # 3. Main Lesson Panel (Top Left)
    if state['mode'] == "tutor":
        panel_overlay = frame.copy()
        cv2.rectangle(panel_overlay, (15, 65), (550, 195), (20, 25, 35), -1)
        cv2.addWeighted(panel_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (15, 65), (550, 195), (0, 180, 255), 2)

        cur_sign = state['current_target']
        info = LESSON_GUIDES.get(cur_sign, {
            "title": cur_sign,
            "guide": f"Demonstrate the sign for {cur_sign}.",
            "tip": "Form the gesture clearly in front of the camera."
        })

        cv2.putText(frame, f"PRACTICE: {info['title']}", (28, 98),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(frame, f"Action: {info['guide'][:55]}", (28, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(frame, f"Tip: {info['tip'][:60]}", (28, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 255), 1)

        # Timer progress bar
        elapsed = time.time() - state['step_start_time']
        remaining = max(0.0, state['time_limit'] - elapsed)
        ratio = remaining / state['time_limit']

        bar_x, bar_y, bar_w, bar_h = 28, 175, 490, 8
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * ratio)
        fill_col = (0, 255, 120) if ratio > 0.3 else (0, 80, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_col, -1)

    # 4. Live Recognition Indicator (Bottom Left)
    live_overlay = frame.copy()
    cv2.rectangle(live_overlay, (15, h - 85), (380, h - 15), (20, 20, 25), -1)
    cv2.addWeighted(live_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (15, h - 85), (380, h - 15), (60, 60, 70), 1)

    detected_text = f"DETECTED: {state['last_pred_label'] if state['last_pred_label'] else 'Detecting...'}"
    conf_col = (0, 255, 120) if state['last_pred_label'] else (180, 180, 180)
    cv2.putText(frame, detected_text, (25, h - 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, conf_col, 2)
    conf_pct = f"CONFIDENCE: {state['last_pred_conf'] * 100:.1f}%" if state['last_pred_label'] else "CONFIDENCE: --"
    cv2.putText(frame, conf_pct, (25, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 5. Celebration Banner on Success
    if state['status'] == "success":
        banner_overlay = frame.copy()
        cy = h // 2
        cv2.rectangle(banner_overlay, (0, cy - 60), (w, cy + 60), (0, 140, 30), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "CORRECT! EXCELLENT EXECUTION!", (w // 2 - 280, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "+100 Points Awarded", (w // 2 - 110, cy + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    # 6. Timeout Alert
    elif state['status'] == "timeout":
        banner_overlay = frame.copy()
        cy = h // 2
        cv2.rectangle(banner_overlay, (0, cy - 50), (w, cy + 50), (0, 60, 180), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "TIME'S UP! Keep Practicing!", (w // 2 - 220, cy + 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    # 7. Navigation Controls (Bottom Right)
    cv2.putText(frame, "[N] Next Sign  |  [M] Change Mode  |  [Q] Quit", (w - 420, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


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

            # Render Landmarks & HUD
            draw_landmarks(frame, detection)
            draw_hud(frame, state)

            cv2.imshow("EchoSign AI - Interactive Tutor", frame)
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
