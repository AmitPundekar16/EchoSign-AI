"""
tutor_app.py
============
EchoSign AI: Interactive Sign Language Learning Companion for the Non-Verbal.

Features:
- Dual-Engine Recognition:
    1. Instant Geometric Engine (<1ms) for Static Signs: Numbers (1-5), Letters (A, B, C, V)
       with real-time finger posture feedback (zero training dataset required).
    2. Deep Neural Network + Spatial Kinematics Engine for Dynamic Signs:
       Family and Essential Signs (Father, Mother, No, Hello, Thank You).
- Interactive Curriculums with Instant Switching ([C] key):
    * ALL SIGNS
    * NUMBERS (1 to 5)
    * ALPHABET (A, B, C, V)
    * WORDS (Father, Mother, No)
- Real-time AI Landmark Detection with unmasked natural face visualization.
- Interactive Click-to-Enlarge Modal for high-definition demo reference photos.
- Self-paced Learning with zero timer pressure.
- Audio Chimes & Gamified scoring/streaks.
"""

import os
import sys

# Suppress TensorFlow C++ and oneDNN verbose startup logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import collections
import json
import pickle
import random
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
from utils.finger_geometry import classify_static_sign
from utils.landmarks import LandmarkDetector, draw_landmarks, hand_bounding_box
from utils.preprocessing import resample_sequence
from utils.pretrained_recognizers import PretrainedGestureEngine
from utils.ui_renderer import render_studio_frame


STATIC_SIGNS = {"1", "2", "3", "4", "5", "A", "B", "C", "V"}


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


def build_curriculums(dynamic_signs):
    """Organizes available signs into switchable category modules."""
    numbers = ["1", "2", "3", "4", "5"]
    alphabet = ["A", "B", "C", "V"]
    words = [s for s in dynamic_signs if s not in numbers and s not in alphabet]
    if not words:
        words = ["FATHER", "MOTHER", "NO"]

    curriculums = {
        "ALL": numbers + alphabet + words,
        "NUMBERS": numbers,
        "ALPHABET": alphabet,
        "WORDS": words
    }
    return curriculums


def on_mouse_click(event, x, y, flags, param):
    """Handles mouse click events to enlarge or dismiss demo reference photos."""
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if state.get("show_enlarged_demo", False):
            # Click anywhere closes the enlarged preview
            state["show_enlarged_demo"] = False
        else:
            # Check if clicked on the thumbnail box or "Click to Enlarge" badge
            # (Thumbnail is at x: 1070..1260, y: 130..320)
            if 1070 <= x <= 1260 and 130 <= y <= 320:
                state["show_enlarged_demo"] = True


def main():
    model, label_encoder, scaler, run_config = load_artifacts()
    sequence_len = run_config["sequence_len"]
    separator = run_config.get("class_separator", "::")

    dynamic_signs = extract_positive_signs(label_encoder, separator)
    if not dynamic_signs:
        dynamic_signs = ["FATHER", "MOTHER", "NO"]

    curriculums = build_curriculums(dynamic_signs)
    category_keys = ["ALL", "NUMBERS", "ALPHABET", "WORDS"]
    current_cat_idx = 0
    current_category = category_keys[current_cat_idx]
    active_signs = curriculums[current_category]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    window_name = "EchoSign AI - Interactive Tutor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    detector = LandmarkDetector()
    pretrained_engine = PretrainedGestureEngine()
    start_app_time = time.time()
    image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    frame_buffer = collections.deque(maxlen=config.MAX_SEQUENCE_LEN)
    prediction_buffer = collections.deque(maxlen=config.PREDICTION_BUFFER_SIZE)

    min_frame_interval = 1.0 / config.TARGET_CAPTURE_FPS
    last_capture = time.time()

    # Tutor State
    state = {
        "mode": "tutor",                  # 'tutor' or 'free'
        "current_category": current_category,
        "current_target": active_signs[0],
        "target_idx": 0,
        "score": 0,
        "streak": 0,
        "status": "active",               # 'active', 'success'
        "step_start_time": time.time(),
        "last_pred_label": "",
        "last_pred_conf": 0.0,
        "feedback_msg": "",
        "cooldown_until": 0.0,
        "show_enlarged_demo": False,       # Click-to-enlarge modal state
        "is_in_position": False
    }

    static_match_frames = 0
    cv2.setMouseCallback(window_name, on_mouse_click, state)

    print("\n" + "=" * 65)
    print("  EchoSign AI - Interactive Sign Language Companion")
    print(f"  Curriculums: {', '.join(category_keys)}")
    print(f"  Current category [{current_category}]: {', '.join(active_signs)}")
    print("  Models: Google Gesture Recognizer + ASL 26-Letter Neural Net")
    print("  Controls: [N] Next Sign  | [C] Switch Category | [D] Enlarge Demo")
    print("=" * 65 + "\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            detection = detector.process(frame)

            now = time.time()
            cur_target = state["current_target"]
            is_static_target = cur_target in STATIC_SIGNS

            # -------------------------------------------------------------
            # AI Inference Pipeline (Multi-Model Ensemble)
            # -------------------------------------------------------------
            pred_label = ""
            pred_conf = 0.0

            if is_static_target:
                # 1. Multi-Model Evaluation: Google Gesture Recognizer + ASL TFLite + Kinematics (<2ms)
                timestamp_ms = int((now - start_app_time) * 1000)
                eval_res = pretrained_engine.evaluate(frame, detection, timestamp_ms, target_sign=cur_target)
                state["feedback_msg"] = eval_res["feedback"]

                s_pred = eval_res["consensus_sign"]
                s_conf = eval_res["consensus_conf"]
                if s_pred:
                    pred_label = s_pred
                    pred_conf = s_conf

                if s_pred == cur_target and s_conf >= 0.70:
                    static_match_frames += 1
                else:
                    static_match_frames = 0

            else:
                # 2. Dynamic Word Signs: Sequence Buffer + Neural Network
                state["feedback_msg"] = ""
                static_match_frames = 0

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

                    votes = collections.Counter(c for c, _ in prediction_buffer if c is not None)
                    if votes:
                        best_class, count = votes.most_common(1)[0]
                        if count >= config.MIN_CONSECUTIVE_AGREEMENT:
                            confs = [c for cls, c in prediction_buffer if cls == best_class]
                            raw_label = best_class.split(separator)[-1] if separator in best_class else best_class
                            if not raw_label.startswith("NOT_"):
                                pred_label = raw_label
                                pred_conf = float(np.mean(confs))

            state["last_pred_label"] = pred_label
            state["last_pred_conf"] = pred_conf

            # -------------------------------------------------------------
            # Tutor Logic Loop (Self-Paced, Multi-Modal Confirmation)
            # -------------------------------------------------------------
            if state["mode"] == "tutor":
                if state["status"] == "active":
                    success_triggered = False

                    if is_static_target:
                        # Static Sign Success: held steady for ~4 frames (~150-200ms)
                        if static_match_frames >= 4:
                            success_triggered = True
                    else:
                        # Dynamic Sign Success: hybrid spatial + neural agreement
                        is_in_target_zone = state.get("is_in_position", False)
                        target_match = (pred_label == cur_target)
                        if (target_match and pred_conf >= 0.45) or \
                           (is_in_target_zone and target_match and pred_conf >= 0.35) or \
                           (is_in_target_zone and pred_conf >= 0.55):
                            success_triggered = True

                    if success_triggered:
                        state["status"] = "success"
                        state["score"] += 100
                        state["streak"] += 1
                        state["cooldown_until"] = now + 2.0
                        play_chime(success=True)

                elif state["status"] == "success":
                    if now >= state["cooldown_until"]:
                        # Advance to next lesson in active category
                        state["target_idx"] = (state["target_idx"] + 1) % len(active_signs)
                        state["current_target"] = active_signs[state["target_idx"]]
                        state["status"] = "active"
                        state["step_start_time"] = now
                        state["feedback_msg"] = ""
                        static_match_frames = 0
                        frame_buffer.clear()
                        prediction_buffer.clear()

            elif state["mode"] == "free":
                # Free Practice: Multi-model detection across static alphabet and gestures
                timestamp_ms = int((now - start_app_time) * 1000)
                eval_res = pretrained_engine.evaluate(frame, detection, timestamp_ms, target_sign=None)
                if eval_res["consensus_sign"] and eval_res["consensus_conf"] >= 0.70:
                    state["last_pred_label"] = eval_res["consensus_sign"]
                    state["last_pred_conf"] = eval_res["consensus_conf"]
                    state["feedback_msg"] = f"Detected: {eval_res['consensus_sign']}"

            # Render landmarks on camera feed (Face mesh disabled, hand skeletons crisp)
            draw_landmarks(frame, detection)

            # Build full modern Studio Canvas
            studio_canvas = render_studio_frame(frame, detection, state, active_signs)

            cv2.imshow(window_name, studio_canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('d') or key == ord(' '):  # Toggle enlarged demo preview
                state['show_enlarged_demo'] = not state.get('show_enlarged_demo', False)
            elif key == 27:  # ESC key
                if state.get('show_enlarged_demo', False):
                    state['show_enlarged_demo'] = False
                else:
                    break
            elif key == ord('c'):  # Switch category (ALL -> NUMBERS -> ALPHABET -> WORDS -> ALL)
                current_cat_idx = (current_cat_idx + 1) % len(category_keys)
                current_category = category_keys[current_cat_idx]
                active_signs = curriculums[current_category]
                state['current_category'] = current_category
                state['target_idx'] = 0
                state['current_target'] = active_signs[0]
                state['status'] = "active"
                state['feedback_msg'] = ""
                state['show_enlarged_demo'] = False
                state['step_start_time'] = time.time()
                static_match_frames = 0
                frame_buffer.clear()
                prediction_buffer.clear()
                print(f"[CATEGORY SWITCH] Active: {current_category} -> {active_signs}")
            elif key == ord('n'):  # Skip / Next sign in current category
                state['target_idx'] = (state['target_idx'] + 1) % len(active_signs)
                state['current_target'] = active_signs[state['target_idx']]
                state['status'] = "active"
                state['feedback_msg'] = ""
                state['show_enlarged_demo'] = False
                state['step_start_time'] = time.time()
                static_match_frames = 0
                frame_buffer.clear()
                prediction_buffer.clear()
            elif key == ord('m'):  # Toggle mode (Tutor <-> Free Practice)
                state['mode'] = "free" if state['mode'] == "tutor" else "tutor"
                state['status'] = "active"
                state['step_start_time'] = time.time()
            elif key == ord('r'):  # Reset score
                state['score'] = 0
                state['streak'] = 0

    finally:
        cap.release()
        detector.close()
        pretrained_engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
