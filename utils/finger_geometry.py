"""
utils/finger_geometry.py
========================
Deterministic Geometric Pose Recognizer for ASL Numbers & Letters.

Features:
- Instant (<1ms) evaluation directly from MediaPipe 21 3D hand keypoints.
- Zero training dataset collection required: instant out-of-the-box accuracy.
- Evaluates finger extension kinematics (Thumb, Index, Middle, Ring, Pinky).
- Provides actionable diagnostic feedback (e.g., "Fold your middle finger", "Extend your thumb").
- Supported Static Signs:
    Numbers: '1', '2', '3', '4', '5'
    Letters: 'A', 'B', 'C', 'L', 'V', 'Y'
"""

import math
import numpy as np


def _euclidean_dist(pt1, pt2):
    """Euclidean distance in 2D/3D."""
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    dz = (pt1[2] - pt2[2]) if (len(pt1) > 2 and len(pt2) > 2) else 0.0
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def analyze_hand_geometry(landmarks, handedness="Right"):
    """
    Analyzes a 21-landmark MediaPipe hand.

    Returns:
        dict: {
            "scale": float,
            "fingers_up": {
                "thumb": bool,
                "index": bool,
                "middle": bool,
                "ring": bool,
                "pinky": bool
            },
            "finger_angles": dict,
            "diagnostic": str
        }
    """
    if landmarks is None or len(landmarks) < 21:
        return None

    # Reference points
    wrist = landmarks[0]
    thumb_cmc, thumb_mcp, thumb_ip, thumb_tip = landmarks[1], landmarks[2], landmarks[3], landmarks[4]
    index_mcp, index_pip, index_dip, index_tip = landmarks[5], landmarks[6], landmarks[7], landmarks[8]
    middle_mcp, middle_pip, middle_dip, middle_tip = landmarks[9], landmarks[10], landmarks[11], landmarks[12]
    ring_mcp, ring_pip, ring_dip, ring_tip = landmarks[13], landmarks[14], landmarks[15], landmarks[16]
    pinky_mcp, pinky_pip, pinky_dip, pinky_tip = landmarks[17], landmarks[18], landmarks[19], landmarks[20]

    # Hand scale: wrist to middle MCP
    hand_scale = _euclidean_dist(wrist, middle_mcp)
    if hand_scale < 1e-4:
        hand_scale = 1.0

    # -------------------------------------------------------------
    # 1. Four Fingers: Index, Middle, Ring, Pinky Extension Logic
    # -------------------------------------------------------------
    # A finger is EXTENDED if tip is substantially further from wrist
    # than PIP, and tip is above PIP/DIP in upright posture.
    def is_finger_extended(tip, dip, pip, mcp):
        dist_tip_wrist = _euclidean_dist(tip, wrist)
        dist_pip_wrist = _euclidean_dist(pip, wrist)
        dist_tip_mcp = _euclidean_dist(tip, mcp)
        dist_pip_mcp = _euclidean_dist(pip, mcp)

        # Distance ratio check
        ratio_ext = (dist_tip_wrist > dist_pip_wrist * 1.1) and (dist_tip_mcp > dist_pip_mcp * 1.15)
        # Upright y-check (image coordinates: smaller y is higher)
        y_ext = (tip[1] < pip[1])

        return ratio_ext or (y_ext and dist_tip_wrist > dist_pip_wrist)

    index_up = is_finger_extended(index_tip, index_dip, index_pip, index_mcp)
    middle_up = is_finger_extended(middle_tip, middle_dip, middle_pip, middle_mcp)
    ring_up = is_finger_extended(ring_tip, ring_dip, ring_pip, ring_mcp)
    pinky_up = is_finger_extended(pinky_tip, pinky_dip, pinky_pip, pinky_mcp)

    # -------------------------------------------------------------
    # 2. Thumb Extension & Position Logic
    # -------------------------------------------------------------
    dist_thumb_wrist = _euclidean_dist(thumb_tip, wrist)
    dist_thumb_mcp = _euclidean_dist(thumb_tip, thumb_mcp)
    dist_thumb_index_mcp = _euclidean_dist(thumb_tip, index_mcp)
    dist_thumb_pinky_mcp = _euclidean_dist(thumb_tip, pinky_mcp)

    # Thumb is considered extended outwards if it spreads away from index base
    thumb_extended = (dist_thumb_index_mcp > hand_scale * 0.55) or (dist_thumb_wrist > hand_scale * 0.95)

    # Thumb tucked across palm (as in 'B' or '4')
    thumb_tucked_palm = (dist_thumb_index_mcp < hand_scale * 0.45) or (dist_thumb_pinky_mcp < hand_scale * 0.6)

    # Thumb upright along fist (as in 'A')
    thumb_along_fist = (thumb_tip[1] < index_mcp[1] + hand_scale * 0.1) and (not index_up and not middle_up and not ring_up and not pinky_up)

    # 'C' shape detection: all fingers gently curved, thumb facing fingertips
    dist_thumb_index_tip = _euclidean_dist(thumb_tip, index_tip)
    c_curve = (0.2 * hand_scale < dist_thumb_index_tip < 0.9 * hand_scale) and \
              (middle_tip[1] < wrist[1]) and (not index_up or middle_tip[1] > index_pip[1] - hand_scale * 0.2)

    return {
        "hand_scale": hand_scale,
        "fingers_up": {
            "thumb": thumb_extended,
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
        },
        "thumb_tucked": thumb_tucked_palm,
        "thumb_along_fist": thumb_along_fist,
        "c_curve": c_curve,
        "dist_thumb_index_tip": dist_thumb_index_tip,
        "dist_thumb_index_mcp": dist_thumb_index_mcp
    }


def classify_static_sign(detection, target_sign=None):
    """
    Evaluates both hands (or dominant detected hand) against static sign rules.

    Returns:
        tuple (predicted_sign, confidence, feedback_message)
    """
    hands = detection.get("hands", {})
    dominant_analysis = None
    dominant_label = None

    for side in ("Right", "Left"):
        lms = hands.get(side)
        if lms and len(lms) >= 21:
            dominant_analysis = analyze_hand_geometry(lms, side)
            dominant_label = side
            break

    if dominant_analysis is None:
        return None, 0.0, "Place your hand in front of the camera"

    f = dominant_analysis["fingers_up"]
    thumb = f["thumb"]
    index = f["index"]
    middle = f["middle"]
    ring = f["ring"]
    pinky = f["pinky"]
    thumb_tucked = dominant_analysis["thumb_tucked"]
    thumb_along_fist = dominant_analysis["thumb_along_fist"]
    c_curve = dominant_analysis["c_curve"]

    # Match Candidates with confidence scoring
    candidates = {}

    # Number 1: Index UP, others DOWN
    if index and not middle and not ring and not pinky:
        score = 0.92 if (not thumb or thumb_tucked) else 0.70
        candidates["1"] = score

    # Number 2 / Letter V: Index & Middle UP, Ring & Pinky DOWN
    if index and middle and not ring and not pinky:
        score = 0.94 if (not thumb or thumb_tucked) else 0.75
        candidates["2"] = score
        candidates["V"] = score

    # Number 3 (ASL Standard): Thumb, Index, Middle EXTENDED, Ring & Pinky DOWN
    if thumb and index and middle and not ring and not pinky:
        candidates["3"] = 0.95
    elif not thumb and index and middle and ring and not pinky:
        # European / colloquial 3 (Index, Middle, Ring)
        candidates["3"] = 0.85

    # Number 4: 4 fingers UP, Thumb tucked
    if index and middle and ring and pinky and not thumb:
        candidates["4"] = 0.93

    # Number 5: All 5 fingers extended and spread
    if thumb and index and middle and ring and pinky:
        candidates["5"] = 0.96

    # Letter A: Closed fist, thumb alongside index
    if not index and not middle and not ring and not pinky:
        if thumb_along_fist or thumb:
            candidates["A"] = 0.92
        else:
            candidates["A"] = 0.75

    # Letter B: 4 fingers straight up together, thumb folded flat across palm
    if index and middle and ring and pinky:
        if thumb_tucked or not thumb:
            candidates["B"] = 0.95
        else:
            candidates["B"] = 0.70

    # Letter C: Arched fingers forming a C
    if c_curve and not pinky and (not index or not middle):
        candidates["C"] = 0.88

    # Letter L: Index UP, Thumb OUT (~90 deg), others DOWN
    if index and thumb and not middle and not ring and not pinky:
        candidates["L"] = 0.96

    # Letter Y: Thumb and Pinky extended, middle 3 DOWN
    if thumb and pinky and not index and not middle and not ring:
        candidates["Y"] = 0.95

    # Target-specific feedback generation
    feedback = ""
    if target_sign:
        target_upper = target_sign.upper()
        if target_upper == "1":
            if not index:
                feedback = "Point your INDEX finger straight up"
            elif middle or ring or pinky:
                feedback = "Fold down middle, ring, and pinky fingers"
        elif target_upper in ("2", "V"):
            if not index or not middle:
                feedback = "Raise both INDEX and MIDDLE fingers"
            elif ring or pinky:
                feedback = "Keep ring and pinky fingers folded"
        elif target_upper == "3":
            if not thumb:
                feedback = "Extend your THUMB along with index and middle"
            elif not (index and middle):
                feedback = "Raise your INDEX and MIDDLE fingers"
            elif ring or pinky:
                feedback = "Keep ring and pinky folded into your palm"
        elif target_upper == "4":
            if not (index and middle and ring and pinky):
                feedback = "Hold all 4 fingers straight up"
            elif thumb:
                feedback = "Tuck your thumb across your palm"
        elif target_upper == "5":
            missing = []
            if not thumb: missing.append("thumb")
            if not index: missing.append("index")
            if not middle: missing.append("middle")
            if not ring: missing.append("ring")
            if not pinky: missing.append("pinky")
            if missing:
                feedback = f"Open hand fully (extend {', '.join(missing)})"
        elif target_upper == "A":
            if index or middle or ring or pinky:
                feedback = "Make a tight fist with all 4 fingers folded"
            else:
                feedback = "Rest your thumb along the side of your fist"
        elif target_upper == "B":
            if not (index and middle and ring and pinky):
                feedback = "Hold 4 fingers straight up together"
            elif not thumb_tucked:
                feedback = "Fold thumb across your palm"
        elif target_upper == "C":
            feedback = "Curve hand like holding a cup into a 'C' shape"
        elif target_upper == "L":
            if not index:
                feedback = "Point index finger straight up"
            elif not thumb:
                feedback = "Stick thumb outward to form an 'L'"
            elif middle or ring or pinky:
                feedback = "Keep other 3 fingers folded down"
        elif target_upper == "Y":
            if not thumb or not pinky:
                feedback = "Stick out thumb and pinky finger"
            elif index or middle or ring:
                feedback = "Fold middle 3 fingers down"

    if not candidates:
        return None, 0.0, feedback or "Adjust hand posture to match the demo photo"

    # If the target sign is matched in candidates, prioritize it
    if target_sign and target_sign.upper() in candidates:
        return target_sign.upper(), candidates[target_sign.upper()], feedback or "Great form! Hold steady"

    # Otherwise return the highest scoring candidate
    best_sign, best_conf = max(candidates.items(), key=lambda item: item[1])
    return best_sign, best_conf, feedback or f"Detected '{best_sign}'"
