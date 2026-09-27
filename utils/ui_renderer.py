"""
utils/ui_renderer.py
====================
Modern, high-resolution visual rendering engine for EchoSign AI.

Features:
- Dual-pane Studio Canvas (1280x720): Widescreen live webcam on the left,
  interactive AI tutor coach dashboard on the right.
- Real-time Spatial Target Guidance: Highlights facial target zones (Forehead, Chin)
  with dynamic pulsing crosshairs and distance tracking.
- Step-by-step visual instruction cards with human-readable guidance.
- Crisp anti-aliased TrueType typography via Pillow.
- Rich gamification cards (Score, Streak, Live Confidence, Timer).
"""

import math
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

# ---------------------------------------------------------------------------
# Visual Styling Tokens (Modern Dark Palette)
# ---------------------------------------------------------------------------
COLOR_BG_DARK = (13, 17, 23)          # #0D1117 (Deep GitHub Canvas)
COLOR_CARD_BG = (22, 27, 34)          # #161B22 (Card Surface)
COLOR_CARD_BORDER = (48, 54, 61)      # #30363D (Subtle Card Border)
COLOR_CARD_ACTIVE = (30, 41, 59)      # Slate Highlight

COLOR_CYAN = (0, 210, 255)            # Primary Brand Accent
COLOR_EMERALD = (46, 213, 115)        # Success / In Position
COLOR_AMBER = (255, 177, 66)          # Caution / Searching
COLOR_CORAL = (255, 71, 87)           # Timeout / Error
COLOR_PURPLE = (165, 94, 234)         # Streak Multiplier

COLOR_TEXT_PRIMARY = (240, 246, 252)  # High Contrast Text
COLOR_TEXT_SECONDARY = (160, 175, 195)# Subtitle Text
COLOR_TEXT_MUTED = (120, 134, 150)    # Caption / Dim Text

# ---------------------------------------------------------------------------
# Font Manager (Cached TrueType Fonts with Fallback)
# ---------------------------------------------------------------------------
class FontManager:
    """Loads system TrueType fonts with clean anti-aliasing."""
    def __init__(self):
        font_dir = "C:/Windows/Fonts"
        bold_path = os.path.join(font_dir, "segoeuib.ttf")
        reg_path = os.path.join(font_dir, "segoeui.ttf")

        # Fallbacks for non-Windows or if Segoe UI is missing
        if not os.path.exists(bold_path):
            bold_path = os.path.join(font_dir, "arialbd.ttf")
        if not os.path.exists(reg_path):
            reg_path = os.path.join(font_dir, "arial.ttf")

        try:
            self.title_font = ImageFont.truetype(bold_path, 28)
            self.header_font = ImageFont.truetype(bold_path, 20)
            self.bold_font = ImageFont.truetype(bold_path, 16)
            self.regular_font = ImageFont.truetype(reg_path, 15)
            self.small_font = ImageFont.truetype(reg_path, 13)
            self.large_sign_font = ImageFont.truetype(bold_path, 36)
            self.stat_font = ImageFont.truetype(bold_path, 22)
        except Exception:
            # Safe standard fallback
            self.title_font = ImageFont.load_default()
            self.header_font = ImageFont.load_default()
            self.bold_font = ImageFont.load_default()
            self.regular_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
            self.large_sign_font = ImageFont.load_default()
            self.stat_font = ImageFont.load_default()

FONTS = FontManager()

# ---------------------------------------------------------------------------
# Lesson Curriculum: Clear, Simple Steps
# ---------------------------------------------------------------------------
SIGN_METADATA = {
    "FATHER": {
        "title": "FATHER",
        "category": "FAMILY SIGN • ASL",
        "steps": [
            "1. Open your dominant hand with 5 fingers spread out.",
            "2. Touch the tip of your thumb to your FOREHEAD.",
            "3. Keep fingers still and face the camera."
        ],
        "target_region": "forehead",
        "target_landmark_id": 10,  # MediaPipe Forehead ID
        "target_label": "TARGET: FOREHEAD"
    },
    "MOTHER": {
        "title": "MOTHER",
        "category": "FAMILY SIGN • ASL",
        "steps": [
            "1. Open your dominant hand with 5 fingers spread out.",
            "2. Touch the tip of your thumb to your CHIN.",
            "3. Keep fingers still and face the camera."
        ],
        "target_region": "chin",
        "target_landmark_id": 152, # MediaPipe Chin ID
        "target_label": "TARGET: CHIN"
    },
    "NO": {
        "title": "NO",
        "category": "DAILY ESSENTIALS • ASL",
        "steps": [
            "1. Extend index and middle fingers together.",
            "2. Snap/pinch them firmly down to meet your thumb.",
            "3. Perform the closing motion in front of your chest."
        ],
        "target_region": "hand_snap",
        "target_landmark_id": None,
        "target_label": "MOTION: PINCH TO THUMB"
    }
}


def get_sign_meta(sign_name):
    """Retrieve structured metadata for a sign, with dynamic fallback."""
    upper = sign_name.upper()
    if upper in SIGN_METADATA:
        return SIGN_METADATA[upper]
    return {
        "title": upper,
        "category": "CUSTOM SIGN PRACTICE",
        "steps": [
            f"1. Demonstrate the gesture for {upper}.",
            "2. Ensure your hands are within the camera frame.",
            "3. Hold the position steady for 1 second."
        ],
        "target_region": None,
        "target_landmark_id": None,
        "target_label": "TARGET GESTURE"
    }


# ---------------------------------------------------------------------------
# In-Frame Target Overlay (OpenCV Direct Rendering on Video Feed)
# ---------------------------------------------------------------------------
def draw_spatial_target_overlay(frame, detection, current_target):
    """
    Draws a glowing visual target ring on the user's face (Forehead or Chin)
    and tracks the distance to their thumb in real-time.
    """
    meta = get_sign_meta(current_target)
    target_lid = meta.get("target_landmark_id")
    if target_lid is None:
        return None, None  # No facial target for this sign (e.g., NO)

    h, w, _ = frame.shape
    face_lms = detection.get("face")
    if not face_lms or target_lid >= len(face_lms):
        return None, None

    # Target landmark pixel coordinate
    tx = int(face_lms[target_lid][0] * w)
    ty = int(face_lms[target_lid][1] * h)

    # Find dominant thumb tip coordinate
    thumb_pt = None
    min_dist = float("inf")
    hands = detection.get("hands", {})
    for side in ("Right", "Left"):
        hand_lms = hands.get(side)
        if hand_lms and len(hand_lms) > 4:
            hx = int(hand_lms[4][0] * w)  # thumb tip
            hy = int(hand_lms[4][1] * h)
            dist = math.hypot(hx - tx, hy - ty)
            if dist < min_dist:
                min_dist = dist
                thumb_pt = (hx, hy)

    # In-position threshold (in pixels on webcam)
    is_in_position = (thumb_pt is not None and min_dist < 65)

    # Pulsing ring animation
    pulse = int(5 * math.sin(time.time() * 6))
    base_radius = 28 + pulse

    if is_in_position:
        ring_col = (50, 255, 80)     # Neon Green (BGR)
        label_text = "IN POSITION! HOLD IT!"
    else:
        ring_col = (0, 215, 255)     # Glowing Cyan/Yellow (BGR)
        label_text = meta["target_label"]

    # Draw target crosshair & circle
    cv2.circle(frame, (tx, ty), base_radius, ring_col, 2, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 6, ring_col, -1, cv2.LINE_AA)
    cv2.line(frame, (tx - base_radius - 8, ty), (tx - base_radius + 4, ty), ring_col, 2, cv2.LINE_AA)
    cv2.line(frame, (tx + base_radius - 4, ty), (tx + base_radius + 8, ty), ring_col, 2, cv2.LINE_AA)
    cv2.line(frame, (tx, ty - base_radius - 8), (tx, ty - base_radius + 4), ring_col, 2, cv2.LINE_AA)
    cv2.line(frame, (tx, ty + base_radius - 4), (tx, ty + base_radius + 8), ring_col, 2, cv2.LINE_AA)

    # Distance beam between thumb and target
    if thumb_pt is not None and not is_in_position:
        cv2.line(frame, thumb_pt, (tx, ty), (100, 180, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, thumb_pt, 7, (0, 215, 255), -1, cv2.LINE_AA)

    # Small badge above target
    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    bx1 = tx - label_size[0] // 2 - 8
    by1 = ty - base_radius - 22
    bx2 = tx + label_size[0] // 2 + 8
    by2 = ty - base_radius - 4
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 24, 32), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), ring_col, 1)
    cv2.putText(frame, label_text, (bx1 + 8, by2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return thumb_pt, is_in_position


# ---------------------------------------------------------------------------
# Studio Canvas Assembly (1280 x 720)
# ---------------------------------------------------------------------------
def render_studio_frame(raw_cam_frame, detection, state, positive_signs):
    """
    Builds the high-definition 1280x720 studio canvas.
    - Left (800x720 area): Nicely scaled camera feed with rounded border and target cues.
    - Right (480x720 area): Tutor Coach instructions, guidance box, meters, and scores.
    """
    CANVAS_W, CANVAS_H = 1280, 720
    canvas_img = Image.new("RGB", (CANVAS_W, CANVAS_H), COLOR_BG_DARK)
    draw = ImageDraw.Draw(canvas_img)

    # 1. Camera Viewport Dimensions
    cam_x = 24
    cam_y = 24
    cam_w = 780
    cam_h = 585  # Standard 4:3 aspect ratio

    # Process camera overlay
    cam_display = raw_cam_frame.copy()
    cur_target = state["current_target"]
    thumb_pt, in_pos = draw_spatial_target_overlay(cam_display, detection, cur_target)

    # Resize camera frame to viewport
    resized_cam = cv2.resize(cam_display, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
    resized_cam_rgb = cv2.cvtColor(resized_cam, cv2.COLOR_BGR2RGB)
    cam_pil = Image.fromarray(resized_cam_rgb)

    # Paste camera feed
    canvas_img.paste(cam_pil, (cam_x, cam_y))

    # Border around camera
    draw.rounded_rectangle([cam_x - 2, cam_y - 2, cam_x + cam_w + 2, cam_y + cam_h + 2],
                           radius=12, outline=COLOR_CARD_BORDER, width=2)

    # Live Camera Pill Badge (Top Left of Camera)
    draw.rounded_rectangle([cam_x + 14, cam_y + 14, cam_x + 175, cam_y + 44],
                           radius=8, fill=(18, 22, 30), outline=COLOR_CARD_BORDER, width=1)
    draw.ellipse([cam_x + 24, cam_y + 24, cam_x + 34, cam_y + 34], fill=COLOR_EMERALD)
    draw.text((cam_x + 42, cam_y + 21), "LIVE CAMERA", font=FONTS.small_font, fill=COLOR_TEXT_PRIMARY)

    # Camera Bottom Info Strip
    draw.rounded_rectangle([cam_x, cam_y + cam_h + 16, cam_x + cam_w, cam_y + cam_h + 84],
                           radius=10, fill=COLOR_CARD_BG, outline=COLOR_CARD_BORDER, width=1)
    controls_txt = "[N] Next Lesson    |    [M] Mode Toggle    |    [R] Reset Score    |    [Q] Quit"
    draw.text((cam_x + 24, cam_y + cam_h + 40), controls_txt,
              font=FONTS.bold_font, fill=COLOR_TEXT_SECONDARY)

    # -----------------------------------------------------------------------
    # 2. Right Side: Interactive AI Coach Dashboard
    # -----------------------------------------------------------------------
    dash_x = cam_x + cam_w + 24
    dash_w = CANVAS_W - dash_x - 24

    # Top Brand Header
    draw.text((dash_x, 24), "EchoSign AI", font=FONTS.title_font, fill=COLOR_CYAN)
    draw.rounded_rectangle([dash_x + 185, 29, dash_x + 285, 55],
                           radius=6, fill=(30, 45, 65), outline=COLOR_CYAN, width=1)
    draw.text((dash_x + 195, 33), "Tutor v2.0", font=FONTS.small_font, fill=COLOR_CYAN)
    draw.text((dash_x, 62), "Interactive Sign Language Companion",
              font=FONTS.regular_font, fill=COLOR_TEXT_SECONDARY)

    # Mode Indicator Pill
    is_tutor_mode = (state["mode"] == "tutor")
    mode_label = "TUTOR MODE" if is_tutor_mode else "FREE PRACTICE MODE"
    mode_color = COLOR_EMERALD if is_tutor_mode else COLOR_PURPLE
    draw.rounded_rectangle([dash_x + dash_w - 170, 26, dash_x + dash_w, 54],
                           radius=6, fill=COLOR_CARD_BG, outline=mode_color, width=1)
    draw.text((dash_x + dash_w - 155, 32), mode_label, font=FONTS.small_font, fill=mode_color)

    # -----------------------------------------------------------------------
    # Card 1: Current Lesson Card
    # -----------------------------------------------------------------------
    card1_y = 96
    card1_h = 240
    draw.rounded_rectangle([dash_x, card1_y, dash_x + dash_w, card1_y + card1_h],
                           radius=12, fill=COLOR_CARD_BG, outline=COLOR_CARD_BORDER, width=1)

    meta = get_sign_meta(cur_target)
    draw.text((dash_x + 20, card1_y + 16), meta["category"], font=FONTS.small_font, fill=COLOR_CYAN)
    draw.text((dash_x + 20, card1_y + 36), meta["title"], font=FONTS.large_sign_font, fill=COLOR_TEXT_PRIMARY)

    # Step-by-Step Instructions
    step_y = card1_y + 88
    for step_text in meta["steps"]:
        draw.text((dash_x + 20, step_y), step_text, font=FONTS.regular_font, fill=COLOR_TEXT_PRIMARY)
        step_y += 32

    # Challenge Timer Bar (Only in Tutor mode)
    if is_tutor_mode:
        elapsed = time.time() - state["step_start_time"]
        remaining = max(0.0, state["time_limit"] - elapsed)
        time_ratio = min(1.0, max(0.0, remaining / state["time_limit"]))
        bar_col = COLOR_EMERALD if time_ratio > 0.3 else COLOR_CORAL

        draw.text((dash_x + 20, card1_y + card1_h - 40),
                  f"Challenge Timer: {remaining:.1f}s", font=FONTS.small_font, fill=COLOR_TEXT_MUTED)

        # Progress bar background & fill
        bx = dash_x + 20
        by = card1_y + card1_h - 20
        bw = dash_w - 40
        draw.rounded_rectangle([bx, by, bx + bw, by + 8], radius=4, fill=(40, 48, 60))
        if time_ratio > 0:
            draw.rounded_rectangle([bx, by, bx + int(bw * time_ratio), by + 8],
                                   radius=4, fill=bar_col)

    # -----------------------------------------------------------------------
    # Card 2: Live AI Coach Guidance
    # -----------------------------------------------------------------------
    card2_y = card1_y + card1_h + 16
    card2_h = 135

    # Determine Coach Feedback State
    has_face = detection.get("face") is not None
    hands = detection.get("hands", {})
    has_hand = (hands.get("Right") is not None) or (hands.get("Left") is not None)

    if state.get("status") == "success":
        coach_badge = "EXCELLENT!"
        coach_msg = "Gesture recognized successfully! +100 Points Awarded."
        coach_theme = COLOR_EMERALD
    elif state.get("status") == "timeout":
        coach_badge = "TIME'S UP"
        coach_msg = "Don't worry! Try the next sign with steady motion."
        coach_theme = COLOR_CORAL
    elif not has_face:
        coach_badge = "LOOKING FOR FACE"
        coach_msg = "Please position your face clearly in the camera view."
        coach_theme = COLOR_AMBER
    elif not has_hand:
        coach_badge = "LOOKING FOR HAND"
        coach_msg = "Raise your hand in front of the camera to begin signing."
        coach_theme = COLOR_AMBER
    elif in_pos:
        coach_badge = "GREAT POSITION"
        coach_msg = "Hand is in the target zone! Hold still for recognition..."
        coach_theme = COLOR_EMERALD
    elif meta.get("target_region") == "forehead":
        coach_badge = "ACTION REQUIRED"
        coach_msg = "Touch your thumb tip to your FOREHEAD (yellow target circle)."
        coach_theme = COLOR_CYAN
    elif meta.get("target_region") == "chin":
        coach_badge = "ACTION REQUIRED"
        coach_msg = "Touch your thumb tip to your CHIN (yellow target circle)."
        coach_theme = COLOR_CYAN
    else:
        coach_badge = "ACTION REQUIRED"
        coach_msg = "Perform the gesture clearly in front of the camera."
        coach_theme = COLOR_CYAN

    draw.rounded_rectangle([dash_x, card2_y, dash_x + dash_w, card2_y + card2_h],
                           radius=12, fill=COLOR_CARD_BG, outline=coach_theme, width=2)

    draw.rounded_rectangle([dash_x + 16, card2_y + 14, dash_x + 175, card2_y + 38],
                           radius=6, fill=(28, 36, 48), outline=coach_theme, width=1)
    draw.text((dash_x + 26, card2_y + 17), coach_badge, font=FONTS.small_font, fill=coach_theme)

    draw.text((dash_x + 18, card2_y + 52), coach_msg, font=FONTS.regular_font, fill=COLOR_TEXT_PRIMARY)

    # Live prediction & confidence meter
    pred_lbl = state.get("last_pred_label", "")
    pred_conf = state.get("last_pred_conf", 0.0)
    conf_str = f"{pred_conf * 100:.0f}%" if pred_lbl else "--"
    pred_display = f"AI Prediction: {pred_lbl if pred_lbl else 'Analyzing motion...'}"

    draw.text((dash_x + 18, card2_y + 85), pred_display, font=FONTS.small_font, fill=COLOR_TEXT_SECONDARY)
    draw.text((dash_x + dash_w - 70, card2_y + 85), f"Conf: {conf_str}", font=FONTS.small_font, fill=COLOR_CYAN)

    # Meter bar
    draw.rounded_rectangle([dash_x + 18, card2_y + 110, dash_x + dash_w - 18, card2_y + 116],
                           radius=3, fill=(40, 48, 60))
    if pred_conf > 0:
        bar_w = int((dash_w - 36) * min(1.0, pred_conf))
        draw.rounded_rectangle([dash_x + 18, card2_y + 110, dash_x + 18 + bar_w, card2_y + 116],
                               radius=3, fill=COLOR_CYAN)

    # -----------------------------------------------------------------------
    # Card 3: Gamified Scoreboard (Score & Streak)
    # -----------------------------------------------------------------------
    card3_y = card2_y + card2_h + 16
    stat_box_w = (dash_w - 12) // 2

    # Score Box
    draw.rounded_rectangle([dash_x, card3_y, dash_x + stat_box_w, card3_y + 90],
                           radius=10, fill=COLOR_CARD_BG, outline=COLOR_CARD_BORDER, width=1)
    draw.text((dash_x + 16, card3_y + 14), "SCORE", font=FONTS.small_font, fill=COLOR_TEXT_MUTED)
    draw.text((dash_x + 16, card3_y + 36), f"{state['score']} pts", font=FONTS.stat_font, fill=COLOR_CYAN)

    # Streak Box
    draw.rounded_rectangle([dash_x + stat_box_w + 12, card3_y, dash_x + dash_w, card3_y + 90],
                           radius=10, fill=COLOR_CARD_BG, outline=COLOR_CARD_BORDER, width=1)
    draw.text((dash_x + stat_box_w + 28, card3_y + 14), "STREAK", font=FONTS.small_font, fill=COLOR_TEXT_MUTED)
    streak_col = COLOR_PURPLE if state["streak"] > 0 else COLOR_TEXT_MUTED
    draw.text((dash_x + stat_box_w + 28, card3_y + 36), f"{state['streak']}x Combo", font=FONTS.stat_font, fill=streak_col)

    # -----------------------------------------------------------------------
    # 3. Floating Celebration / Timeout Banner across Camera
    # -----------------------------------------------------------------------
    if state.get("status") == "success":
        banner_w = cam_w - 80
        banner_h = 100
        bx1 = cam_x + 40
        by1 = cam_y + (cam_h - banner_h) // 2
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=16, fill=(18, 55, 32), outline=COLOR_EMERALD, width=3)
        draw.text((bx1 + banner_w // 2 - 190, by1 + 18), "CORRECT! EXCELLENT EXECUTION!",
                  font=FONTS.header_font, fill=(255, 255, 255))
        draw.text((bx1 + banner_w // 2 - 90, by1 + 54), "+100 Points Awarded",
                  font=FONTS.bold_font, fill=COLOR_EMERALD)

    elif state.get("status") == "timeout":
        banner_w = cam_w - 80
        banner_h = 100
        bx1 = cam_x + 40
        by1 = cam_y + (cam_h - banner_h) // 2
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=16, fill=(55, 22, 28), outline=COLOR_CORAL, width=3)
        draw.text((bx1 + banner_w // 2 - 180, by1 + 18), "TIME'S UP! Keep Practicing!",
                  font=FONTS.header_font, fill=(255, 255, 255))
        draw.text((bx1 + banner_w // 2 - 120, by1 + 54), "Moving to next sign challenge...",
                  font=FONTS.bold_font, fill=(255, 200, 200))

    # Convert back to BGR for OpenCV
    final_canvas_bgr = np.array(canvas_img)[:, :, ::-1]
    return final_canvas_bgr
