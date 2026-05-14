"""3-pose anthropometric calibration for analytical retargeting.

The user is asked to hold three reference poses in sequence:

    1. **T-pose**          arms straight out to the sides     (~3 s)
    2. **A-pose**          arms hanging straight down         (~3 s)
    3. **Tray-holding**    upper arms down, forearms HORIZONTAL
                            forward, elbows ~90°               (~3 s)

UI flow per pose (4 phases):

    GET_READY (2.0 s)  →  COUNTDOWN (3.0 s)  →  MEASURING (1.5 s)  →  DONE (1.0 s)

* GET_READY: large pose name + instructions + reference silhouette, so the
  user sees what to do before any timer starts.
* COUNTDOWN: huge animated number in the centre with a sweeping ring,
  giving the user time to adopt the pose.
* MEASURING: green badge "MEASURING" + sample counter + live validity
  check (red "Adjust pose: …" if the body-frame vectors don't match the
  expected geometry).
* DONE: green ✓ confirmation flashed briefly before moving to the next pose.

Also drawn every frame:
* top banner with "Pose 2 of 3 — A-POSE"
* bottom progress bar showing total calibration progress
* small stick-figure reference of the expected pose in the upper-right corner

From the captured landmarks we measure:

* `arm_length_user`        average shoulder→wrist distance, T-pose
* `shoulder_width_user`    L-shoulder ↔ R-shoulder distance, A-pose
* `torso_height_user`      mid-hip ↔ mid-shoulder distance, A-pose
* `anatomical_zero`        14-D vector — for the SHOULDER joints this is the
                            joint angles produced by the analytical retargeter
                            when the user holds the A-pose; for the ELBOW
                            joints we additionally subtract a per-user elbow
                            bias measured in the tray-holding pose (observed
                            elbow − π/2), so the user's "neutral" elbow rest
                            maps cleanly to the robot's elbow zero.  All other
                            joints (wrists) stay at 0.

Cached to `models/anthropo.json` so subsequent runs skip the calibration.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import config
from .retarget import build_body_frame, compute_anatomical_angles

if TYPE_CHECKING:  # heavy deps only needed at runtime, not for unit tests
    import cv2
    from .pose import PoseEstimator


# ── Public dataclass ─────────────────────────────────────────────────────────

@dataclass
class BodyCalibration:
    arm_length_user:     float          # metres, T-pose shoulder→wrist
    shoulder_width_user: float          # metres, A-pose L↔R shoulder
    torso_height_user:   float          # metres, A-pose mid-hip → mid-shoulder
    arm_length_robot:    float          # metres, taken from the MJCF
    scale:               float          # arm_length_robot / arm_length_user
    anatomical_zero:     list[float] = field(default_factory=lambda: [0.0] * 14)

    def save(self, path=None) -> None:
        path = config.CALIBRATION_PATH if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=config.CALIBRATION_PATH) -> "BodyCalibration | None":
        try:
            d = json.loads(path.read_text())
            return cls(**d)
        except Exception:
            return None


# ── Pose definitions ─────────────────────────────────────────────────────────

# (key, display name, instruction text, hungarian hint)
_POSES = (
    ("tpose", "T-POSE",
     "Stretch arms straight OUT to the sides (horizontal)",
     "T-poz: karok oldalra kinyujtva (vizszintesen)"),
    ("apose", "A-POSE",
     "Arms hanging straight DOWN at your sides (relaxed)",
     "A-poz: karok lazan lefele lognak a test mellett"),
    ("tray",  "TRAY-HOLDING",
     "Upper arms DOWN, forearms HORIZONTAL FORWARD, elbows 90 deg",
     "Talca-poz: felkar log, alkar VIZSZINTESEN ELORE (mint talcat tartva)"),
)

# Minimum number of valid samples per pose; below this we retry (and ultimately
# fail) instead of relying on 1-2 marginal frames that scrape past the
# validity check.  Critical for the tray pose, where a low-quality sample set
# would produce a bad elbow bias and the robot would mix up A-pose / tray.
#
# 3 is the lowest value that gives a usable median while still rejecting the
# "single noisy frame slipped through" failure mode.  Higher values (e.g. 8)
# made the calibration get stuck on A-pose / repeatedly retry, because the
# strict per-pose validity check + a short 1.5 s measuring window only let
# 5-7 frames through under normal lighting / posture.
_MIN_SAMPLES_PER_POSE = 3


# ── Anthropometric measurements (numpy-only, unit-test friendly) ─────────────

def _arm_length(lms: np.ndarray) -> float:
    """Average left+right shoulder→wrist distance (metres)."""
    left  = float(np.linalg.norm(lms[config.LM_LEFT_WRIST]  - lms[config.LM_LEFT_SHOULDER]))
    right = float(np.linalg.norm(lms[config.LM_RIGHT_WRIST] - lms[config.LM_RIGHT_SHOULDER]))
    return 0.5 * (left + right)


def _shoulder_width(lms: np.ndarray) -> float:
    return float(np.linalg.norm(
        lms[config.LM_LEFT_SHOULDER] - lms[config.LM_RIGHT_SHOULDER]
    ))


def _torso_height(lms: np.ndarray) -> float:
    mid_sh  = 0.5 * (lms[config.LM_LEFT_SHOULDER] + lms[config.LM_RIGHT_SHOULDER])
    mid_hip = 0.5 * (lms[config.LM_LEFT_HIP]      + lms[config.LM_RIGHT_HIP])
    return float(np.linalg.norm(mid_sh - mid_hip))


# ── Pose-validity check ──────────────────────────────────────────────────────

def _pose_validity(lms: np.ndarray, pose_key: str) -> tuple[bool, str]:
    """Check whether the current landmark frame matches the expected pose.

    Returns (ok, message). Uses body-frame vectors so it's resilient to where
    the camera is pointing.  Tolerances are deliberately loose — the goal is
    to catch *clearly wrong* poses (e.g. arms at sides during T-pose check),
    not to demand precision.
    """
    R = build_body_frame(lms)
    R_T = R.T

    def unit_body(a_idx: int, b_idx: int) -> np.ndarray:
        v = R_T @ (lms[b_idx] - lms[a_idx])
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.zeros(3)

    L_upper = unit_body(config.LM_LEFT_SHOULDER,  config.LM_LEFT_ELBOW)
    R_upper = unit_body(config.LM_RIGHT_SHOULDER, config.LM_RIGHT_ELBOW)
    L_fore  = unit_body(config.LM_LEFT_ELBOW,     config.LM_LEFT_WRIST)
    R_fore  = unit_body(config.LM_RIGHT_ELBOW,    config.LM_RIGHT_WRIST)

    # Body frame: x = subject left, y = forward (toward camera), z = up.
    # Tolerances are intentionally LOOSE — we only reject clearly wrong poses
    # (e.g. arms at sides during T-pose), not minor imperfections.
    if pose_key == "tpose":
        if L_upper[0] < 0.5:
            return False, "Stretch LEFT arm further out to the side"
        if R_upper[0] > -0.5:
            return False, "Stretch RIGHT arm further out to the side"
        if float(np.dot(L_upper, L_fore)) < 0.6:
            return False, "Keep your LEFT elbow straight"
        if float(np.dot(R_upper, R_fore)) < 0.6:
            return False, "Keep your RIGHT elbow straight"
        return True, "T-pose looks good"

    if pose_key == "apose":
        # Upper arm must CLEARLY point down (-z) but tolerate small
        # forward/back lean.  Earlier thresholds (-0.85, |y|<0.25, dot<0.9)
        # were so strict that natural relaxed standing failed validation
        # under normal MediaPipe noise — the calibration got stuck on
        # A-pose and never progressed to the tray phase.
        if L_upper[2] > -0.80:
            return False, "Drop LEFT arm straighter down (don't lean forward)"
        if R_upper[2] > -0.80:
            return False, "Drop RIGHT arm straighter down (don't lean forward)"
        if abs(L_upper[1]) > 0.35:
            return False, "LEFT upper arm should hang straight (not forward/back)"
        if abs(R_upper[1]) > 0.35:
            return False, "RIGHT upper arm should hang straight (not forward/back)"
        # Forearm should also point down (elbow roughly straight).
        if L_fore[2] > -0.80:
            return False, "Let LEFT forearm hang straight down"
        if R_fore[2] > -0.80:
            return False, "Let RIGHT forearm hang straight down"
        if float(np.dot(L_upper, L_fore)) < 0.85:
            return False, "Keep your LEFT elbow straight (relaxed)"
        if float(np.dot(R_upper, R_fore)) < 0.85:
            return False, "Keep your RIGHT elbow straight (relaxed)"
        return True, "A-pose looks good"

    if pose_key == "tray":
        # Depth-robust tray check.  We do NOT require forearm.y > 0 because
        # MediaPipe's single-RGB depth estimate is unreliable: the wrist
        # often projects BEHIND the elbow in body-frame y even when the user
        # is clearly holding a tray pose toward the camera.  The actual
        # forward/backward direction is reconstructed by the live retarget
        # from the *current* frame anyway, so calibration only needs to
        # capture the elbow-flex MAGNITUDE for the per-user bias.
        #
        # Sufficient conditions:
        #   (1) upper arm hangs DOWN (not raised, not horizontal)
        #   (2) elbow bent close to 90°  (this is the bias we're measuring,
        #       AND it is what distinguishes tray-holding from A-pose)
        L_elbow_dot = float(np.dot(L_upper, L_fore))
        R_elbow_dot = float(np.dot(R_upper, R_fore))
        L_flex_deg  = float(np.degrees(np.arccos(np.clip(L_elbow_dot, -1, 1))))
        R_flex_deg  = float(np.degrees(np.arccos(np.clip(R_elbow_dot, -1, 1))))

        if L_upper[2] > -0.55:
            return False, (
                f"Lower LEFT upper arm (u.z={L_upper[2]:+.2f}, need <-0.55)"
            )
        if R_upper[2] > -0.55:
            return False, (
                f"Lower RIGHT upper arm (u.z={R_upper[2]:+.2f}, need <-0.55)"
            )
        # Elbow ~90°: dot(u_upper, u_fore) = cos(flex).  |dot|<0.5 accepts
        # any flex in [60°, 120°] — generous for natural tray-holding.
        if abs(L_elbow_dot) > 0.5:
            return False, (
                f"LEFT elbow flex ~{L_flex_deg:.0f}°, bend closer to 90°"
            )
        if abs(R_elbow_dot) > 0.5:
            return False, (
                f"RIGHT elbow flex ~{R_flex_deg:.0f}°, bend closer to 90°"
            )
        return True, "Tray-holding looks good"

    return True, ""


# ── Drawing helpers ──────────────────────────────────────────────────────────

# Catppuccin-ish palette (BGR for OpenCV).
_C_BG_DARK   = (35, 35, 50)
_C_BG_MID    = (60, 60, 80)
_C_TEXT      = (244, 214, 205)
_C_ACCENT    = (250, 180, 137)
_C_GREEN     = (161, 227, 166)
_C_RED       = (168, 139, 243)
_C_YELLOW    = (175, 226, 249)
_C_GRAY      = (134, 112, 108)


def _draw_top_banner(frame: "np.ndarray", pose_idx: int, total: int,
                     pose_name: str, instruction: str) -> None:
    import cv2  # noqa: PLC0415
    h, w = frame.shape[:2]
    bh = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), _C_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    badge = f"POSE {pose_idx + 1} / {total}"
    cv2.putText(frame, badge, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, pose_name, (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, _C_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, instruction, (180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_TEXT, 1, cv2.LINE_AA)


def _draw_bottom_status(frame: "np.ndarray", phase_label: str,
                        sub_msg: str, color: tuple) -> None:
    import cv2  # noqa: PLC0415
    h, w = frame.shape[:2]
    bh = 56
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bh), (w, h), _C_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, phase_label, (12, h - bh + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, sub_msg, (12, h - bh + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_TEXT, 1, cv2.LINE_AA)


def _draw_progress_bar(frame: "np.ndarray", frac: float) -> None:
    import cv2  # noqa: PLC0415
    h, w = frame.shape[:2]
    y = h - 6
    cv2.rectangle(frame, (0, y), (w, h), _C_BG_MID, -1)
    cv2.rectangle(frame, (0, y), (int(w * max(0.0, min(1.0, frac))), h),
                  _C_ACCENT, -1)


def _draw_countdown(frame: "np.ndarray", seconds_remaining: float,
                    total_seconds: float) -> None:
    """Big animated countdown number + sweeping ring in the centre."""
    import cv2  # noqa: PLC0415
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    R = 70

    # Translucent dark disc behind the number.
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), R + 12, _C_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Sweeping arc — angle proportional to fraction elapsed.
    elapsed = max(0.0, total_seconds - seconds_remaining)
    sweep_deg = int(360 * min(1.0, elapsed / max(1e-6, total_seconds)))
    cv2.ellipse(frame, (cx, cy), (R, R), -90, 0, sweep_deg,
                _C_ACCENT, 4, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (R, R), -90, sweep_deg, 360,
                _C_BG_MID, 4, cv2.LINE_AA)

    # The integer ceiling of seconds remaining: 3 → 3, 2.4 → 3, 2.0 → 2, etc.
    n = max(0, int(np.ceil(seconds_remaining - 1e-3)))
    text = str(n) if n > 0 else "GO"
    scale = 3.0 if n > 0 else 1.6
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 6)
    cv2.putText(frame, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale,
                _C_YELLOW if n > 0 else _C_GREEN, 6, cv2.LINE_AA)


def _draw_get_ready_hint(
    frame: "np.ndarray",
    pose_name: str,
    instruction: str,
    hu_hint: str,
    seconds_remaining: float,
) -> None:
    """Big centred overlay shown during the GET_READY phase.

    The earlier UX had only the small banner at the top — users couldn't
    process "TRAY: forearms forward, 90° elbows" fast enough before the
    countdown started, and ended up adopting an A-pose-like stance during
    the tray pose.  This panel makes the pose name and instruction
    impossible to miss.
    """
    import cv2  # noqa: PLC0415

    h, w = frame.shape[:2]
    box_w = min(560, w - 40)
    box_h = 200
    x0 = (w - box_w) // 2
    y0 = (h - box_h) // 2 - 30

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                  _C_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h),
                  _C_ACCENT, 2, cv2.LINE_AA)

    # Pose name — huge.
    (tw, th), _ = cv2.getTextSize(pose_name, cv2.FONT_HERSHEY_SIMPLEX,
                                  1.4, 3)
    cv2.putText(frame, pose_name,
                (x0 + (box_w - tw) // 2, y0 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, _C_ACCENT, 3, cv2.LINE_AA)

    # English instruction.
    (tw, _), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.65, 2)
    cv2.putText(frame, instruction,
                (x0 + (box_w - tw) // 2, y0 + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _C_TEXT, 2, cv2.LINE_AA)

    # Hungarian hint.
    (tw, _), _ = cv2.getTextSize(hu_hint, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.55, 1)
    cv2.putText(frame, hu_hint,
                (x0 + (box_w - tw) // 2, y0 + 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _C_YELLOW, 1, cv2.LINE_AA)

    # Countdown to start of measurement.
    start_msg = f"Starts in {max(0, int(np.ceil(seconds_remaining - 1e-3)))}s"
    (tw, _), _ = cv2.getTextSize(start_msg, cv2.FONT_HERSHEY_SIMPLEX,
                                 0.7, 2)
    cv2.putText(frame, start_msg,
                (x0 + (box_w - tw) // 2, y0 + 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _C_GREEN, 2, cv2.LINE_AA)


def _draw_live_body_overlay(
    frame: "np.ndarray",
    lms: "np.ndarray",
    pose_key: str,
) -> None:
    """Live numeric panel showing what the validator sees: body-frame unit
    vectors for both arms + elbow flex angle.  Sticks to the LEFT edge of the
    frame so it doesn't overlap the silhouette reference card on the right.

    For each scalar shown, expected ranges are colour-coded:
        green   → currently passes the threshold for this pose
        red     → currently failing the threshold (this is what's rejecting you)
    """
    import cv2  # noqa: PLC0415

    R = build_body_frame(lms)
    R_T = R.T

    def unit(a: int, b: int) -> np.ndarray:
        v = R_T @ (lms[b] - lms[a])
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.zeros(3)

    L_up = unit(config.LM_LEFT_SHOULDER,  config.LM_LEFT_ELBOW)
    R_up = unit(config.LM_RIGHT_SHOULDER, config.LM_RIGHT_ELBOW)
    L_fo = unit(config.LM_LEFT_ELBOW,     config.LM_LEFT_WRIST)
    R_fo = unit(config.LM_RIGHT_ELBOW,    config.LM_RIGHT_WRIST)

    L_flex_deg = float(np.degrees(np.arccos(
        np.clip(float(np.dot(L_up, L_fo)), -1.0, 1.0))))
    R_flex_deg = float(np.degrees(np.arccos(
        np.clip(float(np.dot(R_up, R_fo)), -1.0, 1.0))))

    # Pose-specific OK ranges for the values we *actually* check.  The
    # forearm body-y component is intentionally NOT enforced for tray
    # (see _pose_validity) because MediaPipe's depth axis is unreliable
    # on monocular RGB and frequently shows the wrist behind the elbow
    # even when the user holds a clean tray pose — that's display-only.
    if pose_key == "tray":
        ok_up_z   = lambda v: v < -0.55
        ok_flex   = lambda d: 60.0 <= d <= 120.0
        flex_hint = "target 90°"
    elif pose_key == "apose":
        ok_up_z   = lambda v: v < -0.80
        ok_flex   = lambda d: d < 32.0   # straight elbow
        flex_hint = "target 0°"
    else:  # tpose
        ok_up_z   = lambda v: True
        ok_flex   = lambda d: True
        flex_hint = ""

    # Card geometry — left edge, below the top banner.
    h, w = frame.shape[:2]
    card_w, card_h = 240, 145
    x0, y0 = 10, 85

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h),
                  _C_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + card_w, y0 + card_h),
                  _C_BG_MID, 1)

    cv2.putText(frame, "Live body-frame", (x0 + 6, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "Green = OK, Red = adjust", (x0 + 6, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, _C_GRAY, 1, cv2.LINE_AA)

    def line(y: int, label: str, val: float, ok: bool) -> None:
        col = _C_GREEN if ok else _C_RED
        cv2.putText(frame, label, (x0 + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{val:+.2f}", (x0 + card_w - 56, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    def line_deg(y: int, label: str, deg: float, ok: bool) -> None:
        col = _C_GREEN if ok else _C_RED
        cv2.putText(frame, label, (x0 + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{deg:5.0f}°", (x0 + card_w - 56, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    line(y0 +  54, "L upper down",      L_up[2], ok_up_z(L_up[2]))
    line(y0 +  74, "R upper down",      R_up[2], ok_up_z(R_up[2]))
    line_deg(y0 +  98, f"L elbow flex   {flex_hint}",
             L_flex_deg, ok_flex(L_flex_deg))
    line_deg(y0 + 118, f"R elbow flex   {flex_hint}",
             R_flex_deg, ok_flex(R_flex_deg))


def _draw_done_check(frame: "np.ndarray", message: str) -> None:
    """Big green ✓ + message for the DONE phase."""
    import cv2  # noqa: PLC0415
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), 90, _C_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Draw a thick checkmark.
    p1 = (cx - 35, cy + 5)
    p2 = (cx - 8,  cy + 30)
    p3 = (cx + 35, cy - 25)
    cv2.line(frame, p1, p2, _C_GREEN, 8, cv2.LINE_AA)
    cv2.line(frame, p2, p3, _C_GREEN, 8, cv2.LINE_AA)

    (tw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, message, (cx - tw // 2, cy + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _C_GREEN, 2, cv2.LINE_AA)


def _draw_reference_silhouette(frame: "np.ndarray", pose_key: str) -> None:
    """Tiny stick-figure preview of the target pose, top-right corner.

    Drawing strategy:

    * T-pose and A-pose are unambiguous from the front, so we use a frontal
      stick figure for both.
    * Tray-holding looks **identical to A-pose from the front** (forearm
      foreshortened to a stub), so we draw a SIDE-VIEW figure for the tray
      pose — the 90° "L" of the arm becomes obvious and impossible to
      confuse with A-pose.
    """
    import cv2  # noqa: PLC0415

    box_w, box_h = 130, 150
    margin = 10
    fh, fw = frame.shape[:2]
    x0 = fw - box_w - margin
    y0 = 80   # below the top banner

    # Background card.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                  _C_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h),
                  _C_BG_MID, 1)

    label = "Target (side view)" if pose_key == "tray" else "Target"
    cv2.putText(frame, label, (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _C_GRAY, 1, cv2.LINE_AA)

    color = _C_ACCENT
    cxf = x0 + box_w // 2

    if pose_key == "tray":
        # ── SIDE-VIEW figure ────────────────────────────────────────────────
        # Person faces RIGHT.  Spine vertical, single shoulder at the top,
        # upper arm hangs straight DOWN, forearm bends 90° and points to the
        # right (FORWARD).  A clearly-drawn tray sits on top of the wrist.
        # This silhouette cannot be mistaken for the frontal A-pose.
        cyf = y0 + 40                      # head centre
        # Head (profile circle + small "nose" tick to the right).
        cv2.circle(frame, (cxf - 10, cyf), 9, color, 2, cv2.LINE_AA)
        cv2.line(frame, (cxf - 1, cyf), (cxf + 4, cyf - 1),
                 color, 2, cv2.LINE_AA)
        # Spine.
        spine_top = (cxf - 10, cyf + 9)
        spine_bot = (cxf - 10, cyf + 70)
        cv2.line(frame, spine_top, spine_bot, color, 2, cv2.LINE_AA)
        # Shoulder anchor and upper arm hanging straight down.
        sh = (cxf - 6, cyf + 15)
        el = (cxf - 6, cyf + 50)
        cv2.line(frame, sh, el, color, 3, cv2.LINE_AA)
        # Forearm HORIZONTAL FORWARD (= toward the right of the figure).
        wr = (el[0] + 38, el[1])
        cv2.line(frame, el, wr, color, 3, cv2.LINE_AA)
        # Right-angle marker at the elbow so the 90° bend is unambiguous.
        cv2.line(frame, (el[0] + 6, el[1]), (el[0] + 6, el[1] - 6),
                 color, 1, cv2.LINE_AA)
        cv2.line(frame, (el[0], el[1] - 6), (el[0] + 6, el[1] - 6),
                 color, 1, cv2.LINE_AA)
        # The tray itself — solid rectangle resting on top of the wrist.
        tray_l, tray_t = wr[0] - 18, wr[1] - 7
        tray_r, tray_b = wr[0] + 12, wr[1] - 2
        cv2.rectangle(frame, (tray_l, tray_t), (tray_r, tray_b),
                      color, -1, cv2.LINE_AA)
        cv2.putText(frame, "tray", (tray_l - 2, tray_t - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        # Foot stub so it doesn't look like a floating torso.
        cv2.line(frame, spine_bot, (spine_bot[0] + 6, spine_bot[1] + 6),
                 color, 2, cv2.LINE_AA)
        return

    # ── FRONTAL view for T-pose and A-pose ──────────────────────────────────
    cyf = y0 + 50              # head centre

    # Head.
    cv2.circle(frame, (cxf, cyf), 7, color, 2, cv2.LINE_AA)
    # Spine.
    spine_top = (cxf, cyf + 7)
    spine_bot = (cxf, cyf + 50)
    cv2.line(frame, spine_top, spine_bot, color, 2, cv2.LINE_AA)
    # Hips.
    hip_l = (cxf - 10, cyf + 55)
    hip_r = (cxf + 10, cyf + 55)
    cv2.line(frame, hip_l, hip_r, color, 2, cv2.LINE_AA)
    # Shoulder line (used as arm anchor).
    sh_l = (cxf - 14, cyf + 13)
    sh_r = (cxf + 14, cyf + 13)
    cv2.line(frame, sh_l, sh_r, color, 2, cv2.LINE_AA)

    if pose_key == "tpose":
        # Arms straight out horizontally (NOTE: drawing is non-mirrored).
        cv2.line(frame, sh_l, (sh_l[0] - 30, sh_l[1]), color, 2, cv2.LINE_AA)
        cv2.line(frame, sh_r, (sh_r[0] + 30, sh_r[1]), color, 2, cv2.LINE_AA)
    elif pose_key == "apose":
        # Arms straight down.
        cv2.line(frame, sh_l, (sh_l[0], sh_l[1] + 38), color, 2, cv2.LINE_AA)
        cv2.line(frame, sh_r, (sh_r[0], sh_r[1] + 38), color, 2, cv2.LINE_AA)


# ── Visibility check ─────────────────────────────────────────────────────────

_REQUIRED_LANDMARKS = (
    config.LM_LEFT_SHOULDER,  config.LM_RIGHT_SHOULDER,
    config.LM_LEFT_ELBOW,     config.LM_RIGHT_ELBOW,
    config.LM_LEFT_WRIST,     config.LM_RIGHT_WRIST,
    config.LM_LEFT_HIP,       config.LM_RIGHT_HIP,
)


def _required_landmarks_visible(vis: np.ndarray | None) -> bool:
    if vis is None:
        return False
    return all(vis[i] >= config.VISIBILITY_THRESHOLD for i in _REQUIRED_LANDMARKS)


# ── Single-pose capture (4-phase state machine) ──────────────────────────────

def _capture_pose(
    cap: "cv2.VideoCapture",
    pose: "PoseEstimator",
    pose_idx: int,
    total_poses: int,
    pose_key: str,
    pose_name: str,
    instruction: str,
    countdown_s: float,
    collect_s: float,
    hu_hint: str = "",
    get_ready_s: float = 3.5,
    done_s: float = 1.0,
    overall_progress: tuple[float, float] = (0.0, 1.0),
    on_frame=None,
    on_abort=None,
    attempt: int = 1,
    max_attempts: int = 3,
) -> list[np.ndarray]:
    """Run the 4-phase capture for a single pose.

    Returns the list of valid (33, 3) landmark arrays collected during the
    MEASURING phase.

    Parameters
    ----------
    on_frame : callable(bgr_image) -> None
        Hands every annotated frame to the caller (so the GUI can show the
        calibration UX inside its own camera preview, no separate cv2 window).
        If None, the function pops a cv2 window itself (CLI mode).
    on_abort : callable() -> bool
        Polled each iteration; if it returns True the function raises
        RuntimeError to abort calibration (e.g. user pressed Stop in the GUI).
    overall_progress : (lo, hi)
        Fraction of the global progress bar this pose occupies — grows linearly
        from `lo` to `hi` over the pose's 4 phases.
    """
    PHASE_GET_READY = 0
    PHASE_COUNTDOWN = 1
    PHASE_MEASURING = 2
    PHASE_DONE      = 3

    durations = [get_ready_s, countdown_s, collect_s, done_s]
    boundaries = np.cumsum(durations)
    total_t = float(boundaries[-1])
    prog_lo, prog_hi = overall_progress

    use_cv2_window = on_frame is None
    if use_cv2_window:
        import cv2  # noqa: PLC0415

    collected: list[np.ndarray] = []
    t_start = time.time()

    while True:
        if on_abort is not None and on_abort():
            raise RuntimeError("Calibration aborted by caller.")

        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - t_start

        if elapsed < boundaries[0]:
            phase = PHASE_GET_READY
        elif elapsed < boundaries[1]:
            phase = PHASE_COUNTDOWN
        elif elapsed < boundaries[2]:
            phase = PHASE_MEASURING
        else:
            phase = PHASE_DONE

        res = pose.process(frame)

        # Mirror first so the operator sees a natural selfie view; overlays
        # drawn afterward stay readable.
        import cv2 as _cv2  # noqa: PLC0415
        shown = _cv2.flip(res.annotated, 1)

        banner_name = (
            f"{pose_name}   (RETRY {attempt}/{max_attempts})"
            if attempt > 1 else pose_name
        )
        _draw_top_banner(shown, pose_idx, total_poses, banner_name, instruction)
        _draw_reference_silhouette(shown, pose_key)
        _draw_progress_bar(shown,
                           prog_lo + (prog_hi - prog_lo) * (elapsed / total_t))

        if phase == PHASE_GET_READY:
            # Big centred panel so the pose name + instruction (in EN and HU)
            # are impossible to miss before the countdown starts.
            _draw_get_ready_hint(
                shown, pose_name, instruction, hu_hint,
                seconds_remaining=boundaries[0] - elapsed,
            )
            _draw_bottom_status(
                shown,
                "GET READY...",
                f"Adopt the {pose_name} pose — countdown starts in "
                f"{boundaries[0] - elapsed:.0f}s",
                _C_YELLOW,
            )

        elif phase == PHASE_COUNTDOWN:
            t_remaining = boundaries[1] - elapsed
            _draw_countdown(shown, t_remaining, countdown_s)
            _draw_bottom_status(
                shown, "HOLD POSITION", instruction, _C_YELLOW,
            )

        elif phase == PHASE_MEASURING:
            ok_landmarks = (
                res.world_landmarks is not None
                and _required_landmarks_visible(res.visibility)
            )
            ok_pose, pose_msg = (False, "Stand fully in view")
            if ok_landmarks:
                ok_pose, pose_msg = _pose_validity(
                    res.world_landmarks, pose_key,
                )
                if ok_pose:
                    collected.append(res.world_landmarks.copy())

            # Live body-frame numeric overlay — shows the EXACT vectors the
            # validator sees so you can tell at a glance whether your pose
            # is being measured correctly (or which axis is off-target).
            if ok_landmarks:
                _draw_live_body_overlay(shown, res.world_landmarks, pose_key)

            color = _C_GREEN if ok_pose else _C_RED
            sub = (
                f"Samples: {len(collected)}  —  {pose_msg}"
                if ok_pose else f"Adjust: {pose_msg}"
            )
            _draw_bottom_status(shown, "MEASURING", sub, color)

        else:  # PHASE_DONE
            samples_msg = (
                f"{len(collected)} samples collected"
                if collected else "No valid samples — will retry"
            )
            _draw_done_check(shown, samples_msg)
            _draw_bottom_status(
                shown, "POSE CAPTURED",
                "Moving to next pose..." if pose_idx + 1 < total_poses
                else "Calibration complete!",
                _C_GREEN,
            )

        if use_cv2_window:
            _cv2.imshow("Real2Sim — Calibration", shown)
            if _cv2.waitKey(1) & 0xFF == ord("q"):
                raise RuntimeError("Calibration aborted by user.")
        else:
            on_frame(shown)

        if elapsed >= total_t:
            break

    return collected


# ── Public entry point ───────────────────────────────────────────────────────

def run_calibration(
    cap: "cv2.VideoCapture",
    pose: "PoseEstimator",
    arm_length_robot: float,
    countdown_s: float = config.CALIB_COUNTDOWN_S,
    collect_s: float = config.CALIB_COLLECT_S,
    on_frame=None,
    on_abort=None,
) -> BodyCalibration:
    """Run the 3-pose anthropometric calibration. Blocks until done.

    Parameters
    ----------
    on_frame : callable(bgr_image) -> None, optional
        If supplied, every overlaid calibration frame is handed to this
        callback (GUI mode — display inside the existing preview).  If None,
        the function pops its own cv2 window (CLI mode).
    on_abort : callable() -> bool, optional
        Polled each iteration; True aborts with RuntimeError.
    """
    use_cv2_window = on_frame is None
    if use_cv2_window:
        import cv2  # noqa: PLC0415

    captures: dict[str, list[np.ndarray]] = {}

    try:
        for i, (key, name, instruction, hu_hint) in enumerate(_POSES):
            prog_lo = i / len(_POSES)
            prog_hi = (i + 1) / len(_POSES)

            frames: list[np.ndarray] = []
            attempt = 0
            max_attempts = 3
            # Require at least _MIN_SAMPLES_PER_POSE valid frames — a handful
            # of marginal frames is not enough to reliably compute the
            # elbow-bias (and a bad elbow bias is what makes the live robot
            # swap A-pose <-> tray pose).
            while len(frames) < _MIN_SAMPLES_PER_POSE and attempt < max_attempts:
                attempt += 1
                frames = _capture_pose(
                    cap, pose,
                    pose_idx=i, total_poses=len(_POSES),
                    pose_key=key, pose_name=name, instruction=instruction,
                    countdown_s=countdown_s, collect_s=collect_s,
                    hu_hint=hu_hint,
                    overall_progress=(prog_lo, prog_hi),
                    on_frame=on_frame,
                    on_abort=on_abort,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if len(frames) < _MIN_SAMPLES_PER_POSE and attempt < max_attempts:
                    print(
                        f"[calibration] '{name}' only collected {len(frames)} "
                        f"valid samples (need {_MIN_SAMPLES_PER_POSE}); "
                        f"retrying ({attempt}/{max_attempts}) — please match "
                        "the on-screen reference more closely."
                    )

            if len(frames) < _MIN_SAMPLES_PER_POSE:
                raise RuntimeError(
                    f"Calibration failed during '{name}': only "
                    f"{len(frames)} valid samples (need "
                    f"{_MIN_SAMPLES_PER_POSE}). Match the on-screen target "
                    "pose more precisely — for TRAY-HOLDING, your upper "
                    "arms must hang straight down and your forearms must "
                    "be HORIZONTAL FORWARD (90° elbows, like holding a tray)."
                )
            captures[key] = frames
    finally:
        if use_cv2_window:
            cv2.destroyWindow("Real2Sim — Calibration")

    # ── Aggregate measurements (median, robust to outliers) ─────────────
    t_frames    = captures["tpose"]
    a_frames    = captures["apose"]
    tray_frames = captures["tray"]

    arm_length_user = float(np.median([_arm_length(f)     for f in t_frames]))
    shoulder_width  = float(np.median([_shoulder_width(f) for f in a_frames]))
    torso_height    = float(np.median([_torso_height(f)   for f in a_frames]))

    if arm_length_user < 0.15:
        raise RuntimeError(
            f"Calibration failed: T-pose arm length {arm_length_user:.3f} m "
            "is impossibly small. MediaPipe may be confused — retry."
        )

    scale = float(arm_length_robot / arm_length_user)

    # Anatomical-zero — shoulder/yaw joints from A-pose, elbow joints from
    # the tray-holding pose (which provides a known elbow target of π/2 so we
    # can subtract a per-user elbow bias that the A-pose can't see).
    a_zero_samples    = np.stack([compute_anatomical_angles(f) for f in a_frames])
    tray_zero_samples = np.stack([compute_anatomical_angles(f) for f in tray_frames])

    anatomical_zero_arr = np.median(a_zero_samples, axis=0)

    # Indices in the 14-D ArmAngles layout (L then R, 7 each).
    L_ELBOW_IDX, R_ELBOW_IDX = 3, 10
    # Median elbow value the user produces while holding the tray (should be
    # very close to π/2 but small posture-dependent offsets are normal).
    tray_elbow_L = float(np.median(tray_zero_samples[:, L_ELBOW_IDX]))
    tray_elbow_R = float(np.median(tray_zero_samples[:, R_ELBOW_IDX]))

    # Sanity check: tray-pose elbow must be close to π/2.  If it is wildly
    # off (e.g. ~0 rad, meaning the user actually held A-pose during the tray
    # phase), the resulting bias of −π/2 would *swap* A-pose and tray on the
    # live robot — exactly the "the robot does it backwards" symptom.  We
    # refuse to save such a calibration and force a retry.
    #
    # Tolerance of 0.8 rad (~46°) is roomy on purpose: a clean tray-pose
    # passes easily, but a near-A-pose (elbow ~0) or a stretched-out arm
    # (elbow ~π) are still firmly rejected.
    HALF_PI = 0.5 * np.pi
    if (abs(tray_elbow_L - HALF_PI) > 0.8
            or abs(tray_elbow_R - HALF_PI) > 0.8):
        raise RuntimeError(
            "Calibration sanity check failed for TRAY-HOLDING pose: "
            f"measured elbow flexion L={tray_elbow_L:.2f} rad, "
            f"R={tray_elbow_R:.2f} rad (expected ~{HALF_PI:.2f} = pi/2). "
            "Your forearms must be HORIZONTAL FORWARD with the elbows bent "
            "to 90 degrees — exactly as if you were holding a tray. Please "
            "recalibrate."
        )

    # Bias = observed − expected.  Subtracting this from live frames lines up
    # the user's "natural" elbow flexion with the robot's elbow zero.
    anatomical_zero_arr[L_ELBOW_IDX] = tray_elbow_L - HALF_PI
    anatomical_zero_arr[R_ELBOW_IDX] = tray_elbow_R - HALF_PI

    anatomical_zero = anatomical_zero_arr.tolist()

    # Diagnostic log — helps spot bad calibration immediately (e.g. very
    # large elbow biases hint that the tray pose was held incorrectly).
    print(
        f"[calibration] tray elbow medians: "
        f"L={tray_elbow_L:+.3f} rad, R={tray_elbow_R:+.3f} rad "
        f"(expected ~{HALF_PI:+.3f})"
    )
    print(
        f"[calibration] anatomical_zero L (pitch/roll/yaw/elbow): "
        f"{anatomical_zero_arr[0]:+.3f} {anatomical_zero_arr[1]:+.3f} "
        f"{anatomical_zero_arr[2]:+.3f} {anatomical_zero_arr[3]:+.3f}"
    )
    print(
        f"[calibration] anatomical_zero R (pitch/roll/yaw/elbow): "
        f"{anatomical_zero_arr[7]:+.3f} {anatomical_zero_arr[8]:+.3f} "
        f"{anatomical_zero_arr[9]:+.3f} {anatomical_zero_arr[10]:+.3f}"
    )

    return BodyCalibration(
        arm_length_user     = arm_length_user,
        shoulder_width_user = shoulder_width,
        torso_height_user   = torso_height,
        arm_length_robot    = float(arm_length_robot),
        scale               = scale,
        anatomical_zero     = anatomical_zero,
    )
