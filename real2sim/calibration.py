"""3-pose anthropometric calibration for analytical retargeting.

The user is asked to hold three reference poses in sequence:

    1. **T-pose**          arms straight out to the sides     (~3 s)
    2. **A-pose**          arms hanging straight down         (~3 s)
    3. **Elbow-90°**       arms forward, elbows bent ~90°     (~3 s)

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
* `anatomical_zero`        14-D vector — the joint angles produced by the
                            analytical retargeter when the user holds the
                            A-pose.  Subtracting this from every live frame
                            maps the user's natural rest pose exactly to the
                            robot's zeros.

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

# (key, display name, instruction text)
_POSES = (
    ("tpose",   "T-POSE",         "Stretch arms straight out to the sides"),
    ("apose",   "A-POSE",         "Let your arms hang straight down (relaxed)"),
    ("elbow90", "ELBOW 90°",      "Arms forward, bend elbows 90°"),
)


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
        # Upper arm must CLEARLY point down: strong -z (≥ ~58°) AND not
        # forward/back (|y| small).  This is the pose that captures the
        # anatomical zero, so be strict — a slightly-forward upper arm
        # here causes the live retargeter to invert A-pose with elbow-90°.
        if L_upper[2] > -0.85:
            return False, "Drop LEFT arm straighter down (don't lean forward)"
        if R_upper[2] > -0.85:
            return False, "Drop RIGHT arm straighter down (don't lean forward)"
        if abs(L_upper[1]) > 0.25:
            return False, "LEFT upper arm should hang straight (not forward/back)"
        if abs(R_upper[1]) > 0.25:
            return False, "RIGHT upper arm should hang straight (not forward/back)"
        # Forearm must also point down (elbow straight, forearm not swung
        # forward).
        if L_fore[2] > -0.85:
            return False, "Let LEFT forearm hang straight down"
        if R_fore[2] > -0.85:
            return False, "Let RIGHT forearm hang straight down"
        if float(np.dot(L_upper, L_fore)) < 0.9:
            return False, "Keep your LEFT elbow straight (relaxed)"
        if float(np.dot(R_upper, R_fore)) < 0.9:
            return False, "Keep your RIGHT elbow straight (relaxed)"
        return True, "A-pose looks good"

    if pose_key == "elbow90":
        # Upper arm clearly forward (y ≥ 0.6 → arm at most ~53° off forward).
        if L_upper[1] < 0.6:
            return False, "Raise LEFT upper arm to point FORWARD (parallel to floor)"
        if R_upper[1] < 0.6:
            return False, "Raise RIGHT upper arm to point FORWARD (parallel to floor)"
        # Forearm must point DOWN (z ≤ -0.6 → ~53° below horizontal).
        if L_fore[2] > -0.6:
            return False, "Drop LEFT forearm straight DOWN (90° at elbow)"
        if R_fore[2] > -0.6:
            return False, "Drop RIGHT forearm straight DOWN (90° at elbow)"
        # Elbow ≈ 90° → u_upper ⟂ u_fore → |dot| close to 0.
        if abs(float(np.dot(L_upper, L_fore))) > 0.4:
            return False, "Bend LEFT elbow closer to 90°"
        if abs(float(np.dot(R_upper, R_fore))) > 0.4:
            return False, "Bend RIGHT elbow closer to 90°"
        return True, "Elbow 90° looks good"

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
    """Tiny stick-figure preview of the target pose, top-right corner."""
    import cv2  # noqa: PLC0415

    box_w, box_h = 110, 130
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

    cv2.putText(frame, "Target", (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _C_GRAY, 1, cv2.LINE_AA)

    # Centre of the figure inside the card.
    cxf = x0 + box_w // 2
    cyf = y0 + 50          # head centre
    color = _C_ACCENT

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
    elif pose_key == "elbow90":
        # Upper arm forward (down-right diagonal to suggest forward shortening),
        # forearm straight down.
        el_l = (sh_l[0] - 14, sh_l[1] + 14)
        el_r = (sh_r[0] + 14, sh_r[1] + 14)
        cv2.line(frame, sh_l, el_l, color, 2, cv2.LINE_AA)
        cv2.line(frame, sh_r, el_r, color, 2, cv2.LINE_AA)
        cv2.line(frame, el_l, (el_l[0], el_l[1] + 28), color, 2, cv2.LINE_AA)
        cv2.line(frame, el_r, (el_r[0], el_r[1] + 28), color, 2, cv2.LINE_AA)


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
    get_ready_s: float = 2.0,
    done_s: float = 1.0,
    overall_progress: tuple[float, float] = (0.0, 1.0),
    on_frame=None,
    on_abort=None,
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

        _draw_top_banner(shown, pose_idx, total_poses, pose_name, instruction)
        _draw_reference_silhouette(shown, pose_key)
        _draw_progress_bar(shown,
                           prog_lo + (prog_hi - prog_lo) * (elapsed / total_t))

        if phase == PHASE_GET_READY:
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
        for i, (key, name, instruction) in enumerate(_POSES):
            prog_lo = i / len(_POSES)
            prog_hi = (i + 1) / len(_POSES)

            frames: list[np.ndarray] = []
            retries = 0
            max_retries = 1
            while not frames and retries <= max_retries:
                frames = _capture_pose(
                    cap, pose,
                    pose_idx=i, total_poses=len(_POSES),
                    pose_key=key, pose_name=name, instruction=instruction,
                    countdown_s=countdown_s, collect_s=collect_s,
                    overall_progress=(prog_lo, prog_hi),
                    on_frame=on_frame,
                    on_abort=on_abort,
                )
                retries += 1

            if not frames:
                raise RuntimeError(
                    f"Calibration failed during '{name}': no valid samples "
                    "collected. Ensure your whole upper body is visible and "
                    "try again."
                )
            captures[key] = frames
    finally:
        if use_cv2_window:
            cv2.destroyWindow("Real2Sim — Calibration")

    # ── Aggregate measurements (median, robust to outliers) ─────────────
    t_frames = captures["tpose"]
    a_frames = captures["apose"]

    arm_length_user = float(np.median([_arm_length(f)     for f in t_frames]))
    shoulder_width  = float(np.median([_shoulder_width(f) for f in a_frames]))
    torso_height    = float(np.median([_torso_height(f)   for f in a_frames]))

    if arm_length_user < 0.15:
        raise RuntimeError(
            f"Calibration failed: T-pose arm length {arm_length_user:.3f} m "
            "is impossibly small. MediaPipe may be confused — retry."
        )

    scale = float(arm_length_robot / arm_length_user)

    # Anatomical-zero from the A-pose frames.
    zero_samples = np.stack([compute_anatomical_angles(f) for f in a_frames])
    anatomical_zero = np.median(zero_samples, axis=0).tolist()

    return BodyCalibration(
        arm_length_user     = arm_length_user,
        shoulder_width_user = shoulder_width,
        torso_height_user   = torso_height,
        arm_length_robot    = float(arm_length_robot),
        scale               = scale,
        anatomical_zero     = anatomical_zero,
    )
