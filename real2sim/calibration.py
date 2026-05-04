"""T-pose calibration: measure user arm length and compute scale factor.

At startup the user holds a T-pose (arms extended straight out) for a few
seconds.  The module collects MediaPipe landmarks during that window, averages
the shoulder-to-wrist distance, and derives a scale factor so that the robot's
arm proportions match the user's.

Usage (from runner.py)::

    from real2sim.calibration import run_calibration, BodyCalibration, load_calibration

    calib = load_calibration()            # returns None if no cache
    if calib is None:
        calib = run_calibration(cap, pose_estimator, arm_length_robot=0.55)
        calib.save()
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from . import config
from .pose import PoseEstimator, PoseResult


# ── Public dataclass ─────────────────────────────────────────────────────────

@dataclass
class BodyCalibration:
    arm_length_user:  float   # metres, shoulder→wrist at T-pose
    arm_length_robot: float   # metres, as measured from the MJCF
    scale:            float   # arm_length_robot / arm_length_user

    def save(self, path=config.CALIBRATION_PATH) -> None:
        path = config.CALIBRATION_PATH if path is None else path
        path = config.PROJECT_ROOT / path if not str(path).startswith(str(config.PROJECT_ROOT)) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path=config.CALIBRATION_PATH) -> "BodyCalibration | None":
        try:
            d = json.loads(path.read_text())
            return cls(**d)
        except Exception:
            return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _arm_length_from_landmarks(lms: np.ndarray) -> float:
    """Average left+right shoulder→wrist distance (metres)."""
    left  = float(np.linalg.norm(lms[config.LM_LEFT_WRIST]  - lms[config.LM_LEFT_SHOULDER]))
    right = float(np.linalg.norm(lms[config.LM_RIGHT_WRIST] - lms[config.LM_RIGHT_SHOULDER]))
    return 0.5 * (left + right)


def _draw_overlay(frame: np.ndarray, msg: str, sub: str = "") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h // 2 - 55), (w, h // 2 + 55), (0, 0, 0), -1)
    cv2.putText(out, msg, (w // 2 - len(msg) * 9, h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (w // 2 - len(sub) * 7, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ── Main entry point ─────────────────────────────────────────────────────────

def run_calibration(
    cap: "cv2.VideoCapture",
    pose: PoseEstimator,
    arm_length_robot: float,
    countdown_s: float = config.CALIB_COUNTDOWN_S,
    collect_s: float = config.CALIB_COLLECT_S,
) -> BodyCalibration:
    """Show countdown on-screen, collect T-pose frames, return calibration.

    Blocks until calibration is complete.  Press 'q' to abort (raises
    RuntimeError).
    """
    lengths: list[float] = []
    t_start = time.time()
    total = countdown_s + collect_s

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - t_start
        res: PoseResult = pose.process(frame)

        if elapsed < countdown_s:
            remaining = int(countdown_s - elapsed) + 1
            shown = _draw_overlay(
                res.annotated,
                f"Hold T-pose — {remaining}s",
                "Stretch arms out straight to the sides",
            )
        else:
            # Collecting phase
            if res.world_landmarks is not None:
                vis = res.visibility
                ok = (
                    vis is not None
                    and vis[config.LM_LEFT_WRIST]   >= config.VISIBILITY_THRESHOLD
                    and vis[config.LM_RIGHT_WRIST]  >= config.VISIBILITY_THRESHOLD
                    and vis[config.LM_LEFT_SHOULDER] >= config.VISIBILITY_THRESHOLD
                    and vis[config.LM_RIGHT_SHOULDER] >= config.VISIBILITY_THRESHOLD
                )
                if ok:
                    lengths.append(_arm_length_from_landmarks(res.world_landmarks))

            shown = _draw_overlay(res.annotated, "Measuring…", f"Samples: {len(lengths)}")

        shown = cv2.flip(shown, 1)
        cv2.imshow("Real2Sim — Calibration", shown)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise RuntimeError("Calibration aborted by user.")

        if elapsed >= total:
            break

    cv2.destroyWindow("Real2Sim — Calibration")

    if not lengths:
        raise RuntimeError(
            "Calibration failed: no valid pose detected during the T-pose window. "
            "Ensure your whole upper body (shoulders + wrists) is visible."
        )

    arm_length_user = float(np.median(lengths))
    raw_scale = arm_length_robot / max(arm_length_user, 0.1)
    scale = float(np.clip(raw_scale, config.CALIB_SCALE_MIN, config.CALIB_SCALE_MAX))

    return BodyCalibration(
        arm_length_user=arm_length_user,
        arm_length_robot=arm_length_robot,
        scale=scale,
    )
