"""Main loop: webcam → MediaPipe → One-Euro filter → retarget → MuJoCo.

Modes
-----
--pose-only     webcam + skeleton overlay, no simulation.
--viewer        live MuJoCo passive viewer + camera preview window.
--record PATH   offscreen split-screen mp4 (camera | sim).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from . import config
from .calibration import BodyCalibration, run_calibration
from .one_euro import OneEuroFilterArray
from .pose import PoseEstimator, PoseResult
from .recorder import SplitScreenRecorder
from .retarget import compute_arm_angles
from .sim import G1Sim


# ── Options dataclass ─────────────────────────────────────────────────────────

@dataclass
class RunOptions:
    camera: int           = config.CAMERA_INDEX
    pose_only: bool       = False
    viewer: bool          = False
    record_path: Path | None = None
    duration: float       = 0.0
    mirror_display: bool  = True
    debug_print: bool     = False
    recalibrate: bool     = False
    angle_debug: bool     = False


# ── Camera helpers ────────────────────────────────────────────────────────────

def _open_camera(idx: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {idx}")
    return cap


# ── HUD overlay ───────────────────────────────────────────────────────────────

def _hud_text(targets: np.ndarray) -> list[str]:
    """14-element target vector → 2 summary lines for the camera overlay."""
    n = len(targets) // 2
    L = targets[:n]
    R = targets[n:]
    fmt = lambda v: " ".join(f"{x:+.2f}" for x in v[:4])  # pitch/roll/yaw/elbow
    return [f"L: {fmt(L)}", f"R: {fmt(R)}"]


def _annotate(
    frame: np.ndarray,
    lines: list[str],
    fps: float,
    mirror: bool,
    calib: BodyCalibration | None,
) -> np.ndarray:
    shown = cv2.flip(frame, 1) if mirror else frame.copy()
    cv2.putText(
        shown, f"fps={fps:5.1f}  q=quit", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    if calib is not None:
        cv2.putText(
            shown,
            f"scale={calib.scale:.2f}  arm={calib.arm_length_user:.2f}m",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
        )
    y = 75
    for ln in lines:
        cv2.putText(
            shown, ln, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )
        y += 18
    return shown


# ── Visibility check helper ──────────────────────────────────────────────────

def _arms_visible(vis: np.ndarray | None, threshold: float) -> bool:
    if vis is None:
        return False
    required = (
        config.LM_LEFT_SHOULDER,  config.LM_RIGHT_SHOULDER,
        config.LM_LEFT_ELBOW,     config.LM_RIGHT_ELBOW,
        config.LM_LEFT_WRIST,     config.LM_RIGHT_WRIST,
        config.LM_LEFT_HIP,       config.LM_RIGHT_HIP,
    )
    return all(vis[i] >= threshold for i in required)


# ── Main run function ─────────────────────────────────────────────────────────

def run(opts: RunOptions) -> int:  # noqa: C901
    cap  = _open_camera(opts.camera)
    pose = PoseEstimator()

    sim: G1Sim | None = None
    calib: BodyCalibration | None = None
    anatomical_zero: np.ndarray | None = None
    viewer   = None
    renderer: mujoco.Renderer | None = None
    recorder: SplitScreenRecorder | None = None

    lm_filter = OneEuroFilterArray(shape=(33, 3))

    try:
        if not opts.pose_only:
            sim = G1Sim()

            # ── Calibration ──────────────────────────────────────────────────
            if not opts.recalibrate:
                calib = BodyCalibration.load(config.CALIBRATION_PATH)
                if calib is not None:
                    print(
                        f"[calibration] loaded from cache — "
                        f"scale={calib.scale:.3f}, "
                        f"user_arm={calib.arm_length_user:.3f} m"
                    )
            if calib is None:
                print("[calibration] running 3-pose anthropometric calibration…")
                calib = run_calibration(
                    cap, pose,
                    arm_length_robot=sim.arm_length_robot,
                )
                calib.save()
                print(
                    f"[calibration] done — "
                    f"scale={calib.scale:.3f}, "
                    f"user_arm={calib.arm_length_user:.3f} m, "
                    f"shoulder_w={calib.shoulder_width_user:.3f} m, "
                    f"torso_h={calib.torso_height_user:.3f} m"
                )
            anatomical_zero = np.asarray(calib.anatomical_zero, dtype=np.float64)

            if opts.viewer:
                viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            if opts.record_path:
                renderer = sim.make_renderer(config.FRAME_W, config.FRAME_H)
                recorder = SplitScreenRecorder(
                    opts.record_path,
                    config.FRAME_W, config.FRAME_H,
                    float(config.TARGET_FPS),
                )

        # ── Main loop ────────────────────────────────────────────────────────
        t0        = time.time()
        next_t    = t0
        frame_idx = 0
        last_targets = (
            sim.data.ctrl[sim.actuator_ids].copy() if sim is not None
            else np.zeros(len(config.ARM_JOINT_NAMES))
        )
        _targets_valid = False
        _t_prev = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed; exiting.", file=sys.stderr)
                break

            _t_now = time.time()
            dt = max(0.005, min(_t_now - _t_prev, 0.5))
            _t_prev = _t_now

            res: PoseResult = pose.process(frame)

            if not opts.pose_only and sim is not None:
                if res.world_landmarks is not None and _arms_visible(
                    res.visibility, config.VISIBILITY_THRESHOLD,
                ):
                    lms_smooth = lm_filter.update(res.world_landmarks, dt)
                    last_targets = compute_arm_angles(
                        lms_smooth, anatomical_zero=anatomical_zero,
                    )
                    _targets_valid = True
                    if opts.angle_debug and frame_idx % 30 == 0:
                        print("targets:", " ".join(f"{v:+.2f}" for v in last_targets))

                n_substeps = max(1, round(dt / sim.dt))
                sim.step_smooth(
                    n_substeps=n_substeps,
                    ik_target=last_targets if _targets_valid else None,
                )

            fps = (frame_idx + 1) / max(1e-6, time.time() - t0)
            shown = _annotate(
                res.annotated,
                _hud_text(last_targets),
                fps,
                opts.mirror_display,
                calib,
            )

            if recorder is not None and renderer is not None and sim is not None:
                renderer.update_scene(sim.data)
                pixels = renderer.render()
                recorder.write(shown, pixels, right_is_rgb=True)

            if viewer is not None:
                viewer.sync()

            cv2.imshow("Real2Sim", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1
            if opts.duration > 0 and time.time() - t0 >= opts.duration:
                break

            # Wall-clock pacing.
            next_t += 1.0 / config.TARGET_FPS
            now = time.time()
            if next_t > now:
                time.sleep(next_t - now)
            else:
                next_t = now

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()

    return 0
