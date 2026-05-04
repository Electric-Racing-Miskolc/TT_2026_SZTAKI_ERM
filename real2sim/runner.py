"""Main loop: webcam → MediaPipe → One-Euro filter → mink IK → MuJoCo.

Modes
-----
--pose-only     webcam + skeleton overlay, no simulation.
--viewer        live MuJoCo passive viewer + camera preview window.
--record PATH   offscreen split-screen mp4 (camera | sim).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from . import config
from .calibration import BodyCalibration, run_calibration
from .ik import ArmIK
from .one_euro import OneEuroFilterArray
from .pose import PoseEstimator, PoseResult
from .recorder import SplitScreenRecorder
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
    ik_debug: bool        = False


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
    fmt = lambda v: " ".join(f"{x:+.2f}" for x in v)
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


# ── Main run function ─────────────────────────────────────────────────────────

def run(opts: RunOptions) -> int:  # noqa: C901
    cap  = _open_camera(opts.camera)
    pose = PoseEstimator()

    sim: G1Sim | None = None
    arm_ik: ArmIK | None = None
    calib: BodyCalibration | None = None
    viewer   = None
    renderer: mujoco.Renderer | None = None
    recorder: SplitScreenRecorder | None = None

    # One-Euro filter for raw MediaPipe world landmarks (33 pts × 3 coords).
    lm_filter = OneEuroFilterArray(shape=(33, 3))

    try:
        if not opts.pose_only:
            sim = G1Sim()
            n_substeps = sim.substeps_for_fps(config.TARGET_FPS)

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
                print("[calibration] running T-pose calibration…")
                calib = run_calibration(
                    cap, pose,
                    arm_length_robot=sim.arm_length_robot,
                )
                calib.save()
                print(
                    f"[calibration] done — "
                    f"scale={calib.scale:.3f}, "
                    f"user_arm={calib.arm_length_user:.3f} m, "
                    f"robot_arm={calib.arm_length_robot:.3f} m"
                )

            arm_ik = ArmIK(sim)

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
        last_targets = np.zeros(len(config.ARM_JOINT_NAMES))
        dt = 1.0 / config.TARGET_FPS

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed; exiting.", file=sys.stderr)
                break

            res: PoseResult = pose.process(frame)

            if not opts.pose_only and sim is not None and arm_ik is not None and calib is not None:
                if res.world_landmarks is not None:
                    # Check that required landmarks are visible.
                    vis = res.visibility
                    arms_visible = (
                        vis is not None
                        and all(
                            vis[i] >= config.VISIBILITY_THRESHOLD
                            for i in (
                                config.LM_LEFT_SHOULDER,  config.LM_RIGHT_SHOULDER,
                                config.LM_LEFT_WRIST,     config.LM_RIGHT_WRIST,
                            )
                        )
                    )
                    if arms_visible:
                        lms_smooth = lm_filter.update(res.world_landmarks, dt)
                        targets = arm_ik.step(lms_smooth, calib, dt=dt)
                        last_targets = targets
                        if opts.ik_debug and frame_idx % 30 == 0:
                            print("IK targets:", " ".join(f"{v:+.2f}" for v in targets))

                sim.step(n_substeps=n_substeps)

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

            if not opts.record_path or opts.viewer:
                cv2.imshow("Real2Sim", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                cv2.imshow("Real2Sim (recording)", shown)
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
