"""Main loop tying webcam -> pose -> retarget -> filter -> sim -> render.

Modes:
    --pose-only            no MuJoCo, just shows the camera + skeleton
    --viewer               live MuJoCo viewer + camera window
    --record OUT [--duration N]  offscreen render + mp4 split-screen
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
from .filter import ExpSmoother, LandmarkSmoother, limit_delta
from .pose import PoseEstimator, PoseResult
from .recorder import SplitScreenRecorder
from .retarget import clip_to_joint_ranges, compute_angles
from .sim import G1Sim


@dataclass
class RunOptions:
    camera: int = config.CAMERA_INDEX
    pose_only: bool = False
    viewer: bool = False
    record_path: Path | None = None
    duration: float = 0.0
    mirror_display: bool = True
    debug_print: bool = False


def _open_camera(idx: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {idx}")
    return cap


def _hud_text(angles: np.ndarray) -> list[str]:
    return [
        f"L pit={angles[0]:+.2f} rol={angles[1]:+.2f} yaw={angles[2]:+.2f} elb={angles[3]:+.2f}",
        f"R pit={angles[4]:+.2f} rol={angles[5]:+.2f} yaw={angles[6]:+.2f} elb={angles[7]:+.2f}",
    ]


def _annotate(frame: np.ndarray, lines: list[str], fps: float, mirror: bool) -> np.ndarray:
    shown = cv2.flip(frame, 1) if mirror else frame.copy()
    cv2.putText(shown, f"fps={fps:5.1f}  q=quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    y = 50
    for ln in lines:
        cv2.putText(shown, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return shown


def run(opts: RunOptions) -> int:
    cap = _open_camera(opts.camera)
    pose = PoseEstimator()
    sim: G1Sim | None = None
    lm_smoother = LandmarkSmoother(alpha=config.LANDMARK_SMOOTH_ALPHA)
    smoother = ExpSmoother(dim=8, alpha=config.SMOOTHING_ALPHA)
    viewer = None
    renderer: mujoco.Renderer | None = None
    recorder: SplitScreenRecorder | None = None

    try:
        if not opts.pose_only:
            sim = G1Sim()
            n_substeps = sim.substeps_for_fps(config.TARGET_FPS)
            if opts.viewer:
                viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            if opts.record_path:
                renderer = sim.make_renderer(config.FRAME_W, config.FRAME_H)
                recorder = SplitScreenRecorder(
                    opts.record_path, config.FRAME_W, config.FRAME_H, float(config.TARGET_FPS)
                )

        t0 = time.time()
        next_t = t0
        frame_idx = 0
        last_angles = np.zeros(8)
        prev_smoothed = np.zeros(8)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed; exiting.", file=sys.stderr)
                break

            res: PoseResult = pose.process(frame)

            if not opts.pose_only and sim is not None:
                if res.world_landmarks is not None:
                    # 1. Pre-smooth landmark positions to stabilise the body frame.
                    smooth_lm = lm_smoother.update(res.world_landmarks)
                    # 2. Compute raw joint angles from smoothed landmarks.
                    raw, valid = compute_angles(smooth_lm, res.visibility)
                    # 3. EMA smooth the angles themselves.
                    smoothed = smoother.update(raw, valid)
                    # 4. Velocity-clamp: prevent per-frame jumps > MAX_ANGLE_DELTA_RAD.
                    vel_clamped = limit_delta(prev_smoothed, smoothed, config.MAX_ANGLE_DELTA_RAD)
                    prev_smoothed = vel_clamped.copy()
                    clipped = clip_to_joint_ranges(vel_clamped, sim.joint_ranges)
                    sim.set_arm_angles(clipped)
                    last_angles = clipped
                    if opts.debug_print and frame_idx % 30 == 0:
                        print("angles:", " ".join(f"{v:+.2f}" for v in clipped))
                sim.step(n_substeps=n_substeps)

            fps = (frame_idx + 1) / max(1e-6, time.time() - t0)
            shown = _annotate(res.annotated, _hud_text(last_angles), fps, opts.mirror_display)

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
                next_t = now  # fell behind; reset

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()

    return 0
