"""End-to-end smoke test WITHOUT a camera.

Generates a synthetic landmark stream (arms drawing arcs in front of the
torso), feeds it through retarget -> filter -> sim, renders frames offscreen,
and writes a split-screen mp4 alongside a fake "camera" panel.

Verifies: G1Sim, retarget integration, ExpSmoother, mujoco.Renderer offscreen,
and SplitScreenRecorder all wired up.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real2sim import config  # noqa: E402
from real2sim.filter import ExpSmoother  # noqa: E402
from real2sim.recorder import SplitScreenRecorder  # noqa: E402
from real2sim.retarget import clip_to_joint_ranges, compute_angles  # noqa: E402
from real2sim.sim import G1Sim  # noqa: E402


def synth_landmarks(t: float) -> np.ndarray:
    """Arms rotate slowly: shoulders pitch from 0 -> ~pi/2 over 2 s."""
    L_sh = np.array([+0.18, 0.0, 0.0], dtype=np.float32)
    R_sh = np.array([-0.18, 0.0, 0.0], dtype=np.float32)
    L_hip = np.array([+0.09, 0.0, -0.50], dtype=np.float32)
    R_hip = np.array([-0.09, 0.0, -0.50], dtype=np.float32)

    a = 0.5 * math.sin(t * 1.2) + 0.5      # 0..1
    pitch = a * math.pi / 2                # 0..pi/2 forward swing
    upper, lower = 0.30, 0.27
    # upper arm direction in body frame for forward pitch:
    #   d = (0, sin(pitch), -cos(pitch))   (rises forward as pitch grows)
    dx, dy, dz = 0.0, math.sin(pitch), -math.cos(pitch)

    L_el = L_sh + np.array([dx, dy, dz]) * upper
    R_el = R_sh + np.array([dx, dy, dz]) * upper
    # forearm continues straight (no elbow bend)
    L_wr = L_el + np.array([dx, dy, dz]) * lower
    R_wr = R_el + np.array([dx, dy, dz]) * lower

    arr = np.zeros((33, 3), dtype=np.float32)
    arr[config.LM_LEFT_SHOULDER] = L_sh
    arr[config.LM_RIGHT_SHOULDER] = R_sh
    arr[config.LM_LEFT_HIP] = L_hip
    arr[config.LM_RIGHT_HIP] = R_hip
    arr[config.LM_LEFT_ELBOW] = L_el
    arr[config.LM_RIGHT_ELBOW] = R_el
    arr[config.LM_LEFT_WRIST] = L_wr
    arr[config.LM_RIGHT_WRIST] = R_wr
    return arr


def fake_camera_frame(angles: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    img = np.full((frame_h, frame_w, 3), 30, dtype=np.uint8)
    cv2.putText(img, "synthetic camera", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"L pit={angles[0]:+.2f} elb={angles[3]:+.2f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(img, f"R pit={angles[4]:+.2f} elb={angles[7]:+.2f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img


def main() -> int:
    sim = G1Sim()
    smoother = ExpSmoother(dim=8, alpha=config.SMOOTHING_ALPHA)
    n_substeps = sim.substeps_for_fps(config.TARGET_FPS)
    renderer = sim.make_renderer(config.FRAME_W, config.FRAME_H)

    out_path = Path(__file__).parent / "smoke_pipeline_synth.mp4"
    duration_s = 5.0
    n_frames = int(duration_s * config.TARGET_FPS)

    last = np.zeros(8)
    t_start = time.time()
    with SplitScreenRecorder(out_path, config.FRAME_W, config.FRAME_H, float(config.TARGET_FPS)) as rec:
        for i in range(n_frames):
            t = i / config.TARGET_FPS
            lms = synth_landmarks(t)
            raw, valid = compute_angles(lms)
            smoothed = smoother.update(raw, valid)
            clipped = clip_to_joint_ranges(smoothed, sim.joint_ranges)
            sim.set_arm_angles(clipped)
            sim.step(n_substeps=n_substeps)

            renderer.update_scene(sim.data)
            pixels = renderer.render()  # RGB
            cam = fake_camera_frame(clipped, config.FRAME_W, config.FRAME_H)
            rec.write(cam, pixels, right_is_rgb=True)
            last = clipped

    wall = time.time() - t_start
    print(f"wrote {n_frames} frames in {wall:.2f}s -> {out_path}")
    print("final angles:", " ".join(f"{v:+.2f}" for v in last))
    print(f"file size: {out_path.stat().st_size} bytes")
    return 0 if out_path.stat().st_size > 1000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
