"""MuJoCo wrapper around the Unitree G1 model.

Responsibilities:
* Load scene.xml and reset to the "stand" keyframe.
* Resolve actuator IDs for the 8 joints we drive.
* Step the simulation while pinning the floating base (so the robot stays
  upright while only the arms move).
* Expose joint ranges so the caller can clip target angles.
* Provide an offscreen renderer for split-screen mp4 recording.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import config


class G1Sim:
    def __init__(self, model_path: str | Path | None = None, keyframe: str = config.KEYFRAME_NAME):
        path = Path(model_path or config.MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"MJCF not found: {path}")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        # Reset to standing keyframe if available, then snapshot torso pose.
        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if kf_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
        mujoco.mj_forward(self.model, self.data)
        self._root_qpos = self.data.qpos[:7].copy()

        # Map joint names -> actuator id, in the canonical config.JOINT_NAMES order.
        self.actuator_ids = np.zeros(len(config.JOINT_NAMES), dtype=np.int32)
        for i, name in enumerate(config.JOINT_NAMES):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Actuator '{name}' not found in model.")
            self.actuator_ids[i] = aid

        # Joint ranges for clipping target angles.
        self.joint_ranges = np.zeros((len(config.JOINT_NAMES), 2), dtype=np.float64)
        for i, name in enumerate(config.JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint '{name}' not found in model.")
            self.joint_ranges[i] = self.model.jnt_range[jid]

        self.dt = float(self.model.opt.timestep)

    def set_arm_angles(self, angles: np.ndarray) -> None:
        if angles.shape != (len(self.actuator_ids),):
            raise ValueError(f"expected ({len(self.actuator_ids)},), got {angles.shape}")
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = float(angles[i])

    def step(self, n_substeps: int = 1) -> None:
        # Pin the floating base so the robot can't tip over while we drive only arms.
        for _ in range(max(1, n_substeps)):
            self.data.qpos[:7] = self._root_qpos
            self.data.qvel[:6] = 0.0
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[:7] = self._root_qpos
        self.data.qvel[:6] = 0.0

    def substeps_for_fps(self, fps: float) -> int:
        return max(1, int(round((1.0 / fps) / self.dt)))

    def make_renderer(self, width: int = config.FRAME_W, height: int = config.FRAME_H) -> mujoco.Renderer:
        return mujoco.Renderer(self.model, height=height, width=width)
