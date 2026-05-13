"""MuJoCo wrapper for the Unitree G1 (analytical retargeting edition).

Responsibilities:
* Load the menagerie scene + G1 model verbatim (no runtime MJCF patching).
* Reset to the "stand" keyframe and snapshot the torso pose for pinning.
* Expose `arm_length_robot` (shoulder → wrist link distance at rest) for
  the calibration scale.
* `step_smooth()` advances physics while keeping the floating base frozen,
  with a per-substep exponential blend from current ctrl to the requested
  joint targets (avoids hard ctrl jumps between retargeter updates).

Compared to the previous mink-IK version: no `fix_base` second model, no
left_palm/right_palm runtime site injection — the analytical retargeter
drives joint angles directly via the position-PD actuators.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import config


class G1Sim:
    def __init__(
        self,
        model_path: str | Path | None = None,
        keyframe: str = config.KEYFRAME_NAME,
    ) -> None:
        path = Path(model_path or config.MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"MJCF scene not found: {path}")

        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data  = mujoco.MjData(self.model)

        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if kf_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
        mujoco.mj_forward(self.model, self.data)

        # Floating-base pose snapshot (used to pin the torso each step).
        self._root_qpos = self.data.qpos[:7].copy()

        # Arm actuator IDs (order matches config.ARM_JOINT_NAMES).
        self.actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in config.ARM_JOINT_NAMES
        ], dtype=np.int32)
        missing = [
            config.ARM_JOINT_NAMES[i]
            for i, aid in enumerate(self.actuator_ids) if aid < 0
        ]
        if missing:
            raise RuntimeError(f"Actuators not found in model: {missing}")

        # Robot arm length for the calibration HUD: shoulder pitch link
        # body → wrist yaw link body at the rest pose.
        left_sh   = self.data.body("left_shoulder_pitch_link").xpos
        left_wrist = self.data.body("left_wrist_yaw_link").xpos
        self.arm_length_robot = float(np.linalg.norm(left_wrist - left_sh))

        self.dt = float(self.model.opt.timestep)

    # ── Simulation step ──────────────────────────────────────────────────────

    def step_smooth(
        self,
        n_substeps: int = 1,
        ik_target: np.ndarray | None = None,
        tau: float = 0.04,
    ) -> None:
        """Advance physics at real-time rate with per-substep ctrl smoothing.

        Exponentially blends ctrl toward `ik_target` at every physics
        timestep, instead of writing it once per camera frame.  This avoids
        the visible "twitch" each time a new retargeter sample lands.

        tau : time constant in seconds (0.04 = 40 ms).  Smaller = faster
              tracking, less smoothing.
        """
        if ik_target is not None:
            alpha = float(1.0 - np.exp(-self.dt / tau))
            tgt   = np.asarray(ik_target, dtype=np.float64)
            ids   = self.actuator_ids

        for _ in range(max(1, n_substeps)):
            if ik_target is not None:
                self.data.ctrl[ids] += alpha * (tgt - self.data.ctrl[ids])
            # Pin floating base so the robot can't translate / fall over.
            self.data.qpos[:7] = self._root_qpos
            self.data.qvel[:6] = 0.0
            mujoco.mj_step(self.model, self.data)

        self.data.qpos[:7] = self._root_qpos
        self.data.qvel[:6] = 0.0

    # ── Renderer ─────────────────────────────────────────────────────────────

    def make_renderer(
        self,
        width: int  = config.FRAME_W,
        height: int = config.FRAME_H,
    ) -> mujoco.Renderer:
        return mujoco.Renderer(self.model, height=height, width=width)
