"""mink-based differential IK retargeter.

Each frame:
  1. Read filtered MediaPipe world landmarks.
  2. Express wrist vectors relative to shoulders in the person's body frame.
  3. Scale by arm-length ratio (calibration).
  4. Map to the robot's MuJoCo world frame.
  5. Feed the 3-D targets to mink FrameTasks for left_palm / right_palm.
  6. Solve the QP (daqp solver, CPU) using a FIXED-BASE model so the
     optimizer cannot cheat by translating the whole robot body.
  7. Integrate qvel → qpos inside the mink Configuration.
  8. Write the resulting arm joint angles back to data.ctrl so that the
     position-PD actuators track them through physics.

Fixed-base model
----------------
The full G1 has a floating base (freejoint, 7 qpos / 6 qvel DOFs).  mink's
QP naturally satisfies end-effector tasks by translating the root — arm joints
get near-zero velocities.  The fix: load a second copy of the MJCF with the
freejoint stripped out.  The IK then has only 29 DOFs (legs + waist + arms),
and the QP must move arm joints to reach the palm targets.

Arm qpos addresses in the fixed-base model equal the physics-model addresses
minus 7 (the 7 qpos slots consumed by the freejoint).

Coordinate system notes
-----------------------
MediaPipe world_landmarks body frame:
    e_x = subject's left  (L_shoulder − R_shoulder)
    e_y = subject's forward  (cross(e_z, e_x))
    e_z = up along spine

G1 MuJoCo world frame (standing, torso pinned):
    +X = robot forward
    +Y = robot left
    +Z = up

Mapping: mp_x → robot Y,  mp_y → robot X,  mp_z → robot Z.
"""

from __future__ import annotations

import numpy as np
import mujoco

try:
    import mink
except ImportError as e:
    raise ImportError(
        "mink is required: pip install mink daqp\n"
        f"Original error: {e}"
    ) from e

from . import config
from .calibration import BodyCalibration
from .filter import limit_delta
from .sim import G1Sim, _load_patched_model

EPS = 1e-6

# How many dof the freejoint contributes to qpos (7) vs qvel (6).
_FREE_QPOS = 7


# ── Body-frame helper ─────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros_like(v)


def build_body_frame(lms: np.ndarray) -> np.ndarray:
    """Return 3×3 matrix R whose columns are [e_x, e_y, e_z] in world coords.

    v_body = R.T @ v_world
    """
    L_sh  = lms[config.LM_LEFT_SHOULDER]
    R_sh  = lms[config.LM_RIGHT_SHOULDER]
    L_hip = lms[config.LM_LEFT_HIP]
    R_hip = lms[config.LM_RIGHT_HIP]

    e_x = _normalize(L_sh - R_sh)
    e_z = _normalize(0.5 * (L_sh + R_sh) - 0.5 * (L_hip + R_hip))
    e_y = _normalize(np.cross(e_z, e_x))
    e_x = _normalize(np.cross(e_y, e_z))
    return np.column_stack([e_x, e_y, e_z]).astype(np.float64)


# ── IK class ─────────────────────────────────────────────────────────────────

class ArmIK:
    """Wraps a mink Configuration (fixed-base G1) and two FrameTasks."""

    def __init__(self, sim: G1Sim) -> None:
        self._model = sim.model   # full physics model
        self._data  = sim.data

        # ── Fixed-base IK model (no freejoint → base welded to world) ────────
        _ik_model = _load_patched_model(config.MODEL_PATH, fix_base=True)
        _ik_data  = mujoco.MjData(_ik_model)
        # Initialise arm joints from the physics keyframe pose.
        # physics qpos[7:] maps 1-to-1 to fixed-base qpos[0:].
        q_init = _ik_data.qpos.copy()
        q_init[:] = sim.data.qpos[_FREE_QPOS:]
        _ik_data.qpos[:] = q_init
        mujoco.mj_forward(_ik_model, _ik_data)

        # mink Configuration on the fixed-base model.
        self._cfg = mink.Configuration(_ik_model)
        self._cfg.update(q_init)

        # ── Tasks ─────────────────────────────────────────────────────────────
        self._left_task = mink.FrameTask(
            frame_name=config.IK_SITE_LEFT,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        self._right_task = mink.FrameTask(
            frame_name=config.IK_SITE_RIGHT,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        self._posture_task = mink.PostureTask(_ik_model, cost=config.IK_POSTURE_COST)
        self._posture_task.set_target_from_configuration(self._cfg)

        self._limits = [mink.ConfigurationLimit(_ik_model)]
        self._tasks  = [self._left_task, self._right_task, self._posture_task]

        # ── Elbow FrameTasks (added dynamically in step() based on visibility) ─
        self._left_elbow_task = mink.FrameTask(
            frame_name="left_elbow_link",
            frame_type="body",
            position_cost=0.5,
            orientation_cost=0.0,
        )
        self._right_elbow_task = mink.FrameTask(
            frame_name="right_elbow_link",
            frame_type="body",
            position_cost=0.5,
            orientation_cost=0.0,
        )

        # Rate-limiter state: prevents sudden joint-angle jumps when MediaPipe
        # briefly loses tracking (e.g. hands near face).
        self._prev_ctrl: np.ndarray | None = None
        self._max_delta = 0.12  # rad/frame max change (~3.6 rad/s at 30 fps)

        # Debug info populated every step(), readable by GUI/logging code.
        self.last_debug: dict = {}

        # ── Fixed geometry from IK model (consistent with IK frame) ──────────
        self._left_sh_xpos  = _ik_data.body("left_shoulder_pitch_link").xpos.copy()
        self._right_sh_xpos = _ik_data.body("right_shoulder_pitch_link").xpos.copy()

        torso_xmat = _ik_data.body("torso_link").xmat.reshape(3, 3)
        self._R_mp_to_world = np.column_stack([
            torso_xmat[:, 1],   # mp_x (person left)    → robot left    (+Y)
            -torso_xmat[:, 0],  # mp_y (person forward)  → robot forward (+X)  [negated: MediaPipe body-frame e_y points toward camera, so rel_body[1] is negative for forward motion]
            torso_xmat[:, 2],   # mp_z (person up)        → robot up     (+Z)
        ])

        # ── Actuator / qpos address mapping ──────────────────────────────────
        self._act_ids  = sim.actuator_ids         # (14,) actuator IDs in full model
        # Arm qpos addresses in fixed-base model = physics addrs - FREE_QPOS.
        self._qpos_adr = sim.arm_qpos_addrs - _FREE_QPOS   # (14,)
        # Full physics qpos addresses (for syncing leg/waist joints into IK cfg).
        self._phys_qpos_slice = slice(_FREE_QPOS, None)    # qpos[7:]

    # ── Per-frame step ───────────────────────────────────────────────────────

    def step(
        self,
        lms_filtered: np.ndarray,          # (33, 3) One-Euro filtered world landmarks
        calib: BodyCalibration,
        dt: float = config.IK_DT,
        visibility: np.ndarray | None = None,  # (33,) MediaPipe visibility scores
    ) -> np.ndarray:
        """Solve IK for one timestep.

        Sets data.ctrl for all arm actuators and returns the 14-D target
        joint-position vector (for HUD display / debug).
        """
        R_body = build_body_frame(lms_filtered)

        left_target  = self._wrist_target(
            lms_filtered[config.LM_LEFT_SHOULDER],
            lms_filtered[config.LM_LEFT_WRIST],
            R_body, calib, self._left_sh_xpos,
        )
        right_target = self._wrist_target(
            lms_filtered[config.LM_RIGHT_SHOULDER],
            lms_filtered[config.LM_RIGHT_WRIST],
            R_body, calib, self._right_sh_xpos,
        )

        self._left_task.set_target(mink.SE3.from_translation(left_target))
        self._right_task.set_target(mink.SE3.from_translation(right_target))

        # Debug info: rel_body = wrist position in person body-frame (meters).
        # In B-mode the Y component (forward/back) should be ~0.
        # L/R_tgt_x = robot forward position of the wrist target (constant in B-mode).
        self.last_debug = {
            "L_body": (R_body.T @ (lms_filtered[config.LM_LEFT_WRIST]  - lms_filtered[config.LM_LEFT_SHOULDER])).tolist(),
            "R_body": (R_body.T @ (lms_filtered[config.LM_RIGHT_WRIST] - lms_filtered[config.LM_RIGHT_SHOULDER])).tolist(),
            "L_tgt_x": float(left_target[0]),
            "R_tgt_x": float(right_target[0]),
            "L_tgt":   left_target.tolist(),
            "R_tgt":   right_target.tolist(),
        }

        # ── Elbow tasks: only when elbow is visible ───────────────────────────
        tasks = list(self._tasks)
        thr = config.VISIBILITY_THRESHOLD
        if visibility is not None:
            if visibility[config.LM_LEFT_ELBOW] >= thr:
                el_tgt = self._wrist_target(
                    lms_filtered[config.LM_LEFT_SHOULDER],
                    lms_filtered[config.LM_LEFT_ELBOW],
                    R_body, calib, self._left_sh_xpos,
                )
                self._left_elbow_task.set_target(mink.SE3.from_translation(el_tgt))
                tasks.append(self._left_elbow_task)
            if visibility[config.LM_RIGHT_ELBOW] >= thr:
                el_tgt = self._wrist_target(
                    lms_filtered[config.LM_RIGHT_SHOULDER],
                    lms_filtered[config.LM_RIGHT_ELBOW],
                    R_body, calib, self._right_sh_xpos,
                )
                self._right_elbow_task.set_target(mink.SE3.from_translation(el_tgt))
                tasks.append(self._right_elbow_task)

        # Sync ALL joints (leg + waist + arm) from physics into fixed-base cfg.
        # physics qpos[7:] maps 1-to-1 to fixed-base qpos[:].
        self._cfg.update(self._data.qpos[self._phys_qpos_slice].copy())

        try:
            vel = mink.solve_ik(
                self._cfg,
                tasks,
                dt=dt,
                solver=config.IK_SOLVER,
                limits=self._limits,
            )
        except Exception as _ik_exc:
            import traceback as _tb
            print(f"[IK] solve_ik failed: {_ik_exc}\n{_tb.format_exc()}", flush=True)
            return self._data.ctrl[self._act_ids].copy()

        self._cfg.integrate_inplace(vel, dt)

        # Write solved arm qpos as ctrl targets for the PD actuators.
        targets = self._cfg.q[self._qpos_adr].copy()

        # Rate-limit: clamp per-joint delta to avoid sudden jumps when
        # MediaPipe re-acquires landmarks after brief occlusion.
        if self._prev_ctrl is not None:
            targets = limit_delta(self._prev_ctrl, targets, self._max_delta)
        self._prev_ctrl = targets.copy()

        for i, aid in enumerate(self._act_ids):
            self._data.ctrl[aid] = float(targets[i])

        return targets

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _wrist_target(
        self,
        mp_shoulder: np.ndarray,
        mp_wrist: np.ndarray,
        R_body: np.ndarray,
        calib: BodyCalibration,
        robot_sh_xpos: np.ndarray,
    ) -> np.ndarray:
        """3-D palm target position in MuJoCo world frame."""
        rel_world  = mp_wrist - mp_shoulder
        rel_body   = R_body.T @ rel_world
        rel_scaled = rel_body * calib.scale
        rel_robot  = self._R_mp_to_world @ rel_scaled
        return robot_sh_xpos + rel_robot
