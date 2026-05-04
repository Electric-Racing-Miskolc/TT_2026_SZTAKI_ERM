"""MuJoCo wrapper for the Unitree G1.

Key responsibilities:
* Patch the menagerie MJCF at runtime to inject left_palm / right_palm sites
  (the submodule is never modified on disk).
* Reset to the "stand" keyframe and snapshot the torso pose for pinning.
* Expose arm_length_robot (shoulder → palm at rest) for calibration.
* step(): advance physics while keeping the floating base frozen.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import config

# Exact strings in g1.xml that we inject sites after.
_L_ANCHOR = '<geom pos="0.0415 0.003 0" quat="1 0 0 0" class="visual" mesh="left_rubber_hand"/>'
_R_ANCHOR = '<geom pos="0.0415 -0.003 0" quat="1 0 0 0" class="visual" mesh="right_rubber_hand"/>'

_L_SITE = '<site name="left_palm"  pos="0.06 0 0"/>'
_R_SITE = '<site name="right_palm" pos="0.06 0 0"/>'


def _load_patched_model(
    scene_path: Path,
    fix_base: bool = False,
) -> mujoco.MjModel:
    """Return MjModel with left_palm / right_palm sites injected at runtime.

    If fix_base=True the freejoint is removed so the pelvis is welded to the
    world — useful for arm-only IK (prevents the QP from moving the base).

    Writes two temp files next to the originals so that meshdir and include
    paths resolve correctly, then deletes them immediately after loading.
    """
    menagerie_dir = scene_path.parent
    g1_path = menagerie_dir / "g1.xml"

    g1_xml = g1_path.read_text(encoding="utf-8")

    if "left_palm" not in g1_xml:
        g1_xml = g1_xml.replace(
            _L_ANCHOR,
            _L_ANCHOR + "\n                          " + _L_SITE,
        )
    if "right_palm" not in g1_xml:
        g1_xml = g1_xml.replace(
            _R_ANCHOR,
            _R_ANCHOR + "\n                          " + _R_SITE,
        )

    if fix_base:
        # Remove freejoint so pelvis is welded to world (nq drops from 36→29).
        # Also strip keyframe qpos/ctrl (36 values → wrong size without freejoint).
        g1_xml = g1_xml.replace(
            "      <freejoint name=\"floating_base_joint\"/>\n", ""
        )
        # Remove keyframe block entirely to avoid qpos size mismatch.
        import re
        g1_xml = re.sub(r"<keyframe>.*?</keyframe>", "", g1_xml, flags=re.DOTALL)

    suffix = "_fb" if fix_base else ""
    tmp_g1    = menagerie_dir / f"_g1_patched{suffix}_tmp.xml"
    tmp_scene = menagerie_dir / f"_scene_patched{suffix}_tmp.xml"

    scene_xml = scene_path.read_text(encoding="utf-8").replace(
        '<include file="g1.xml"/>',
        f'<include file="_g1_patched{suffix}_tmp.xml"/>',
    )

    try:
        tmp_g1.write_text(g1_xml, encoding="utf-8")
        tmp_scene.write_text(scene_xml, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(tmp_scene))
    finally:
        tmp_g1.unlink(missing_ok=True)
        tmp_scene.unlink(missing_ok=True)

    return model


class G1Sim:
    def __init__(
        self,
        model_path: str | Path | None = None,
        keyframe: str = config.KEYFRAME_NAME,
    ) -> None:
        path = Path(model_path or config.MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"MJCF scene not found: {path}")

        self.model = _load_patched_model(path)
        self.data  = mujoco.MjData(self.model)

        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if kf_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
        mujoco.mj_forward(self.model, self.data)

        # Snapshot floating-base pose for pinning.
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


        # qpos address for each arm joint (needed by IK to read back solved pose).
        self.arm_qpos_addrs = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in config.ARM_JOINT_NAMES
        ], dtype=np.int32)

        # Robot arm length for calibration (shoulder pitch link → palm site).
        left_sh  = self.data.body("left_shoulder_pitch_link").xpos
        left_palm = self.data.site("left_palm").xpos
        self.arm_length_robot = float(np.linalg.norm(left_palm - left_sh))

        self.dt = float(self.model.opt.timestep)

    # ── Simulation step ──────────────────────────────────────────────────────

    def step(self, n_substeps: int = 1) -> None:
        """Advance physics while keeping the floating base frozen."""
        for _ in range(max(1, n_substeps)):
            self.data.qpos[:7] = self._root_qpos
            self.data.qvel[:6] = 0.0
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[:7] = self._root_qpos
        self.data.qvel[:6] = 0.0

    def substeps_for_fps(self, fps: float) -> int:
        return max(1, int(round((1.0 / fps) / self.dt)))

    # ── Renderer ─────────────────────────────────────────────────────────────

    def make_renderer(
        self,
        width: int  = config.FRAME_W,
        height: int = config.FRAME_H,
    ) -> mujoco.Renderer:
        return mujoco.Renderer(self.model, height=height, width=width)
