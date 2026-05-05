"""Integration tests for the mink IK retargeter.

These tests load the real G1 MuJoCo model (requires mujoco_menagerie) and
verify that the IK produces plausible joint angles for several canonical poses.

Skipped automatically if the menagerie MJCF is not present (CI without assets).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real2sim import config

# Skip the whole module if the MJCF or mink is unavailable.
pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="mujoco_menagerie not found; skipping IK tests",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sim_and_ik():
    pytest.importorskip("mink")
    from real2sim.sim import G1Sim
    from real2sim.ik import ArmIK
    from real2sim.calibration import BodyCalibration

    sim = G1Sim()
    ik  = ArmIK(sim)
    calib = BodyCalibration(
        arm_length_user=0.60,
        arm_length_robot=sim.arm_length_robot,
        scale=sim.arm_length_robot / 0.60,
    )
    return sim, ik, calib


def _make_lms(
    L_sh, R_sh, L_hip, R_hip, L_el, R_el, L_wr, R_wr
) -> np.ndarray:
    lms = np.zeros((33, 3), dtype=np.float32)
    lms[config.LM_LEFT_SHOULDER]  = L_sh
    lms[config.LM_RIGHT_SHOULDER] = R_sh
    lms[config.LM_LEFT_HIP]       = L_hip
    lms[config.LM_RIGHT_HIP]      = R_hip
    lms[config.LM_LEFT_ELBOW]     = L_el
    lms[config.LM_RIGHT_ELBOW]    = R_el
    lms[config.LM_LEFT_WRIST]     = L_wr
    lms[config.LM_RIGHT_WRIST]    = R_wr
    return lms


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ik_returns_14_targets(sim_and_ik):
    """step() should always return a 14-element vector."""
    sim, ik, calib = sim_and_ik
    arm = 0.30
    lms = _make_lms(
        L_sh=[ 0.1, 0, 0], R_sh=[-0.1, 0, 0],
        L_hip=[0.05, 0, -0.5], R_hip=[-0.05, 0, -0.5],
        L_el=[ 0.1, 0, -arm], R_el=[-0.1, 0, -arm],
        L_wr=[ 0.1, 0, -2*arm], R_wr=[-0.1, 0, -2*arm],
    )
    sim.step(n_substeps=1)
    targets = ik.step(lms.astype(np.float32), calib)
    assert targets.shape == (len(config.ARM_JOINT_NAMES),)
    assert not np.any(np.isnan(targets)), "IK returned NaN"


def test_ik_joint_limits(sim_and_ik):
    """IK must never produce joint angles outside the model's allowed range."""
    import mujoco
    sim, ik, calib = sim_and_ik
    # Extreme pose: arms stretched forward.
    arm = 0.30
    lms = _make_lms(
        L_sh=[ 0.1, 0, 0], R_sh=[-0.1, 0, 0],
        L_hip=[0.05, 0, -0.5], R_hip=[-0.05, 0, -0.5],
        L_el=[ 0.1, arm, 0], R_el=[-0.1, arm, 0],
        L_wr=[ 0.1, 2*arm, 0], R_wr=[-0.1, 2*arm, 0],
    )
    targets = ik.step(lms.astype(np.float32), calib)

    for i, jname in enumerate(config.ARM_JOINT_NAMES):
        jid  = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = sim.model.jnt_range[jid]
        assert lo - 1e-3 <= targets[i] <= hi + 1e-3, (
            f"Joint {jname}: target {targets[i]:.3f} outside [{lo:.3f}, {hi:.3f}]"
        )


def test_ik_palm_site_error(sim_and_ik):
    """IK algorithm should converge palm close to target (pure IK, no physics).

    We test the IK algorithm in isolation by running the differential IK
    iteration directly on the fixed-base configuration (bypassing physics).
    This avoids the gravity-compliance offset of the position PD actuators.
    """
    import mujoco as _mujoco
    from real2sim.sim import _load_patched_model
    sim, ik, calib = sim_and_ik

    # T-pose: arms straight out laterally.
    arm = 0.30
    lms = _make_lms(
        L_sh=[ 0.1, 0, 0], R_sh=[-0.1, 0, 0],
        L_hip=[0.05, 0, -0.5], R_hip=[-0.05, 0, -0.5],
        L_el=[ 0.1+arm, 0, 0], R_el=[-0.1-arm, 0, 0],
        L_wr=[ 0.1+2*arm, 0, 0], R_wr=[-0.1-2*arm, 0, 0],
    )

    # Build a fresh fixed-base IK model and run the IK directly.
    mink = pytest.importorskip("mink")
    ik_model = _load_patched_model(config.MODEL_PATH, fix_base=True)
    ik_data  = _mujoco.MjData(ik_model)
    q0 = sim.data.qpos[7:].copy()  # keyframe arm+leg+waist state
    ik_data.qpos[:] = q0
    _mujoco.mj_forward(ik_model, ik_data)

    cfg = mink.Configuration(ik_model)
    cfg.update(q0)

    # Replicate the IK task setup from ArmIK.
    from real2sim.ik import build_body_frame
    left_sh  = ik_data.body("left_shoulder_pitch_link").xpos.copy()
    right_sh = ik_data.body("right_shoulder_pitch_link").xpos.copy()
    torso_xmat = ik_data.body("torso_link").xmat.reshape(3, 3)
    R_mp_to_world = np.column_stack([
        torso_xmat[:, 1], -torso_xmat[:, 0], torso_xmat[:, 2]
    ])
    R_body = build_body_frame(lms.astype(np.float64))

    def _target(mp_sh, mp_wr, sh_xpos):
        rel = R_body.T @ (mp_wr - mp_sh)
        return sh_xpos + R_mp_to_world @ (rel * calib.scale)

    tl = _target(lms[config.LM_LEFT_SHOULDER],  lms[config.LM_LEFT_WRIST],  left_sh)
    tr = _target(lms[config.LM_RIGHT_SHOULDER], lms[config.LM_RIGHT_WRIST], right_sh)

    lt = mink.FrameTask(config.IK_SITE_LEFT,  "site", position_cost=1.0, orientation_cost=0.0)
    rt = mink.FrameTask(config.IK_SITE_RIGHT, "site", position_cost=1.0, orientation_cost=0.0)
    pt = mink.PostureTask(ik_model, cost=config.IK_POSTURE_COST)
    pt.set_target_from_configuration(cfg)
    limits = [mink.ConfigurationLimit(ik_model)]
    lt.set_target(mink.SE3.from_translation(tl))
    rt.set_target(mink.SE3.from_translation(tr))

    # Iterate pure differential IK (no physics).
    for _ in range(500):
        vel = mink.solve_ik(cfg, [lt, rt, pt], dt=config.IK_DT,
                            solver=config.IK_SOLVER, limits=limits)
        cfg.integrate_inplace(vel, config.IK_DT)

    ik_data.qpos[:] = cfg.q
    _mujoco.mj_forward(ik_model, ik_data)

    actual = ik_data.site("left_palm").xpos
    err = float(np.linalg.norm(actual - tl))
    # Expect < 1 cm error (differential IK converges to sub-mm for reachable targets).
    assert err < 0.01, f"Pure IK palm error too large: {err*100:.2f} cm"
