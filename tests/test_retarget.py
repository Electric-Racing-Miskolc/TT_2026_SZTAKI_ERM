"""Unit tests for the analytical retargeter.

We hand-build synthetic landmark configurations for canonical poses and check
that the anatomical angles produced by `compute_anatomical_angles` match
expectations.  No camera, no MediaPipe — kamera-mentes, CI-kompatibilis.

World convention used by these tests (matches MediaPipe's hip-centred
world_landmarks):  +x = subject's right, +y = down, +z = forward.  We let the
body-frame builder figure out e_x/e_y/e_z; the tests only assert the joint
angles, not the intermediate frame.

Joint-angle conventions returned by compute_anatomical_angles follow G1's
joint axes directly (see retarget.ArmAngles docstring):

  +pitch (shoulder) → arm swings BACKWARD from hanging-down (around body-LEFT)
  ±π pitch          → overhead
  +roll  (left arm) → abduct AWAY from torso (toward subject-left)
  yaw               → rotation around the upper-arm axis (elbow-bend plane)
  elbow             → flexion ∈ [0, π]
"""

from __future__ import annotations

import sys
from math import pi
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real2sim import config
from real2sim.retarget import (
    compute_anatomical_angles,
    build_body_frame,
)

# Indices into the 14-D angle vector (matches order in ArmAngles.as_array()).
L_PITCH, L_ROLL, L_YAW, L_ELBOW = 0, 1, 2, 3
R_PITCH, R_ROLL, R_YAW, R_ELBOW = 7, 8, 9, 10


# ── Pose builder helpers ─────────────────────────────────────────────────────

def _make_lms(positions: dict[int, list[float]]) -> np.ndarray:
    lms = np.zeros((33, 3), dtype=np.float64)
    for idx, p in positions.items():
        lms[idx] = p
    return lms


def _torso_frame(shoulder_width: float = 0.4, torso_height: float = 0.55):
    """Standard torso layout, used as common scaffolding for every test pose.

    Hips fixed at z = -h, shoulders at z = 0.  Subject faces camera => MediaPipe
    world axes:  +x = subject's right (so LEFT shoulder is at negative x in
    MediaPipe world frame).  +y is down.  +z is forward.

    The build_body_frame function constructs:
        e_x = (L_hip - R_hip) / |...|     → subject's LEFT direction
        e_z = upward along spine
        e_y = forward
    so it is INSENSITIVE to which way MediaPipe labels its axes — the body
    frame is anatomical, not world-aligned.
    """
    s  = shoulder_width / 2
    return {
        config.LM_LEFT_SHOULDER:  [-s, 0, 0],   # MP world: left = -x
        config.LM_RIGHT_SHOULDER: [ s, 0, 0],
        config.LM_LEFT_HIP:       [-s * 0.7, torso_height, 0],
        config.LM_RIGHT_HIP:      [ s * 0.7, torso_height, 0],
    }


# ── Body-frame sanity test ───────────────────────────────────────────────────

def test_body_frame_orthonormal():
    """build_body_frame should produce an orthonormal matrix."""
    lms = _make_lms(_torso_frame())
    R = build_body_frame(lms)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


# ── Canonical pose tests ─────────────────────────────────────────────────────

def test_a_pose_zero_angles():
    """Arms hanging straight down → all anatomical angles ~ 0."""
    base = _torso_frame()
    # Elbows directly below shoulders, wrists below elbows.
    base[config.LM_LEFT_ELBOW]   = [-0.2, 0.3,  0]
    base[config.LM_LEFT_WRIST]   = [-0.2, 0.6,  0]
    base[config.LM_RIGHT_ELBOW]  = [ 0.2, 0.3,  0]
    base[config.LM_RIGHT_WRIST]  = [ 0.2, 0.6,  0]
    q = compute_anatomical_angles(_make_lms(base))
    assert abs(q[L_PITCH]) < 1e-6, f"L pitch should be 0, got {q[L_PITCH]}"
    assert abs(q[L_ROLL])  < 1e-6, f"L roll should be 0, got {q[L_ROLL]}"
    assert abs(q[L_ELBOW]) < 1e-6, f"L elbow should be 0, got {q[L_ELBOW]}"
    assert abs(q[R_PITCH]) < 1e-6
    assert abs(q[R_ROLL])  < 1e-6
    assert abs(q[R_ELBOW]) < 1e-6


def test_t_pose_roll_pi_over_2():
    """T-pose (arms straight out to sides) → shoulder_roll = ±π/2, others 0.

    LEFT  arm extends to subject's LEFT  (MP world -x) ⇒ +π/2 anatomical roll
    RIGHT arm extends to subject's RIGHT (MP world +x) ⇒ −π/2 anatomical roll
    """
    base = _torso_frame()
    base[config.LM_LEFT_ELBOW]  = [-0.5, 0, 0]
    base[config.LM_LEFT_WRIST]  = [-0.8, 0, 0]
    base[config.LM_RIGHT_ELBOW] = [ 0.5, 0, 0]
    base[config.LM_RIGHT_WRIST] = [ 0.8, 0, 0]
    q = compute_anatomical_angles(_make_lms(base))

    assert abs(q[L_PITCH])              < 1e-6
    assert abs(q[L_ROLL]  - pi / 2)     < 1e-4, f"L roll: {q[L_ROLL]}"
    assert abs(q[L_ELBOW])              < 1e-4

    assert abs(q[R_PITCH])              < 1e-6
    assert abs(q[R_ROLL]  + pi / 2)     < 1e-4, f"R roll: {q[R_ROLL]}"
    assert abs(q[R_ELBOW])              < 1e-4


def test_arms_forward_pitch_neg_pi_over_2():
    """Arms straight forward → shoulder_pitch = −π/2, roll/elbow = 0.

    G1 convention: +pitch swings the arm BACKWARD from the hanging-down
    rest pose, so "arms forward" is −π/2.

    MediaPipe world: +z points AWAY from the camera (subject's back).  Subject
    faces the camera, so "arm forward" means the wrist/elbow is at NEGATIVE
    MP-z relative to the shoulder.
    """
    base = _torso_frame()
    base[config.LM_LEFT_ELBOW]  = [-0.2, 0, -0.3]
    base[config.LM_LEFT_WRIST]  = [-0.2, 0, -0.6]
    base[config.LM_RIGHT_ELBOW] = [ 0.2, 0, -0.3]
    base[config.LM_RIGHT_WRIST] = [ 0.2, 0, -0.6]
    q = compute_anatomical_angles(_make_lms(base))

    assert abs(q[L_PITCH] + pi / 2) < 1e-4, f"L pitch: {q[L_PITCH]}"
    assert abs(q[L_ROLL])           < 1e-4, f"L roll:  {q[L_ROLL]}"
    assert abs(q[L_ELBOW])          < 1e-4
    assert abs(q[R_PITCH] + pi / 2) < 1e-4
    assert abs(q[R_ROLL])           < 1e-4
    assert abs(q[R_ELBOW])          < 1e-4


def test_elbow_flexion_90deg():
    """Upper arm forward, forearm pointing down → elbow ≈ π/2.

    Upper arm along -z (forward, toward camera), forearm along +y (down).
    """
    base = _torso_frame()
    base[config.LM_LEFT_ELBOW]  = [-0.2, 0,   -0.3]
    base[config.LM_LEFT_WRIST]  = [-0.2, 0.3, -0.3]
    base[config.LM_RIGHT_ELBOW] = [ 0.2, 0,   -0.3]
    base[config.LM_RIGHT_WRIST] = [ 0.2, 0.3, -0.3]
    q = compute_anatomical_angles(_make_lms(base))

    # Pitch is −π/2 (upper arm forward), elbow is π/2 (forearm perpendicular).
    assert abs(q[L_PITCH] + pi / 2) < 1e-4
    assert abs(q[L_ELBOW] - pi / 2) < 1e-3, f"L elbow: {q[L_ELBOW]}"
    assert abs(q[R_PITCH] + pi / 2) < 1e-4
    assert abs(q[R_ELBOW] - pi / 2) < 1e-3


def test_elbow_full_flexion():
    """Forearm folded back onto upper arm → elbow = π."""
    base = _torso_frame()
    base[config.LM_LEFT_ELBOW]  = [-0.2, 0, -0.3]
    base[config.LM_LEFT_WRIST]  = [-0.2, 0,  0]      # forearm points back toward shoulder
    base[config.LM_RIGHT_ELBOW] = [ 0.2, 0, -0.3]
    base[config.LM_RIGHT_WRIST] = [ 0.2, 0,  0]
    q = compute_anatomical_angles(_make_lms(base))
    assert abs(q[L_ELBOW] - pi) < 1e-3
    assert abs(q[R_ELBOW] - pi) < 1e-3


def test_arms_up_overhead():
    """Both arms straight overhead → |pitch| = π, roll = 0 (singular).

    The decomposition has a singularity along the vertical axis; with the
    G1 chain (pitch around body-LEFT first, then roll around body-FORWARD),
    overhead falls on |pitch| = π with roll = 0.
    """
    base = _torso_frame()
    base[config.LM_LEFT_ELBOW]  = [-0.2, -0.3, 0]  # above shoulder (negative y = up)
    base[config.LM_LEFT_WRIST]  = [-0.2, -0.6, 0]
    base[config.LM_RIGHT_ELBOW] = [ 0.2, -0.3, 0]
    base[config.LM_RIGHT_WRIST] = [ 0.2, -0.6, 0]
    q = compute_anatomical_angles(_make_lms(base))
    # |pitch| = π, roll ≈ 0 for both arms.
    assert abs(abs(q[L_PITCH]) - pi) < 1e-3, f"L pitch: {q[L_PITCH]}"
    assert abs(q[L_ROLL])             < 1e-3
    assert abs(abs(q[R_PITCH]) - pi) < 1e-3, f"R pitch: {q[R_PITCH]}"
    assert abs(q[R_ROLL])             < 1e-3


def test_clamping_respects_joint_limits():
    """compute_arm_angles must wrap+clamp to MJCF joint limits.

    Arms overhead produces pitch near ±π — outside the [-3.08, 2.67] joint
    range.  _wrap_to_joint_range picks the (pitch ± 2π) variant closer to
    range, and the final clip brings it into [-3.08, 2.67].
    """
    from real2sim.retarget import compute_arm_angles
    base = _torso_frame()
    # Arms up overhead.
    base[config.LM_LEFT_ELBOW]  = [-0.2, -0.3, 0]
    base[config.LM_LEFT_WRIST]  = [-0.2, -0.6, 0]
    base[config.LM_RIGHT_ELBOW] = [ 0.2, -0.3, 0]
    base[config.LM_RIGHT_WRIST] = [ 0.2, -0.6, 0]
    q = compute_arm_angles(_make_lms(base), anatomical_zero=None)
    # Every joint must lie inside its G1 range.
    assert np.all(q >= config.G1_JOINT_LOWER - 1e-6)
    assert np.all(q <= config.G1_JOINT_UPPER + 1e-6)


def test_yaw_orients_elbow_bend_plane():
    """Forearm-DOWN with arm horizontal sideways → yaw ≠ 0.

    The user holds the LEFT arm horizontal (T-pose-like) and bends the
    forearm straight DOWN.  Without shoulder_yaw the G1 elbow would bend the
    forearm BACKWARD; the analytical retargeter must compute a non-zero yaw
    that reorients the elbow bend plane to point down.
    """
    base = _torso_frame()
    # LEFT arm: shoulder (-s, 0, 0), elbow at -s − 0.3 in MP-x (subject left),
    # wrist below the elbow (positive MP-y = subject's down).
    base[config.LM_LEFT_ELBOW]  = [-0.5, 0.0, 0.0]
    base[config.LM_LEFT_WRIST]  = [-0.5, 0.3, 0.0]
    # RIGHT arm mirrored.
    base[config.LM_RIGHT_ELBOW] = [ 0.5, 0.0, 0.0]
    base[config.LM_RIGHT_WRIST] = [ 0.5, 0.3, 0.0]
    q = compute_anatomical_angles(_make_lms(base))

    # Roll should still be ~±π/2 (arm horizontal), elbow ~π/2 (90° bend),
    # and yaw must be non-zero so the forearm points DOWN, not backward.
    assert abs(q[L_ROLL]  - pi / 2) < 1e-3, f"L roll:  {q[L_ROLL]}"
    assert abs(q[L_ELBOW] - pi / 2) < 1e-3, f"L elbow: {q[L_ELBOW]}"
    assert abs(q[L_YAW]) > 0.1,             f"L yaw:   {q[L_YAW]}"
    assert abs(q[R_ROLL] + pi / 2) < 1e-3
    assert abs(q[R_ELBOW] - pi / 2) < 1e-3
    assert abs(q[R_YAW]) > 0.1


def test_anatomical_zero_offset_cancels_bias():
    """Subtracting anatomical_zero collected on the SAME pose returns ~0."""
    base = _torso_frame()
    base[config.LM_LEFT_ELBOW]  = [-0.21, 0.3, 0]   # slightly off perfect A-pose
    base[config.LM_LEFT_WRIST]  = [-0.22, 0.6, 0]
    base[config.LM_RIGHT_ELBOW] = [ 0.19, 0.3, 0]
    base[config.LM_RIGHT_WRIST] = [ 0.18, 0.6, 0]
    lms = _make_lms(base)
    bias = compute_anatomical_angles(lms)
    from real2sim.retarget import compute_arm_angles
    q = compute_arm_angles(lms, anatomical_zero=bias)
    # All shoulders/elbows should be at the per-joint OFFSET (default zeros).
    assert np.allclose(q, config.G1_JOINT_OFFSET, atol=1e-9)
