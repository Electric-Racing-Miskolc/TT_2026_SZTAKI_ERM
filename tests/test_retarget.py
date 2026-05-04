"""Synthetic-pose unit tests for the retargeting math.

Builds 33-landmark arrays where only the indices we care about
(shoulders, elbows, wrists, hips) are populated. The body frame
the math expects is:
  e_x_body = subject's left   (along shoulder line, +X = subject's left)
  e_z_body = up along spine   (+Z = head)
  e_y_body = forward          (out of the chest)

We place the subject so that the world frame coincides with the body
frame, which keeps the expected angles easy to reason about.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real2sim import retarget  # noqa: E402
from real2sim.config import (  # noqa: E402
    LM_LEFT_ELBOW,
    LM_LEFT_HIP,
    LM_LEFT_SHOULDER,
    LM_LEFT_WRIST,
    LM_RIGHT_ELBOW,
    LM_RIGHT_HIP,
    LM_RIGHT_SHOULDER,
    LM_RIGHT_WRIST,
)

TOL_RAD = math.radians(5)


def make_landmarks(
    L_sh, R_sh, L_hip, R_hip,
    L_el, R_el,
    L_wr, R_wr,
):
    arr = np.zeros((33, 3), dtype=np.float32)
    arr[LM_LEFT_SHOULDER] = L_sh
    arr[LM_RIGHT_SHOULDER] = R_sh
    arr[LM_LEFT_HIP] = L_hip
    arr[LM_RIGHT_HIP] = R_hip
    arr[LM_LEFT_ELBOW] = L_el
    arr[LM_RIGHT_ELBOW] = R_el
    arr[LM_LEFT_WRIST] = L_wr
    arr[LM_RIGHT_WRIST] = R_wr
    return arr


def torso_only(shoulder_half: float = 0.18, torso_h: float = 0.5):
    """Common torso (shoulders + hips). +X subject left, +Y forward, +Z up."""
    L_sh = np.array([+shoulder_half, 0.0, 0.0])
    R_sh = np.array([-shoulder_half, 0.0, 0.0])
    L_hip = np.array([+0.5 * shoulder_half, 0.0, -torso_h])
    R_hip = np.array([-0.5 * shoulder_half, 0.0, -torso_h])
    return L_sh, R_sh, L_hip, R_hip


# ----------------------------------------------------------------------------
# Test 1: arms hanging straight down -> all angles ~ 0
# ----------------------------------------------------------------------------
def test_arms_down_all_zero():
    L_sh, R_sh, L_hip, R_hip = torso_only()
    upper, lower = 0.30, 0.27
    L_el = L_sh + np.array([0, 0, -upper])
    R_el = R_sh + np.array([0, 0, -upper])
    L_wr = L_el + np.array([0, 0, -lower])
    R_wr = R_el + np.array([0, 0, -lower])
    lms = make_landmarks(L_sh, R_sh, L_hip, R_hip, L_el, R_el, L_wr, R_wr)

    angles, valid = retarget.compute_angles(lms)
    assert valid.all()
    # arms down -> pitch = 0 in G1 (arm points straight down at zero)
    assert abs(angles[0]) < TOL_RAD, f"L pitch {angles[0]}"
    assert abs(angles[1]) < TOL_RAD, f"L roll  {angles[1]}"
    # yaw is undefined when arm is straight; just check elbow
    assert abs(angles[3]) < TOL_RAD, f"L elbow {angles[3]}"
    assert abs(angles[4]) < TOL_RAD, f"R pitch {angles[4]}"
    assert abs(angles[5]) < TOL_RAD, f"R roll  {angles[5]}"
    assert abs(angles[7]) < TOL_RAD, f"R elbow {angles[7]}"


# ----------------------------------------------------------------------------
# Test 2: arms forward horizontal -> shoulder pitch = -pi/2
#         G1 pitch=0 = arm down, pitch=-pi/2 = arm forward.
# ----------------------------------------------------------------------------
def test_arms_forward_pitch_90():
    L_sh, R_sh, L_hip, R_hip = torso_only()
    upper, lower = 0.30, 0.27
    L_el = L_sh + np.array([0, +upper, 0])    # +Y = forward in body frame
    R_el = R_sh + np.array([0, +upper, 0])
    L_wr = L_el + np.array([0, +lower, 0])
    R_wr = R_el + np.array([0, +lower, 0])
    lms = make_landmarks(L_sh, R_sh, L_hip, R_hip, L_el, R_el, L_wr, R_wr)

    angles, valid = retarget.compute_angles(lms)
    assert valid.all()
    # arms forward = pitch=-pi/2 (G1: 0=down, -pi/2=forward)
    assert abs(angles[0] + math.pi / 2) < TOL_RAD, f"L pitch {angles[0]}"
    assert abs(angles[4] + math.pi / 2) < TOL_RAD, f"R pitch {angles[4]}"
    # roll should be ~0 (no abduction)
    assert abs(angles[1]) < TOL_RAD, f"L roll {angles[1]}"
    assert abs(angles[5]) < TOL_RAD, f"R roll {angles[5]}"


# ----------------------------------------------------------------------------
# Test 3: T-pose (arms out laterally) -> shoulder roll = +pi/2 (left), -pi/2 (right)
# ----------------------------------------------------------------------------
def test_tpose_roll_90():
    L_sh, R_sh, L_hip, R_hip = torso_only()
    upper, lower = 0.30, 0.27
    L_el = L_sh + np.array([+upper, 0, 0])    # subject's left
    R_el = R_sh + np.array([-upper, 0, 0])    # subject's right
    L_wr = L_el + np.array([+lower, 0, 0])
    R_wr = R_el + np.array([-lower, 0, 0])
    lms = make_landmarks(L_sh, R_sh, L_hip, R_hip, L_el, R_el, L_wr, R_wr)

    angles, valid = retarget.compute_angles(lms)
    assert valid.all()
    # Convention: positive roll on left arm = arm abducting outward (asin(+1) = pi/2).
    # On the right arm we flip the output sign so that *outward abduction*
    # produces a negative joint value (matches G1's right_shoulder_roll
    # range [-2.25, 1.59] where negative is outward).
    assert abs(angles[1] - math.pi / 2) < TOL_RAD, f"L roll {angles[1]}"
    assert abs(angles[5] + math.pi / 2) < TOL_RAD, f"R roll {angles[5]}"
    assert abs(angles[3]) < TOL_RAD, f"L elbow {angles[3]}"
    assert abs(angles[7]) < TOL_RAD, f"R elbow {angles[7]}"


# ----------------------------------------------------------------------------
# Test 4: arm down + elbow bent 90deg with forearm forward -> elbow ~ pi/2
# ----------------------------------------------------------------------------
def test_elbow_90():
    L_sh, R_sh, L_hip, R_hip = torso_only()
    upper, lower = 0.30, 0.27
    L_el = L_sh + np.array([0, 0, -upper])
    R_el = R_sh + np.array([0, 0, -upper])
    # Forearm forward (+Y), perpendicular to upper arm -> 90 deg flexion
    L_wr = L_el + np.array([0, +lower, 0])
    R_wr = R_el + np.array([0, +lower, 0])
    lms = make_landmarks(L_sh, R_sh, L_hip, R_hip, L_el, R_el, L_wr, R_wr)

    angles, valid = retarget.compute_angles(lms)
    assert valid.all()
    assert abs(angles[3] - math.pi / 2) < TOL_RAD, f"L elbow {angles[3]}"
    assert abs(angles[7] - math.pi / 2) < TOL_RAD, f"R elbow {angles[7]}"


# ----------------------------------------------------------------------------
# Test 5: visibility gating
# ----------------------------------------------------------------------------
def test_visibility_gating():
    L_sh, R_sh, L_hip, R_hip = torso_only()
    upper, lower = 0.30, 0.27
    L_el = L_sh + np.array([0, 0, -upper])
    R_el = R_sh + np.array([0, 0, -upper])
    L_wr = L_el + np.array([0, 0, -lower])
    R_wr = R_el + np.array([0, 0, -lower])
    lms = make_landmarks(L_sh, R_sh, L_hip, R_hip, L_el, R_el, L_wr, R_wr)

    vis = np.ones(33, dtype=np.float32)
    vis[LM_LEFT_WRIST] = 0.1     # below threshold
    angles, valid = retarget.compute_angles(lms, visibility=vis)
    assert not valid[0] and not valid[1] and not valid[2] and not valid[3]
    assert valid[4] and valid[5] and valid[6] and valid[7]


# ----------------------------------------------------------------------------
# Test 6: clipping
# ----------------------------------------------------------------------------
def test_clipping():
    angles = np.array([5.0, -5.0, 0.0, 5.0, -5.0, 5.0, 0.0, -5.0])
    ranges = np.array([
        [-3.0892, 2.6704],
        [-1.5882, 2.2515],
        [-2.618, 2.618],
        [-1.0472, 2.0944],
        [-3.0892, 2.6704],
        [-2.2515, 1.5882],
        [-2.618, 2.618],
        [-1.0472, 2.0944],
    ])
    out = retarget.clip_to_joint_ranges(angles, ranges)
    assert out[0] == 2.6704
    assert out[1] == -1.5882
    assert out[3] == 2.0944
    assert out[4] == -3.0892
    assert out[5] == 1.5882
    assert out[7] == -1.0472
