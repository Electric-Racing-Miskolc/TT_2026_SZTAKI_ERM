"""Analytical joint-angle retargeting (Method A).

Direct, deterministic computation of Unitree G1 upper-body joint targets from
MediaPipe Pose world landmarks. No inverse kinematics, no QP solver — every
joint angle is a closed-form function of the user's joint positions.

Method
------
1.  Build a body frame from the operator's shoulders + hips:
        e_x = subject's LEFT direction   (L_hip - R_hip, normalised)
        e_z = subject's UP direction     (mid_shoulder - mid_hip, normalised)
        e_y = subject's FORWARD direction(e_z × e_x)

2.  Express each upper-arm and forearm as a unit vector in this body frame.

3.  Analytically invert the G1 shoulder kinematic chain.  G1's chain (per
    `g1.xml`) is shoulder_pitch (axis Y_G1 = LEFT) → shoulder_roll (X_G1 =
    FWD) → shoulder_yaw (Z_G1 = upper-arm axis) → elbow (Y_G1).  In the
    BODY frame (x=LEFT, y=FWD, z=UP) this is equivalent to:

        u_upper = Rx(-pitch) · Ry(-roll) · (0, 0, -1)
        u_fore  = Rx(-pitch) · Ry(-roll) · Rz(-yaw) · Rx(-elbow) · (0, 0, -1)

    The four shoulder/elbow joint angles are recovered in closed form:

        roll  = arcsin(u_upper.x)              ∈ [-π/2, π/2]
        pitch = atan2(-u_upper.y, -u_upper.z)  ∈ (-π, π]

    Project u_fore into the upper-arm-local frame and read off:

        elbow = arccos(-u_fore_local.z)             ∈ [0, π]
        yaw   = atan2(-u_fore_local.x, -u_fore_local.y)

    Computing yaw makes the elbow bend in the user's actual bend direction
    (e.g. forearm-DOWN in a goalpost pose) instead of always G1's default
    backward direction.

4.  Pitch values near ±π reach G1's overhead range only via wrap-around —
    `_wrap_to_range` picks pitch ± 2π if the wrapped value is closer to the
    joint range (e.g. atan2 returns +3.05 for "arms up with slight backward
    lean", but G1 reaches that pose via pitch ≈ -3.05 since the negative
    pitch limit is closer to ±π than the positive limit).

5.  Apply per-joint sign / offset table (config.G1_JOINT_SIGN / _OFFSET),
    wrap to range, then clamp to the joint limits read straight from MJCF.

References
----------
*   Humandroid: <https://github.com/vellons/Humandroid>  — same vector-based
    decomposition for a real humanoid robot.
*   Temuge Batpurev, "Estimating joint angles from 3D body poses" (2021):
    <https://temugeb.github.io/python/motion_capture/2021/09/16/joint_rotations.html>
*   Multi-Humanoid Robot Arm Motion Imitation (Biomimetics 10(3):190, 2025):
    geometric joint-angle retargeting baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config

EPS = 1e-6


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros_like(v)


def build_body_frame(lms: np.ndarray) -> np.ndarray:
    """3×3 matrix with columns [e_x, e_y, e_z] expressed in world coordinates.

    v_body = R_body.T @ v_world
    """
    L_sh  = lms[config.LM_LEFT_SHOULDER]
    R_sh  = lms[config.LM_RIGHT_SHOULDER]
    L_hip = lms[config.LM_LEFT_HIP]
    R_hip = lms[config.LM_RIGHT_HIP]

    e_x = _normalize(L_hip - R_hip)
    e_z = _normalize(0.5 * (L_sh + R_sh) - 0.5 * (L_hip + R_hip))
    e_y = _normalize(np.cross(e_z, e_x))
    # Re-orthogonalise e_x against the corrected e_y/e_z to kill any tilt error.
    e_x = _normalize(np.cross(e_y, e_z))
    return np.column_stack([e_x, e_y, e_z]).astype(np.float64)


# ── Per-arm decomposition ────────────────────────────────────────────────────

@dataclass
class ArmAngles:
    """G1-joint 7-vector for a single arm.

    Returned values follow the G1 joint-axis convention directly (no extra
    sign flips in compute_arm_angles).  Body frame is x=LEFT, y=FORWARD,
    z=UP; G1 frame is X=FORWARD, Y=LEFT, Z=UP.

      shoulder_pitch  rotation around G1-Y (= body-LEFT axis).
                       +pitch → upper arm swings BACKWARD from hanging-down;
                       -pitch → swings FORWARD.  ±π → overhead.
      shoulder_roll   rotation around G1-X (= body-FORWARD axis).
                       +roll on LEFT arm  abducts AWAY (toward subject-left);
                       -roll on RIGHT arm abducts AWAY (toward subject-right).
      shoulder_yaw    rotation around the upper-arm long axis.  Orients the
                       elbow bend plane so the forearm bends in the user's
                       actual direction (down vs. back vs. up).
      elbow           flexion ∈ [0, π], 0 = straight, π = fully folded.
      wrist_roll/pitch/yaw  0 in V1.
    """
    pitch: float
    roll:  float
    yaw:   float
    elbow: float
    wrist_roll:  float = 0.0
    wrist_pitch: float = 0.0
    wrist_yaw:   float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([
            self.pitch, self.roll, self.yaw, self.elbow,
            self.wrist_roll, self.wrist_pitch, self.wrist_yaw,
        ], dtype=np.float64)


def _decompose_arm(
    sh_body: np.ndarray,
    el_body: np.ndarray,
    wr_body: np.ndarray,
) -> ArmAngles:
    """Decompose one arm into G1 (pitch, roll, yaw, elbow).

    Vectors are passed in the BODY frame (x=LEFT, y=FORWARD, z=UP).

    G1 chain in body-frame coordinates (see module docstring):

        u_upper = Rx(-pitch) · Ry(-roll) · (0, 0, -1)
        u_fore  = Rx(-pitch) · Ry(-roll) · Rz(-yaw) · Rx(-elbow) · (0, 0, -1)
    """
    u_upper = _normalize(el_body - sh_body)
    u_fore  = _normalize(wr_body - el_body)

    # ── shoulder pitch & roll from upper-arm direction ─────────────────────
    # u_upper = (sin r, -sin p · cos r, -cos p · cos r)
    #   ⇒ roll  = arcsin(u_upper.x)
    #     pitch = atan2(-u_upper.y, -u_upper.z)
    roll  = float(np.arcsin(np.clip(u_upper[0], -1.0, 1.0)))
    # Gimbal-lock guard: when the upper arm lies exactly along ±body-X
    # (perfect T-pose), cos(roll) ≈ 0 makes both pitch terms ≈ 0 and atan2
    # falls back to numerical noise (or the IEEE-754 quirk where
    # atan2(0, -0.0) returns π).  Default to 0 in that singular case.
    if u_upper[1] * u_upper[1] + u_upper[2] * u_upper[2] < 1e-10:
        pitch = 0.0
    else:
        pitch = float(np.arctan2(-u_upper[1], -u_upper[2]))

    # ── express forearm in the upper-arm-LOCAL frame ───────────────────────
    # The local frame is the body frame after (Rx(-pitch) · Ry(-roll)).
    # Inverse rotation: Ry(roll) · Rx(pitch).
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    # Rx(pitch) · u_fore  (rotates around body-LEFT axis)
    af = np.array([
        u_fore[0],
        cp * u_fore[1] - sp * u_fore[2],
        sp * u_fore[1] + cp * u_fore[2],
    ])
    # Ry(roll)  · af      (rotates around body-FORWARD axis)
    u_loc = np.array([
        cr * af[0] + sr * af[2],
        af[1],
       -sr * af[0] + cr * af[2],
    ])

    # ── shoulder yaw & elbow from local-frame forearm direction ────────────
    # u_loc = Rz(-yaw) · Rx(-elbow) · (0, 0, -1)
    #       = (-sin(e)·sin(y), -sin(e)·cos(y), -cos(e))
    cos_elbow = float(np.clip(-u_loc[2], -1.0, 1.0))
    elbow = float(np.arccos(cos_elbow))
    sin_elbow = float(np.sin(elbow))
    if sin_elbow > 1e-6:
        yaw = float(np.arctan2(-u_loc[0], -u_loc[1]))
    else:
        # Elbow is straight (or fully folded) → yaw is indeterminate; the
        # forearm direction is independent of yaw. Picking 0 keeps the
        # upper-arm Z axis stable for downstream wrist joints.
        yaw = 0.0

    return ArmAngles(pitch=pitch, roll=roll, yaw=yaw, elbow=elbow)


# ── Public API ───────────────────────────────────────────────────────────────

def compute_anatomical_angles(lms_world: np.ndarray) -> np.ndarray:
    """Compute the 14-D *anatomical* joint-angle vector from one frame.

    The returned values follow the conventions in the ArmAngles docstring
    and are independent of the G1 sign/offset/limit table.  Use this for
    calibration (recording the user's natural-rest offsets) and for unit
    tests that assert anatomical correctness.
    """
    R_body = build_body_frame(lms_world)
    R_T    = R_body.T

    L_sh = R_T @ lms_world[config.LM_LEFT_SHOULDER]
    L_el = R_T @ lms_world[config.LM_LEFT_ELBOW]
    L_wr = R_T @ lms_world[config.LM_LEFT_WRIST]

    R_sh = R_T @ lms_world[config.LM_RIGHT_SHOULDER]
    R_el = R_T @ lms_world[config.LM_RIGHT_ELBOW]
    R_wr = R_T @ lms_world[config.LM_RIGHT_WRIST]

    left  = _decompose_arm(L_sh, L_el, L_wr)
    right = _decompose_arm(R_sh, R_el, R_wr)

    return np.concatenate([left.as_array(), right.as_array()])


def _wrap_to_joint_range(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """For each revolute joint, pick the (q ± 2π·k) representative closest to
    [lower, upper].

    Critical for the shoulder pitch joint when the user is reaching overhead:
    atan2 returns values near +π for "arms up with slight backward lean", but
    G1's pitch range ([-3.08, 2.67]) is wider on the negative side, so the
    pose is more accurately reached via pitch ≈ -π.  Without this step, those
    poses get clamped to a far-from-vertical arm.
    """
    result = np.array(q, dtype=np.float64, copy=True)
    two_pi = 2.0 * np.pi
    for i in range(result.size):
        a = result[i]
        lo, up = float(lower[i]), float(upper[i])
        best, best_score = a, max(lo - a, a - up, 0.0)
        for delta in (-two_pi, two_pi):
            cand = a + delta
            score = max(lo - cand, cand - up, 0.0)
            if score < best_score:
                best, best_score = cand, score
        result[i] = best
    return result


def compute_arm_angles(
    lms_world: np.ndarray,
    anatomical_zero: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the 14-D G1 arm joint target vector from one frame of landmarks.

    Parameters
    ----------
    lms_world : (33, 3) array
        Filtered MediaPipe world landmarks (metres, hip-centred).
    anatomical_zero : (14,) array or None
        Per-joint *anatomical* offset measured during A-pose calibration.
        Subtracting it makes the user's natural rest pose map to the robot's
        zero (rather than to whatever residual angles MediaPipe produces for
        that pose, e.g. shoulders slightly raised).

    Returns
    -------
    (14,) float64 — joint targets in the order of config.ARM_JOINT_NAMES,
    expressed in G1 joint convention and clamped to the MJCF joint limits.
    """
    q_anatomical = compute_anatomical_angles(lms_world)

    if anatomical_zero is not None:
        q_anatomical = q_anatomical - anatomical_zero

    q_g1 = config.G1_JOINT_SIGN * q_anatomical + config.G1_JOINT_OFFSET
    q_g1 = _wrap_to_joint_range(q_g1, config.G1_JOINT_LOWER, config.G1_JOINT_UPPER)
    q_g1 = np.clip(q_g1, config.G1_JOINT_LOWER, config.G1_JOINT_UPPER)
    return q_g1


# ── Debug helper (for HUD overlay) ───────────────────────────────────────────

def debug_arm_vectors(lms_world: np.ndarray) -> dict:
    """Return body-frame coordinates of key points for overlay / CSV logging."""
    R_body = build_body_frame(lms_world)
    R_T    = R_body.T

    def b(i: int) -> list[float]:
        return (R_T @ lms_world[i]).tolist()

    return {
        "L_shoulder_body": b(config.LM_LEFT_SHOULDER),
        "L_elbow_body":    b(config.LM_LEFT_ELBOW),
        "L_wrist_body":    b(config.LM_LEFT_WRIST),
        "R_shoulder_body": b(config.LM_RIGHT_SHOULDER),
        "R_elbow_body":    b(config.LM_RIGHT_ELBOW),
        "R_wrist_body":    b(config.LM_RIGHT_WRIST),
    }


# ── Rich live-debug info for the GUI ─────────────────────────────────────────

# Order: (display label, MediaPipe landmark index)
_DEBUG_VIS_LANDMARKS = (
    ("L_sh", config.LM_LEFT_SHOULDER),
    ("R_sh", config.LM_RIGHT_SHOULDER),
    ("L_el", config.LM_LEFT_ELBOW),
    ("R_el", config.LM_RIGHT_ELBOW),
    ("L_wr", config.LM_LEFT_WRIST),
    ("R_wr", config.LM_RIGHT_WRIST),
    ("L_hp", config.LM_LEFT_HIP),
    ("R_hp", config.LM_RIGHT_HIP),
)


def compute_debug_info(
    lms_world: np.ndarray,
    anatomical_zero: np.ndarray | None,
    visibility: np.ndarray | None = None,
) -> dict:
    """Return every intermediate value the GUI debug panel needs.

    Pipeline laid out top-to-bottom:
        visibility (MediaPipe)
           ↓
        body frame (e_x / e_y / e_z)
           ↓
        unit arm vectors in body frame  (L_upper, L_fore, R_upper, R_fore)
           ↓
        q_raw      = compute_anatomical_angles(lms)              [14-D]
           ↓
        q_zeroed   = q_raw - anatomical_zero                     [14-D]
           ↓
        q_g1       = SIGN * q_zeroed + OFFSET, clamped to limits [14-D, final]

    Everything is dumped into the returned dict so the GUI can display it
    without ever recomputing.
    """
    R_body = build_body_frame(lms_world)
    R_T = R_body.T

    def unit_vec(a_idx: int, b_idx: int) -> np.ndarray:
        v = R_T @ (lms_world[b_idx] - lms_world[a_idx])
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.zeros(3)

    L_upper = unit_vec(config.LM_LEFT_SHOULDER,  config.LM_LEFT_ELBOW)
    L_fore  = unit_vec(config.LM_LEFT_ELBOW,     config.LM_LEFT_WRIST)
    R_upper = unit_vec(config.LM_RIGHT_SHOULDER, config.LM_RIGHT_ELBOW)
    R_fore  = unit_vec(config.LM_RIGHT_ELBOW,    config.LM_RIGHT_WRIST)

    q_raw    = compute_anatomical_angles(lms_world)
    if anatomical_zero is not None:
        zero    = np.asarray(anatomical_zero, dtype=np.float64)
        q_zeroed = q_raw - zero
    else:
        zero    = np.zeros(14, dtype=np.float64)
        q_zeroed = q_raw.copy()
    q_g1 = config.G1_JOINT_SIGN * q_zeroed + config.G1_JOINT_OFFSET
    q_g1 = _wrap_to_joint_range(q_g1, config.G1_JOINT_LOWER, config.G1_JOINT_UPPER)
    q_g1 = np.clip(q_g1, config.G1_JOINT_LOWER, config.G1_JOINT_UPPER)

    vis_dict: dict[str, float] = {}
    if visibility is not None:
        for label, idx in _DEBUG_VIS_LANDMARKS:
            vis_dict[label] = float(visibility[idx])

    return {
        "visibility":     vis_dict,
        "L_upper_body":   L_upper.tolist(),
        "L_fore_body":    L_fore.tolist(),
        "R_upper_body":   R_upper.tolist(),
        "R_fore_body":    R_fore.tolist(),
        "q_raw":          q_raw.tolist(),
        "q_zeroed":       q_zeroed.tolist(),
        "q_g1_target":    q_g1.tolist(),
        "anatomical_zero": zero.tolist(),
    }
