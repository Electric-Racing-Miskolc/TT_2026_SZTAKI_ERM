"""Map MediaPipe pose world-landmarks to 8 Unitree G1 upper-body joint angles.

The math is documented in the plan file. Summary:

* Build a body frame from shoulders + hips: e_x = subject's left,
  e_z = up along spine, e_y = forward (out of chest).
* For each arm, express the upper-arm direction u_b in the body frame.
* G1 shoulder_pitch rotates in the sagittal plane around the body Y-axis
  where pitch=0 is arm-straight-down, pitch=-pi/2 is arm-forward.
  Formula: pitch = atan2(-u_b.y, -u_b.z)
* Shoulder roll = asin(±u_b.x)  (positive = abduction; right arm negated).
* Elbow flexion = acos(dot(upper_arm, forearm))  (0=straight, pi=fully bent).
* Right arm mirrors the X sign for roll and yaw.
* Output is 8 floats in the order config.JOINT_NAMES.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config

EPS = 1e-6


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v)
    return v / n


def _build_body_frame(landmarks: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix R_body whose columns are e_x, e_y, e_z.

    To express a world vector v_w in body coords: v_b = R_body.T @ v_w.
    """
    L_sh = landmarks[config.LM_LEFT_SHOULDER]
    R_sh = landmarks[config.LM_RIGHT_SHOULDER]
    L_hip = landmarks[config.LM_LEFT_HIP]
    R_hip = landmarks[config.LM_RIGHT_HIP]

    mid_sh = 0.5 * (L_sh + R_sh)
    mid_hip = 0.5 * (L_hip + R_hip)

    e_x = _normalize(L_sh - R_sh)            # subject's left
    e_z = _normalize(mid_sh - mid_hip)       # up along spine
    e_y = _normalize(np.cross(e_z, e_x))     # forward (out of chest)
    e_x = _normalize(np.cross(e_y, e_z))     # re-orthogonalize

    return np.column_stack([e_x, e_y, e_z]).astype(np.float64)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _arm_angles(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    R_body: np.ndarray,
    is_right: bool,
    yaw_scale: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute (pitch, roll, yaw, elbow) for one arm.

    Convention: zero pose has the upper arm pointing straight down along -e_z_body.

    yaw_scale: shoulder yaw is the axial twist of the upper arm, which is
    unreliable from a single camera (requires stable wrist-depth estimate).
    Default 0.0 zeros it out; set to 1.0 to fully include it.
    """
    u_w = elbow - shoulder
    f_w = wrist - elbow
    u_n = _normalize(u_w)
    f_n = _normalize(f_w)
    u_b = R_body.T @ u_n
    ux, uy, uz = float(u_b[0]), float(u_b[1]), float(u_b[2])

    # Sign convention: for the LEFT arm, ux>0 means abduction outward (lateral).
    # The G1 left_shoulder_roll has range [-1.59, 2.25] with positive = abduction.
    # For the RIGHT arm, ux<0 corresponds to outward abduction (mirror), and
    # right_shoulder_roll range is [-2.25, 1.59] with negative = abduction. So
    # we feed (-ux) into asin() for the right arm to keep "outward" -> positive
    # at the formula level, then negate at the output to match the joint sign.
    sign = -1.0 if is_right else 1.0

    # When arm is nearly pure-lateral (uy≈uz≈0), atan2(0,0) is undefined.
    # Snap pitch to 0 (arms-down G1 neutral) so roll drives the motion.
    if uy * uy + uz * uz < EPS:
        pitch = 0.0
    else:
        pitch = float(np.arctan2(-uy, -uz))
    roll = float(np.arcsin(np.clip(sign * ux, -1.0, 1.0)))

    if yaw_scale != 0.0:
        R_pr = _rot_y(pitch) @ _rot_x(roll)
        f_b = R_body.T @ f_n
        f_pr = R_pr.T @ f_b
        yaw = float(np.arctan2(f_pr[0], f_pr[1])) * yaw_scale
    else:
        yaw = 0.0

    cos_elbow = float(np.clip(np.dot(u_n, f_n), -1.0, 1.0))
    elbow_angle = float(np.arccos(cos_elbow))

    if is_right:
        roll = -roll
        yaw = -yaw

    return pitch, roll, yaw, elbow_angle


def compute_angles(
    world_landmarks: np.ndarray,
    visibility: np.ndarray | None = None,
    visibility_threshold: float = config.VISIBILITY_THRESHOLD,
    yaw_scale: float = config.YAW_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (angles, valid_mask).

    angles    : (8,) float array in the order config.JOINT_NAMES.
    valid_mask: (8,) bool array. False entries should be ignored / frozen by
                the caller (typical pattern: keep last filtered value).
    """
    R_body = _build_body_frame(world_landmarks)

    out = np.zeros(8, dtype=np.float64)
    valid = np.ones(8, dtype=bool)

    sides = [
        ("left", False, 0,
         config.LM_LEFT_SHOULDER, config.LM_LEFT_ELBOW, config.LM_LEFT_WRIST),
        ("right", True, 4,
         config.LM_RIGHT_SHOULDER, config.LM_RIGHT_ELBOW, config.LM_RIGHT_WRIST),
    ]

    for _name, is_right, base, sh_i, el_i, wr_i in sides:
        if visibility is not None:
            ok = (visibility[sh_i] >= visibility_threshold and
                  visibility[el_i] >= visibility_threshold and
                  visibility[wr_i] >= visibility_threshold)
            if not ok:
                valid[base:base + 4] = False
                continue
        sh = world_landmarks[sh_i]
        el = world_landmarks[el_i]
        wr = world_landmarks[wr_i]
        try:
            pitch, roll, yaw, elbow = _arm_angles(sh, el, wr, R_body, is_right, yaw_scale)
        except Exception:
            valid[base:base + 4] = False
            continue
        out[base + 0] = pitch
        out[base + 1] = roll
        out[base + 2] = yaw
        out[base + 3] = elbow

    return out, valid


def clip_to_joint_ranges(angles: np.ndarray, joint_ranges: np.ndarray) -> np.ndarray:
    """Clamp each angle to its joint's allowed [lo, hi] range.

    joint_ranges: (8, 2) array, rows in the same order as config.JOINT_NAMES.
    """
    lo = joint_ranges[:, 0]
    hi = joint_ranges[:, 1]
    return np.minimum(np.maximum(angles, lo), hi)
