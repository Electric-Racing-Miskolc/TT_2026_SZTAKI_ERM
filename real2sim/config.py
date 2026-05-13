"""Project-wide constants for Real2Sim (analytical joint-angle edition)."""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
KEYFRAME_NAME = "stand"

# Calibration cache (overwritten each re-calibration).
CALIBRATION_PATH = PROJECT_ROOT / "models" / "anthropo.json"

# ── Arm joints driven by retargeting (14 total, L then R, in MJCF tree order)
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# ── Anatomical → G1 joint sign / offset table ─────────────────────────────────
# After retarget._decompose_arm we have a 7-vector of "anatomical" angles per
# arm; this table converts them to G1's joint sign convention. Indices match
# ARM_JOINT_NAMES order.
#
# Initial guesses derived from:
#   * the joint axes in g1.xml (L/R shoulder pitch axis = (0,1,0), L/R
#     shoulder roll axis = (1,0,0), elbow axis = (0,1,0)),
#   * the "stand" keyframe (left/right shoulder_roll = ±0.2 → both arms
#     slightly abducted), so a positive anatomical roll maps to a +1 sign on
#     LEFT and naturally yields the mirrored ±π/2 on RIGHT (no extra flip).
#
# If a joint moves the wrong direction at runtime, flip its sign here — do
# NOT hack the maths in retarget.py.  Run `tests/test_retarget.py -v` to
# confirm anatomical conventions before tuning signs.
G1_JOINT_SIGN = np.array([
    +1.0,  # L shoulder_pitch  (arm forward → +pitch)
    +1.0,  # L shoulder_roll   (arm out-to-left = +roll)
    +1.0,  # L shoulder_yaw
    +1.0,  # L elbow           (flexion → +)
    +1.0,  # L wrist_roll
    +1.0,  # L wrist_pitch
    +1.0,  # L wrist_yaw
    +1.0,  # R shoulder_pitch
    +1.0,  # R shoulder_roll   (formula gives -π/2 for T-pose right, matches MJCF)
    +1.0,  # R shoulder_yaw
    +1.0,  # R elbow
    +1.0,  # R wrist_roll
    +1.0,  # R wrist_pitch
    +1.0,  # R wrist_yaw
], dtype=np.float64)

# Per-joint constant offset added AFTER sign multiplication. Use this only
# when the user's natural rest pose is consistently shifted from zero (e.g.
# shoulders slightly raised). Prefer the per-user anatomical-zero offset
# captured by run_anthropo_calibration() over hard-coding here.
G1_JOINT_OFFSET = np.zeros(14, dtype=np.float64)

# Joint limits straight from g1.xml (radians). Order = ARM_JOINT_NAMES.
G1_JOINT_LOWER = np.array([
    -3.0892, -1.5882, -2.6180, -1.0472, -1.9722, -1.6144, -1.6144,  # left
    -3.0892, -2.2515, -2.6180, -1.0472, -1.9722, -1.6144, -1.6144,  # right
], dtype=np.float64)
G1_JOINT_UPPER = np.array([
     2.6704,  2.2515,  2.6180,  2.0944,  1.9722,  1.6144,  1.6144,  # left
     2.6704,  1.5882,  2.6180,  2.0944,  1.9722,  1.6144,  1.6144,  # right
], dtype=np.float64)

# ── MediaPipe landmark indices ───────────────────────────────────────────────
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW     = 13
LM_RIGHT_ELBOW    = 14
LM_LEFT_WRIST     = 15
LM_RIGHT_WRIST    = 16
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24

VISIBILITY_THRESHOLD = 0.5

# ── Calibration ──────────────────────────────────────────────────────────────
CALIB_COUNTDOWN_S = 3.0   # "Hold pose" on-screen timer before each pose
CALIB_COLLECT_S   = 1.5   # Seconds of frames averaged per pose

# ── One-Euro filter (raw MediaPipe world landmarks) ──────────────────────────
OEF_MIN_CUTOFF = 1.0   # Hz — base smoothing
OEF_BETA       = 0.05  # Speed coupling (higher → less lag on fast motion)
OEF_D_CUTOFF   = 1.0   # Hz — derivative cutoff

# ── Camera / recording ───────────────────────────────────────────────────────
CAMERA_INDEX = 2   # 0 = beépített RGB, 1 = IR (Windows Hello), 2 = USB kamera
FRAME_W      = 640
FRAME_H      = 480
TARGET_FPS   = 30
