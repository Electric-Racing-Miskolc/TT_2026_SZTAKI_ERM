"""Project-wide constants for Real2Sim (mink IK edition)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
KEYFRAME_NAME = "stand"

# Calibration cache (overwritten each recalibration).
CALIBRATION_PATH = PROJECT_ROOT / "models" / "calibration.json"

# ── Arm joints driven by IK (14 total, L then R) ────────────────────────────
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

# Sites that mink targets (must be present in the patched MJCF).
IK_SITE_LEFT  = "left_palm"
IK_SITE_RIGHT = "right_palm"

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

# ── IK solver ────────────────────────────────────────────────────────────────
IK_SOLVER       = "daqp"       # CPU-friendly QP solver
IK_POSTURE_COST = 1e-3         # Regularisation; keep arms near natural rest
IK_DT           = 1.0 / 30.0  # Integration step (matches TARGET_FPS)

# ── Calibration ──────────────────────────────────────────────────────────────
CALIB_COUNTDOWN_S = 3.0   # "Hold T-pose" on-screen timer before collecting
CALIB_COLLECT_S   = 1.5   # Seconds of frames averaged for calibration
CALIB_SCALE_MIN   = 0.4   # Safety clamp: robot / person arm-length ratio
CALIB_SCALE_MAX   = 1.8

# ── One-Euro filter (applied to raw MediaPipe world landmarks) ───────────────
OEF_MIN_CUTOFF = 1.0   # Hz — base smoothing
OEF_BETA       = 0.05  # Speed coupling (higher → less lag on fast motion)
OEF_D_CUTOFF   = 1.0   # Hz — derivative cutoff

# ── Camera / recording ───────────────────────────────────────────────────────
CAMERA_INDEX = 2   # 0 = beépített RGB, 1 = IR (Windows Hello), 2 = USB kamera
FRAME_W      = 640
FRAME_H      = 480
TARGET_FPS   = 30
