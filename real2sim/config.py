"""Project-wide constants for Real2Sim."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
KEYFRAME_NAME = "stand"

# Eight upper-body actuators we drive. Order matters: it defines the angle vector layout.
JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

# MediaPipe Pose landmark indices we care about.
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24

VISIBILITY_THRESHOLD = 0.5

# Joint-angle smoother: lower = smoother/more lag. 0.15 keeps motion fluid
# while cutting out per-frame noise.
SMOOTHING_ALPHA = 0.15

# Landmark position smoother (applied BEFORE angle math).
# Stabilises the body frame and reduces input noise.
LANDMARK_SMOOTH_ALPHA = 0.25

# Maximum joint-angle change per frame (velocity clamp). Prevents the sudden
# jumps that occur near gimbal-lock configurations or when MediaPipe drops
# and re-detects a landmark.
import math as _math
MAX_ANGLE_DELTA_RAD = _math.radians(8.0)

# Shoulder yaw (axial twist of upper arm) is unreliable from a single camera.
# Scale it toward zero to avoid chaotic motion on that axis.
YAW_SCALE = 0.0

CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480
TARGET_FPS = 30
