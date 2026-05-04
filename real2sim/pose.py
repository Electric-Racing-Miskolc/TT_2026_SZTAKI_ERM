"""MediaPipe Pose wrapper (Tasks API).

Returns world-coordinate landmarks (meters, hip-centered) which are far more
stable for joint-angle math than the image-normalized landmarks.

Uses mediapipe.tasks.vision.PoseLandmarker in VIDEO mode for sequential frames.
The legacy `mediapipe.solutions.pose` API was removed in mediapipe>=0.10.20.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"


@dataclass
class PoseResult:
    """One frame of pose data.

    world_landmarks: (33, 3) numpy array in meters, hip-centered. None if no detection.
    visibility:     (33,)   numpy array in [0, 1]. None if no detection.
    annotated:      BGR image with skeleton drawn (always present).
    """

    world_landmarks: np.ndarray | None
    visibility: np.ndarray | None
    annotated: np.ndarray


class PoseEstimator:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, min_detection_confidence: float = 0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. "
                "Download with the snippet in README.md."
            )
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._connections = PoseLandmarksConnections.POSE_LANDMARKS

    def _timestamp_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def process(self, bgr_frame: np.ndarray) -> PoseResult:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms())

        annotated = bgr_frame.copy()
        world = None
        vis = None

        if result.pose_landmarks:
            lms_2d = result.pose_landmarks[0]
            h, w = bgr_frame.shape[:2]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms_2d]
            for c in self._connections:
                p1, p2 = pts[c.start], pts[c.end]
                cv2.line(annotated, p1, p2, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(annotated, p, 3, (0, 0, 255), -1)
            vis = np.array([lm.visibility for lm in lms_2d], dtype=np.float32)

        if result.pose_world_landmarks:
            lms_w = result.pose_world_landmarks[0]
            world = np.array([[lm.x, lm.y, lm.z] for lm in lms_w], dtype=np.float32)

        return PoseResult(world_landmarks=world, visibility=vis, annotated=annotated)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
