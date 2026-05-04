"""Verify the MediaPipe pose pipeline on a synthetic test image (no camera).

Generates a stick-figure person with a colored body on white background,
runs PoseEstimator, prints whether landmarks were detected. This isolates
the pose model from any webcam issues.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import urllib.request

import cv2
import numpy as np

from real2sim.pose import PoseEstimator


SAMPLE_URL = (
    "https://storage.googleapis.com/mediapipe-assets/pose_landmarker.jpg"
)


def main() -> int:
    sample = Path(__file__).parent / "_sample_person.jpg"
    if not sample.exists():
        print(f"downloading sample image -> {sample}")
        urllib.request.urlretrieve(SAMPLE_URL, sample)

    img = cv2.imread(str(sample))
    if img is None:
        print(f"FAIL: could not load {sample}")
        return 2

    print(f"loaded image {img.shape}")

    with PoseEstimator() as pose:
        # We're in VIDEO mode; feed the same image once.
        res = pose.process(img)

    out = Path(__file__).parent / "smoke_pose_static_out.jpg"
    cv2.imwrite(str(out), res.annotated)
    print(f"detected = {res.world_landmarks is not None}")
    if res.world_landmarks is not None:
        print(f"shape    = {res.world_landmarks.shape}")
        print(f"L sh (11)= {res.world_landmarks[11]}")
        print(f"R sh (12)= {res.world_landmarks[12]}")
    print(f"annotated image -> {out}")

    return 0 if res.world_landmarks is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
