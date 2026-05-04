"""Headless smoke test: grab 10 frames from the webcam, run MediaPipe Pose,
report how many had detections. Saves the last annotated frame to disk for
visual sanity check.

Run:  real2sim_env/Scripts/python tests/smoke_pose.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from real2sim import config  # noqa: E402
from real2sim.pose import PoseEstimator  # noqa: E402


def main() -> int:
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)

    if not cap.isOpened():
        print("FAIL: camera did not open")
        return 2

    detected = 0
    last_frame = None
    landmarks_seen = None

    with PoseEstimator() as pose:
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                print(f"FAIL: camera read failed at frame {i}")
                cap.release()
                return 2
            t0 = time.time()
            res = pose.process(frame)
            dt = (time.time() - t0) * 1000.0
            print(f"frame {i}: pose_detected={res.world_landmarks is not None} "
                  f"({dt:.1f} ms)")
            if res.world_landmarks is not None:
                detected += 1
                landmarks_seen = res.world_landmarks
            last_frame = res.annotated

    cap.release()

    if last_frame is not None:
        out_path = Path(__file__).parent / "smoke_pose_last.jpg"
        cv2.imwrite(str(out_path), last_frame)
        print(f"saved last annotated frame -> {out_path}")

    print(f"\nSummary: {detected}/10 frames had pose landmarks.")
    if landmarks_seen is not None:
        ls = landmarks_seen
        print(f"world_landmarks shape: {ls.shape}")
        print(f"left  shoulder (idx 11): {ls[11]}")
        print(f"right shoulder (idx 12): {ls[12]}")
        print(f"left  elbow    (idx 13): {ls[13]}")
        print(f"right elbow    (idx 14): {ls[14]}")

    return 0 if detected > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
