"""Split-screen mp4 recorder.

Composes (annotated webcam | rendered MuJoCo) horizontally and writes via
cv2.VideoWriter. Both panels are resized to the same height; the final width
is panel_w * 2.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class SplitScreenRecorder:
    def __init__(self, out_path: str | Path, panel_w: int, panel_h: int, fps: float):
        self.out_path = Path(out_path)
        self.panel_w = int(panel_w)
        self.panel_h = int(panel_h)
        self.fps = float(fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.out_path),
            fourcc,
            self.fps,
            (self.panel_w * 2, self.panel_h),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {self.out_path}")
        self._closed = False

    def _fit(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if (w, h) != (self.panel_w, self.panel_h):
            return cv2.resize(img, (self.panel_w, self.panel_h))
        return img

    def write(self, left_bgr: np.ndarray, right_rgb_or_bgr: np.ndarray, right_is_rgb: bool = True):
        if self._closed:
            raise RuntimeError("recorder already closed")
        left = self._fit(left_bgr)
        right = right_rgb_or_bgr
        if right_is_rgb:
            right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        right = self._fit(right)
        composite = np.hstack([left, right])
        self._writer.write(composite)

    def close(self):
        if not self._closed:
            self._writer.release()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
