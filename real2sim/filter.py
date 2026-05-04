"""Filtering utilities for Real2Sim.

ExpSmoother      — per-channel exponential moving average for joint angles.
LandmarkSmoother — exponential moving average over the (33, 3) landmark array;
                   smooths landmark positions before angle computation so the
                   body frame stays stable.
limit_delta      — per-channel angular velocity clamp; prevents sudden jumps.
"""

from __future__ import annotations

import numpy as np


class ExpSmoother:
    """Per-channel exponential smoother for the 8-dim joint-angle vector.

    theta_t = alpha * theta_raw + (1 - alpha) * theta_{t-1}

    Invalid channels (visibility drop) freeze at their last value.
    On the first valid frame after a drop, the channel is re-seeded instantly.
    """

    def __init__(self, dim: int, alpha: float = 0.15):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._state: np.ndarray | None = None
        self._fresh = np.ones(dim, dtype=bool)
        self._dim = dim

    def reset(self):
        self._state = None
        self._fresh = np.ones(self._dim, dtype=bool)

    def update(self, raw: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        if raw.shape != (self._dim,):
            raise ValueError(f"expected ({self._dim},), got {raw.shape}")
        if valid is None:
            valid = np.ones(self._dim, dtype=bool)

        if self._state is None:
            self._state = np.zeros(self._dim, dtype=np.float64)

        for i in range(self._dim):
            if not valid[i]:
                self._fresh[i] = True
                continue
            if self._fresh[i]:
                self._state[i] = float(raw[i])
                self._fresh[i] = False
            else:
                self._state[i] = self._alpha * float(raw[i]) + (1.0 - self._alpha) * self._state[i]

        return self._state.copy()


class LandmarkSmoother:
    """Exponential smoother over the (33, 3) pose world-landmark array.

    Smoothing landmark POSITIONS before computing joint angles stabilises the
    body frame and reduces high-frequency noise in the angle outputs.
    """

    def __init__(self, alpha: float = 0.25):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._state: np.ndarray | None = None

    def reset(self):
        self._state = None

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """landmarks: (33, 3) float array. Returns smoothed (33, 3) array."""
        if self._state is None:
            self._state = landmarks.astype(np.float64).copy()
        else:
            self._state = self._alpha * landmarks + (1.0 - self._alpha) * self._state
        return self._state.copy()


def limit_delta(prev: np.ndarray, curr: np.ndarray, max_delta: float) -> np.ndarray:
    """Clamp per-channel change to [-max_delta, +max_delta].

    Prevents instantaneous jumps near gimbal-lock configurations or when
    MediaPipe re-detects a previously dropped landmark.
    """
    delta = curr - prev
    clipped = np.clip(delta, -max_delta, max_delta)
    return prev + clipped
