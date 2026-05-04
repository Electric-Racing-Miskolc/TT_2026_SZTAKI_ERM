"""Filtering utilities for Real2Sim.

Only limit_delta is kept from the old module; the EMA-based smoothers have
been replaced by OneEuroFilterArray (see one_euro.py).
"""

from __future__ import annotations

import numpy as np


def limit_delta(prev: np.ndarray, curr: np.ndarray, max_delta: float) -> np.ndarray:
    """Clamp per-channel change to [-max_delta, +max_delta].

    Safety net against sudden jumps when MediaPipe re-detects a landmark
    after a temporary occlusion.
    """
    return prev + np.clip(curr - prev, -max_delta, max_delta)
