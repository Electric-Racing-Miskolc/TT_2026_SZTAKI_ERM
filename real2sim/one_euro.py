"""One-Euro filter — low-latency adaptive smoothing for landmark arrays.

Reference: Casiez et al. 2012, "1€ Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems."
https://gery.casiez.net/1euro/
"""

from __future__ import annotations

import numpy as np

from . import config


class _LowPass:
    """Single-channel first-order low-pass filter."""

    def __init__(self) -> None:
        self._y: float | None = None
        self._dy: float | None = None

    def apply(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
        self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Per-channel One-Euro filter for a 1-D float sequence."""

    def __init__(
        self,
        min_cutoff: float = config.OEF_MIN_CUTOFF,
        beta: float = config.OEF_BETA,
        d_cutoff: float = config.OEF_D_CUTOFF,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_lp = _LowPass()
        self._dx_lp = _LowPass()
        self._last: float | None = None

    def reset(self) -> None:
        self._x_lp = _LowPass()
        self._dx_lp = _LowPass()
        self._last = None

    def __call__(self, x: float, dt: float) -> float:
        if self._last is None:
            self._last = x
        dx = (x - self._last) / max(dt, 1e-9)
        self._last = x
        da = _alpha(self._d_cutoff, dt)
        dx_hat = self._dx_lp.apply(dx, da)
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        return self._x_lp.apply(x, a)


class OneEuroFilterArray:
    """Vectorised One-Euro filter for an (N,) or (N, D) numpy array.

    Call ``update(arr, dt)`` each frame; returns the smoothed array.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        min_cutoff: float = config.OEF_MIN_CUTOFF,
        beta: float = config.OEF_BETA,
        d_cutoff: float = config.OEF_D_CUTOFF,
    ) -> None:
        self._shape = shape
        n = int(np.prod(shape))
        self._filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n)
        ]

    def reset(self) -> None:
        for f in self._filters:
            f.reset()

    def update(self, arr: np.ndarray, dt: float) -> np.ndarray:
        flat = arr.ravel().astype(np.float64)
        out = np.array([f(float(v), dt) for f, v in zip(self._filters, flat)])
        return out.reshape(self._shape)
