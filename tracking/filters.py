"""Adaptive low-pass filtering for noisy landmark streams.

The One-Euro filter (Casiez et al., CHI 2012) is the standard answer to the
jitter-vs-lag dilemma of tracking: a fixed smoothing factor either leaves the
cursor trembling when the hand is still, or makes it lag when the hand moves.

The One-Euro filter makes the cutoff frequency grow with the signal's speed:

    cutoff = min_cutoff + beta * |speed|

When the hand is slow the cutoff is low (heavy smoothing, no jitter); when it is
fast the cutoff is high (little smoothing, little lag).
"""

from __future__ import annotations

import math

import numpy as np


def _smoothing_alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    """Exponential-smoothing factor for a first-order low-pass at ``cutoff`` Hz."""
    tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-Euro filter for arrays of points shaped ``(..., D)``.

    Speed is measured per point (norm over the last axis) so each landmark adapts
    on its own - a fingertip can be moving while the wrist is still.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or x.shape != self._x.shape:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()

        dt = t - self._t
        if dt <= 1e-6:  # duplicate / out-of-order timestamp: keep the previous estimate
            return self._x.copy()
        self._t = t

        # 1) smoothed derivative
        dx_raw = (x - self._x) / dt
        a_d = _smoothing_alpha(self.d_cutoff, dt)
        self._dx = a_d * dx_raw + (1.0 - a_d) * self._dx

        # 2) speed-adaptive cutoff, per point
        speed = np.linalg.norm(self._dx, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed

        # 3) smooth the signal itself
        a = _smoothing_alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()
