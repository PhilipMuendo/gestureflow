"""A fake camera for ``--simulate`` mode, demos and tests.

Produces a dim, slightly noisy "room" backdrop.  With ``realtime=False`` the
timestamps advance by exactly ``1/fps`` per frame regardless of how fast the
caller reads, which makes simulated runs deterministic (used by the end-to-end
tests and for generating screenshots).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from camera.capture import Frame


class SimulatedCamera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, realtime: bool = True, seed: int = 3) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.realtime = realtime
        self.status = "ok"
        self._index = 0
        self._started = 0.0
        rng = np.random.default_rng(seed)

        # soft vertical gradient + vignette + a few blurred blobs: reads as a dim room
        y = np.linspace(0.0, 1.0, height)[:, None]
        x = np.linspace(-1.0, 1.0, width)[None, :]
        base = 52 + 30 * y - 22 * (x**2 + (y - 0.5) ** 2 * 2)
        blobs = cv2.GaussianBlur(rng.uniform(0, 1, (height // 8, width // 8)).astype(np.float32), (0, 0), 3)
        blobs = cv2.resize(blobs, (width, height), interpolation=cv2.INTER_CUBIC)
        gray = np.clip(base + 26 * (blobs - 0.5), 0, 255)
        self._backdrop = np.dstack([gray * 1.08, gray * 0.98, gray * 0.88]).astype(np.uint8)  # cool tint (BGR)
        self._rng = rng

    def start(self) -> None:
        self._started = time.perf_counter()

    def stop(self) -> None:
        self.status = "closed"

    def read(self, timeout: float | None = None) -> Frame | None:
        self._index += 1
        t = self._index / self.fps
        if self.realtime:
            delay = (self._started + t) - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        image = self._backdrop.copy()
        # a touch of per-frame sensor noise on a small patch keeps the feed from looking frozen
        flicker = int(self._rng.integers(-2, 3))
        if flicker:
            image = cv2.add(image, np.full_like(image, abs(flicker))) if flicker > 0 else cv2.subtract(image, np.full_like(image, abs(flicker)))
        return Frame(image, t if not self.realtime else time.perf_counter(), self._index)
