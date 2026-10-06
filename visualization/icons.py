"""Tiny monoline gesture icons.

Each icon is a few strokes on a unit square.  They are rasterised 4x oversampled and
area-downscaled to an alpha mask (cached per size), then tinted and blended like text, so
they stay crisp and consistent at 14-20 px without shipping image assets.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from visualization import typeface
from visualization.theme import Color

SS = 4  # oversampling factor

# stroke = list of (x, y) points on the unit square; closed strokes repeat the first point
_P = lambda *pts: [pts[i : i + 2] for i in range(0, len(pts), 2)]  # noqa: E731

ICONS: dict[str, list[list[tuple[float, float]]]] = {
    "palm": [
        _P(0.30, 0.58, 0.30, 0.16), _P(0.43, 0.52, 0.43, 0.08), _P(0.57, 0.52, 0.57, 0.08), _P(0.70, 0.58, 0.70, 0.16),
        _P(0.30, 0.58, 0.30, 0.78, 0.50, 0.92, 0.70, 0.78, 0.70, 0.58),
        _P(0.30, 0.66, 0.14, 0.44),
    ],
    "point": [
        _P(0.50, 0.08, 0.50, 0.56),
        _P(0.30, 0.56, 0.30, 0.78, 0.50, 0.92, 0.70, 0.78, 0.70, 0.56, 0.30, 0.56),
    ],
    "pinch": [
        _P(0.08, 0.50, 0.36, 0.50), _P(0.24, 0.36, 0.38, 0.50, 0.24, 0.64),
        _P(0.92, 0.50, 0.64, 0.50), _P(0.76, 0.36, 0.62, 0.50, 0.76, 0.64),
        _P(0.48, 0.47, 0.52, 0.47, 0.52, 0.53, 0.48, 0.53, 0.48, 0.47),
    ],
    "fist": [
        _P(0.24, 0.40, 0.76, 0.40, 0.76, 0.84, 0.24, 0.84, 0.24, 0.40),
        _P(0.41, 0.40, 0.41, 0.58), _P(0.59, 0.40, 0.59, 0.58),
        _P(0.24, 0.40, 0.34, 0.20, 0.66, 0.20, 0.76, 0.40),
    ],
    "two": [
        _P(0.38, 0.90, 0.28, 0.10), _P(0.62, 0.90, 0.72, 0.10),
        _P(0.30, 0.64, 0.70, 0.64),
    ],
    "swipe": [
        _P(0.12, 0.50, 0.88, 0.50),
        _P(0.26, 0.34, 0.12, 0.50, 0.26, 0.66),
        _P(0.74, 0.34, 0.88, 0.50, 0.74, 0.66),
    ],
    "zoom": [
        _P(0.14, 0.86, 0.46, 0.54), _P(0.86, 0.14, 0.54, 0.46),
        _P(0.14, 0.62, 0.14, 0.86, 0.38, 0.86), _P(0.86, 0.38, 0.86, 0.14, 0.62, 0.14),
    ],
}


@lru_cache(maxsize=64)
def _mask(name: str, size: int, stroke: float) -> np.ndarray:
    big = size * SS
    canvas = np.zeros((big, big), np.uint8)
    pad = 0.06  # keep strokes off the very edge
    for stroke_pts in ICONS[name]:
        pts = np.array([[(pad + (1 - 2 * pad) * x) * big, (pad + (1 - 2 * pad) * y) * big] for x, y in stroke_pts], np.int32)
        cv2.polylines(canvas, [pts], False, 255, max(1, int(round(stroke * SS))), cv2.LINE_AA)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def draw_icon(img: np.ndarray, name: str, cx: int, cy: int, size: int, color: Color, alpha: float = 1.0, stroke: float = 1.3) -> None:
    """Draw icon ``name`` centred at (cx, cy), ``size`` px square."""
    typeface.blend_mask(img, _mask(name, size, stroke), cx - size // 2, cy - size // 2, color, alpha)
