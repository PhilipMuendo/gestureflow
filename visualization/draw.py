"""Small OpenCV drawing helpers shared by the renderer and the HUD (BGR colours)."""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from visualization.theme import Color

AA = cv2.LINE_AA


def scale_color(color: Color, k: float) -> Color:
    return (int(min(255, color[0] * k)), int(min(255, color[1] * k)), int(min(255, color[2] * k)))


def mix_color(a: Color, b: Color, t: float) -> Color:
    t = min(max(t, 0.0), 1.0)
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


@lru_cache(maxsize=64)
def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    r = max(0, min(r, w // 2, h // 2))
    cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, AA)
    return mask


def fill(img: np.ndarray, x: int, y: int, w: int, h: int, color: Color, alpha: float = 1.0, radius: int = 0) -> None:
    """Alpha-blended (optionally rounded) rectangle; only the ROI is touched."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    roi = img[y0:y1, x0:x1]
    if radius <= 0 and alpha >= 1.0:
        roi[:] = color
        return
    tint = np.empty_like(roi)
    tint[:] = color
    blended = cv2.addWeighted(roi, 1.0 - alpha, tint, alpha, 0)
    if radius > 0:
        cv2.copyTo(blended, _rounded_mask(w, h, radius)[y0 - y : y1 - y, x0 - x : x1 - x], roi)
    else:
        roi[:] = blended


def glass(img: np.ndarray, x: int, y: int, w: int, h: int, color: Color, alpha: float = 0.78, radius: int = 6,
          border: Color | None = None, blur: float = 7.0) -> None:
    """Translucent panel over a video: the backdrop is blurred, then tinted dark."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    # blur at 1/4 resolution: visually identical for a soft backdrop, ~10x cheaper
    small = cv2.resize(roi, (max(1, (x1 - x0) // 4), max(1, (y1 - y0) // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, blur / 4.0))
    soft = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    tint = np.empty_like(soft)
    tint[:] = color
    blended = cv2.addWeighted(soft, 1.0 - alpha, tint, alpha, 0)
    cv2.copyTo(blended, _rounded_mask(w, h, radius)[y0 - y : y1 - y, x0 - x : x1 - x], roi)
    if border is not None:
        outline(img, x, y, w, h, border, radius)


def outline(img: np.ndarray, x: int, y: int, w: int, h: int, color: Color, radius: int = 0) -> None:
    if radius <= 0:
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, 1, AA)
        return
    r = radius
    cv2.line(img, (x + r, y), (x + w - r - 1, y), color, 1, AA)
    cv2.line(img, (x + r, y + h - 1), (x + w - r - 1, y + h - 1), color, 1, AA)
    cv2.line(img, (x, y + r), (x, y + h - r - 1), color, 1, AA)
    cv2.line(img, (x + w - 1, y + r), (x + w - 1, y + h - r - 1), color, 1, AA)
    for cx, cy, a0 in ((x + r, y + r, 180), (x + w - r - 1, y + r, 270), (x + w - r - 1, y + h - r - 1, 0), (x + r, y + h - r - 1, 90)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, color, 1, AA)


def arc(img: np.ndarray, center: tuple[int, int], radius: int, start_deg: float, end_deg: float, color: Color, thickness: int = 1) -> None:
    cv2.ellipse(img, center, (radius, radius), 0, start_deg, end_deg, color, thickness, AA)


def progress_arc(img: np.ndarray, center: tuple[int, int], radius: int, fraction: float, color: Color, thickness: int = 2) -> None:
    """Ring starting at 12 o'clock covering ``fraction`` of a circle."""
    fraction = min(max(fraction, 0.0), 1.0)
    if fraction > 0.0:
        cv2.ellipse(img, center, (radius, radius), -90, 0, 360.0 * fraction, color, thickness, AA)


def brackets(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Color, length: int = 10, thickness: int = 1) -> None:
    """Four corner brackets around a rectangle: the 'detected object' mark."""
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + sx * length, cy), color, thickness, AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * length), color, thickness, AA)


def reticle(img: np.ndarray, center: tuple[int, int], color: Color, radius: int = 9) -> None:
    """Small precise pointer: a thin ring with a gap-cut crosshair and a centre dot."""
    cx, cy = center
    cv2.circle(img, center, radius, color, 1, AA)
    for ang in (0, 90, 180, 270):
        a = math.radians(ang)
        cv2.line(img, (int(cx + math.cos(a) * (radius - 4)), int(cy + math.sin(a) * (radius - 4))),
                 (int(cx + math.cos(a) * (radius + 5)), int(cy + math.sin(a) * (radius + 5))), color, 1, AA)
    cv2.circle(img, center, 1, color, -1, AA)
