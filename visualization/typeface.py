"""Anti-aliased UI typography on top of OpenCV frames.

OpenCV's built-in Hershey fonts cannot produce a modern sans-serif, so text is rasterised
with Pillow (FreeType) into small alpha masks, cached, and alpha-blended onto the frame.
Only the text *region* is touched, and rendered strings are cached, so per-frame cost is a
handful of tiny blends.

Font resolution: the bundled Inter (``assets/fonts``, SIL Open Font License) -> common system
UI fonts -> Pillow's default.  Weights: regular / medium / semibold.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import PROJECT_ROOT

Color = tuple[int, int, int]  # BGR
FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
_WEIGHT_FILES = {"regular": "Inter-Regular.otf", "medium": "Inter-Medium.otf", "semibold": "Inter-SemiBold.otf"}
_SYSTEM_FALLBACKS = {
    "regular": ["segoeui.ttf", "SFNS.ttf", "Helvetica.ttc", "Roboto-Regular.ttf", "DejaVuSans.ttf", "arial.ttf"],
    "medium": ["segoeui.ttf", "SFNS.ttf", "Helvetica.ttc", "Roboto-Medium.ttf", "DejaVuSans.ttf", "arial.ttf"],
    "semibold": ["seguisb.ttf", "segoeuib.ttf", "SFNS.ttf", "Helvetica.ttc", "Roboto-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
}
_SYSTEM_DIRS = (
    [Path("C:/Windows/Fonts")] if sys.platform.startswith("win")
    else [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"]
    if sys.platform == "darwin"
    else [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"]
)


def _find_font(weight: str) -> str | None:
    bundled = FONT_DIR / _WEIGHT_FILES[weight]
    if bundled.exists():
        return str(bundled)
    for name in _SYSTEM_FALLBACKS[weight]:
        for directory in _SYSTEM_DIRS:
            for candidate in ([directory / name] if directory.exists() else []):
                if candidate.exists():
                    return str(candidate)
            if directory.exists():
                found = next(directory.rglob(name), None)
                if found is not None:
                    return str(found)
    return None


@lru_cache(maxsize=64)
def _font(weight: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _find_font(weight)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


@lru_cache(maxsize=1024)
def _mask(text: str, weight: str, size: int, tracking: int) -> np.ndarray:
    """Alpha mask (h, w) uint8 of ``text``.  ``tracking`` adds letter-spacing in pixels."""
    font = _font(weight, size)
    if tracking:
        widths = [font.getlength(ch) + tracking for ch in text]
        w = int(sum(widths)) + 2
    else:
        w = int(font.getlength(text)) + 2
    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (size, size // 4)
    h = ascent + descent + 2
    img = Image.new("L", (max(w, 1), h), 0)
    draw = ImageDraw.Draw(img)
    if tracking:
        x = 0.0
        for ch, adv in zip(text, widths):
            draw.text((x, 0), ch, font=font, fill=255)
            x += adv
    else:
        draw.text((0, 0), text, font=font, fill=255)
    return np.asarray(img)


def measure(text: str, size: int, weight: str = "regular", tracking: int = 0) -> tuple[int, int]:
    """(width, height) in pixels of rendered ``text``."""
    m = _mask(text, weight, size, tracking)
    return m.shape[1], m.shape[0]


def blend_mask(img: np.ndarray, mask: np.ndarray, x: int, y: int, color: Color, alpha: float = 1.0) -> None:
    """Alpha-blend ``color`` through ``mask`` (uint8) onto ``img`` with the mask's top-left at (x, y)."""
    H, W = img.shape[:2]
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.uint16)
    if alpha < 1.0:
        m = (m * alpha).astype(np.uint16)
    roi = img[y0:y1, x0:x1]
    inv = 255 - m
    col = np.asarray(color, np.uint16)
    roi[:] = ((roi.astype(np.uint16) * inv[..., None] + col * m[..., None]) // 255).astype(np.uint8)


def text(
    img: np.ndarray,
    s: str,
    x: int,
    y: int,
    size: int = 13,
    color: Color = (235, 233, 231),
    weight: str = "regular",
    anchor: str = "l",
    alpha: float = 1.0,
    tracking: int = 0,
) -> int:
    """Draw ``s`` so that its *vertical centre line* is at ``y``.  ``anchor``: l / c / r.

    Returns the rendered width.
    """
    if not s:
        return 0
    mask = _mask(s, weight, size, tracking)
    h, w = mask.shape[:2]
    left = x - (w // 2 if anchor == "c" else w if anchor == "r" else 0)
    blend_mask(img, mask, left, y - int(h * 0.52), color, alpha)
    return w
