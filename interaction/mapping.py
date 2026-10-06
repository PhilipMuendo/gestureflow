"""Mapping between camera-frame coordinates and scene coordinates.

The viewport shows only a centred part of the camera frame (the *active region*).  Its size
is set by ``margin``: a float (same on both axes) or an ``(mx, my)`` pair.  With a viewport
that is not the camera's aspect ratio the two margins differ (see :func:`view_margins`).
"""

from __future__ import annotations

import numpy as np

Margin = float | tuple[float, float]


def view_margins(view_aspect: float, camera_aspect: float, base: float) -> tuple[float, float]:
    """Per-axis margins so the visible camera region has the viewport's aspect ratio.

    The camera image is *cropped to cover* the viewport (no stretching, no letterbox bars).
    ``base`` is the margin on the axis that has room to spare, which keeps the screen
    corners reachable without leaving the camera's view.
    """
    visible = 1.0 - 2.0 * base
    if view_aspect >= camera_aspect:  # viewport is wider than the camera: crop the height
        width_frac, height_frac = visible, visible * camera_aspect / view_aspect
    else:  # viewport is narrower: crop the width
        width_frac, height_frac = visible * view_aspect / camera_aspect, visible
    return (1.0 - width_frac) / 2.0, (1.0 - height_frac) / 2.0


def camera_to_view(xy: np.ndarray, margin: Margin) -> np.ndarray:
    """Unclipped camera -> view mapping.  The viewport shows exactly the active region of
    the camera frame, so drawing the hand skeleton through this mapping keeps it aligned
    with the hand in the video *and* with the cursor."""
    m = np.asarray(margin, dtype=float)
    return (np.asarray(xy, dtype=float) - m) / (1.0 - 2.0 * m)


def camera_to_scene(xy: np.ndarray, margin: Margin) -> np.ndarray:
    """Map normalised camera coordinates to normalised scene coordinates.

    The hand is hard to track near the image border (it is partly cut off), so only the
    central region maps onto the full scene; a small overshoot is allowed so the cursor can
    rest against the edge.
    """
    return np.clip(camera_to_view(xy, margin), -0.03, 1.03)


def scene_to_camera(xy: np.ndarray, margin: Margin) -> np.ndarray:
    """Inverse of :func:`camera_to_scene` (used by the simulator to aim the hand)."""
    m = np.asarray(margin, dtype=float)
    return m + np.asarray(xy, dtype=float) * (1.0 - 2.0 * m)
