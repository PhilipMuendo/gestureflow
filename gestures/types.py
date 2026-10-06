"""Vocabulary shared by the gesture engine, the controller and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from gestures.features import HandFeatures


class Gesture(str, Enum):
    NONE = "none"
    # static: recognised from a single frame's hand shape
    OPEN_PALM = "open_palm"
    POINT = "point"
    PINCH = "pinch"
    FIST = "fist"
    TWO_FINGERS = "two_fingers"
    # dynamic: recognised from motion over several frames
    PINCH_DRAG = "pinch_drag"
    PINCH_ROTATE = "pinch_rotate"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"

    @property
    def is_dynamic(self) -> bool:
        return self in DYNAMIC_GESTURES

    @property
    def is_static(self) -> bool:
        return self in STATIC_GESTURES


STATIC_GESTURES: tuple[Gesture, ...] = (
    Gesture.OPEN_PALM,
    Gesture.POINT,
    Gesture.PINCH,
    Gesture.FIST,
    Gesture.TWO_FINGERS,
)
DYNAMIC_GESTURES: tuple[Gesture, ...] = (
    Gesture.PINCH_DRAG,
    Gesture.PINCH_ROTATE,
    Gesture.SWIPE_LEFT,
    Gesture.SWIPE_RIGHT,
    Gesture.ZOOM_IN,
    Gesture.ZOOM_OUT,
)
# Order shown in the UI legend.
LEGEND_ORDER: tuple[Gesture, ...] = (
    Gesture.OPEN_PALM,
    Gesture.POINT,
    Gesture.PINCH,
    Gesture.PINCH_DRAG,
    Gesture.PINCH_ROTATE,
    Gesture.FIST,
    Gesture.TWO_FINGERS,
    Gesture.SWIPE_LEFT,
    Gesture.SWIPE_RIGHT,
    Gesture.ZOOM_IN,
    Gesture.ZOOM_OUT,
)


@dataclass(frozen=True)
class GestureInfo:
    label: str  # short display name
    action: str  # what it does in the interface


GESTURE_INFO: Mapping[Gesture, GestureInfo] = MappingProxyType(
    {
        Gesture.NONE: GestureInfo("No gesture", "-"),
        Gesture.OPEN_PALM: GestureInfo("Open palm", "Pause  |  hold to reset"),
        Gesture.POINT: GestureInfo("Point", "Move cursor / select"),
        Gesture.PINCH: GestureInfo("Pinch", "Grab object"),
        Gesture.PINCH_DRAG: GestureInfo("Pinch + move", "Drag object"),
        Gesture.PINCH_ROTATE: GestureInfo("Pinch + twist", "Rotate object"),
        Gesture.FIST: GestureInfo("Fist", "Hold: activate / standby"),
        Gesture.TWO_FINGERS: GestureInfo("Two fingers", "Move up/down: switch object"),
        Gesture.SWIPE_LEFT: GestureInfo("Swipe left", "Next scene"),
        Gesture.SWIPE_RIGHT: GestureInfo("Swipe right", "Previous scene"),
        Gesture.ZOOM_IN: GestureInfo("Spread fingers", "Zoom in"),
        Gesture.ZOOM_OUT: GestureInfo("Fingers together", "Zoom out"),
    }
)


@dataclass(frozen=True)
class MotionState:
    """Per-frame motion measurements, all in palm lengths unless noted."""

    velocity: tuple[float, float] = (0.0, 0.0)  # palm centre, palm lengths / s
    speed: float = 0.0
    roll_deg: float = 0.0  # absolute wrist roll
    roll_delta_deg: float = 0.0  # change since the previous frame (wrapped)
    roll_rate: float = 0.0  # deg / s, smoothed
    spread: float = 0.0
    spread_delta: float = 0.0  # change of the smoothed spread since the previous frame
    spread_rate: float = 0.0  # per second, over the zoom window
    scroll_delta: float = 0.0  # vertical travel of the two-finger tip this frame (palm lengths, + = up)


@dataclass(frozen=True)
class GestureState:
    """Output of the gesture engine for one frame.

    ``gesture`` is the single most meaningful label: an active dynamic gesture wins
    over the underlying static pose.  ``events`` holds one-shot gestures (swipes)
    that fired on *this* frame.
    """

    timestamp: float = 0.0
    hand_present: bool = False
    stale: bool = False  # hand missing but still inside the grace period
    gesture: Gesture = Gesture.NONE
    confidence: float = 0.0
    static: Gesture = Gesture.NONE
    static_confidence: float = 0.0
    dynamic: Gesture = Gesture.NONE
    dynamic_confidence: float = 0.0
    hold_s: float = 0.0  # how long the current static gesture has been held
    events: tuple[Gesture, ...] = ()
    motion: MotionState = field(default_factory=MotionState)
    features: HandFeatures | None = None
    scores: Mapping[Gesture, float] = field(default_factory=dict)  # smoothed static scores (debug)
    handedness: str = ""
    tracking_score: float = 0.0
    enabled: frozenset[Gesture] = frozenset(Gesture)  # gestures the user has switched on

    # convenient screen-space anchors (normalised frame coordinates, mirrored view)
    @property
    def cursor(self) -> np.ndarray | None:
        return None if self.features is None else self.features.index_tip

    @property
    def pinch_point(self) -> np.ndarray | None:
        return None if self.features is None else self.features.pinch_point
