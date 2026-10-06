"""UI state owned by the application and read by the renderer / HUD."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from gestures.types import Gesture


class ViewMode(Enum):
    GESTURE = "Gesture mode"  # live camera feed is the backdrop; the scene is overlaid on it
    VISUALIZATION = "Visualization mode"  # immersive scene; the camera shrinks to a picture-in-picture


@dataclass(frozen=True)
class GestureGroup:
    """What the side panel lists: related gestures presented as one row."""

    id: str
    name: str
    action: str  # one-word effect shown on the row
    icon: str
    members: tuple[Gesture, ...]
    how: str  # how to perform it (shown when the row is selected)


GESTURE_GROUPS: tuple[GestureGroup, ...] = (
    GestureGroup("palm", "Open palm", "Pause", "palm", (Gesture.OPEN_PALM,),
                 "Show all fingers. The scene slows to a near-stop. Hold it still for 2 seconds to reset the scene."),
    GestureGroup("point", "Point", "Select", "point", (Gesture.POINT,),
                 "Extend only your index finger. Hold it over an object for a moment to select it."),
    GestureGroup("pinch", "Pinch", "Grab", "pinch", (Gesture.PINCH, Gesture.PINCH_DRAG, Gesture.PINCH_ROTATE),
                 "Touch thumb and index finger over an object to grab it. Keep pinching to move or twist."),
    GestureGroup("fist", "Fist", "Standby", "fist", (Gesture.FIST,),
                 "Close your hand and hold for half a second to pause or resume all gestures."),
    GestureGroup("two", "Two fingers", "Switch", "two", (Gesture.TWO_FINGERS,),
                 "Raise index and middle fingers, then move up or down to step through objects."),
    GestureGroup("swipe", "Swipe", "Scene", "swipe", (Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT),
                 "Sweep an open hand quickly sideways to change scene."),
    GestureGroup("zoom", "Zoom", "Scale", "zoom", (Gesture.ZOOM_IN, Gesture.ZOOM_OUT),
                 "With an open hand, spread your fingers apart to zoom in, bring them together to zoom out."),
)
GROUP_BY_ID = {g.id: g for g in GESTURE_GROUPS}


def group_of(gesture: Gesture) -> GestureGroup | None:
    return next((g for g in GESTURE_GROUPS if gesture in g.members), None)


@dataclass
class UIState:
    mode: ViewMode = ViewMode.GESTURE
    show_help: bool = False
    show_debug: bool = False
    show_panel: bool = True
    selected_group: str | None = None  # the side-panel row whose details are open
    enabled: frozenset[Gesture] = field(default_factory=lambda: frozenset(g for g in Gesture if g is not Gesture.NONE))
    toast: str = ""
    toast_sub: str = ""
    toast_until: float = 0.0
    toast_start: float = 0.0

    def say(self, text: str, t: float, sub: str = "", duration: float = 1.6) -> None:
        self.toast, self.toast_sub, self.toast_start, self.toast_until = text, sub, t, t + duration

    def toggle_gesture(self, gesture: Gesture) -> None:
        self.enabled = self.enabled - {gesture} if gesture in self.enabled else self.enabled | {gesture}

    def toggle_group(self, group: GestureGroup) -> None:
        """Switch a whole row on, or off if any member is currently on."""
        members = frozenset(group.members)
        self.enabled = self.enabled - members if self.enabled & members else self.enabled | members

    def select_group(self, group_id: str) -> None:
        self.selected_group = None if self.selected_group == group_id else group_id


@dataclass(frozen=True)
class FrameStats:
    fps: float = 0.0
    frame_ms: float = 0.0
    track_ms: float = 0.0
    gesture_ms: float = 0.0
    render_ms: float = 0.0
    camera_status: str = "ok"
    simulated: bool = False


@dataclass(frozen=True)
class Hitbox:
    """A clickable rectangle in canvas pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int
    action: tuple

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1
