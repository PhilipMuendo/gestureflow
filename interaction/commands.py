"""The command vocabulary between the interaction controller and the scene.

The controller never touches the scene directly: it emits small immutable
:class:`Command` objects and the application hands them to the scene.  That keeps
gesture logic testable (assert on the commands) and the scene reusable (drive it
from a mouse, a test, or a replayed recording).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class CommandType(Enum):
    HOVER = auto()  # object_id (None clears): pointer is over an object
    SELECT = auto()  # object_id
    DESELECT = auto()
    GRAB = auto()  # object_id, position = where the pinch is
    MOVE = auto()  # position = new target for the grabbed object
    RELEASE = auto()  # velocity = throw velocity (scene units / s)
    ROTATE = auto()  # object_id, value = radians about the view axis
    SCALE = auto()  # object_id (None = whole scene), value = multiplicative factor
    CYCLE = auto()  # value = +1 / -1: select next / previous object
    NEXT_SCENE = auto()
    PREV_SCENE = auto()
    RESET = auto()  # reset the current scene
    SET_PAUSED = auto()  # value = 1.0 paused, 0.0 running
    SET_ACTIVE = auto()  # value = 1.0 engaged, 0.0 standby
    GRAB_MISS = auto()  # pinch closed on empty space (visual feedback only)


@dataclass(frozen=True)
class Command:
    type: CommandType
    object_id: int | None = None
    position: tuple[float, float] | None = None
    velocity: tuple[float, float] | None = None
    value: float = 0.0
    label: str = ""  # human-readable text for the HUD
