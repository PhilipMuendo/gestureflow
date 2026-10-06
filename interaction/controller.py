"""Interaction controller: gesture state in, scene commands out.

This is the layer that gives gestures their *meaning*.  It owns the interaction
state machine (hovering, grabbing, dragging, standby...) while the gesture engine
only reports what the hand is doing and the scene only knows how to carry out
commands.

======================  ==================================================
Gesture                 Command(s)
======================  ==================================================
Point                   cursor, HOVER; dwell on an object -> SELECT
Pinch (on an object)    GRAB; the object then follows the pinch (MOVE)
Pinch + move            MOVE; releasing while moving throws the object
Pinch + twist           ROTATE (translation is frozen while rotating)
Open palm               SET_PAUSED; hold steady -> RESET
Fist (hold)             SET_ACTIVE toggles engaged / standby
Two fingers up / down   CYCLE next / previous object
Swipe left / right      NEXT_SCENE / PREV_SCENE
Spread / close fingers  SCALE the grabbed/selected object (or the whole scene)
======================  ==================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from config import InteractionConfig
from gestures.types import Gesture, GestureState
from interaction.commands import Command, CommandType
from interaction.mapping import Margin, camera_to_scene
from tracking.filters import OneEuroFilter

DESELECT_DWELL_S = 1.0  # point at empty space this long to clear the selection


class Phase(Enum):
    STANDBY = "STANDBY"
    IDLE = "READY"
    POINTING = "POINTING"
    GRABBED = "GRABBED"
    DRAGGING = "DRAGGING"
    ROTATING = "ROTATING"
    ZOOMING = "ZOOMING"
    PAUSED = "PAUSED"
    SWITCHING = "SWITCHING"


class SceneQuery(Protocol):
    """The read-only view of the scene the controller needs."""

    @property
    def selected_id(self) -> int | None: ...

    def pick(self, x: float, y: float, slop: float = 1.4) -> int | None: ...

    def object_position(self, object_id: int) -> tuple[float, float] | None: ...

    def object_ids(self) -> list[int]: ...


@dataclass(frozen=True)
class InteractionState:
    """Snapshot consumed by the renderer / HUD."""

    active: bool = True
    phase: Phase = Phase.IDLE
    cursor: tuple[float, float] | None = None  # scene coordinates of the index fingertip
    pinch_point: tuple[float, float] | None = None
    pinching: bool = False
    hovered_id: int | None = None
    selected_id: int | None = None
    grabbed_id: int | None = None
    dwell_progress: float = 0.0  # 0..1 while pointing at an object
    fist_progress: float = 0.0  # 0..1 while holding a fist (toggle active)
    reset_progress: float = 0.0  # 0..1 while holding an open palm (reset scene)
    status: str = ""  # most recent command, empty once it has faded
    status_age: float = 0.0


class InteractionController:
    def __init__(self, scene: SceneQuery, cfg: InteractionConfig | None = None) -> None:
        self.scene = scene
        self.cfg = cfg or InteractionConfig()
        self.state = InteractionState()
        self._margin: Margin = self.cfg.active_margin
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        cfg = self.cfg
        self._cursor_filter = OneEuroFilter(cfg.cursor_min_cutoff, cfg.cursor_beta)
        self._pinch_filter = OneEuroFilter(cfg.cursor_min_cutoff, cfg.cursor_beta)
        self._last_cursor: np.ndarray | None = None
        self._last_pinch: np.ndarray | None = None
        self._t: float | None = None

        self._active = True
        self._hovered: int | None = None
        self._grabbed: int | None = None
        self._grab_offset = np.zeros(2)
        self._pinch_handled = False
        self._last_pinch_end = -math.inf
        self._pinch_velocity = np.zeros(2)
        self._prev_pinch: np.ndarray | None = None

        self._dwell_target: int | None = None
        self._dwell_s = 0.0
        self._empty_s = 0.0
        self._fist_fired = False
        self._fist_progress = 0.0
        self._palm_s = 0.0
        self._palm_reset_fired = False
        self._paused = False
        self._scroll_acc = 0.0

        self._status = ""
        self._status_t = -math.inf

    def set_margin(self, margin: Margin) -> None:
        """Change the camera region that maps to the viewport (call when the window is resized)."""
        self._margin = margin

    @property
    def active(self) -> bool:
        return self._active

    # ----------------------------------------------------------------- update
    def update(self, g: GestureState, t: float) -> list[Command]:
        """Consume one frame of gesture state; return the commands to apply."""
        cfg = self.cfg
        dt = 0.0 if self._t is None else min(max(t - self._t, 0.0), 0.1)
        self._t = t
        cmds: list[Command] = []

        if not g.hand_present or g.features is None:
            self._on_hand_lost(cmds, t)
            self._publish(t, Phase.STANDBY if not self._active else Phase.IDLE)
            return cmds

        cursor = self._track(self._cursor_filter, g.features.index_tip, "cursor", t)
        pinch = self._track(self._pinch_filter, g.features.pinch_point, "pinch", t)
        static = g.static

        self._update_fist_toggle(g, cmds, t)
        if not self._active:
            self._release(cmds, t, throw=False)
            self._set_hover(None, cmds)
            self._publish(t, Phase.STANDBY, cursor=cursor, pinch=pinch)
            return cmds

        phase = Phase.IDLE
        hovered: int | None = None

        # --- one-shot gestures
        for event in g.events:
            if event is Gesture.SWIPE_LEFT:
                self._release(cmds, t, throw=False)
                cmds.append(Command(CommandType.NEXT_SCENE, label="NEXT SCENE"))
                self._say("NEXT SCENE", t)
                phase = Phase.SWITCHING
            elif event is Gesture.SWIPE_RIGHT:
                self._release(cmds, t, throw=False)
                cmds.append(Command(CommandType.PREV_SCENE, label="PREVIOUS SCENE"))
                self._say("PREVIOUS SCENE", t)
                phase = Phase.SWITCHING

        # --- pose-driven behaviour
        if static is Gesture.POINT:
            phase = Phase.POINTING
            hovered = self.scene.pick(cursor[0], cursor[1], cfg.pick_slop)
            self._update_dwell(hovered, dt, cmds, t)
        else:
            self._dwell_target, self._dwell_s, self._empty_s = None, 0.0, 0.0

        if static is Gesture.PINCH:
            phase, hovered = self._update_pinch(g, pinch, dt, cmds, t)
        else:
            self._release(cmds, t, throw=True)
            self._pinch_handled = False

        if static is Gesture.OPEN_PALM:
            phase = Phase.PAUSED
            self._update_palm(g, dt, cmds, t)
        else:
            self._leave_palm(cmds)

        if static is Gesture.TWO_FINGERS:
            phase = Phase.SWITCHING
            self._update_scroll(g, cmds, t)
        else:
            self._scroll_acc = 0.0

        if g.dynamic in (Gesture.ZOOM_IN, Gesture.ZOOM_OUT):
            phase = Phase.ZOOMING
            self._update_zoom(g, cmds, t)

        self._set_hover(hovered, cmds)
        self._publish(t, phase, cursor=cursor, pinch=pinch, g=g)
        return cmds

    # -------------------------------------------------------------- tracking
    def _track(self, flt: OneEuroFilter, camera_xy: np.ndarray, which: str, t: float) -> np.ndarray:
        """Camera coords -> scene coords -> smoothed.  A sudden big jump (hand
        re-acquired elsewhere) restarts the filter instead of sweeping across."""
        point = camera_to_scene(camera_xy, self._margin)
        last = self._last_cursor if which == "cursor" else self._last_pinch
        if last is not None and float(np.linalg.norm(point - last)) > self.cfg.max_cursor_step:
            flt.reset()
        smoothed = flt(point, t)
        if which == "cursor":
            self._last_cursor = smoothed
        else:
            self._last_pinch = smoothed
        return smoothed

    def _on_hand_lost(self, cmds: list[Command], t: float) -> None:
        self._release(cmds, t, throw=False)
        self._leave_palm(cmds)
        self._set_hover(None, cmds)
        for flt in (self._cursor_filter, self._pinch_filter):
            flt.reset()
        self._last_cursor = self._last_pinch = None
        self._pinch_handled = False
        self._dwell_target, self._dwell_s, self._empty_s = None, 0.0, 0.0
        self._scroll_acc = 0.0
        self._fist_fired = False

    # -------------------------------------------------------------- behaviours
    def _update_fist_toggle(self, g: GestureState, cmds: list[Command], t: float) -> None:
        """Hold a fist for ``fist_toggle_hold_s`` to flip between engaged and standby.

        Closing the hand into a fist is also how a grab usually ends, so the hold clock does
        not start until ``fist_after_pinch_lockout_s`` after the last pinch.  Releasing an
        object with a fist therefore never switches the interface off by accident.
        """
        cfg = self.cfg
        if g.static is not Gesture.FIST or self._grabbed is not None:
            self._fist_progress = 0.0
            if g.static is not Gesture.FIST:
                self._fist_fired = False
            return
        held = min(g.hold_s, t - (self._last_pinch_end + cfg.fist_after_pinch_lockout_s))
        self._fist_progress = min(1.0, max(0.0, held) / cfg.fist_toggle_hold_s)
        if self._fist_progress >= 1.0 and not self._fist_fired:
            self._fist_fired = True
            self._active = not self._active
            text = "ACTIVATED" if self._active else "STANDBY"
            cmds.append(Command(CommandType.SET_ACTIVE, value=1.0 if self._active else 0.0, label=text))
            self._say(text, t)

    def _update_dwell(self, hovered: int | None, dt: float, cmds: list[Command], t: float) -> None:
        cfg = self.cfg
        if hovered is None:
            self._dwell_target, self._dwell_s = None, 0.0
            self._empty_s += dt
            if self._empty_s >= DESELECT_DWELL_S and self.scene.selected_id is not None:
                cmds.append(Command(CommandType.DESELECT, label="DESELECT"))
                self._say("DESELECT", t)
            return
        self._empty_s = 0.0
        if hovered != self._dwell_target:
            self._dwell_target, self._dwell_s = hovered, 0.0
        else:
            self._dwell_s += dt
        if self._dwell_s >= cfg.select_dwell_s and self.scene.selected_id != hovered:
            cmds.append(Command(CommandType.SELECT, object_id=hovered, label="SELECT"))
            self._say("SELECT", t)

    def _update_pinch(
        self, g: GestureState, pinch: np.ndarray, dt: float, cmds: list[Command], t: float
    ) -> tuple[Phase, int | None]:
        cfg = self.cfg
        enabled = g.enabled
        phase = Phase.IDLE
        hovered: int | None = None

        if self._grabbed is None:
            hovered = self.scene.pick(pinch[0], pinch[1], cfg.grab_slop)  # preview what a pinch would grab
            if not self._pinch_handled:
                self._pinch_handled = True  # one attempt per pinch
                if hovered is None:
                    cmds.append(Command(CommandType.GRAB_MISS, position=(float(pinch[0]), float(pinch[1]))))
                else:
                    pos = self.scene.object_position(hovered)
                    if pos is not None:
                        self._grabbed = hovered
                        self._grab_offset = np.asarray(pos) - pinch
                        self._pinch_velocity = np.zeros(2)
                        self._prev_pinch = None
                        cmds.append(Command(CommandType.GRAB, object_id=hovered, position=tuple(pinch), label="GRAB"))
                        self._say("GRAB", t)
            if self._grabbed is not None:
                phase = Phase.GRABBED
            return phase, hovered

        # holding an object ------------------------------------------------
        hovered = self._grabbed
        phase = Phase.GRABBED
        if self._prev_pinch is not None and dt > 1e-4:
            self._pinch_velocity = 0.6 * self._pinch_velocity + 0.4 * (pinch - self._prev_pinch) / dt
        self._prev_pinch = pinch.copy()

        rotating = g.dynamic is Gesture.PINCH_ROTATE and Gesture.PINCH_ROTATE in enabled
        if rotating:
            phase = Phase.ROTATING
            delta = math.radians(g.motion.roll_delta_deg) * cfg.rotate_gain
            if delta != 0.0:
                cmds.append(Command(CommandType.ROTATE, object_id=self._grabbed, value=delta, label="ROTATE"))
            self._say("ROTATE", t)
            pos = self.scene.object_position(self._grabbed)
            if pos is not None:  # keep the object where it is and re-anchor so there is no jump afterwards
                self._grab_offset = np.asarray(pos) - pinch
        elif Gesture.PINCH_DRAG in enabled:
            target = np.clip(pinch + self._grab_offset, 0.02, 0.98)
            cmds.append(Command(CommandType.MOVE, object_id=self._grabbed, position=(float(target[0]), float(target[1]))))
            if g.dynamic is Gesture.PINCH_DRAG:
                phase = Phase.DRAGGING
                self._say("DRAG", t)
        return phase, hovered

    def _release(self, cmds: list[Command], t: float, throw: bool) -> None:
        if self._grabbed is None:
            return
        velocity = tuple(float(x) for x in self._pinch_velocity * self.cfg.throw_velocity_gain) if throw else (0.0, 0.0)
        cmds.append(Command(CommandType.RELEASE, object_id=self._grabbed, velocity=velocity, label="RELEASE"))
        self._say("RELEASE", t)
        self._grabbed = None
        self._last_pinch_end = t
        self._pinch_velocity = np.zeros(2)
        self._prev_pinch = None

    def _update_palm(self, g: GestureState, dt: float, cmds: list[Command], t: float) -> None:
        cfg = self.cfg
        if not self._paused:
            self._paused = True
            cmds.append(Command(CommandType.SET_PAUSED, value=1.0, label="PAUSE"))
            self._say("PAUSE", t)
        zooming = g.dynamic in (Gesture.ZOOM_IN, Gesture.ZOOM_OUT)
        steady = g.motion.speed < cfg.palm_steady_speed and not zooming
        if steady and not self._palm_reset_fired:
            self._palm_s += dt
        else:
            self._palm_s = max(0.0, self._palm_s - 2.0 * dt)
        if self._palm_s >= cfg.palm_reset_hold_s and not self._palm_reset_fired:
            self._palm_reset_fired = True
            self._palm_s = 0.0
            cmds.append(Command(CommandType.RESET, label="RESET SCENE"))
            self._say("RESET SCENE", t)
            self._grabbed = None

    def _leave_palm(self, cmds: list[Command]) -> None:
        if self._paused:
            self._paused = False
            cmds.append(Command(CommandType.SET_PAUSED, value=0.0, label="RESUME"))
        self._palm_s = 0.0
        self._palm_reset_fired = False

    def _update_scroll(self, g: GestureState, cmds: list[Command], t: float) -> None:
        step = self.cfg.scroll_step
        self._scroll_acc += g.motion.scroll_delta
        while self._scroll_acc >= step:  # hand moved up -> next object
            self._scroll_acc -= step
            cmds.append(Command(CommandType.CYCLE, value=1.0, label="NEXT OBJECT"))
            self._say("NEXT OBJECT", t)
        while self._scroll_acc <= -step:
            self._scroll_acc += step
            cmds.append(Command(CommandType.CYCLE, value=-1.0, label="PREVIOUS OBJECT"))
            self._say("PREVIOUS OBJECT", t)

    def _update_zoom(self, g: GestureState, cmds: list[Command], t: float) -> None:
        delta = g.motion.spread_delta
        if abs(delta) < 1e-5:
            return
        factor = math.exp(self.cfg.zoom_gain * delta)
        target = self._grabbed if self._grabbed is not None else self.scene.selected_id
        cmds.append(Command(CommandType.SCALE, object_id=target, value=factor, label="ZOOM"))
        self._say("ZOOM IN" if g.dynamic is Gesture.ZOOM_IN else "ZOOM OUT", t)

    # --------------------------------------------------------------- publishing
    def _set_hover(self, hovered: int | None, cmds: list[Command]) -> None:
        if hovered != self._hovered:
            self._hovered = hovered
            cmds.append(Command(CommandType.HOVER, object_id=hovered))

    def _say(self, text: str, t: float) -> None:
        self._status, self._status_t = text, t

    def _publish(
        self,
        t: float,
        phase: Phase,
        cursor: np.ndarray | None = None,
        pinch: np.ndarray | None = None,
        g: GestureState | None = None,
    ) -> None:
        age = t - self._status_t
        expired = age > self.cfg.command_display_s
        dwell = 0.0
        if self._dwell_target is not None:
            dwell = min(1.0, self._dwell_s / self.cfg.select_dwell_s)
        self.state = InteractionState(
            active=self._active,
            phase=phase if self._active else Phase.STANDBY,
            cursor=None if cursor is None else (float(cursor[0]), float(cursor[1])),
            pinch_point=None if pinch is None else (float(pinch[0]), float(pinch[1])),
            pinching=g is not None and g.static is Gesture.PINCH,
            hovered_id=self._hovered,
            selected_id=self.scene.selected_id,
            grabbed_id=self._grabbed,
            dwell_progress=dwell,
            fist_progress=self._fist_progress,
            reset_progress=min(1.0, self._palm_s / self.cfg.palm_reset_hold_s),
            status="" if expired else self._status,
            status_age=0.0 if expired else age,
        )
