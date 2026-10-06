"""Dynamic (multi-frame) gesture recognition.

A single frame cannot tell a *swipe* from a hand that is merely at the left of the
screen: motion only exists across time.  This module keeps a short history of
hand samples and derives velocity, roll rate and finger-spread rate from it.

Recognisers (all driven by :class:`DynamicGestureRecognizer`):

========== ======================================================== =============
Gesture    Signal                                                   Output kind
========== ======================================================== =============
swipe      palm-centre velocity + net travel, open hand             one-shot event
zoom       rate of change of the finger-spread metric, open hand    continuous
rotate     wrist-roll rate while pinching, little translation       continuous
drag       pinch-point travel/speed while pinching                  continuous
scroll     vertical motion of two raised fingertips                 continuous delta
========== ======================================================== =============

All distances are in **palm lengths**, so the thresholds hold at any distance from
the camera.  Time always comes from the caller (frame timestamps), never from the
wall clock, which keeps the logic deterministic and unit-testable.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from config import DynamicGestureConfig
from gestures.features import HandFeatures
from gestures.types import Gesture, MotionState


def wrap_degrees(angle: float) -> float:
    """Wrap an angle difference into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class _Sample:
    t: float
    center: np.ndarray  # palm centre, isotropic frame units
    pinch_point: np.ndarray  # isotropic
    roll: float  # unwrapped degrees
    spread: float  # smoothed spread metric
    min_extension: float  # extension of the least-extended of the four fingers
    mean_extension: float  # mean extension of the four fingers


@dataclass(frozen=True)
class DynamicResult:
    active: Gesture = Gesture.NONE  # continuous dynamic gesture currently in progress
    confidence: float = 0.0
    events: tuple[Gesture, ...] = ()  # one-shot gestures fired this frame
    motion: MotionState = MotionState()


class _SwipePhase(Enum):
    IDLE = 0
    TRACKING = 1
    REFRACTORY = 2


class DynamicGestureRecognizer:
    def __init__(self, cfg: DynamicGestureConfig | None = None) -> None:
        self.cfg = cfg or DynamicGestureConfig()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._history: deque[_Sample] = deque()
        self._palm_ref: float | None = None  # smoothed palm size (isotropic units)
        self._spread_smooth: float | None = None
        self._roll_unwrapped: float | None = None
        self._last_roll_raw: float | None = None
        self._last_spread_smooth: float | None = None
        self._prev_two_tip: np.ndarray | None = None

        self._swipe_phase = _SwipePhase.IDLE
        self._swipe_start: np.ndarray | None = None
        self._swipe_start_t = 0.0
        self._swipe_path = 0.0
        self._swipe_prev: np.ndarray | None = None
        self._swipe_cooldown_until = -math.inf
        self._last_swipe: tuple[Gesture, float] | None = None

        self._zoom_active = Gesture.NONE
        self._zoom_last_active = -math.inf

        self._pinch_active = False
        self._pinch_anchor: np.ndarray | None = None
        self._rotate_active = False
        self._rotate_last_active = -math.inf
        self._drag_active = False
        self._drag_last_active = -math.inf

    # ------------------------------------------------------------------ update
    def update(self, f: HandFeatures, t: float, static: Gesture) -> DynamicResult:
        """Feed one frame and return what is happening *now*.

        Args:
            f: features of the current frame.
            t: frame timestamp in seconds.
            static: the stabilised static gesture (decides which dynamic
                recognisers are allowed to run: drag/rotate need a pinch, swipe
                and zoom need an open hand).
        """
        cfg = self.cfg
        aspect = np.array([f.aspect, 1.0])
        # Smooth the palm size so scale noise does not leak into every velocity.
        self._palm_ref = f.palm_size if self._palm_ref is None else 0.7 * self._palm_ref + 0.3 * f.palm_size
        palm = max(self._palm_ref, 1e-6)

        spread = f.spread if self._spread_smooth is None else 0.5 * self._spread_smooth + 0.5 * f.spread
        self._spread_smooth = spread

        roll_delta = 0.0
        if self._last_roll_raw is None or self._roll_unwrapped is None:
            self._roll_unwrapped = f.roll_deg
        else:
            roll_delta = wrap_degrees(f.roll_deg - self._last_roll_raw)
            self._roll_unwrapped += roll_delta
        self._last_roll_raw = f.roll_deg

        sample = _Sample(
            t=t,
            center=f.palm_center_iso.copy(),
            pinch_point=f.pinch_point * aspect,
            roll=self._roll_unwrapped,
            spread=spread,
            min_extension=f.min_extension,
            mean_extension=f.mean_extension,
        )
        self._history.append(sample)
        while self._history and t - self._history[0].t > cfg.history_s:
            self._history.popleft()

        velocity = self._velocity(0.15, palm)
        speed = float(np.hypot(*velocity))
        pinching = static is Gesture.PINCH

        events = self._update_swipe(sample, velocity, speed, palm, f, pinching)
        zoom, spread_delta, spread_rate = self._update_zoom(sample, t, pinching)
        rotate, drag, roll_rate = self._update_pinch(sample, t, palm, pinching)
        scroll = self._update_scroll(f, palm, static)

        # Only one continuous dynamic label is reported; pinch gestures and zoom are
        # mutually exclusive by construction (zoom requires an open hand).
        active, confidence = Gesture.NONE, 0.0
        if rotate:
            active, confidence = Gesture.PINCH_ROTATE, min(1.0, 0.6 + abs(roll_rate) / (4 * cfg.rotate_enter_rate))
        elif drag:
            active, confidence = Gesture.PINCH_DRAG, min(1.0, 0.6 + speed / 4.0)
        elif zoom is not Gesture.NONE:
            active, confidence = zoom, min(1.0, 0.6 + abs(spread_rate) / (4 * cfg.zoom_enter_rate))

        motion = MotionState(
            velocity=velocity,
            speed=speed,
            roll_deg=f.roll_deg,
            roll_delta_deg=roll_delta,
            roll_rate=roll_rate,
            spread=spread,
            spread_delta=spread_delta if zoom is not Gesture.NONE else 0.0,
            spread_rate=spread_rate,
            scroll_delta=scroll,
        )
        return DynamicResult(active, confidence, events, motion)

    # ---------------------------------------------------------------- helpers
    def _window(self, seconds: float) -> list[_Sample]:
        """Samples from the last ``seconds`` seconds (oldest first)."""
        if not self._history:
            return []
        t_now = self._history[-1].t
        return [s for s in self._history if t_now - s.t <= seconds]

    def _velocity(self, seconds: float, palm: float) -> tuple[float, float]:
        win = self._window(seconds)
        if len(win) < 2 or win[-1].t - win[0].t < 1e-3:
            return (0.0, 0.0)
        v = (win[-1].center - win[0].center) / (win[-1].t - win[0].t) / palm
        return float(v[0]), float(v[1])

    # ------------------------------------------------------------------ swipe
    def _update_swipe(
        self, s: _Sample, velocity: tuple[float, float], speed: float, palm: float, f: HandFeatures, pinching: bool
    ) -> tuple[Gesture, ...]:
        cfg = self.cfg
        open_hand = s.mean_extension >= cfg.swipe_min_open_extension and not pinching

        if self._swipe_phase is _SwipePhase.REFRACTORY:
            if s.t >= self._swipe_cooldown_until and speed < cfg.swipe_end_speed:
                self._swipe_phase = _SwipePhase.IDLE
            return ()

        if self._swipe_phase is _SwipePhase.IDLE:
            if open_hand and speed >= cfg.swipe_start_speed:
                win = self._window(0.15)
                self._swipe_phase = _SwipePhase.TRACKING
                self._swipe_start = win[0].center.copy()
                self._swipe_start_t = win[0].t
                self._swipe_prev = win[0].center.copy()
                self._swipe_path = 0.0
            else:
                return ()

        # TRACKING: accumulate path and test the stroke
        assert self._swipe_start is not None and self._swipe_prev is not None
        self._swipe_path += float(np.linalg.norm(s.center - self._swipe_prev))
        self._swipe_prev = s.center.copy()
        net = (s.center - self._swipe_start) / palm
        path = max(self._swipe_path / palm, 1e-6)
        dx, dy = float(net[0]), float(net[1])
        elapsed = s.t - self._swipe_start_t

        stroke_over = speed < cfg.swipe_end_speed or not open_hand
        if elapsed > cfg.swipe_max_duration_s:
            self._swipe_phase = _SwipePhase.IDLE
            return ()

        direction = Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT
        needed = cfg.swipe_min_distance
        if self._last_swipe is not None:
            last_dir, last_t = self._last_swipe
            if last_dir is not direction and s.t - last_t < cfg.swipe_return_window_s:
                needed *= cfg.swipe_return_penalty  # the "hand coming back" stroke needs to be more deliberate

        straight = math.hypot(dx, dy) / path
        if (
            abs(dx) >= needed
            and abs(dx) >= cfg.swipe_horizontal_ratio * abs(dy)
            and straight >= cfg.swipe_min_straightness
        ):
            self._swipe_phase = _SwipePhase.REFRACTORY
            self._swipe_cooldown_until = s.t + cfg.swipe_cooldown_s
            self._last_swipe = (direction, s.t)
            return (direction,)

        if stroke_over:
            self._swipe_phase = _SwipePhase.IDLE
        return ()

    # ------------------------------------------------------------------- zoom
    def _update_zoom(self, s: _Sample, t: float, pinching: bool) -> tuple[Gesture, float, float]:
        cfg = self.cfg
        win = self._window(cfg.zoom_window_s)
        prev_smooth = self._last_spread_smooth
        self._last_spread_smooth = s.spread
        delta = 0.0 if prev_smooth is None else s.spread - prev_smooth

        if len(win) < 3 or win[-1].t - win[0].t < 0.5 * cfg.zoom_window_s:
            return Gesture.NONE, 0.0, 0.0
        rate = (win[-1].spread - win[0].spread) / (win[-1].t - win[0].t)

        # A real zoom changes only the *spacing* between straight fingers.  So the whole
        # window must show an open hand whose finger shape is steady - otherwise
        # straightening a finger while relaxing a pinch (or a fist) into an open palm
        # would also move the fingertips apart and look like a zoom.
        extensions = [w.mean_extension for w in win]
        fully_open = (
            all(w.min_extension >= cfg.zoom_min_open_extension for w in win)
            and max(extensions) - min(extensions) <= cfg.zoom_max_extension_drift
            and not pinching
        )
        if not fully_open:
            self._zoom_active = Gesture.NONE
            return Gesture.NONE, 0.0, rate

        direction = Gesture.ZOOM_IN if rate > 0 else Gesture.ZOOM_OUT
        if self._zoom_active is Gesture.NONE:
            if abs(rate) >= cfg.zoom_enter_rate:
                self._zoom_active, self._zoom_last_active = direction, t
        else:
            same_way = direction is self._zoom_active
            if same_way and abs(rate) >= cfg.zoom_exit_rate:
                self._zoom_last_active = t
            elif not same_way and abs(rate) >= cfg.zoom_enter_rate:
                self._zoom_active, self._zoom_last_active = direction, t  # reversed direction
            elif t - self._zoom_last_active > cfg.zoom_hold_s:
                self._zoom_active = Gesture.NONE
        return self._zoom_active, delta, rate

    # ------------------------------------------------------- pinch: drag/rotate
    def _update_pinch(self, s: _Sample, t: float, palm: float, pinching: bool) -> tuple[bool, bool, float]:
        cfg = self.cfg
        if not pinching:
            self._pinch_active = False
            self._pinch_anchor = None
            self._rotate_active = self._drag_active = False
            return False, False, 0.0

        if not self._pinch_active:  # rising edge of the pinch
            self._pinch_active = True
            self._pinch_anchor = s.pinch_point.copy()
            self._rotate_active = self._drag_active = False
            self._rotate_last_active = self._drag_last_active = -math.inf

        win = self._window(cfg.rotate_window_s)
        roll_rate = 0.0
        roll_total = 0.0
        translation = 0.0
        if len(win) >= 3 and win[-1].t - win[0].t > 1e-3:
            dt = win[-1].t - win[0].t
            roll_total = win[-1].roll - win[0].roll
            roll_rate = roll_total / dt
            translation = float(np.linalg.norm(win[-1].center - win[0].center)) / palm

        # --- rotate (wrist roll with the hand held roughly in place)
        # Entering uses the longer window (robust against noise); leaving uses a short
        # one so the label does not linger after the twist has finished.
        fast = self._window(0.15)
        fast_rate = 0.0
        if len(fast) >= 2 and fast[-1].t - fast[0].t > 1e-3:
            fast_rate = (fast[-1].roll - fast[0].roll) / (fast[-1].t - fast[0].t)
        rotating_now = (
            abs(roll_rate) >= cfg.rotate_enter_rate
            and abs(roll_total) >= cfg.rotate_min_total_deg
            and translation <= cfg.rotate_max_translation
        )
        if rotating_now:
            self._rotate_active = True
            self._rotate_last_active = t
        elif self._rotate_active:
            if abs(fast_rate) >= cfg.rotate_exit_rate:
                self._rotate_last_active = t
            elif t - self._rotate_last_active > cfg.rotate_hold_s:
                self._rotate_active = False

        # --- drag (pinch point travelling while the pinch is held)
        pinch_speed = self._pinch_speed(0.15, palm)
        travelled = float(np.linalg.norm(s.pinch_point - self._pinch_anchor)) / palm if self._pinch_anchor is not None else 0.0
        if not self._rotate_active:
            if travelled >= cfg.drag_deadzone and pinch_speed >= cfg.drag_enter_speed:
                self._drag_active = True
                self._drag_last_active = t
            elif self._drag_active:
                if pinch_speed >= 0.5 * cfg.drag_enter_speed:
                    self._drag_last_active = t
                elif t - self._drag_last_active > cfg.drag_hold_s:
                    self._drag_active = False
        else:
            self._drag_active = False
        return self._rotate_active, self._drag_active, roll_rate

    def _pinch_speed(self, seconds: float, palm: float) -> float:
        win = self._window(seconds)
        if len(win) < 2 or win[-1].t - win[0].t < 1e-3:
            return 0.0
        return float(np.linalg.norm(win[-1].pinch_point - win[0].pinch_point)) / (win[-1].t - win[0].t) / palm

    # ----------------------------------------------------------------- scroll
    def _update_scroll(self, f: HandFeatures, palm: float, static: Gesture) -> float:
        """Vertical travel of the raised fingertips this frame in palm lengths
        (positive = up the screen).  Zero unless the two-finger pose is held."""
        if static is not Gesture.TWO_FINGERS:
            self._prev_two_tip = None
            return 0.0
        tip = f.two_finger_tip * np.array([f.aspect, 1.0])
        previous, self._prev_two_tip = self._prev_two_tip, tip
        if previous is None:
            return 0.0
        return float(-(tip[1] - previous[1]) / palm)
