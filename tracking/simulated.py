"""Scripted synthetic hand: a :class:`~tracking.hand_tracker.Tracker` without a camera.

``python app.py --simulate`` runs the real gesture engine, controller, scene and
renderer against a hand performing a scripted routine.  It exists so the project
can be demoed and tested on a machine with no webcam (or no hands in front of it),
and it doubles as the end-to-end regression test.

A script is a chain of *steps*.  Each step morphs the hand from where the previous
step ended to a new pose / position / roll over some duration.  Positions are in
**scene coordinates** and may be callables, so the hand can chase a moving object.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from config import InteractionConfig
from interaction.mapping import scene_to_camera
from tracking.synthetic import POSES, HandPose, make_observation
from tracking.types import HandObservation, Landmark, TrackingResult

Position = tuple[float, float] | Callable[[], tuple[float, float]]

# Which part of the hand should land on the target position.
_AIM_POINTS = {
    "palm": None,
    "index_tip": (Landmark.INDEX_TIP,),
    "pinch": (Landmark.THUMB_TIP, Landmark.INDEX_TIP),
    "two_fingers": (Landmark.INDEX_TIP, Landmark.MIDDLE_TIP),
}


@dataclass(frozen=True)
class Endpoint:
    pose: HandPose
    pos: Position
    aim: str
    roll: float
    present: bool


@dataclass(frozen=True)
class Step:
    t0: float
    duration: float
    start: Endpoint
    end: Endpoint
    ease: str  # "smooth" | "linear"
    label: str


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class GestureScript:
    """Builds and evaluates a scripted hand performance."""

    def __init__(self, margin: float = InteractionConfig().active_margin, palm_px: float = 0.21) -> None:
        self.margin = margin
        self.palm_px = palm_px
        self.steps: list[Step] = []
        self._end = Endpoint(POSES["open_palm"], (0.5, 0.5), "palm", 0.0, False)

    # ----------------------------------------------------------------- build
    @property
    def duration(self) -> float:
        return self.steps[-1].t0 + self.steps[-1].duration if self.steps else 0.0

    def _add(self, duration: float, end: Endpoint, ease: str, label: str) -> "GestureScript":
        self.steps.append(Step(self.duration, duration, self._end, end, ease, label))
        self._end = end
        return self

    def absent(self, duration: float) -> "GestureScript":
        return self._add(duration, Endpoint(self._end.pose, self._end.pos, self._end.aim, self._end.roll, False), "linear", "absent")

    def hold(self, duration: float, label: str = "hold") -> "GestureScript":
        return self._add(duration, self._end, "linear", label)

    def to(
        self,
        pose: HandPose | str | None = None,
        at: Position | None = None,
        duration: float = 0.4,
        aim: str | None = None,
        roll: float | None = None,
        ease: str = "smooth",
        label: str = "",
    ) -> "GestureScript":
        """Morph to a new pose / position / roll.  Anything omitted stays as it was."""
        e = self._end
        end = Endpoint(
            POSES[pose] if isinstance(pose, str) else (pose or e.pose),
            at if at is not None else e.pos,
            aim or e.aim,
            e.roll if roll is None else roll,
            True,
        )
        # A hand appearing from nothing starts at its destination.
        if not e.present:
            self._end = Endpoint(end.pose, end.pos, end.aim, end.roll, False)
            self.steps.append(Step(self.duration, duration, self._end, end, ease, label))
            self._end = end
            return self
        return self._add(duration, end, ease, label or (pose if isinstance(pose, str) else "move"))

    # ------------------------------------------------------------- evaluate
    def _step_at(self, t: float) -> Step | None:
        for step in self.steps:
            if step.t0 <= t < step.t0 + step.duration:
                return step
        return None

    def _center(self, ep: Endpoint) -> np.ndarray:
        """Palm centre (normalised camera coordinates) that puts the aimed-at part of
        the hand on ``ep.pos``."""
        target_scene = np.asarray(ep.pos() if callable(ep.pos) else ep.pos, dtype=float)
        target = scene_to_camera(target_scene, self.margin)
        points = _AIM_POINTS[ep.aim]
        if points is None:
            return target
        probe = make_observation(ep.pose, center=(0.5, 0.5), palm_px=self.palm_px, roll_deg=ep.roll)
        aimed = np.mean([probe.landmarks[int(i), :2] for i in points], axis=0)
        return np.array([0.5, 0.5]) + (target - aimed)

    def observation(self, t: float, aspect: float = 4.0 / 3.0) -> HandObservation | None:
        step = self._step_at(t)
        if step is None or not step.end.present:
            return None
        u = (t - step.t0) / step.duration
        u = _smoothstep(u) if step.ease == "smooth" else u
        a, b = step.start, step.end
        center = self._center(a) * (1 - u) + self._center(b) * u
        return make_observation(
            a.pose.lerp(b.pose, u),
            center=(float(center[0]), float(center[1])),
            palm_px=self.palm_px,
            roll_deg=a.roll + (b.roll - a.roll) * u,
            aspect=aspect,
            timestamp=t,
        )

    def label_at(self, t: float) -> str:
        step = self._step_at(t)
        return "" if step is None else step.label


class SimulatedTracker:
    """Tracker that ignores the image and plays back a :class:`GestureScript`."""

    def __init__(self, script: GestureScript, loop: bool = True, noise: float = 0.004, seed: int = 5) -> None:
        self.script = script
        self.loop = loop
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self._t0: float | None = None

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> TrackingResult:
        if self._t0 is None:
            self._t0 = timestamp
        local = timestamp - self._t0
        if self.loop and self.script.duration > 0:
            local %= self.script.duration
        h, w = frame_bgr.shape[:2]
        obs = self.script.observation(local, aspect=w / h)
        if obs is None:
            return TrackingResult(brightness=float(frame_bgr.mean()))
        if self.noise > 0:  # small landmark jitter, like a real tracker
            obs = HandObservation(
                obs.landmarks + self._rng.normal(0, self.noise * 0.2, obs.landmarks.shape),
                obs.world + self._rng.normal(0, self.noise * 0.2, obs.world.shape),
                obs.handedness, obs.score, timestamp, obs.aspect,
            )
        else:
            obs = HandObservation(obs.landmarks, obs.world, obs.handedness, obs.score, timestamp, obs.aspect)
        return TrackingResult(hands=(obs,), primary=obs, brightness=float(frame_bgr.mean()))

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# The demo routine
# --------------------------------------------------------------------------- #
def demo_script(object_position: Callable[[int], tuple[float, float]], margin: float | None = None) -> GestureScript:
    """A tour of every gesture.  ``object_position(i)`` returns the *current* scene
    position of object ``i`` so the hand can aim at it even as it floats around."""
    script = GestureScript() if margin is None else GestureScript(margin=margin)
    obj = lambda i: (lambda: object_position(i))  # noqa: E731

    (script
        .absent(0.6)
        # 1. open palm: pause
        .to("open_palm", at=(0.5, 0.5), duration=0.01, aim="palm", label="open palm: pause")
        .hold(1.0, "open palm: pause")
        # 2. point at an object -> select
        .to("point", at=obj(0), duration=0.6, aim="index_tip", label="point: select")
        .hold(0.9, "point: select")
        # 3. pinch it and drag it across the scene, then let go (throw)
        .to("pinch", at=obj(0), duration=0.35, aim="pinch", label="pinch: grab")
        .hold(0.35, "pinch: grab")
        .to(at=(0.74, 0.30), duration=0.9, label="pinch + move: drag")
        .to(at=(0.30, 0.62), duration=0.9, label="pinch + move: drag")
        .hold(0.3, "pinch: hold")
        # 4. twist the wrist to rotate it
        .to(roll=75.0, duration=1.0, label="pinch + twist: rotate")
        .to(roll=-35.0, duration=1.0, label="pinch + twist: rotate")
        .to(roll=0.0, duration=0.6, label="pinch + twist: rotate")
        .hold(0.2, "pinch: hold")
        # 5. release, open the hand, spread the fingers -> zoom the selected object
        .to("together", at=(0.5, 0.6), duration=0.5, aim="palm", label="open hand")
        .hold(0.7, "open hand")
        .to("spread", duration=0.55, label="spread fingers: zoom in")
        .hold(0.7, "open hand")
        .to("together", duration=0.55, label="fingers together: zoom out")
        .hold(0.5, "open hand")
        # 6. two fingers: move up/down to switch the selected object
        .to("two_fingers", at=(0.5, 0.78), duration=0.5, aim="two_fingers", label="two fingers: switch object")
        .hold(0.4, "two fingers")
        .to(at=(0.5, 0.40), duration=1.0, label="two fingers: switch object")
        .hold(0.4, "two fingers")
        # 7. swipe left / right -> change scene
        .to("open_palm", at=(0.82, 0.5), duration=0.5, aim="palm", label="open palm")
        .hold(0.6, "open palm")
        .to(at=(0.18, 0.5), duration=0.32, ease="linear", label="swipe left: next scene")
        .hold(1.4, "open palm")
        .to(at=(0.18, 0.5), duration=0.2, label="open palm")
        .to(at=(0.82, 0.5), duration=0.32, ease="linear", label="swipe right: previous scene")
        .hold(1.2, "open palm")
        # 8. fist: standby, then activate again
        .to("fist", at=(0.5, 0.55), duration=0.4, aim="palm", label="fist: standby")
        .hold(1.0, "fist: standby")
        .to("open_palm", duration=0.4, label="open palm")
        .hold(0.7, "standby: gestures ignored")
        .to("fist", duration=0.4, label="fist: activate")
        .hold(1.0, "fist: activate")
        # 9. hold an open palm still: reset the scene
        .to("open_palm", at=(0.5, 0.5), duration=0.4, label="open palm: hold to reset")
        .hold(2.2, "open palm: hold to reset")
        .absent(0.8))
    return script
