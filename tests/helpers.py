"""Helpers for driving the gesture engine with synthetic hand timelines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from gestures.gesture_engine import GestureEngine
from gestures.types import GestureState
from tracking.synthetic import HandPose, POSES, make_observation

FPS = 30.0
DT = 1.0 / FPS


def run(
    engine: GestureEngine,
    duration: float,
    pose: HandPose | str | Callable[[float], HandPose | str] = "open_palm",
    *,
    start: float = 0.0,
    center: Callable[[float], tuple[float, float]] | tuple[float, float] = (0.5, 0.55),
    roll: Callable[[float], float] | float = 0.0,
    noise: float = 0.0,
    palm_px: float = 0.22,
    handedness: str = "Right",
    present: Callable[[float], bool] | None = None,
    seed: int = 0,
) -> list[GestureState]:
    """Feed ``duration`` seconds of a synthetic hand into the engine at 30 FPS.

    Any of ``pose``, ``center`` and ``roll`` may be a function of the local time ``u``
    (seconds since the start of this call).  Returns the engine output per frame.
    """
    rng = np.random.default_rng(seed)
    states: list[GestureState] = []
    n = int(round(duration * FPS))
    for i in range(n):
        u = i * DT
        t = start + u
        if present is not None and not present(u):
            states.append(engine.process(None, t))
            continue
        p = pose(u) if callable(pose) else pose
        c = center(u) if callable(center) else center
        r = roll(u) if callable(roll) else roll
        obs = make_observation(
            p, center=c, roll_deg=r, palm_px=palm_px, handedness=handedness,
            timestamp=t, noise=noise, rng=rng,
        )
        states.append(engine.process(obs, t))
    return states


def lerp_pose(a: str, b: str, u: float) -> HandPose:
    return POSES[a].lerp(POSES[b], u)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# --------------------------------------------------------------------------- #
# interaction helpers
# --------------------------------------------------------------------------- #
from config import InteractionConfig  # noqa: E402
from interaction.controller import InteractionController  # noqa: E402
from interaction.mapping import scene_to_camera  # noqa: E402

MARGIN = InteractionConfig().active_margin
_AIM = {"index_tip": [8], "pinch": [4, 8], "palm": None}


def aim_center(pose, scene_xy, aim: str = "index_tip", roll: float = 0.0, palm_px: float = 0.22) -> tuple[float, float]:
    """Palm centre (camera coordinates) that puts the aimed-at fingertip on ``scene_xy``."""
    target = scene_to_camera(np.asarray(scene_xy, float), MARGIN)
    if _AIM[aim] is None:
        return float(target[0]), float(target[1])
    probe = make_observation(pose, center=(0.5, 0.5), palm_px=palm_px, roll_deg=roll)
    aimed = probe.landmarks[_AIM[aim], :2].mean(axis=0)
    c = np.array([0.5, 0.5]) + target - aimed
    return float(c[0]), float(c[1])


class FakeScene:
    """Minimal SceneQuery with objects at fixed scene positions (aspect 4:3)."""

    aspect = 4.0 / 3.0

    def __init__(self, objects: dict[int, tuple[float, float]], radius: float = 0.09) -> None:
        self.objects = dict(objects)
        self.radius = radius
        self.selected_id: int | None = None

    def pick(self, x: float, y: float, slop: float = 1.4) -> int | None:
        best, best_d = None, 1e9
        for oid, (ox, oy) in self.objects.items():
            d = float(np.hypot((x - ox) * self.aspect, y - oy))
            if d <= self.radius * slop and d < best_d:
                best, best_d = oid, d
        return best

    def object_position(self, object_id: int):
        return self.objects.get(object_id)

    def object_ids(self) -> list[int]:
        return list(self.objects)

    def apply(self, cmd) -> None:
        """Apply the commands that change what the controller can observe (selection, positions)."""
        from interaction.commands import CommandType as C

        if cmd.type is C.SELECT:
            self.selected_id = cmd.object_id
        elif cmd.type is C.DESELECT:
            self.selected_id = None
        elif cmd.type is C.GRAB:
            self.selected_id = self.grabbed = cmd.object_id
        elif cmd.type is C.MOVE and getattr(self, "grabbed", None) is not None and cmd.position is not None:
            self.objects[self.grabbed] = cmd.position
        elif cmd.type in (C.RELEASE, C.RESET):
            self.grabbed = None
        elif cmd.type is C.CYCLE:
            ids = list(self.objects)
            cur = ids.index(self.selected_id) if self.selected_id in ids else -1 if cmd.value > 0 else 0
            self.selected_id = ids[(cur + int(cmd.value)) % len(ids)]


def drive(engine, controller: InteractionController, states) -> list:
    """Feed engine states through the controller (applying its commands to the scene, as the
    application does every frame), returning every command it emits."""
    commands = []
    for s in states:
        cmds = controller.update(s, s.timestamp)
        for c in cmds:
            if hasattr(controller.scene, "apply"):
                controller.scene.apply(c)
        commands.extend(cmds)
    return commands


def kinds(commands) -> list[str]:
    return [c.type.name for c in commands]
