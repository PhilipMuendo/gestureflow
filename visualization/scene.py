"""The interactive scene model: objects, physics and scene switching.

Pure state + simulation - nothing here draws.  All positions are in *normalised viewport
coordinates* (x, y in [0, 1], origin top-left); distances that must be isotropic multiply x
by the viewport ``aspect``.  The renderer reads this state; the interaction controller changes
it by sending :class:`~interaction.commands.Command` objects to :meth:`SceneManager.apply`.

Everything the scene keeps exists because it communicates interaction state (selected /
hovered / grabbed levels, scale, rotation).  There are deliberately no decorative effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from config import DisplayConfig, InteractionConfig
from interaction.commands import Command, CommandType

Color = tuple[int, int, int]  # BGR

SCENE_CENTER = np.array([0.5, 0.5])
BOUNDS_X = (0.07, 0.93)
BOUNDS_Y = (0.10, 0.90)


@dataclass(frozen=True)
class ScenePreset:
    name: str
    shapes: tuple[str, ...]
    tint: Color  # object line colour: a near-white with a slight tone, so scenes differ quietly
    background: Color  # base tone of the backdrop in visualisation mode


SCENE_PRESETS: tuple[ScenePreset, ...] = (
    ScenePreset("Nebula", ("icosahedron", "cube", "octahedron", "torus", "tetrahedron", "sphere"), (232, 228, 224), (20, 18, 17)),
    ScenePreset("Aurora", ("sphere", "torus", "bipyramid", "icosahedron", "cube", "octahedron"), (226, 232, 226), (18, 19, 17)),
    ScenePreset("Ember", ("tetrahedron", "bipyramid", "cube", "torus", "octahedron", "icosahedron"), (222, 228, 236), (17, 17, 20)),
)


# --------------------------------------------------------------------------- #
# Objects
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class SceneObject:
    id: int
    shape: str
    color: Color
    home: np.ndarray  # layout position, used by "reset"
    base_radius: float  # fraction of viewport height
    depth: float  # 0 near .. 1 far: affects size and brightness only
    pos: np.ndarray
    drift: np.ndarray  # ambient drift velocity
    spin: np.ndarray  # idle angular velocity (rad/s) about x, y, z
    angles: np.ndarray
    phase: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 1.0
    target_scale: float = 1.0
    user_roll: float = 0.0  # extra rotation about the view axis driven by the user
    user_roll_target: float = 0.0
    target: np.ndarray = field(default_factory=lambda: np.zeros(2))  # where a grabbed object is heading
    seek_s: float = 0.0  # >0 while animating back to ``home`` after a reset
    # animated 0..1 state levels (eased, so state changes never pop)
    select_level: float = 0.0
    hover_level: float = 0.0
    grab_level: float = 0.0
    appear: float = 0.0  # 0..1 intro animation

    def __post_init__(self) -> None:
        self.target = self.pos.copy()

    @property
    def display_scale(self) -> float:
        return self.scale * (1.0 - 0.22 * self.depth) * (0.6 + 0.4 * ease_out(self.appear))


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _approach(current: float, target: float, rate: float, dt: float) -> float:
    return current + (target - current) * (1.0 - math.exp(-rate * dt))


class Scene:
    def __init__(self, preset: ScenePreset, aspect: float, seed: int, limits: InteractionConfig | None = None) -> None:
        self.preset = preset
        self.aspect = aspect
        self.limits = limits or InteractionConfig()
        self.rng = np.random.default_rng(seed)
        self.objects: list[SceneObject] = self._build_objects()

        self.selected_id: int | None = None
        self.hovered_id: int | None = None
        self.grabbed_id: int | None = None
        self.zoom = 1.0
        self.zoom_target = 1.0
        self.time = 0.0
        self.time_scale = 1.0
        self.time_scale_target = 1.0

    # ----------------------------------------------------------------- build
    def _build_objects(self) -> list[SceneObject]:
        p, rng = self.preset, self.rng
        positions: list[np.ndarray] = []
        for _ in range(len(p.shapes)):
            for _attempt in range(200):
                cand = np.array([rng.uniform(0.15, 0.85), rng.uniform(0.2, 0.8)])
                if all(np.hypot((cand[0] - q[0]) * self.aspect, cand[1] - q[1]) > 0.26 for q in positions):
                    break
            positions.append(cand)
        objects = []
        for i, (shape, pos) in enumerate(zip(p.shapes, positions)):
            angle = rng.uniform(0, 2 * math.pi)
            objects.append(
                SceneObject(
                    id=i,
                    shape=shape,
                    color=p.tint,
                    home=pos.copy(),
                    base_radius=float(rng.uniform(0.06, 0.085)),
                    depth=float(rng.uniform(0.0, 0.6)),
                    pos=pos.copy(),
                    drift=np.array([math.cos(angle), math.sin(angle)]) * rng.uniform(0.004, 0.011),
                    spin=rng.uniform(-0.5, 0.5, 3),
                    angles=rng.uniform(0, 2 * math.pi, 3),
                    phase=float(rng.uniform(0, 2 * math.pi)),
                    appear=-0.08 * i,  # staggered intro
                )
            )
        return objects

    # ----------------------------------------------------------- geometry
    def screen_pos(self, o: SceneObject) -> np.ndarray:
        bob = np.array([0.0, 0.004 * math.sin(self.time * 0.9 + o.phase)])
        return SCENE_CENTER + (o.pos + bob - SCENE_CENTER) * self.zoom

    def screen_radius(self, o: SceneObject) -> float:
        """Radius in fractions of the viewport height."""
        return o.base_radius * o.display_scale * self.zoom

    def get(self, object_id: int | None) -> SceneObject | None:
        if object_id is None:
            return None
        return next((o for o in self.objects if o.id == object_id), None)

    def pick(self, x: float, y: float, slop: float = 1.4) -> int | None:
        """Id of the object under the point (x, y), preferring the closest centre."""
        best, best_d = None, math.inf
        for o in self.objects:
            sp = self.screen_pos(o)
            d = math.hypot((x - sp[0]) * self.aspect, y - sp[1])
            if d <= self.screen_radius(o) * slop and d < best_d:
                best, best_d = o.id, d
        return best

    # ----------------------------------------------------------- simulation
    def update(self, dt: float) -> None:
        dt = min(max(dt, 0.0), 0.1)
        self.time_scale = _approach(self.time_scale, self.time_scale_target, 5.0, dt)
        sdt = dt * self.time_scale  # ambient motion slows down while "paused"
        self.time += sdt
        self.zoom = _approach(self.zoom, self.zoom_target, 9.0, dt)
        for o in self.objects:
            self._step_object(o, dt, sdt)

    def _step_object(self, o: SceneObject, dt: float, sdt: float) -> None:
        grabbed = o.id == self.grabbed_id
        o.appear = min(1.0, o.appear + dt / 0.7)

        if grabbed:
            previous = o.pos.copy()
            o.pos += (o.target - o.pos) * (1.0 - math.exp(-24.0 * dt))
            if dt > 1e-4:
                o.vel = 0.65 * o.vel + 0.35 * (o.pos - previous) / dt  # estimate for the throw
        else:
            o.pos += o.vel * sdt
            o.vel += (o.drift - o.vel) * (1.0 - math.exp(-1.6 * sdt))
            for axis, (lo, hi) in enumerate((BOUNDS_X, BOUNDS_Y)):  # soft wall bounce
                if o.pos[axis] < lo:
                    o.pos[axis], o.vel[axis], o.drift[axis] = lo, abs(o.vel[axis]), abs(o.drift[axis])
                elif o.pos[axis] > hi:
                    o.pos[axis], o.vel[axis], o.drift[axis] = hi, -abs(o.vel[axis]), -abs(o.drift[axis])

        if o.seek_s > 0.0:  # animate home after a reset
            o.pos += (o.home - o.pos) * (1.0 - math.exp(-7.0 * dt))
            o.seek_s -= dt

        o.scale = _approach(o.scale, o.target_scale, 10.0, dt)
        o.user_roll = _approach(o.user_roll, o.user_roll_target, 14.0, dt)
        o.angles += o.spin * sdt * (0.35 if grabbed else 1.0)
        o.select_level = _approach(o.select_level, 1.0 if o.id == self.selected_id else 0.0, 12.0, dt)
        o.hover_level = _approach(o.hover_level, 1.0 if o.id == self.hovered_id else 0.0, 14.0, dt)
        o.grab_level = _approach(o.grab_level, 1.0 if grabbed else 0.0, 16.0, dt)

    # ------------------------------------------------------------- commands
    def apply(self, cmd: Command) -> None:
        t = cmd.type
        obj = self.get(cmd.object_id)
        if t is CommandType.HOVER:
            self.hovered_id = cmd.object_id if obj is not None else None
        elif t is CommandType.SELECT and obj is not None:
            self.selected_id = obj.id
        elif t is CommandType.DESELECT:
            self.selected_id = None
        elif t is CommandType.GRAB and obj is not None:
            self.grabbed_id = self.selected_id = obj.id
            obj.seek_s = 0.0
            obj.target = obj.pos.copy()
        elif t is CommandType.MOVE and cmd.position is not None:
            grabbed = self.get(self.grabbed_id)
            if grabbed is not None:
                # the command carries a *screen* position; undo the zoom to get scene space
                grabbed.target = SCENE_CENTER + (np.asarray(cmd.position) - SCENE_CENTER) / self.zoom
        elif t is CommandType.RELEASE:
            grabbed = self.get(self.grabbed_id)
            if grabbed is not None and cmd.velocity is not None:
                v = np.asarray(cmd.velocity, dtype=float)
                mag = float(np.hypot(*v))
                grabbed.vel = v if mag < 1.4 else v / mag * 1.4
            self.grabbed_id = None
        elif t is CommandType.ROTATE and obj is not None:
            obj.user_roll_target += cmd.value
        elif t is CommandType.SCALE:
            lim = self.limits
            if obj is not None:
                obj.target_scale = float(np.clip(obj.target_scale * cmd.value, lim.min_scale, lim.max_scale))
            else:
                self.zoom_target = float(np.clip(self.zoom_target * cmd.value, lim.min_camera_zoom, lim.max_camera_zoom))
        elif t is CommandType.CYCLE:
            self._cycle(int(cmd.value))
        elif t is CommandType.RESET:
            self.reset()
        elif t is CommandType.SET_PAUSED:
            self.time_scale_target = self.limits.paused_time_scale if cmd.value > 0.0 else 1.0

    def _cycle(self, step: int) -> None:
        ids = [o.id for o in self.objects]
        if not ids:
            return
        if self.selected_id not in ids:
            self.selected_id = ids[0] if step > 0 else ids[-1]
        else:
            self.selected_id = ids[(ids.index(self.selected_id) + step) % len(ids)]

    def reset(self) -> None:
        """Send every object back to its starting layout with default size/orientation."""
        self.grabbed_id = self.selected_id = self.hovered_id = None
        self.zoom_target = 1.0
        self.time_scale_target = 1.0
        for o in self.objects:
            o.seek_s = 1.4
            o.target_scale = 1.0
            o.user_roll_target = 0.0
            o.vel = np.zeros(2)


# --------------------------------------------------------------------------- #
# Scene manager (multiple scenes + transitions)
# --------------------------------------------------------------------------- #
@dataclass
class Transition:
    previous: int  # index of the scene sliding out
    direction: int  # +1: new scene enters from the right (next), -1: from the left
    progress: float = 0.0


class SceneManager:
    """Owns the scenes, routes commands to the current one and runs transitions.

    Also satisfies the ``SceneQuery`` protocol the interaction controller uses.
    """

    def __init__(
        self,
        cfg: DisplayConfig,
        aspect: float,
        limits: InteractionConfig | None = None,
        presets: tuple[ScenePreset, ...] = SCENE_PRESETS,
    ) -> None:
        self.cfg = cfg
        self.aspect = aspect
        self.scenes = [Scene(p, aspect, cfg.scene_seed + 17 * i, limits) for i, p in enumerate(presets)]
        self.index = 0
        self.transition: Transition | None = None
        self.paused = False
        self.active = True

    def set_aspect(self, aspect: float) -> None:
        """The viewport changed shape (window resized): keep picking geometry isotropic."""
        self.aspect = aspect
        for scene in self.scenes:
            scene.aspect = aspect

    # -- views ---------------------------------------------------------------
    @property
    def current(self) -> Scene:
        return self.scenes[self.index]

    @property
    def previous(self) -> Scene | None:
        return None if self.transition is None else self.scenes[self.transition.previous]

    # -- SceneQuery ----------------------------------------------------------
    @property
    def selected_id(self) -> int | None:
        return self.current.selected_id

    def pick(self, x: float, y: float, slop: float = 1.4) -> int | None:
        return self.current.pick(x, y, slop)

    def object_position(self, object_id: int) -> tuple[float, float] | None:
        obj = self.current.get(object_id)
        return None if obj is None else tuple(self.current.screen_pos(obj))  # type: ignore[return-value]

    def object_ids(self) -> list[int]:
        return [o.id for o in self.current.objects]

    # -- simulation ------------------------------------------------------------
    def update(self, dt: float) -> None:
        self.current.update(dt)
        if self.transition is not None:
            self.scenes[self.transition.previous].update(dt)
            self.transition.progress += dt / self.cfg.transition_s
            if self.transition.progress >= 1.0:
                self.transition = None

    def apply(self, cmd: Command) -> None:
        if cmd.type in (CommandType.NEXT_SCENE, CommandType.PREV_SCENE):
            self.switch(+1 if cmd.type is CommandType.NEXT_SCENE else -1)
        elif cmd.type is CommandType.SET_ACTIVE:
            self.active = cmd.value > 0.0
        elif cmd.type is CommandType.SET_PAUSED:
            self.paused = cmd.value > 0.0
            self.current.apply(cmd)
        else:
            self.current.apply(cmd)

    def switch(self, direction: int) -> None:
        if self.transition is not None and self.transition.progress < 0.5:
            return  # let the running transition finish first
        previous = self.index
        self.current.grabbed_id = self.current.hovered_id = None
        self.index = (self.index + direction) % len(self.scenes)
        self.transition = Transition(previous, direction)
        new = self.current
        new.time_scale_target = 1.0
        for o in new.objects:  # replay the intro animation
            o.appear = -0.04 * o.id

    def reset_current(self) -> None:
        self.current.reset()
