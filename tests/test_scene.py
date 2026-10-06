"""Scene model tests: physics, commands, scene switching.  No rendering involved."""

import numpy as np
import pytest

from config import DisplayConfig, InteractionConfig
from interaction.commands import Command, CommandType as C
from visualization.scene import BOUNDS_X, BOUNDS_Y, SCENE_PRESETS, SceneManager
from visualization.shapes import MESHES

ASPECT = 4 / 3
DT = 1 / 30


@pytest.fixture
def mgr() -> SceneManager:
    m = SceneManager(DisplayConfig(), ASPECT, InteractionConfig())
    for _ in range(40):  # let the intro animation finish
        m.update(DT)
    return m


def step(mgr, seconds):
    for _ in range(int(seconds / DT)):
        mgr.update(DT)


def test_every_preset_builds_non_overlapping_objects():
    m = SceneManager(DisplayConfig(), ASPECT)
    assert len(m.scenes) == len(SCENE_PRESETS) >= 3
    for scene in m.scenes:
        assert len(scene.objects) >= 5
        for i, a in enumerate(scene.objects):
            assert a.shape in MESHES
            for b in scene.objects[i + 1:]:
                d = np.hypot((a.pos[0] - b.pos[0]) * ASPECT, a.pos[1] - b.pos[1])
                assert d > 0.12


def test_meshes_are_well_formed():
    for name, mesh in MESHES.items():
        assert mesh.vertices.shape[1] == 3 and mesh.edges.shape[1] == 2, name
        assert mesh.edges.max() < len(mesh.vertices) and mesh.edges.min() >= 0, name
        assert np.linalg.norm(mesh.vertices, axis=1).max() <= 1.0 + 1e-9, name
        assert not mesh.vertices.flags.writeable  # shared, so must be read-only


def test_pick_returns_the_object_under_the_point_or_none(mgr):
    obj = mgr.current.objects[0]
    pos = mgr.object_position(obj.id)
    assert mgr.pick(*pos) == obj.id
    assert mgr.pick(-0.5, -0.5) is None


def test_select_and_deselect(mgr):
    oid = mgr.current.objects[2].id
    mgr.apply(Command(C.SELECT, object_id=oid))
    assert mgr.selected_id == oid
    mgr.apply(Command(C.DESELECT))
    assert mgr.selected_id is None


def test_unknown_object_ids_are_ignored(mgr):
    mgr.apply(Command(C.SELECT, object_id=999))
    mgr.apply(Command(C.GRAB, object_id=999, position=(0.5, 0.5)))
    assert mgr.selected_id is None and mgr.current.grabbed_id is None


def test_grabbed_object_follows_the_move_target_smoothly(mgr):
    obj = mgr.current.objects[0]
    mgr.apply(Command(C.GRAB, object_id=obj.id, position=tuple(obj.pos)))
    assert mgr.current.grabbed_id == obj.id and mgr.selected_id == obj.id
    target = (0.5, 0.5)
    mgr.apply(Command(C.MOVE, object_id=obj.id, position=target))
    first = obj.pos.copy()
    mgr.update(DT)
    assert not np.allclose(obj.pos, first)  # moving...
    step(mgr, 0.5)
    assert np.allclose(obj.pos, target, atol=0.01)  # ...and arrives


def test_released_object_keeps_momentum_then_settles_inside_bounds(mgr):
    obj = mgr.current.objects[0]
    obj.pos = np.array([0.5, 0.5])
    mgr.apply(Command(C.GRAB, object_id=obj.id, position=(0.5, 0.5)))
    mgr.apply(Command(C.RELEASE, velocity=(0.8, 0.0)))
    assert mgr.current.grabbed_id is None
    x0 = obj.pos[0]
    step(mgr, 0.4)
    assert obj.pos[0] > x0 + 0.05  # kept moving right
    step(mgr, 20.0)
    assert BOUNDS_X[0] - 1e-6 <= obj.pos[0] <= BOUNDS_X[1] + 1e-6
    assert BOUNDS_Y[0] - 1e-6 <= obj.pos[1] <= BOUNDS_Y[1] + 1e-6


def test_throw_speed_is_capped(mgr):
    obj = mgr.current.objects[0]
    mgr.apply(Command(C.GRAB, object_id=obj.id, position=tuple(obj.pos)))
    mgr.apply(Command(C.RELEASE, velocity=(40.0, 0.0)))
    assert np.hypot(*obj.vel) <= 1.4 + 1e-9


def test_objects_stay_on_screen_over_a_long_run(mgr):
    step(mgr, 120)
    for o in mgr.current.objects:
        assert BOUNDS_X[0] - 1e-6 <= o.pos[0] <= BOUNDS_X[1] + 1e-6
        assert BOUNDS_Y[0] - 1e-6 <= o.pos[1] <= BOUNDS_Y[1] + 1e-6


def test_rotation_accumulates_and_is_smoothed(mgr):
    obj = mgr.current.objects[1]
    mgr.apply(Command(C.ROTATE, object_id=obj.id, value=0.5))
    mgr.apply(Command(C.ROTATE, object_id=obj.id, value=0.25))
    assert obj.user_roll_target == pytest.approx(0.75)
    mgr.update(DT)
    assert 0.0 < obj.user_roll < 0.75
    step(mgr, 1.0)
    assert obj.user_roll == pytest.approx(0.75, abs=0.01)


def test_scale_is_clamped_to_the_configured_limits(mgr):
    lim = InteractionConfig()
    obj = mgr.current.objects[0]
    for _ in range(20):
        mgr.apply(Command(C.SCALE, object_id=obj.id, value=1.5))
    assert obj.target_scale == pytest.approx(lim.max_scale)
    for _ in range(40):
        mgr.apply(Command(C.SCALE, object_id=obj.id, value=0.5))
    assert obj.target_scale == pytest.approx(lim.min_scale)


def test_scene_wide_zoom_changes_picking_geometry(mgr):
    obj = mgr.current.objects[0]
    before = mgr.object_position(obj.id)
    mgr.apply(Command(C.SCALE, value=1.5))  # no object id: zoom the whole scene
    step(mgr, 1.5)
    after = mgr.object_position(obj.id)
    assert mgr.current.zoom == pytest.approx(1.5, abs=0.02)
    assert not np.allclose(before, after)
    assert mgr.pick(*after) == obj.id  # picking follows what is drawn


def test_cycle_wraps_around_in_both_directions(mgr):
    ids = mgr.object_ids()
    mgr.apply(Command(C.CYCLE, value=1.0))
    assert mgr.selected_id == ids[0]
    for _ in range(len(ids)):
        mgr.apply(Command(C.CYCLE, value=1.0))
    assert mgr.selected_id == ids[0]
    mgr.apply(Command(C.CYCLE, value=-1.0))
    assert mgr.selected_id == ids[-1]


def test_reset_restores_layout_scale_and_clears_selection(mgr):
    obj = mgr.current.objects[0]
    home = obj.home.copy()
    mgr.apply(Command(C.GRAB, object_id=obj.id, position=tuple(obj.pos)))
    mgr.apply(Command(C.MOVE, position=(0.9, 0.9)))
    mgr.apply(Command(C.SCALE, object_id=obj.id, value=2.0))
    mgr.apply(Command(C.ROTATE, object_id=obj.id, value=1.0))
    step(mgr, 0.5)
    mgr.apply(Command(C.RESET))
    step(mgr, 3.0)
    assert mgr.selected_id is None and mgr.current.grabbed_id is None
    assert np.allclose(obj.pos, home, atol=0.02)
    assert obj.scale == pytest.approx(1.0, abs=0.02) and abs(obj.user_roll) < 0.02


def test_pause_slows_ambient_motion(mgr):
    scene = mgr.current
    def distance_travelled():
        before = [o.pos.copy() for o in scene.objects]
        step(mgr, 2.0)
        return sum(float(np.linalg.norm(o.pos - b)) for o, b in zip(scene.objects, before))
    free = distance_travelled()
    mgr.apply(Command(C.SET_PAUSED, value=1.0))
    step(mgr, 1.0)
    paused = distance_travelled()
    assert paused < 0.4 * free
    mgr.apply(Command(C.SET_PAUSED, value=0.0))
    step(mgr, 2.0)
    assert distance_travelled() > 0.7 * free


def test_scene_switching_wraps_and_runs_a_transition(mgr):
    n = len(mgr.scenes)
    mgr.apply(Command(C.NEXT_SCENE))
    assert mgr.index == 1 and mgr.transition is not None and mgr.transition.direction == 1
    assert mgr.previous is mgr.scenes[0]
    step(mgr, DisplayConfig().transition_s + 0.2)
    assert mgr.transition is None and mgr.previous is None
    mgr.apply(Command(C.PREV_SCENE))
    mgr.apply(Command(C.PREV_SCENE))  # ignored: the first transition is still in its first half
    step(mgr, 1.0)
    assert mgr.index == (1 - 1) % n
    mgr.apply(Command(C.PREV_SCENE))
    assert mgr.index == n - 1  # wrapped


def test_switching_scenes_releases_the_grab(mgr):
    obj = mgr.current.objects[0]
    mgr.apply(Command(C.GRAB, object_id=obj.id, position=tuple(obj.pos)))
    mgr.apply(Command(C.NEXT_SCENE))
    assert mgr.scenes[0].grabbed_id is None


def test_activation_command_updates_manager_state(mgr):
    mgr.apply(Command(C.SET_ACTIVE, value=0.0))
    assert not mgr.active
    mgr.apply(Command(C.SET_ACTIVE, value=1.0))
    assert mgr.active


def test_the_scene_has_no_decorative_effects():
    """Every element of the scene state communicates interaction; there are no particles/pulses."""
    scene = SceneManager(DisplayConfig(), ASPECT).current
    assert not hasattr(scene, "particles") and not hasattr(scene, "pulses") and not hasattr(scene, "dust_pos")


def test_aspect_change_keeps_picking_isotropic(mgr):
    obj = mgr.current.objects[0]
    pos = mgr.object_position(obj.id)
    mgr.set_aspect(2.0)
    assert all(s.aspect == 2.0 for s in mgr.scenes)
    probe_x = pos[0] + obj.base_radius * 1.1 / 2.0 * mgr.current.zoom  # just inside the radius at aspect 2
    assert mgr.pick(probe_x, pos[1]) == obj.id
    assert mgr.pick(pos[0] + obj.base_radius * 1.1 / 1.0, pos[1]) is None  # same offset would have hit at aspect 1
