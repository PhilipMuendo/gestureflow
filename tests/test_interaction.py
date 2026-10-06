"""Controller tests: gesture timelines in, scene commands out (no scene, no camera)."""

import math

import numpy as np
import pytest

from config import GestureConfig
from gestures.gesture_engine import ALL_GESTURES, GestureEngine
from gestures.types import Gesture
from interaction.commands import CommandType as C
from interaction.controller import InteractionController, Phase
from tests.helpers import FakeScene, aim_center, drive, kinds, lerp_pose, run, smoothstep

OBJECTS = {0: (0.3, 0.4), 1: (0.7, 0.4), 2: (0.5, 0.75)}


@pytest.fixture
def scene() -> FakeScene:
    return FakeScene(OBJECTS)


@pytest.fixture
def ctrl(scene) -> InteractionController:
    return InteractionController(scene)


def point_at(xy):
    return aim_center("point", xy, "index_tip")


def pinch_at(xy, roll=0.0):
    return aim_center("pinch", xy, "pinch", roll=roll)


def of(commands, kind):
    return [c for c in commands if c.type is kind]


# =========================================================================== #
# selecting
# =========================================================================== #
def test_pointing_at_an_object_selects_it_after_the_dwell_time(engine, ctrl):
    states = run(engine, 1.5, "point", center=point_at(OBJECTS[1]))
    select_times, hovers = [], []
    for s in states:
        for c in ctrl.update(s, s.timestamp):
            ctrl.scene.apply(c)
            if c.type is C.SELECT:
                select_times.append((s.timestamp, c.object_id))
            elif c.type is C.HOVER:
                hovers.append(c.object_id)
    assert [oid for _, oid in select_times] == [1]  # selected exactly once, and the right object
    assert hovers[0] == 1
    first_pointing = next(s.timestamp for s in states if s.static is Gesture.POINT)
    assert select_times[0][0] - first_pointing >= 0.25  # the dwell is real, not instantaneous


def test_dwell_progress_is_reported_and_no_early_select(engine, scene):
    ctrl = InteractionController(scene)
    states = run(engine, 0.55, "point", center=point_at(OBJECTS[0]))
    seen = []
    for s in states:
        ctrl.update(s, s.timestamp)
        seen.append(ctrl.state)  # (no scene feedback needed to read the dwell progress)
    assert max(st.dwell_progress for st in seen) > 0.5
    assert seen[0].dwell_progress == 0.0
    assert ctrl.state.phase is Phase.POINTING


def test_pointing_at_empty_space_selects_nothing_but_clears_selection_after_a_while(engine, scene, ctrl):
    scene.selected_id = 0
    cmds = drive(engine, ctrl, run(engine, 2.0, "point", center=point_at((0.5, 0.15))))
    assert not of(cmds, C.SELECT)
    assert of(cmds, C.DESELECT)


def test_sweeping_over_objects_does_not_select_unless_dwelling(engine, ctrl):
    path = lambda u: point_at((0.1 + 0.8 * u / 1.0, 0.4))  # noqa: E731  sweeps across in 1 s
    cmds = drive(engine, ctrl, run(engine, 1.0, "point", center=path))
    assert not of(cmds, C.SELECT)


# =========================================================================== #
# grab / drag / release
# =========================================================================== #
def test_pinch_on_an_object_grabs_it_and_it_follows_the_hand(engine, ctrl):
    start, end = OBJECTS[0], (0.6, 0.6)
    cmds = drive(engine, ctrl, run(engine, 0.6, "pinch", center=pinch_at(start)))
    assert [c.object_id for c in of(cmds, C.GRAB)] == [0]
    moves = drive(engine, ctrl, run(engine, 1.0, "pinch", start=0.6,
                                    center=lambda u: pinch_at((start[0] + (end[0] - start[0]) * smoothstep(u / 0.8),
                                                               start[1] + (end[1] - start[1]) * smoothstep(u / 0.8)))))
    targets = [c.position for c in of(moves, C.MOVE)]
    assert targets, "dragging must emit MOVE commands"
    # the grabbed object keeps its offset from the pinch point, so it ends near the destination
    assert targets[-1] == pytest.approx(end, abs=0.06)
    assert ctrl.state.grabbed_id == 0


def test_object_does_not_jump_when_grabbed_off_centre(engine, scene, ctrl):
    off = (OBJECTS[0][0] + 0.05, OBJECTS[0][1] + 0.04)  # pinch slightly off the object's centre
    cmds = drive(engine, ctrl, run(engine, 0.8, "pinch", center=pinch_at(off)))
    moves = of(cmds, C.MOVE)
    assert moves
    assert moves[-1].position == pytest.approx(OBJECTS[0], abs=0.03)  # stays where it was


def test_releasing_the_pinch_releases_the_object_with_throw_velocity(engine, ctrl):
    start = OBJECTS[0]
    drive(engine, ctrl, run(engine, 0.6, "pinch", center=pinch_at(start)))
    fast = lambda u: pinch_at((start[0] + 0.5 * u / 0.4, start[1]))  # noqa: E731  moving right quickly
    drive(engine, ctrl, run(engine, 0.4, "pinch", start=0.6, center=fast))
    cmds = drive(engine, ctrl, run(engine, 0.5, "open_palm", start=1.0, center=pinch_at((0.8, 0.4))))
    release = of(cmds, C.RELEASE)
    assert len(release) == 1 and release[0].object_id == 0
    assert release[0].velocity[0] > 0.1  # thrown towards +x
    assert ctrl.state.grabbed_id is None


def test_pinching_empty_space_grabs_nothing(engine, ctrl):
    cmds = drive(engine, ctrl, run(engine, 0.8, "pinch", center=pinch_at((0.5, 0.12))))
    assert not of(cmds, C.GRAB) and of(cmds, C.GRAB_MISS)
    assert not of(cmds, C.MOVE)


def test_pinch_grabs_only_once_per_pinch(engine, ctrl):
    cmds = drive(engine, ctrl, run(engine, 1.5, "pinch", center=pinch_at(OBJECTS[1])))
    assert len(of(cmds, C.GRAB)) == 1


def test_losing_the_hand_releases_a_grabbed_object(engine, ctrl):
    drive(engine, ctrl, run(engine, 0.7, "pinch", center=pinch_at(OBJECTS[0])))
    assert ctrl.state.grabbed_id == 0
    cmds = drive(engine, ctrl, run(engine, 1.0, "pinch", start=0.7, present=lambda u: False))
    assert of(cmds, C.RELEASE)
    assert ctrl.state.grabbed_id is None


def test_a_one_frame_dropout_does_not_drop_the_object(engine, ctrl):
    cmds = drive(engine, ctrl, run(engine, 1.2, "pinch", center=pinch_at(OBJECTS[0]),
                                   present=lambda u: not (0.6 <= u < 0.7)))
    assert not of(cmds, C.RELEASE)
    assert ctrl.state.grabbed_id == 0


def test_drag_can_be_disabled():
    engine = GestureEngine(GestureConfig(), enabled=ALL_GESTURES - {Gesture.PINCH_DRAG})
    ctrl = InteractionController(FakeScene(OBJECTS))
    cmds = drive(engine, ctrl, run(engine, 1.2, "pinch", center=lambda u: pinch_at((0.3 + 0.2 * u, 0.4))))
    assert of(cmds, C.GRAB) and not of(cmds, C.MOVE)


# =========================================================================== #
# rotate
# =========================================================================== #
def test_pinch_and_twist_rotates_the_grabbed_object_without_moving_it(engine, ctrl):
    pos = OBJECTS[1]
    drive(engine, ctrl, run(engine, 0.7, "pinch", center=pinch_at(pos)))
    twist = lambda u: 60.0 * smoothstep(u / 0.9)  # noqa: E731
    cmds = drive(engine, ctrl, run(engine, 1.1, "pinch", start=0.7, roll=twist,
                                   center=lambda u: pinch_at(pos, roll=twist(u))))
    rotates = of(cmds, C.ROTATE)
    assert rotates and all(c.object_id == 1 for c in rotates)
    assert sum(c.value for c in rotates) == pytest.approx(math.radians(60), rel=0.35)  # most of the twist is applied
    assert not [c for c in cmds if c.type is C.MOVE and c.position and abs(c.position[0] - pos[0]) > 0.08]  # object stays put


def test_rotation_direction_follows_the_wrist(engine, ctrl):
    pos = OBJECTS[1]
    drive(engine, ctrl, run(engine, 0.7, "pinch", center=pinch_at(pos)))
    twist = lambda u: -50.0 * smoothstep(u / 0.8)  # noqa: E731
    cmds = drive(engine, ctrl, run(engine, 1.0, "pinch", start=0.7, roll=twist, center=lambda u: pinch_at(pos, roll=twist(u))))
    assert sum(c.value for c in of(cmds, C.ROTATE)) < 0


# =========================================================================== #
# palm: pause + hold to reset
# =========================================================================== #
def test_open_palm_pauses_and_resumes(engine, ctrl):
    cmds = drive(engine, ctrl, run(engine, 0.8, "open_palm"))
    assert of(cmds, C.SET_PAUSED)[0].value == 1.0 and ctrl.state.phase is Phase.PAUSED
    cmds = drive(engine, ctrl, run(engine, 0.8, "fist", start=0.8))
    assert any(c.value == 0.0 for c in of(cmds, C.SET_PAUSED))


def test_holding_a_steady_open_palm_resets_exactly_once(engine, ctrl):
    cmds = drive(engine, ctrl, run(engine, 5.0, "open_palm"))
    assert len(of(cmds, C.RESET)) == 1  # not again while the palm stays up


def test_reset_can_fire_again_after_the_palm_is_withdrawn(engine, ctrl):
    drive(engine, ctrl, run(engine, 3.0, "open_palm"))
    drive(engine, ctrl, run(engine, 1.0, "point", start=3.0, center=point_at((0.5, 0.15))))
    cmds = drive(engine, ctrl, run(engine, 3.0, "open_palm", start=4.0))
    assert len(of(cmds, C.RESET)) == 1


def test_moving_the_palm_prevents_the_reset(engine, ctrl):
    wave = lambda u: (0.5 + 0.18 * math.sin(2 * math.pi * 0.8 * u), 0.55)  # noqa: E731  slow wave side to side
    cmds = drive(engine, ctrl, run(engine, 4.0, "open_palm", center=wave))
    assert not of(cmds, C.RESET)


def test_reset_progress_is_visible_while_holding(engine, ctrl):
    seen = []
    for s in run(engine, 1.5, "open_palm"):
        ctrl.update(s, s.timestamp)
        seen.append(ctrl.state.reset_progress)
    assert 0.2 < max(seen) < 1.0


# =========================================================================== #
# fist: activate / standby
# =========================================================================== #
def test_holding_a_fist_toggles_standby_and_back(engine, scene, ctrl):
    cmds = drive(engine, ctrl, run(engine, 1.5, "fist"))
    assert [c.value for c in of(cmds, C.SET_ACTIVE)] == [0.0] and not ctrl.state.active
    assert ctrl.state.phase is Phase.STANDBY
    drive(engine, ctrl, run(engine, 0.8, "open_palm", start=1.5))
    cmds = drive(engine, ctrl, run(engine, 1.5, "fist", start=2.3))
    assert [c.value for c in of(cmds, C.SET_ACTIVE)] == [1.0] and ctrl.state.active


def test_a_brief_fist_does_not_toggle(engine, ctrl):
    cmds = drive(engine, ctrl, run(engine, 0.4, "fist"))
    assert not of(cmds, C.SET_ACTIVE)


def test_holding_one_fist_toggles_only_once(engine, ctrl):
    cmds = drive(engine, ctrl, run(engine, 4.0, "fist"))
    assert len(of(cmds, C.SET_ACTIVE)) == 1


def test_gestures_are_ignored_in_standby(engine, ctrl):
    drive(engine, ctrl, run(engine, 1.5, "fist"))
    assert not ctrl.state.active
    cmds = drive(engine, ctrl, run(engine, 2.0, "point", start=1.5, center=point_at(OBJECTS[0])))
    assert not of(cmds, C.SELECT) and not of(cmds, C.GRAB)


def test_closing_into_a_fist_releases_the_object_without_toggling_standby(engine, ctrl):
    drive(engine, ctrl, run(engine, 0.8, "pinch", center=pinch_at(OBJECTS[0])))
    cmds = drive(engine, ctrl, run(engine, 0.7, "fist", start=0.8, center=pinch_at(OBJECTS[0])))
    assert of(cmds, C.RELEASE) and not of(cmds, C.SET_ACTIVE)
    assert ctrl.state.active
    # ...but deliberately keeping the fist closed afterwards does toggle, once
    later = drive(engine, ctrl, run(engine, 2.0, "fist", start=1.5, center=pinch_at(OBJECTS[0])))
    assert len(of(later, C.SET_ACTIVE)) == 1


# =========================================================================== #
# two fingers, swipe, zoom
# =========================================================================== #
def test_two_fingers_moving_up_selects_next_object_and_down_selects_previous(engine, ctrl):
    drive(engine, ctrl, run(engine, 0.6, "two_fingers", center=(0.5, 0.75)))
    up = drive(engine, ctrl, run(engine, 0.8, "two_fingers", start=0.6, center=lambda u: (0.5, 0.75 - 0.35 * smoothstep(u / 0.7))))
    assert [c.value for c in of(up, C.CYCLE)] and all(c.value == 1.0 for c in of(up, C.CYCLE))
    down = drive(engine, ctrl, run(engine, 0.8, "two_fingers", start=1.4, center=lambda u: (0.5, 0.40 + 0.35 * smoothstep(u / 0.7))))
    assert of(down, C.CYCLE) and all(c.value == -1.0 for c in of(down, C.CYCLE))


def test_two_fingers_held_still_does_not_cycle(engine, ctrl):
    assert not of(drive(engine, ctrl, run(engine, 2.0, "two_fingers", noise=0.01)), C.CYCLE)


def test_swipes_change_scene(engine, ctrl):
    run(engine, 0.6, "open_palm", center=(0.8, 0.55))
    right_to_left = run(engine, 0.6, "open_palm", start=0.6, center=lambda u: (0.8 - 0.55 * smoothstep(u / 0.3), 0.55))
    cmds = drive(engine, ctrl, right_to_left)
    assert kinds(cmds).count("NEXT_SCENE") == 1  # swipe LEFT = next scene
    engine.reset()
    run(engine, 1.5, "open_palm", start=2.0, center=(0.25, 0.55))
    left_to_right = run(engine, 0.6, "open_palm", start=3.6, center=lambda u: (0.2 + 0.6 * smoothstep(u / 0.3), 0.55))
    assert kinds(drive(engine, ctrl, left_to_right)).count("PREV_SCENE") == 1


def test_spreading_fingers_scales_the_selected_object_up_and_closing_scales_it_down(engine, scene, ctrl):
    scene.selected_id = 2
    grow = lambda u: lerp_pose("together", "spread", smoothstep((u - 0.8) / 0.5))  # noqa: E731
    cmds = drive(engine, ctrl, run(engine, 2.0, grow))
    scales = of(cmds, C.SCALE)
    assert scales and all(c.object_id == 2 for c in scales)
    assert np.prod([c.value for c in scales]) > 1.3

    engine.reset()
    shrink = lambda u: lerp_pose("spread", "together", smoothstep((u - 0.8) / 0.5))  # noqa: E731
    cmds = drive(engine, ctrl, run(engine, 2.0, shrink, start=3.0))
    assert np.prod([c.value for c in of(cmds, C.SCALE)]) < 0.8


def test_zoom_without_a_selection_scales_the_whole_scene(engine, ctrl):
    grow = lambda u: lerp_pose("together", "spread", smoothstep((u - 0.8) / 0.5))  # noqa: E731
    scales = of(drive(engine, ctrl, run(engine, 2.0, grow)), C.SCALE)
    assert scales and all(c.object_id is None for c in scales)


def test_status_text_reports_the_latest_command(engine, ctrl):
    drive(engine, ctrl, run(engine, 1.0, "point", center=point_at(OBJECTS[0])))
    assert ctrl.state.status == "SELECT"
    states = run(engine, 2.0, "point", start=1.0, center=point_at(OBJECTS[0]))
    drive(engine, ctrl, states)
    assert ctrl.state.status == ""  # faded after command_display_s


def test_cursor_and_pinch_point_are_published_in_scene_coordinates(engine, ctrl):
    drive(engine, ctrl, run(engine, 1.0, "point", center=point_at((0.62, 0.33))))
    assert ctrl.state.cursor == pytest.approx((0.62, 0.33), abs=0.03)


def test_controller_reset_clears_state(engine, ctrl):
    drive(engine, ctrl, run(engine, 1.5, "fist"))
    ctrl.reset()
    assert ctrl.active and ctrl.state.phase in (Phase.STANDBY, Phase.IDLE)
