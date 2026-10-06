"""End-to-end: the scripted demo hand drives the real engine, controller, scene and renderer.

This is the integration test of the whole project (everything except MediaPipe and the
webcam).  If a refactor breaks how the pieces fit together, this is where it shows.
"""

import numpy as np
import pytest

import app as gestureflow_app
from interaction.commands import CommandType as C
from visualization.ui_state import GESTURE_GROUPS, FrameStats, ViewMode


def make_app(*extra: str) -> gestureflow_app.GestureFlowApp:
    args = gestureflow_app.parse_args(["--simulate", "--headless", *extra])
    app = gestureflow_app.GestureFlowApp(gestureflow_app.build_config(args))
    app.setup()
    app.apply_cli_ui(args)
    return app


@pytest.fixture(scope="module")
def demo_run():
    """Play the whole demo script once, returning the commands applied and a few probes."""
    app = make_app()
    scene = app.scenes
    first_obj = scene.current.objects[0]
    start_pos = first_obj.pos.copy()
    script = app.tracker.script
    n = int(script.duration * 30) + 5
    commands, phases, positions = [], set(), []
    for _ in range(n):
        item = app.pipeline.get()
        t = item.frame.timestamp
        state = app.engine.process(item.tracking.primary, t)
        cmds = app.controller.update(state, t)
        for c in cmds:
            scene.apply(c)
        commands.extend(cmds)
        scene.update(1 / 30)
        phases.add(app.controller.state.phase.name)
        positions.append(first_obj.pos.copy())
    app.close()
    return commands, phases, start_pos, positions, scene


def kinds(commands):
    return [c.type for c in commands]


def test_demo_exercises_every_command_type(demo_run):
    commands, *_ = demo_run
    seen = set(kinds(commands))
    expected = {C.HOVER, C.SELECT, C.GRAB, C.MOVE, C.RELEASE, C.ROTATE, C.SCALE, C.CYCLE, C.NEXT_SCENE, C.PREV_SCENE,
                C.RESET, C.SET_PAUSED, C.SET_ACTIVE}
    assert expected <= seen, f"missing: {sorted(t.name for t in expected - seen)}"


def test_demo_goes_through_the_expected_interaction_phases(demo_run):
    _, phases, *_ = demo_run
    assert {"POINTING", "GRABBED", "DRAGGING", "ROTATING", "ZOOMING", "PAUSED", "SWITCHING", "STANDBY"} <= phases


def test_demo_is_free_of_spurious_actions(demo_run):
    commands, *_ = demo_run
    count = lambda t: sum(c.type is t for c in commands)  # noqa: E731
    assert count(C.GRAB) == 1  # one deliberate grab, none caused by transitional hand shapes
    assert count(C.NEXT_SCENE) == 1 and count(C.PREV_SCENE) == 1  # the swipe's return stroke is ignored
    assert count(C.RESET) == 1
    assert [c.value for c in commands if c.type is C.SET_ACTIVE] == [0.0, 1.0]  # standby, then active


def test_demo_zooms_in_and_out(demo_run):
    scales = [c.value for c in demo_run[0] if c.type is C.SCALE]
    assert any(v > 1 for v in scales) and any(v < 1 for v in scales)
    assert np.prod([v for v in scales if v > 1]) > 1.2


def test_demo_drags_the_object_a_long_way(demo_run):
    _, _, start, positions, _ = demo_run
    travelled = max(float(np.linalg.norm(p - start)) for p in positions)
    assert travelled > 0.25


def test_demo_ends_in_a_consistent_state(demo_run):
    *_, scene = demo_run
    assert scene.current.grabbed_id is None and scene.active
    assert not scene.paused


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_one(app, frames=140, **ui):
    for k, v in ui.items():
        setattr(app.ui, k, v)
    out = None
    for _ in range(frames):
        item = app.pipeline.get()
        t = item.frame.timestamp
        state = app.engine.process(item.tracking.primary, t)
        for c in app.controller.update(state, t):
            app.scenes.apply(c)
        app.scenes.update(1 / 30)
        out = app.renderer.render(item.frame.image, item.tracking, state, app.controller.state, app.scenes, app.ui,
                                  FrameStats(fps=30, simulated=True), t)
    return out


@pytest.mark.parametrize("ui", [
    {}, {"mode": ViewMode.VISUALIZATION}, {"show_help": True}, {"show_debug": True}, {"show_panel": False},
    {"mode": ViewMode.VISUALIZATION, "show_debug": True, "show_panel": False},
])
def test_every_view_renders_a_valid_frame(ui):
    app = make_app()
    canvas = render_one(app, **ui)
    h, w, c = canvas.shape
    assert c == 3 and canvas.dtype == np.uint8
    lay = app.renderer.layout
    assert (w, h) == (lay.width, lay.height) == app.renderer.canvas_size
    assert lay.sidebar > 0 if app.ui.show_panel else lay.sidebar == 0
    assert canvas.mean() > 8 and canvas.std() > 10  # something is actually drawn
    app.close()


def test_rendering_does_not_change_the_scene():
    app = make_app()
    render_one(app, frames=100)
    before = [o.pos.copy() for o in app.scenes.current.objects]
    app.renderer.render(None, app.tracker.process(np.zeros((480, 640, 3), np.uint8), 0.0), app.engine.process(None, 99.0),
                        app.controller.state, app.scenes, app.ui, FrameStats(), 99.0)
    assert all(np.array_equal(b, o.pos) for b, o in zip(before, app.scenes.current.objects))
    app.close()


def test_rendering_survives_a_stalled_camera_and_standby():
    app = make_app()
    render_one(app, frames=40)
    canvas = app.renderer.render(None, app.tracker.process(np.zeros((480, 640, 3), np.uint8), 0.0), app.engine.process(None, 50.0),
                                 app.controller.state, app.scenes, app.ui, FrameStats(camera_status="stalled"), 50.0)
    assert canvas.shape[:2] == (app.renderer.layout.height, app.renderer.layout.width)
    app.close()


def test_side_panel_exposes_clickable_rows_modes_and_scene_arrows():
    app = make_app()
    render_one(app, frames=10)
    actions = [b.action for b in app.renderer.hitboxes]
    assert {a[1] for a in actions if a[0] == "group_select"} == {g.id for g in GESTURE_GROUPS}
    assert {a[1] for a in actions if a[0] == "group_toggle"} == {g.id for g in GESTURE_GROUPS}
    assert {a[1] for a in actions if a[0] == "mode"} == {ViewMode.GESTURE, ViewMode.VISUALIZATION}
    assert {a[1] for a in actions if a[0] == "scene"} == {-1, 1}
    sidebar_x = app.renderer.layout.view_w
    assert all(b.x0 >= 0 and b.x1 <= app.renderer.layout.width and b.y1 <= app.renderer.layout.height for b in app.renderer.hitboxes)
    assert all(b.x1 > sidebar_x for b in app.renderer.hitboxes)  # all controls live in the side panel
    app.close()


def _click(app, action):
    import cv2

    box = next(b for b in app.renderer.hitboxes if b.action == action)
    app._on_mouse(cv2.EVENT_LBUTTONDOWN, (box.x0 + box.x1) // 2, (box.y0 + box.y1) // 2, 0, None)


def test_clicking_a_status_dot_toggles_a_whole_gesture_family_in_the_engine():
    from gestures.types import Gesture

    app = make_app()
    render_one(app, frames=10)
    pinch_family = {Gesture.PINCH, Gesture.PINCH_DRAG, Gesture.PINCH_ROTATE}
    _click(app, ("group_toggle", "pinch"))
    assert not (pinch_family & app.engine.enabled) and not (pinch_family & app.ui.enabled)
    assert Gesture.POINT in app.engine.enabled  # other families untouched
    render_one(app, frames=2)
    _click(app, ("group_toggle", "pinch"))
    assert pinch_family <= app.engine.enabled
    app.close()


def test_selecting_a_row_reveals_per_gesture_controls_only_for_that_row():
    from gestures.types import Gesture

    app = make_app()
    render_one(app, frames=10)
    assert not [b for b in app.renderer.hitboxes if b.action[0] == "member_toggle"]  # nothing configurable until selected
    _click(app, ("group_select", "pinch"))
    render_one(app, frames=2)
    members = {b.action[1] for b in app.renderer.hitboxes if b.action[0] == "member_toggle"}
    assert members == {Gesture.PINCH, Gesture.PINCH_DRAG, Gesture.PINCH_ROTATE}
    _click(app, ("member_toggle", Gesture.PINCH_ROTATE))
    assert Gesture.PINCH_ROTATE not in app.engine.enabled and Gesture.PINCH_DRAG in app.engine.enabled
    _click(app, ("group_select", "pinch"))  # selecting again closes the details
    render_one(app, frames=2)
    assert not [b for b in app.renderer.hitboxes if b.action[0] == "member_toggle"]
    app.close()


def test_mode_buttons_and_scene_arrows_work():
    app = make_app()
    render_one(app, frames=10)
    _click(app, ("mode", ViewMode.VISUALIZATION))
    assert app.ui.mode is ViewMode.VISUALIZATION
    start = app.scenes.index
    _click(app, ("scene", 1))
    assert app.scenes.index == (start + 1) % len(app.scenes.scenes)
    app.close()


def test_keyboard_shortcuts():
    app = make_app()
    render_one(app, frames=5)
    press = lambda ch: app._on_key(ord(ch), 1.0)  # noqa: E731
    press("2"); assert app.ui.mode is ViewMode.VISUALIZATION
    press("1"); assert app.ui.mode is ViewMode.GESTURE
    press("h"); assert app.ui.show_help
    press("h"); assert not app.ui.show_help
    press("d"); assert app.ui.show_debug
    press("g"); assert not app.ui.show_panel
    first = app.scenes.index
    press("n"); assert app.scenes.index == (first + 1) % len(app.scenes.scenes)
    press("r"); assert app.controller.active
    press("q"); assert not app.running
    app.close()


def test_screenshot_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(gestureflow_app, "SCREENSHOT_DIR", tmp_path)
    app = make_app()
    app.last_canvas = render_one(app, frames=10)  # run() normally records this
    path = app.save_screenshot(1.0)
    assert path is not None and path.exists() and path.stat().st_size > 5_000
    app.close()


def test_headless_run_honours_max_frames_and_saves_the_last_frame(tmp_path):
    out = tmp_path / "last.png"
    args = gestureflow_app.parse_args(["--simulate", "--headless", "--max-frames", "25", "--save-frame", str(out)])
    app = gestureflow_app.GestureFlowApp(gestureflow_app.build_config(args))
    app.setup()
    assert app.run() == 0
    app.close()
    assert app.timing.frames == 25 and out.exists()
