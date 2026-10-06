"""Presentation-layer tests: responsive layout, camera->view mapping, typography, feedback cues."""

import numpy as np
import pytest

import app as gestureflow_app
from gestures.types import Gesture
from interaction.mapping import camera_to_scene, camera_to_view, scene_to_camera, view_margins
from visualization import icons, typeface
from visualization.hud import Layout
from visualization.renderer import highlighted_landmarks
from visualization.ui_state import GESTURE_GROUPS, FrameStats, UIState, ViewMode, group_of

SIZES = [(640, 480), (760, 480), (800, 520), (960, 600), (1280, 720), (1366, 768), (1600, 900), (1920, 1080), (1000, 1000), (1400, 600)]


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("w,h", SIZES)
def test_canvas_is_the_hero_at_every_window_size(w, h):
    lay = Layout.compute(w, h, show_panel=True)
    assert lay.view_w + lay.sidebar == w and lay.top + lay.view_h == h
    assert lay.view_w / w >= 0.65  # the camera/visualisation gets the clear majority
    assert lay.top <= 0.1 * h + 24  # a slim bar, not a banner
    if lay.sidebar:
        assert lay.sidebar >= 238  # never squeezed below a readable width
        assert lay.sidebar / w <= 0.36


def test_sidebar_is_about_a_quarter_on_a_typical_window_and_collapses_when_narrow():
    typical = Layout.compute(1280, 720, True)
    assert 0.22 <= typical.sidebar / 1280 <= 0.30 and typical.top <= 48
    assert Layout.compute(700, 500, True).sidebar == 0  # too narrow for a panel
    assert Layout.compute(1280, 720, False).sidebar == 0  # hidden on request
    assert Layout.compute(1280, 720, False).view_w == 1280


def test_ui_scale_follows_the_window_within_sane_limits():
    assert Layout.compute(1280, 720, True).scale == pytest.approx(1.0)
    assert Layout.compute(3840, 2160, True).scale <= 1.35
    assert Layout.compute(640, 480, True).scale >= 0.85
    lay = Layout.compute(1920, 1080, True)
    assert lay.u(10) > 10 and Layout.compute(640, 480, True).u(10) <= 10


RENDER_SIZES = [(640, 480), (800, 520), (1280, 720), (1000, 1000), (1920, 1080)]  # includes collapsed panel, square, 1080p


@pytest.mark.parametrize("w,h", RENDER_SIZES)
@pytest.mark.parametrize("mode", [ViewMode.GESTURE, ViewMode.VISUALIZATION])
def test_every_size_and_mode_renders_exactly_the_requested_canvas(w, h, mode):
    args = gestureflow_app.parse_args(["--simulate", "--headless", "--window", f"{w}x{h}"])
    app = gestureflow_app.GestureFlowApp(gestureflow_app.build_config(args))
    app.setup()
    app.ui.mode = mode
    canvas = None
    for _ in range(45):  # the demo hand appears after ~0.6 s
        app._sync_layout()
        item = app.pipeline.get()
        t = item.frame.timestamp
        state = app.engine.process(item.tracking.primary, t)
        for cmd in app.controller.update(state, t):
            app._apply([cmd], t)
        app.scenes.update(1 / 30)
        canvas = app.renderer.render(item.frame.image, item.tracking, state, app.controller.state, app.scenes, app.ui, FrameStats(fps=30), t)
    assert canvas.shape == (h, w, 3)
    assert canvas.std() > 8
    app.close()


def test_resizing_the_window_updates_mapping_and_scene_geometry():
    args = gestureflow_app.parse_args(["--simulate", "--headless", "--window", "1280x720"])
    app = gestureflow_app.GestureFlowApp(gestureflow_app.build_config(args))
    app.setup()
    before = (app.renderer.view_aspect, tuple(app.renderer.margin), app.scenes.aspect)
    app.renderer.ensure_size(900, 900, True)
    app._sync_layout()
    after = (app.renderer.view_aspect, tuple(app.renderer.margin), app.scenes.aspect)
    assert after[0] != before[0] and after[1] != before[1]
    assert app.scenes.aspect == pytest.approx(app.renderer.view_aspect)
    assert app.controller._margin == tuple(app.renderer.margin)
    app.close()


def test_toggling_the_panel_changes_the_canvas_width_not_the_window():
    args = gestureflow_app.parse_args(["--simulate", "--headless"])
    app = gestureflow_app.GestureFlowApp(gestureflow_app.build_config(args))
    app.setup()
    wide = app.renderer.layout.view_w
    app.ui.show_panel = False
    app._sync_layout()
    assert app.renderer.layout.view_w == app.renderer.layout.width > wide
    app.close()


# --------------------------------------------------------------------------- #
# camera -> view mapping with a non-camera aspect ratio
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view_aspect", [0.8, 1.0, 4 / 3, 1.6, 2.2])
def test_view_margins_make_the_visible_camera_region_match_the_viewport_aspect(view_aspect):
    cam = 4 / 3
    mx, my = view_margins(view_aspect, cam, 0.1)
    visible_aspect = (1 - 2 * mx) * cam / (1 - 2 * my)  # pixels of the visible camera crop
    assert visible_aspect == pytest.approx(view_aspect, rel=1e-6)  # no stretching
    assert 0.0 <= mx < 0.5 and 0.0 <= my < 0.5
    assert min(mx, my) == pytest.approx(0.1)  # the axis with room to spare keeps the base margin


def test_matching_aspect_gives_the_plain_square_margin():
    assert view_margins(4 / 3, 4 / 3, 0.1) == pytest.approx((0.1, 0.1))


def test_per_axis_mapping_round_trips():
    margin = (0.15, 0.1)
    pts = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]])
    scene = camera_to_scene(pts, margin)
    assert np.allclose(scene_to_camera(scene, margin), pts, atol=1e-9)
    assert np.allclose(camera_to_view(np.array([0.5, 0.5]), margin), [0.5, 0.5])  # centre stays centred
    assert np.allclose(camera_to_view(np.array([0.15, 0.1]), margin), [0.0, 0.0])  # region corner -> viewport corner


def test_scalar_margin_still_works_everywhere():
    assert np.allclose(camera_to_scene(np.array([0.1, 0.1]), 0.1), [0.0, 0.0])


# --------------------------------------------------------------------------- #
# typography + icons
# --------------------------------------------------------------------------- #
def test_text_is_drawn_only_inside_its_own_region():
    img = np.full((60, 200, 3), 20, np.uint8)
    w = typeface.text(img, "Tracking", 30, 30, 14, (255, 255, 255), "medium")
    changed = np.argwhere((img != 20).any(axis=2))
    assert w > 20 and len(changed)
    assert changed[:, 1].min() >= 28 and changed[:, 1].max() <= 30 + w + 2  # starts at x=30, stays within its width
    assert changed[:, 0].min() >= 15 and changed[:, 0].max() <= 45


def test_text_anchors_and_alpha():
    base = np.zeros((40, 240, 3), np.uint8)
    for anchor in ("l", "c", "r"):
        img = base.copy()
        typeface.text(img, "Hand", 120, 20, 13, (255, 255, 255), anchor=anchor)
        cols = np.argwhere(img.any(axis=2))[:, 1]
        centre = (cols.min() + cols.max()) / 2
        assert {"l": centre > 120, "c": abs(centre - 120) < 4, "r": centre < 120}[anchor]
    faint, full = base.copy(), base.copy()
    typeface.text(faint, "Hand", 20, 20, 13, (255, 255, 255), alpha=0.3)
    typeface.text(full, "Hand", 20, 20, 13, (255, 255, 255), alpha=1.0)
    assert faint.max() < full.max()


def test_every_weight_resolves_to_a_font_and_measures_consistently():
    for weight in ("regular", "medium", "semibold"):
        w, h = typeface.measure("GestureFlow", 14, weight)
        assert w > 40 and 10 < h < 30
    assert typeface.measure("Wide text here", 14)[0] > typeface.measure("Wide", 14)[0]
    assert typeface.measure("ABC", 14, tracking=2)[0] > typeface.measure("ABC", 14)[0]


def test_bundled_font_is_present_and_licensed():
    assert (typeface.FONT_DIR / "Inter-Regular.otf").exists() and (typeface.FONT_DIR / "OFL.txt").exists()


@pytest.mark.parametrize("name", list(icons.ICONS))
@pytest.mark.parametrize("size", [14, 16, 20])
def test_every_icon_renders_inside_its_box(name, size):
    img = np.zeros((size + 8, size + 8, 3), np.uint8)
    icons.draw_icon(img, name, (size + 8) // 2, (size + 8) // 2, size, (255, 255, 255))
    ys, xs = np.nonzero(img.any(axis=2))
    assert len(ys) > size  # something visible
    assert xs.min() >= 3 and xs.max() <= size + 4 and ys.min() >= 3 and ys.max() <= size + 4


# --------------------------------------------------------------------------- #
# gesture families and landmark highlighting
# --------------------------------------------------------------------------- #
def test_every_gesture_belongs_to_exactly_one_listed_family():
    listed = [g for grp in GESTURE_GROUPS for g in grp.members]
    assert len(listed) == len(set(listed))
    assert set(listed) == {g for g in Gesture if g is not Gesture.NONE}
    assert group_of(Gesture.PINCH_DRAG).id == "pinch" and group_of(Gesture.NONE) is None


def test_ui_group_toggle_is_all_or_nothing_and_selection_is_a_toggle():
    ui = UIState()
    pinch = next(g for g in GESTURE_GROUPS if g.id == "pinch")
    ui.toggle_gesture(Gesture.PINCH_ROTATE)  # partially disabled family
    ui.toggle_group(pinch)  # any member on -> everything off
    assert not (ui.enabled & set(pinch.members))
    ui.toggle_group(pinch)
    assert set(pinch.members) <= ui.enabled
    ui.select_group("pinch")
    assert ui.selected_group == "pinch"
    ui.select_group("pinch")
    assert ui.selected_group is None


@pytest.mark.parametrize("gesture,points", [
    (Gesture.PINCH, {4, 8}), (Gesture.PINCH_DRAG, {4, 8}), (Gesture.PINCH_ROTATE, {4, 8}),
    (Gesture.POINT, {5, 6, 7, 8}), (Gesture.OPEN_PALM, {4, 8, 12, 16, 20}),
])
def test_only_the_landmarks_that_define_a_gesture_are_highlighted(gesture, points):
    hl, segs = highlighted_landmarks(gesture)
    assert hl == points and len(hl) < 21
    assert all(a in hl and b in hl for a, b in segs)


def test_pinch_highlights_the_thumb_to_index_connection():
    assert highlighted_landmarks(Gesture.PINCH)[1] == [(4, 8)]
    assert highlighted_landmarks(Gesture.NONE) == (set(), [])


def test_waiting_state_is_small_and_disappears_when_a_hand_is_tracked(monkeypatch):
    from tests.test_simulation import make_app
    from visualization.hud import Hud

    def frame_at(app, n):
        out = state = None
        for _ in range(n):
            app._sync_layout()
            item = app.pipeline.get()
            t = item.frame.timestamp
            state = app.engine.process(item.tracking.primary, t)
            out = app.renderer.render(item.frame.image, item.tracking, state, app.controller.state, app.scenes, app.ui, FrameStats(fps=30), t)
        return out.copy(), state

    app = make_app()
    app.ui.show_panel = False
    with_text, state = frame_at(app, 4)
    assert not state.hand_present  # the demo starts with an empty frame
    app.close()

    app2 = make_app()
    app2.ui.show_panel = False
    monkeypatch.setattr(Hud, "_empty_state", staticmethod(lambda view, lay: None))
    without, _ = frame_at(app2, 4)
    changed = (np.abs(with_text.astype(int) - without.astype(int)).max(axis=2) > 12)
    assert 0 < changed.mean() < 0.01  # a small mark and one quiet line - under 1% of the canvas
    app2.close()

    monkeypatch.undo()
    app3 = make_app()
    app3.ui.show_panel = False
    shown, state = frame_at(app3, 150)  # a hand is tracked by now
    assert state.hand_present
    app3.close()
