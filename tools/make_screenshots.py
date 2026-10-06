"""Render README screenshots from the simulated demo (no webcam involved).

    python tools/make_screenshots.py                 # writes docs/images/*.png at 1280x720
    python tools/make_screenshots.py --window 960x600

Plays the scripted demo hand through the real pipeline and saves frames at moments that show
each feature.  The hand you see is the synthetic hand model, not a person.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import app as gestureflow_app  # noqa: E402
from visualization.ui_state import FrameStats, ViewMode  # noqa: E402

# (file name, demo time in seconds, UI overrides)
SHOTS = [
    ("waiting", 0.3, {}),
    ("point-select", 2.6, {}),
    ("pinch-drag", 4.4, {}),
    ("rotate", 7.0, {}),
    ("visualization-mode", 5.0, {"mode": ViewMode.VISUALIZATION}),
    ("scene-switch", 15.8, {}),
    ("details", 3.4, {"selected_group": "pinch"}),
    ("debug-overlay", 3.2, {"show_debug": True}),
    ("help", 4.8, {"show_help": True}),
    ("standby", 20.7, {}),
]


def main(argv: list[str]) -> None:
    out_dir = ROOT / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = gestureflow_app.parse_args(["--simulate", "--headless", *argv])
    app = gestureflow_app.GestureFlowApp(gestureflow_app.build_config(args))
    app.setup()
    wanted = {round(t * 30): (name, ui) for name, t, ui in SHOTS}
    last_t = None
    for index in range(1, 30 * 28):
        app._sync_layout()
        item = app.pipeline.get()
        t = item.frame.timestamp
        state = app.engine.process(item.tracking.primary, t)
        for cmd in app.controller.update(state, t):
            app._apply([cmd], t)
        app.scenes.update(0.0 if last_t is None else t - last_t)
        last_t = t
        name, ui = wanted.get(index, (None, {}))
        app.ui.mode = ui.get("mode", ViewMode.GESTURE)
        app.ui.show_debug = ui.get("show_debug", False)
        app.ui.show_help = ui.get("show_help", False)
        app.ui.selected_group = ui.get("selected_group")
        canvas = app.renderer.render(
            item.frame.image, item.tracking, state, app.controller.state, app.scenes, app.ui,
            FrameStats(fps=30.0, track_ms=9.0, gesture_ms=1.0, render_ms=6.0, simulated=True), t,
        )
        if name:
            cv2.imwrite(str(out_dir / f"{name}.png"), canvas)
            print("wrote", out_dir / f"{name}.png")
    app.close()


if __name__ == "__main__":
    main(sys.argv[1:])
