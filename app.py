"""GestureFlow - touchless gesture interface.

    python app.py                 # webcam
    python app.py --simulate      # scripted synthetic hand, no webcam needed
    python app.py --help          # all options

This module only wires the pieces together; each stage lives in its own package:

    camera/        frames in             (threaded capture, simulator)
    tracking/      frames -> hand landmarks (MediaPipe, filtering, primary hand)
    gestures/      landmarks -> features -> stable gesture state
    interaction/   gesture state -> scene commands
    visualization/ scene model, renderer, HUD
"""

from __future__ import annotations

import os

if os.name == "nt":  # must precede the first `import cv2` (see camera/__init__.py)
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse
import datetime
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from camera.capture import CameraCapture, CameraError, Frame, FrameSource, list_cameras
from camera.simulated import SimulatedCamera
from config import PROJECT_ROOT, AppConfig, CameraConfig, DisplayConfig, GestureConfig, TrackingConfig
from gestures.gesture_engine import GestureEngine
from gestures.static_gestures import RuleBasedClassifier, StaticClassifier
from gestures.types import GestureState
from interaction.commands import Command, CommandType
from interaction.controller import InteractionController
from tracking.hand_tracker import HandTracker, Tracker, TrackerError
from tracking.pipeline import TrackingPipeline
from tracking.simulated import SimulatedTracker, demo_script
from tracking.types import TrackingResult
from visualization.renderer import Renderer
from visualization.scene import SceneManager
from visualization.ui_state import GESTURE_GROUPS, FrameStats, UIState, ViewMode

SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
KEY_ESC = 27
KEY_TAB = 9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gestureflow",
        description="Real-time webcam hand-gesture interface (MediaPipe + OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cam = p.add_argument_group("camera")
    cam.add_argument("--camera", type=int, default=0, help="camera index")
    cam.add_argument("--width", type=int, default=640, help="requested capture width")
    cam.add_argument("--height", type=int, default=480, help="requested capture height")
    cam.add_argument("--fps", type=int, default=30, help="requested capture frame rate")
    cam.add_argument("--no-mirror", action="store_true", help="do not mirror the image (default is selfie view)")
    cam.add_argument("--list-cameras", action="store_true", help="list working camera indices and exit")

    trk = p.add_argument_group("tracking")
    trk.add_argument("--hand", choices=["Right", "Left", "Any"], default="Right", help="which hand controls the scene")
    trk.add_argument("--max-hands", type=int, default=TrackingConfig().max_hands, choices=[1, 2],
                     help="hands to track; 2 enables dominant-hand selection but roughly doubles inference time")
    trk.add_argument("--detection-confidence", type=float, default=0.6, help="min hand-detection confidence")
    trk.add_argument("--tracking-confidence", type=float, default=0.5, help="min landmark-tracking confidence")
    trk.add_argument("--no-smoothing", action="store_true", help="disable One-Euro landmark smoothing")
    trk.add_argument("--no-low-light-boost", action="store_true", help="do not brighten dark frames before detection")

    ges = p.add_argument_group("gestures")
    ges.add_argument("--classifier", choices=["rules", "learned", "hybrid"], default="rules",
                     help="static-gesture classifier (learned/hybrid need a trained model, see README)")
    ges.add_argument("--learned-model", type=Path, default=None, help="path to a trained static classifier (.joblib)")
    ges.add_argument("--score-smoothing", type=float, default=0.5, help="EMA alpha for gesture scores (lower = smoother)")

    ui = p.add_argument_group("display")
    ui.add_argument("--window", default=f"{DisplayConfig().window_width}x{DisplayConfig().window_height}", metavar="WxH",
                    help="initial window size; the layout follows the window when you resize it")
    ui.add_argument("--no-panel", action="store_true", help="hide the side panel at start")
    ui.add_argument("--mode", choices=["gesture", "visualization"], default="gesture", help="initial view mode")
    ui.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    ui.add_argument("--fullscreen", action="store_true")

    run = p.add_argument_group("run")
    run.add_argument("--simulate", action="store_true", help="drive the app with a scripted synthetic hand (no webcam)")
    run.add_argument("--sync", action="store_true", help="track on the main thread (slower; for debugging)")
    run.add_argument("--headless", action="store_true", help="no window; useful with --max-frames / --save-frame")
    run.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
    run.add_argument("--save-frame", type=str, default="", help="write the last composed frame to this image path")
    return p.parse_args(argv)


def parse_window(spec: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in spec.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--window expects WIDTHxHEIGHT (e.g. 1280x720), got {spec!r}") from None
    return max(w, 480), max(h, 320)


def build_config(args: argparse.Namespace) -> AppConfig:
    width, height = parse_window(args.window)
    camera = CameraConfig(index=args.camera, width=args.width, height=args.height, fps=args.fps, mirror=not args.no_mirror)
    tracking = replace(
        TrackingConfig(),
        dominant_hand=args.hand,
        max_hands=args.max_hands,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        smoothing_enabled=not args.no_smoothing,
        enhance_low_light=not args.no_low_light_boost,
    )
    gestures = replace(GestureConfig(), score_ema_alpha=args.score_smoothing)
    display = replace(DisplayConfig(), window_width=width, window_height=height, show_panel=not args.no_panel)
    cfg = AppConfig(
        camera=camera, tracking=tracking, gestures=gestures, display=display,
        classifier=args.classifier, simulate=args.simulate, headless=args.headless, threaded=not args.sync,
        max_frames=args.max_frames, save_frame=args.save_frame,
    )
    if args.learned_model is not None:
        cfg = replace(cfg, learned_model_path=args.learned_model)
    return cfg


def build_classifier(cfg: AppConfig) -> StaticClassifier:
    rules = RuleBasedClassifier(cfg.gestures.static)
    if cfg.classifier == "rules":
        return rules
    from gestures.learned_classifier import HybridClassifier, LearnedClassifier, LearnedModelError

    try:
        learned = LearnedClassifier.load(cfg.learned_model_path)
    except LearnedModelError as exc:
        print(f"[gestures] {exc}\n[gestures] Falling back to the rule-based classifier.")
        return rules
    return learned if cfg.classifier == "learned" else HybridClassifier(rules, learned)


@dataclass
class _Timing:
    """Running totals for the end-of-run performance summary."""

    frames: int = 0
    track: float = 0.0
    gesture: float = 0.0
    render: float = 0.0
    start: float = 0.0

    def summary(self) -> str:
        if not self.frames:
            return "no frames processed"
        wall = max(time.perf_counter() - self.start, 1e-9)
        n = self.frames
        return (f"{n} frames, {n / wall:.1f} FPS average | per frame: tracking {self.track / n:.1f} ms, "
                f"gestures+interaction {self.gesture / n:.2f} ms, rendering {self.render / n:.1f} ms")


class GestureFlowApp:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.source: FrameSource | None = None
        self.tracker: Tracker | None = None
        self.pipeline: TrackingPipeline | None = None
        self.running = False
        self.last_canvas: np.ndarray | None = None
        self.timing = _Timing()
        self.fullscreen = False
        self.last_t = 0.0

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        cfg = self.cfg
        if cfg.simulate:
            self.source = SimulatedCamera(cfg.camera.width, cfg.camera.height, cfg.camera.fps, realtime=not cfg.headless)
        else:
            self.source = CameraCapture(cfg.camera)
        self.source.start()  # may raise CameraError
        aspect = self.source.width / self.source.height

        self.scenes = SceneManager(cfg.display, aspect, cfg.interaction)
        if cfg.simulate:
            script = demo_script(lambda i: self.scenes.object_position(i % len(self.scenes.current.objects)) or (0.5, 0.5))
            self.tracker = SimulatedTracker(script)
        else:
            self.tracker = HandTracker(cfg.tracking)  # may raise TrackerError

        # Track on a worker thread so inference overlaps rendering (inline for deterministic headless simulation).
        self.pipeline = TrackingPipeline(self.source, self.tracker, threaded=cfg.threaded and not (cfg.headless and cfg.simulate), read_timeout_s=cfg.camera.read_timeout_s)
        self.pipeline.start()

        self.engine = GestureEngine(cfg.gestures, build_classifier(cfg))
        self.controller = InteractionController(self.scenes, cfg.interaction)
        self.renderer = Renderer(cfg.display, aspect, cfg.interaction.active_margin)
        self._synced_layout = -1
        self.ui = UIState(show_panel=cfg.display.show_panel)
        self._sync_layout()

    def apply_cli_ui(self, args: argparse.Namespace) -> None:
        self.ui.mode = ViewMode.GESTURE if args.mode == "gesture" else ViewMode.VISUALIZATION
        self.ui.show_debug = args.debug
        self.fullscreen = args.fullscreen

    # ---------------------------------------------------------------- main loop
    def run(self) -> int:
        cfg = self.cfg
        window = cfg.display.window_name
        if not cfg.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window, self.cfg.display.window_width, self.cfg.display.window_height)
            cv2.setMouseCallback(window, self._on_mouse)
            if self.fullscreen:
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.running = True
        self.timing.start = time.perf_counter()
        last_t: float | None = None
        last_frame: Frame | None = None
        fps = 0.0
        last_wall = time.perf_counter()
        tracking, state = TrackingResult(), GestureState()

        assert self.pipeline is not None
        frame_no = 0
        while self.running:
            frame_no += 1
            if not cfg.headless and frame_no % 6 == 1:
                self._follow_window_size()
            self._sync_layout()
            item = self.pipeline.get()
            stalled = item is None
            if item is not None:
                last_frame = item.frame
                tracking = item.tracking
                t = item.frame.timestamp
                dt = 0.0 if last_t is None else t - last_t
                last_t = self.last_t = t

                t1 = time.perf_counter()
                state = self.engine.process(tracking.primary, t)
                commands = self.controller.update(state, t)
                self._apply(commands, t)
                self.scenes.update(dt)
                t2 = time.perf_counter()
                track_ms = item.track_ms
            else:
                t = last_t or 0.0
                t1 = t2 = time.perf_counter()
                track_ms = 0.0

            now = time.perf_counter()
            if now > last_wall:
                inst = 1.0 / (now - last_wall)
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            last_wall = now

            stats = FrameStats(
                fps=fps,
                frame_ms=track_ms + (t2 - t1) * 1000,
                track_ms=track_ms,
                gesture_ms=(t2 - t1) * 1000,
                render_ms=self.timing.render / max(self.timing.frames, 1),
                camera_status=self.pipeline.status,
                simulated=cfg.simulate,
            )
            r0 = time.perf_counter()
            canvas = self.renderer.render(
                last_frame.image if last_frame else None, tracking, state, self.controller.state, self.scenes, self.ui, stats, t
            )
            r1 = time.perf_counter()
            self.last_canvas = canvas

            if not stalled:
                self.timing.frames += 1
                self.timing.track += track_ms
                self.timing.gesture += (t2 - t1) * 1000
                self.timing.render += (r1 - r0) * 1000

            if not cfg.headless:
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._on_key(key, t)
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
            if cfg.max_frames and self.timing.frames >= cfg.max_frames:
                self.running = False
        return 0

    # ----------------------------------------------------------------- layout
    def _follow_window_size(self) -> None:
        """Render at the live window size (capped), so text stays crisp and the layout is responsive."""
        try:
            _, _, w, h = cv2.getWindowImageRect(self.cfg.display.window_name)
        except cv2.error:
            return
        if w < 320 or h < 240:  # minimised / not yet realised
            return
        cap = self.cfg.display.max_render_width
        if w > cap:
            h, w = int(h * cap / w), cap
        self.renderer.ensure_size(w, h, self.ui.show_panel)

    def _sync_layout(self) -> None:
        """Keep the camera->scene mapping and the scene's aspect in step with the canvas shape."""
        self.renderer.ensure_size(*self.renderer.size, self.ui.show_panel)
        if self.renderer.layout_version == self._synced_layout:
            return
        self._synced_layout = self.renderer.layout_version
        self.controller.set_margin(self.renderer.margin)
        self.scenes.set_aspect(self.renderer.view_aspect)
        script = getattr(self.tracker, "script", None)
        if script is not None:  # the simulated hand aims through the same mapping
            script.margin = self.renderer.margin

    # --------------------------------------------------------------- commands
    def _apply(self, commands: list[Command], t: float) -> None:
        for cmd in commands:
            self.scenes.apply(cmd)
            if cmd.type in (CommandType.NEXT_SCENE, CommandType.PREV_SCENE):
                n = len(self.scenes.scenes)
                self.ui.say(self.scenes.current.preset.name, t, f"scene {self.scenes.index + 1} of {n}")
            elif cmd.type is CommandType.SET_ACTIVE:
                if cmd.value > 0:  # entering standby is already announced on the canvas
                    self.ui.say("Gestures on", t)
            elif cmd.type is CommandType.RESET:
                self.ui.say("Scene reset", t)

    # ------------------------------------------------------------------- input
    def _on_key(self, key: int, t: float) -> None:
        ui = self.ui
        k = chr(key).lower() if 32 <= key < 127 else key
        if k in ("q", KEY_ESC):
            self.running = False
        elif k == "r":
            self.controller.reset()
            self.scenes.apply(Command(CommandType.RESET))
            ui.say("Scene reset", t)
        elif k == "1":
            ui.mode = ViewMode.GESTURE
            ui.say("Gesture mode", t, duration=1.0)
        elif k == "2":
            ui.mode = ViewMode.VISUALIZATION
            ui.say("Visualization mode", t, duration=1.0)
        elif k == "h":
            ui.show_help = not ui.show_help
        elif k == "d":
            ui.show_debug = not ui.show_debug
            ui.say("Debug on" if ui.show_debug else "Debug off", t, duration=0.9)
        elif k == "g":
            ui.show_panel = not ui.show_panel
        elif k in ("n", "p"):
            self._apply([Command(CommandType.NEXT_SCENE if k == "n" else CommandType.PREV_SCENE)], t)
        elif k == KEY_TAB:
            self.scenes.apply(Command(CommandType.CYCLE, value=1.0))
        elif k == "m":
            self._toggle_two_hands(t)
        elif k == "f":
            self._toggle_fullscreen()
        elif k == "s":
            self.save_screenshot(t)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for box in self.renderer.hitboxes:
            if not box.contains(x, y):
                continue
            kind, value = box.action
            if kind == "group_select":
                self.ui.select_group(value)
            elif kind == "group_toggle":
                self.ui.toggle_group(next(g for g in GESTURE_GROUPS if g.id == value))
                self.engine.set_enabled(self.ui.enabled)
            elif kind == "member_toggle":
                self.ui.toggle_gesture(value)
                self.engine.set_enabled(self.ui.enabled)
            elif kind == "mode":
                self.ui.mode = value
            elif kind == "scene":
                self._apply([Command(CommandType.NEXT_SCENE if value > 0 else CommandType.PREV_SCENE)], self.last_t)
            return

    def _toggle_two_hands(self, t: float) -> None:
        """Switch between one-hand (fast) and two-hand (dominant-hand selection) tracking."""
        tracker = self.tracker
        if not hasattr(tracker, "set_max_hands"):
            self.ui.say("Two-hand mode", t, "not available in simulation", 1.2)
            return
        two = tracker.max_hands != 2
        tracker.set_max_hands(2 if two else 1)
        self.ui.say("Two-hand tracking" if two else "One-hand tracking", t,
                    "slower; the dominant hand controls" if two else "fastest", 1.4)

    def _toggle_fullscreen(self) -> None:
        window = self.cfg.display.window_name
        current = cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN)
        mode = cv2.WINDOW_NORMAL if current == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, mode)

    def save_screenshot(self, t: float) -> Path | None:
        if self.last_canvas is None:
            return None
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        path = SCREENSHOT_DIR / f"gestureflow_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
        cv2.imwrite(str(path), self.last_canvas)
        self.ui.say("Screenshot saved", t, path.name, 1.4)
        return path

    # ---------------------------------------------------------------- shutdown
    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()  # stop the worker before closing what it uses
        if self.tracker is not None:
            self.tracker.close()
        if self.source is not None:
            self.source.stop()
        if not self.cfg.headless:
            cv2.destroyAllWindows()
        if self.cfg.save_frame and self.last_canvas is not None:
            cv2.imwrite(self.cfg.save_frame, self.last_canvas)
            print(f"[app] saved final frame to {self.cfg.save_frame}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cameras:
        found = list_cameras()
        print("Working cameras:", found if found else "none found")
        return 0

    cfg = build_config(args)
    app = GestureFlowApp(cfg)
    try:
        app.setup()
    except CameraError as exc:
        print(f"\n[camera] {exc}\n\nTip: run `python app.py --simulate` to try GestureFlow without a webcam.", file=sys.stderr)
        app.close()
        return 1
    except TrackerError as exc:
        print(f"\n[tracking] {exc}", file=sys.stderr)
        app.close()
        return 2

    app.apply_cli_ui(args)
    try:
        code = app.run()
    except KeyboardInterrupt:
        code = 0
    except RuntimeError as exc:  # e.g. the tracker crashed on its worker thread
        print(f"\n[error] {exc}: {exc.__cause__ or ''}", file=sys.stderr)
        code = 3
    finally:
        app.close()
    print("[app] " + app.timing.summary())
    return code


if __name__ == "__main__":
    sys.exit(main())
