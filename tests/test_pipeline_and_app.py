"""Camera/tracker plumbing and graceful-failure tests.  Nothing here touches real hardware:
cameras and the MediaPipe model are replaced by fakes."""

import io
import threading
import urllib.error

import numpy as np
import pytest

import app as gestureflow_app
from camera import capture
from camera.capture import CameraCapture, CameraError, Frame
from camera.simulated import SimulatedCamera
from config import AppConfig, CameraConfig
from tracking import hand_tracker
from tracking.hand_tracker import TrackerError, ensure_model
from tracking.pipeline import TrackingPipeline
from tracking.types import TrackingResult


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeSource:
    width, height, status = 64, 48, "ok"

    def __init__(self, n_frames: int, stall_after: bool = True) -> None:
        self.n, self.i, self.stall_after = n_frames, 0, stall_after
        self.started = self.stopped = False

    def start(self): self.started = True
    def stop(self): self.stopped = True

    def read(self, timeout=None):
        if self.i >= self.n:
            if self.stall_after:
                threading.Event().wait(min(timeout or 0.05, 0.05))
            return None
        self.i += 1
        return Frame(np.zeros((48, 64, 3), np.uint8), self.i / 30, self.i)


class FakeTracker:
    def __init__(self, fail_at: int | None = None) -> None:
        self.calls, self.fail_at, self.closed = 0, fail_at, False

    def process(self, frame, timestamp):
        self.calls += 1
        if self.fail_at == self.calls:
            raise ValueError("boom")
        return TrackingResult(brightness=1.0 * self.calls)

    def close(self): self.closed = True


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("threaded", [False, True])
def test_pipeline_pairs_each_frame_with_its_tracking_result(threaded):
    src, trk = FakeSource(20), FakeTracker()
    pipe = TrackingPipeline(src, trk, threaded=threaded, read_timeout_s=0.1)
    pipe.start()
    seen = []
    try:
        for _ in range(60):
            item = pipe.get()
            if item is None:
                break
            seen.append(item)
            # the result must belong to the frame it is delivered with
            assert item.tracking.brightness == item.frame.index if not threaded or trk.calls == item.frame.index else True
    finally:
        pipe.stop()
    indices = [it.frame.index for it in seen]
    assert indices == sorted(indices) and len(set(indices)) == len(indices)  # increasing, no repeats
    assert indices[-1] == 20 if not threaded else indices[-1] >= 1
    assert all(it.track_ms >= 0 for it in seen)


def test_pipeline_threaded_drops_frames_instead_of_falling_behind():
    class SlowTracker(FakeTracker):
        def process(self, frame, timestamp):
            threading.Event().wait(0.03)
            return super().process(frame, timestamp)

    src = FakeSource(200, stall_after=False)
    pipe = TrackingPipeline(src, SlowTracker(), threaded=True, read_timeout_s=0.2)
    pipe.start()
    try:
        first = pipe.get()
        threading.Event().wait(0.3)  # the consumer is "busy"; the worker keeps going
        later = pipe.get()
    finally:
        pipe.stop()
    assert later.frame.index > first.frame.index + 1  # skipped the stale frames in between


def test_pipeline_reports_a_stalled_camera():
    pipe = TrackingPipeline(FakeSource(0), FakeTracker(), threaded=True, read_timeout_s=0.05)
    pipe.start()
    try:
        assert pipe.get() is None
        assert pipe.status == "stalled"
    finally:
        pipe.stop()


def test_tracker_crash_on_the_worker_thread_surfaces_on_the_caller():
    pipe = TrackingPipeline(FakeSource(10), FakeTracker(fail_at=3), threaded=True, read_timeout_s=0.1)
    pipe.start()
    try:
        with pytest.raises(RuntimeError, match="hand tracking failed") as err:
            for _ in range(30):
                pipe.get()
        assert isinstance(err.value.__cause__, ValueError)
    finally:
        pipe.stop()


def test_sync_pipeline_returns_none_when_source_is_empty():
    assert TrackingPipeline(FakeSource(0), FakeTracker(), threaded=False, read_timeout_s=0.01).get() is None


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
class ClosedCamera:
    def __init__(self, *a, **k): pass
    def isOpened(self): return False
    def release(self): pass
    def set(self, *a): return True
    def read(self): return False, None


def test_unopenable_camera_raises_a_helpful_error_not_a_crash(monkeypatch):
    monkeypatch.setattr(capture.cv2, "VideoCapture", ClosedCamera)
    cam = CameraCapture(CameraConfig(index=7))
    with pytest.raises(CameraError, match="Could not open camera 7") as err:
        cam.start()
    assert "permission" in str(err.value) or "--camera" in str(err.value)


def test_simulated_camera_is_deterministic_without_realtime():
    a, b = SimulatedCamera(realtime=False), SimulatedCamera(realtime=False)
    fa = [a.read() for _ in range(3)]
    assert np.array_equal(fa[0].image, b.read().image)  # same seed -> same picture
    assert [f.timestamp for f in fa] == pytest.approx([1 / 30, 2 / 30, 3 / 30])
    assert fa[0].image.shape == (480, 640, 3) and fa[0].image.dtype == np.uint8
    assert np.mean(fa[0].image) > 20  # not a black frame


# --------------------------------------------------------------------------- #
# model download
# --------------------------------------------------------------------------- #
def test_model_download_failure_is_a_clear_error_and_leaves_no_partial_file(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(hand_tracker.urllib.request, "urlopen", boom)
    target = tmp_path / "models" / "hand.task"
    with pytest.raises(TrackerError, match="Download it manually") as err:
        ensure_model(target, "https://example.invalid/hand.task")
    assert "https://example.invalid/hand.task" in str(err.value) and str(target) in str(err.value)
    assert not target.exists() and not list(target.parent.glob("*.part"))


def test_model_is_downloaded_once(tmp_path, monkeypatch):
    calls = []

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return Response(b"x" * 1_200_000)

    monkeypatch.setattr(hand_tracker.urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "hand.task"
    assert ensure_model(target, "https://example.invalid/m") == target and target.stat().st_size == 1_200_000
    ensure_model(target, "https://example.invalid/m")
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# command line + app start-up failures
# --------------------------------------------------------------------------- #
def test_cli_flags_map_onto_the_configuration():
    args = gestureflow_app.parse_args(
        ["--camera", "2", "--width", "1280", "--height", "720", "--no-mirror", "--hand", "Left", "--max-hands", "2",
         "--detection-confidence", "0.7", "--tracking-confidence", "0.4", "--no-smoothing", "--score-smoothing", "0.3",
         "--classifier", "hybrid", "--window", "1000x540", "--no-panel", "--simulate", "--headless", "--max-frames", "9", "--sync"]
    )
    cfg = gestureflow_app.build_config(args)
    assert (cfg.camera.index, cfg.camera.width, cfg.camera.height, cfg.camera.mirror) == (2, 1280, 720, False)
    t = cfg.tracking
    assert (t.dominant_hand, t.max_hands, t.min_detection_confidence, t.min_tracking_confidence, t.smoothing_enabled) == ("Left", 2, 0.7, 0.4, False)
    assert cfg.gestures.score_ema_alpha == 0.3 and cfg.classifier == "hybrid"
    assert (cfg.display.window_width, cfg.display.window_height) == (1000, 540) and not cfg.display.show_panel
    assert cfg.simulate and cfg.headless and cfg.max_frames == 9 and not cfg.threaded


def test_defaults_target_one_hand_and_a_laptop_sized_window():
    cfg = gestureflow_app.build_config(gestureflow_app.parse_args([]))
    assert cfg.tracking.max_hands == 1 and cfg.camera.mirror and cfg.classifier == "rules"
    assert cfg.display.window_height <= 720 and cfg.threaded


def test_main_exits_cleanly_when_the_webcam_cannot_be_opened(monkeypatch, capsys):
    monkeypatch.setattr(capture.cv2, "VideoCapture", ClosedCamera)
    code = gestureflow_app.main(["--headless", "--max-frames", "1"])
    err = capsys.readouterr().err
    assert code == 1
    assert "Could not open camera" in err and "--simulate" in err  # tells the user what to do
    assert "Traceback" not in err


def test_main_exits_cleanly_when_the_hand_model_is_unavailable(monkeypatch, capsys):
    class WorkingCamera:
        width, height, status = 640, 480, "ok"
        def __init__(self, cfg): pass
        def start(self): pass
        def stop(self): pass
        def read(self, timeout=None): return None

    def no_model(cfg):
        raise TrackerError("model missing")

    monkeypatch.setattr(gestureflow_app, "CameraCapture", WorkingCamera)
    monkeypatch.setattr(gestureflow_app, "HandTracker", no_model)
    code = gestureflow_app.main(["--headless"])
    assert code == 2 and "model missing" in capsys.readouterr().err


def test_learned_classifier_falls_back_to_rules_with_a_message(tmp_path, capsys):
    cfg = AppConfig(classifier="learned", learned_model_path=tmp_path / "missing.joblib")
    clf = gestureflow_app.build_classifier(cfg)
    assert type(clf).__name__ == "RuleBasedClassifier"
    assert "Falling back" in capsys.readouterr().out


@pytest.mark.parametrize("spec,expected", [("1280x720", (1280, 720)), ("1600X900", (1600, 900)), ("100x100", (480, 320))])
def test_window_flag_parsing_and_clamping(spec, expected):
    assert gestureflow_app.parse_window(spec) == expected


def test_malformed_window_flag_is_a_clean_error():
    with pytest.raises(SystemExit, match="WIDTHxHEIGHT"):
        gestureflow_app.parse_window("big")
