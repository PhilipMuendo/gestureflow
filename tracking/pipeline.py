"""Camera -> tracker pipeline, optionally on a worker thread.

Why a thread?  Hand-landmark inference is the single most expensive step (10-40 ms on a
laptop CPU) and rendering is the second.  Run one after the other and the frame rate is
``1 / (track + render)``.  Run them on different threads and it becomes
``1 / max(track, render)``: while the main thread draws frame N, the worker is already
tracking frame N+1.  MediaPipe and OpenCV release the GIL during their native work, so the
two really do run in parallel.

The worker always works on the **newest** camera frame; if the main thread is slower than
the tracker it simply sees the latest result (intermediate results are dropped).  A frame
and its tracking result travel together, so the skeleton drawn over a video frame always
belongs to that frame.

``threaded=False`` runs everything inline, which keeps tests and non-realtime simulation
deterministic.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from camera.capture import Frame, FrameSource
from tracking.hand_tracker import Tracker
from tracking.types import TrackingResult


@dataclass(frozen=True)
class TrackedFrame:
    frame: Frame
    tracking: TrackingResult
    track_ms: float


class TrackingPipeline:
    def __init__(self, source: FrameSource, tracker: Tracker, threaded: bool = True, read_timeout_s: float = 1.0) -> None:
        self.source = source
        self.tracker = tracker
        self.threaded = threaded
        self.read_timeout_s = read_timeout_s
        self._cond = threading.Condition()
        self._latest: TrackedFrame | None = None
        self._returned = -1
        self._stalled = False
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.threaded:
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, name="tracking", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------- read
    def get(self) -> TrackedFrame | None:
        """Next frame + tracking result, or ``None`` if the camera stalled.

        Raises:
            RuntimeError: if the tracker crashed on the worker thread.
        """
        if not self.threaded:
            frame = self.source.read(self.read_timeout_s)
            return None if frame is None else self._track(frame)

        with self._cond:
            self._cond.wait_for(
                lambda: self._error is not None
                or self._stalled
                or self._stop.is_set()
                or (self._latest is not None and self._latest.frame.index > self._returned),
                timeout=self.read_timeout_s + 1.0,
            )
            if self._error is not None:
                raise RuntimeError("hand tracking failed") from self._error
            if self._latest is None or self._latest.frame.index <= self._returned:
                return None
            self._returned = self._latest.frame.index
            return self._latest

    @property
    def status(self) -> str:
        return "stalled" if self._stalled else self.source.status

    # --------------------------------------------------------------- internals
    def _track(self, frame: Frame) -> TrackedFrame:
        started = time.perf_counter()
        result = self.tracker.process(frame.image, frame.timestamp)
        return TrackedFrame(frame, result, (time.perf_counter() - started) * 1000.0)

    def _worker(self) -> None:
        while not self._stop.is_set():
            frame = self.source.read(self.read_timeout_s)
            if frame is None:
                with self._cond:
                    self._stalled = True
                    self._cond.notify_all()
                continue
            try:
                item = self._track(frame)
            except BaseException as exc:  # surface tracker crashes on the main thread
                with self._cond:
                    self._error = exc
                    self._cond.notify_all()
                return
            with self._cond:
                self._latest = item
                self._stalled = False
                self._cond.notify_all()
