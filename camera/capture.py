"""Threaded webcam capture.

Reading from a webcam blocks until the next frame is ready.  Doing that in the
main loop would stall tracking and rendering, and a slow frame would queue behind
older ones.  Instead a background thread reads continuously and keeps only the
**latest** frame; the main loop picks it up when it is ready.  Frames the main loop
is too slow to consume are dropped, which is the right behaviour for a live
interface (low latency beats completeness).

The capture also copes with a camera that cannot be opened (clear error message,
no traceback) or that disappears mid-session (automatic re-open attempts).
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from config import CameraConfig


class CameraError(RuntimeError):
    """The camera could not be opened or has stopped delivering frames."""


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR, already mirrored if configured
    timestamp: float  # seconds, monotonic
    index: int


class FrameSource(Protocol):
    """Anything that yields frames: a real webcam or the simulator."""

    width: int
    height: int
    status: str

    def start(self) -> None: ...

    def read(self, timeout: float | None = None) -> Frame | None: ...

    def stop(self) -> None: ...


def _backends() -> list[int]:
    """Capture backends to try, best first for each platform.

    On Windows, Media Foundation (MSMF) is listed before DirectShow: DirectShow caps many
    webcams at ~15 FPS in OpenCV, while MSMF delivers the full 30.  MSMF's one weakness - a
    multi-second start-up caused by hardware video transforms - is avoided by the
    environment variable set in ``camera/__init__.py`` before OpenCV is imported.
    """
    if sys.platform.startswith("win"):
        return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def list_cameras(max_index: int = 5) -> list[int]:
    """Indices of cameras that open successfully (used for error messages)."""
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, _backends()[0])
        if cap.isOpened():
            found.append(index)
        cap.release()
    return found


class CameraCapture:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.fps = float(cfg.fps)
        self.status = "closed"  # closed | ok | reconnecting | failed
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._returned = -1

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if not self._open():
            available = list_cameras()
            hint = (
                f"Cameras that opened successfully: {available}. Try --camera <index>."
                if available
                else "No camera could be opened at all. Check that it is connected, not in use by another "
                "application (Zoom, Teams, browser tab), and that this terminal has camera permission."
            )
            raise CameraError(f"Could not open camera {self.cfg.index}. {hint}")
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status = "closed"

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------- read
    def read(self, timeout: float | None = None) -> Frame | None:
        """Block until a frame newer than the last returned one exists.

        Returns ``None`` on timeout (the camera is stalled) so the caller can keep the
        UI responsive.
        """
        timeout = self.cfg.read_timeout_s if timeout is None else timeout
        with self._cond:
            self._cond.wait_for(lambda: self._stop.is_set() or (self._frame is not None and self._frame.index > self._returned), timeout)
            if self._frame is None or self._frame.index <= self._returned:
                return None
            self._returned = self._frame.index
            return self._frame

    # --------------------------------------------------------------- internals
    def _open(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        for backend in _backends():
            cap = cv2.VideoCapture(self.cfg.index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if self.cfg.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok = False
            for _ in range(10):  # cameras often return a few black/empty frames first
                ok, image = cap.read()
                if ok and image is not None:
                    break
            if not ok:
                cap.release()
                continue
            self._cap = cap
            self.height, self.width = image.shape[:2]
            self.status = "ok"
            return True
        return False

    def _reader(self) -> None:
        failures = 0
        index = 0
        while not self._stop.is_set():
            cap = self._cap
            ok, image = (False, None) if cap is None else cap.read()
            if not ok or image is None:
                failures += 1
                if failures >= self.cfg.reopen_after_failures:
                    self.status = "reconnecting"
                    if self._open():
                        failures = 0
                    else:
                        time.sleep(1.0)
                else:
                    time.sleep(0.02)
                continue
            failures = 0
            self.status = "ok"
            if self.cfg.mirror:
                image = cv2.flip(image, 1)
            index += 1
            with self._cond:
                self._frame = Frame(image, time.perf_counter(), index)
                self._cond.notify_all()
