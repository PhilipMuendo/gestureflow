"""MediaPipe HandLandmarker wrapper.

Responsibilities (and nothing else - no gesture logic lives here):

* make sure the ``hand_landmarker.task`` model file exists (one-time download),
* run the detector on a BGR frame and convert the result into
  :class:`~tracking.types.HandObservation` objects,
* smooth the landmarks with a One-Euro filter,
* pick the *primary* hand when several are visible,
* brighten dark frames before inference.

MediaPipe is imported lazily so that the rest of the code base (and the unit
tests) work without it installed.
"""

from __future__ import annotations

import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol
from collections.abc import Sequence

import numpy as np

from config import TrackingConfig
from tracking.filters import OneEuroFilter
from tracking.types import HandObservation, Landmark, TrackingResult


class TrackerError(RuntimeError):
    """The hand tracker could not be created (missing model, missing dependency...)."""


class Tracker(Protocol):
    """Anything that turns a frame into a :class:`TrackingResult`."""

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> TrackingResult: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Model file
# --------------------------------------------------------------------------- #
def ensure_model(path: Path, url: str, timeout_s: float = 30.0) -> Path:
    """Return ``path``, downloading the model first if it is missing.

    The download happens once; inference itself is fully local.  The file is
    written to ``<name>.part`` first so an interrupted download never leaves a
    corrupt model behind.
    """
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tracking] Downloading hand-landmark model (~8 MB) to {path} ...")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response, open(tmp, "wb") as out:
            while chunk := response.read(1 << 16):
                out.write(chunk)
        tmp.replace(path)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise TrackerError(
            f"Could not download the MediaPipe hand model ({exc}).\n"
            f"Download it manually from:\n  {url}\nand save it as:\n  {path}"
        ) from exc
    return path


# --------------------------------------------------------------------------- #
# Primary-hand selection
# --------------------------------------------------------------------------- #
def _palm_extent(hand: HandObservation) -> float:
    return float(np.linalg.norm(hand.landmarks[Landmark.MIDDLE_MCP, :2] - hand.landmarks[Landmark.WRIST, :2]))


def select_primary(
    hands: Sequence[HandObservation],
    dominant: str = "Right",
    previous_wrist: np.ndarray | None = None,
    stickiness: float = 0.25,
) -> HandObservation | None:
    """Choose the hand that controls the interface.

    Policy, in order:
      1. *Stickiness* - if a hand is close to where the primary hand was in the
         previous frame, keep it.  This stops control from jumping between hands
         when both are in view.
      2. *Dominant hand* - prefer the hand whose handedness label matches
         ``dominant`` (unless ``dominant == "Any"``).
      3. *Closest hand* - the largest palm in the image.
    """
    if not hands:
        return None
    if previous_wrist is not None:
        nearest = min(hands, key=lambda h: float(np.linalg.norm(h.wrist_xy - previous_wrist)))
        if np.linalg.norm(nearest.wrist_xy - previous_wrist) <= stickiness:
            return nearest
    candidates = [h for h in hands if h.handedness == dominant] if dominant != "Any" else []
    pool = candidates or list(hands)
    return max(pool, key=_palm_extent)


# --------------------------------------------------------------------------- #
# Low-light helpers
# --------------------------------------------------------------------------- #
def mean_luma(frame_bgr: np.ndarray) -> float:
    """Cheap brightness estimate from a heavily sub-sampled frame."""
    small = frame_bgr[::16, ::16]
    # BT.601 luma from BGR
    return float(small[..., 0].mean() * 0.114 + small[..., 1].mean() * 0.587 + small[..., 2].mean() * 0.299)


def gamma_lut(luma: float, target: float) -> np.ndarray:
    """Lookup table that lifts a frame with mean ``luma`` towards ``target``.

    Solves (luma/255) ** gamma == target/255 for gamma, then clamps it so very dark
    frames are not amplified into pure noise.
    """
    luma = max(luma, 1.0)
    gamma = math.log(target / 255.0) / math.log(luma / 255.0)
    gamma = min(max(gamma, 0.4), 1.0)
    return (np.power(np.arange(256) / 255.0, gamma) * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
class _HandFilters:
    """A pair of One-Euro filters (image + world coordinates) for one hand."""

    def __init__(self, cfg: TrackingConfig) -> None:
        self.image = OneEuroFilter(cfg.image_min_cutoff, cfg.image_beta)
        self.world = OneEuroFilter(cfg.world_min_cutoff, cfg.world_beta)
        self.last_seen = -math.inf


class HandTracker:
    """MediaPipe HandLandmarker in VIDEO mode (synchronous, with tracking)."""

    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise TrackerError("MediaPipe is not installed.  Run: pip install -r requirements.txt") from exc

        self._mp = mp
        self._vision = vision
        self._base_options = BaseOptions(model_asset_path=str(ensure_model(cfg.model_path, cfg.model_url)))
        self._lock = threading.Lock()  # process() runs on a worker thread, set_max_hands() on the UI thread
        self._max_hands = cfg.max_hands
        self._landmarker = self._create_landmarker(self._max_hands)
        self._filters: dict[str, _HandFilters] = {}
        self._previous_wrist: np.ndarray | None = None
        self._last_ms = -1

    @property
    def max_hands(self) -> int:
        return self._max_hands

    def set_max_hands(self, n: int) -> None:
        """Switch between one- and two-hand tracking at runtime (re-creates the detector)."""
        with self._lock:
            if n == self._max_hands:
                return
            self._landmarker.close()
            self._max_hands = n
            self._landmarker = self._create_landmarker(n)
            self._filters.clear()
            self._previous_wrist = None

    def _create_landmarker(self, num_hands: int):
        c = self.cfg
        options = self._vision.HandLandmarkerOptions(
            base_options=self._base_options,
            running_mode=self._vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=c.min_detection_confidence,
            min_hand_presence_confidence=c.min_presence_confidence,
            min_tracking_confidence=c.min_tracking_confidence,
        )
        return self._vision.HandLandmarker.create_from_options(options)

    # -- public API ---------------------------------------------------------- #
    def process(self, frame_bgr: np.ndarray, timestamp: float) -> TrackingResult:
        import cv2  # local import keeps module importable without OpenCV

        height, width = frame_bgr.shape[:2]
        luma = mean_luma(frame_bgr)
        low_light = luma < self.cfg.low_light_luma
        detect_frame = frame_bgr
        if low_light and self.cfg.enhance_low_light:
            detect_frame = cv2.LUT(frame_bgr, gamma_lut(luma, self.cfg.target_luma))

        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode demands strictly increasing integer millisecond timestamps.
        ts_ms = max(int(timestamp * 1000.0), self._last_ms + 1)
        self._last_ms = ts_ms

        with self._lock:
            started = time.perf_counter()
            result = self._landmarker.detect_for_video(image, ts_ms)
            inference_ms = (time.perf_counter() - started) * 1000.0

            hands = self._convert(result, timestamp, width / height)
            primary = select_primary(
                hands, self.cfg.dominant_hand, self._previous_wrist, self.cfg.primary_stickiness
            )
            self._previous_wrist = None if primary is None else primary.wrist_xy.copy()
        return TrackingResult(
            hands=tuple(hands),
            primary=primary,
            inference_ms=inference_ms,
            brightness=luma,
            low_light=low_light,
        )

    def close(self) -> None:
        with self._lock:
            self._landmarker.close()

    # -- internals ----------------------------------------------------------- #
    def _convert(self, result, timestamp: float, aspect: float) -> list[HandObservation]:
        hands: list[HandObservation] = []
        used_keys: set[str] = set()
        for i, lm in enumerate(result.hand_landmarks):
            image_pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
            world_pts = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]], dtype=np.float64)
            category = result.handedness[i][0]
            label, score = category.category_name, float(category.score)

            if self.cfg.smoothing_enabled:
                key = label if label not in used_keys else f"{label}#{i}"
                used_keys.add(key)
                filters = self._filters.setdefault(key, _HandFilters(self.cfg))
                if timestamp - filters.last_seen > self.cfg.filter_reset_after_s:
                    filters.image.reset()
                    filters.world.reset()
                filters.last_seen = timestamp
                image_pts = filters.image(image_pts, timestamp)
                world_pts = filters.world(world_pts, timestamp)

            hands.append(HandObservation(image_pts, world_pts, label, score, timestamp, aspect))
        return hands
