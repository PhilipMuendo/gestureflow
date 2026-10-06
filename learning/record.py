"""Record labelled hand-landmark data from the webcam.

    python -m learning.record                         # all five gestures + "none"
    python -m learning.record --labels fist pinch     # just these
    python -m learning.record --sequence --labels swipe_left swipe_right --seconds 1.2

Keys:  SPACE  start a take (3-second countdown, then it records)
       N / P  next / previous label
       Q      quit

A *take* records ``--frames`` frames of one pose (static mode; each frame becomes a sample)
or ``--seconds`` of motion (sequence mode; the whole clip is one sample).  For good data:
record several takes per label, vary distance / angle / position, use both hands if you use
both, and record a "none" class of in-between and relaxed hand shapes so the model learns
what is *not* a gesture.  Every take is its own *session* so the trainer can hold whole takes
out for an honest accuracy estimate.
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import camera  # noqa: F401  (sets OpenCV env vars before cv2 is imported)
import cv2

from camera.capture import CameraCapture, CameraError
from config import CameraConfig, TrackingConfig
from learning.dataset import FrameRecord, Sample, write_samples
from tracking.hand_tracker import HandTracker, TrackerError
from tracking.types import HAND_CONNECTIONS, HandObservation

DEFAULT_LABELS = ["open_palm", "point", "pinch", "fist", "two_fingers", "none"]
COUNTDOWN_S = 3.0
ACCENT, WHITE, GOOD, BAD, WARN = (255, 194, 76), (240, 240, 240), (120, 210, 120), (100, 100, 230), (90, 170, 230)


def _put(img, text, org, scale=0.6, color=WHITE, thickness=1, center=False):
    """Minimal text helper for this developer tool (the app itself uses visualization.typeface)."""
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = org[0] - w // 2 if center else org[0]
    cv2.putText(img, text, (x, org[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class TakeRecorder:
    """Collects one take.  Independent of camera and window so it is unit-testable."""

    def __init__(self, label: str, mode: str, frames: int, seconds: float, session: str) -> None:
        if mode not in ("static", "sequence"):
            raise ValueError("mode must be 'static' or 'sequence'")
        self.label, self.mode, self.session = label, mode, session
        self.target_frames, self.target_seconds = frames, seconds
        self._records: list[tuple[HandObservation, float]] = []
        self._t0: float | None = None

    def feed(self, obs: HandObservation | None, t: float) -> None:
        if self.done:  # a take is exactly as long as requested
            return
        if self._t0 is None:
            self._t0 = t
        if obs is None:
            return  # frames without a hand are simply skipped
        self._records.append((obs, t - self._t0))

    @property
    def progress(self) -> float:
        if self._t0 is None:
            return 0.0
        if self.mode == "static":
            return min(1.0, len(self._records) / self.target_frames)
        return min(1.0, (self._records[-1][1] if self._records else 0.0) / self.target_seconds)

    @property
    def done(self) -> bool:
        return self.progress >= 1.0

    def samples(self) -> list[Sample]:
        if self.mode == "static":
            return [
                Sample(self.label, (FrameRecord(t, o.landmarks, o.world),), o.handedness, "static", self.session, o.aspect)
                for o, t in self._records
            ]
        if len(self._records) < 5:
            return []
        first = self._records[0][0]
        frames = tuple(FrameRecord(t, o.landmarks, o.world) for o, t in self._records)
        return [Sample(self.label, frames, first.handedness, "sequence", self.session, first.aspect)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", nargs="+", default=None, help="labels to record (default: the five gestures + none)")
    p.add_argument("--sequence", action="store_true", help="record motion clips instead of single poses")
    p.add_argument("--frames", type=int, default=150, help="frames per static take")
    p.add_argument("--seconds", type=float, default=1.5, help="length of a sequence take")
    p.add_argument("--out", type=Path, default=None, help="output JSONL (default: data/recordings/<timestamp>.jsonl)")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    labels = args.labels or (["swipe_left", "swipe_right"] if args.sequence else DEFAULT_LABELS)
    run_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = args.out or Path("data/recordings") / f"{run_id}.jsonl"
    mode = "sequence" if args.sequence else "static"

    try:
        cam = CameraCapture(CameraConfig(index=args.camera))
        cam.start()
        tracker = HandTracker(TrackingConfig(smoothing_enabled=True))
    except (CameraError, TrackerError) as exc:
        print(exc)
        return 1

    counts = {label: 0 for label in labels}
    takes = {label: 0 for label in labels}
    current, recorder, countdown_end = 0, None, 0.0
    window = "GestureFlow - record"
    cv2.namedWindow(window)
    print(f"Recording to {out}\nSPACE start a take | N/P change label | Q quit")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            result = tracker.process(frame.image, frame.timestamp)
            obs = result.primary
            now = time.perf_counter()
            img = frame.image.copy()
            h, w = img.shape[:2]
            if obs is not None:
                pts = (obs.landmarks[:, :2] * (w, h)).astype("int32")
                cv2.polylines(img, [pts[[a, b]] for a, b in HAND_CONNECTIONS], False, ACCENT, 2, cv2.LINE_AA)

            label = labels[current]
            status = f"label: {label.upper()}  ({counts[label]} saved)"
            if recorder is None and countdown_end:
                remaining = countdown_end - now
                if remaining <= 0:
                    takes[label] += 1
                    recorder = TakeRecorder(label, mode, args.frames, args.seconds, f"{run_id}-{label}-{takes[label]}")
                    countdown_end = 0.0
                else:
                    _put(img, f"{remaining:.0f}", (w // 2, h // 2 + 30), 3.0, WARN, 3, center=True)
            if recorder is not None:
                recorder.feed(obs, frame.timestamp)
                cv2.rectangle(img, (20, h - 30), (20 + int((w - 40) * recorder.progress), h - 20), BAD, -1)
                status += "  RECORDING"
                if recorder.done:
                    samples = recorder.samples()
                    write_samples(out, samples)
                    counts[recorder.label] += len(samples)
                    print(f"saved {len(samples)} samples for '{recorder.label}'")
                    recorder = None
            _put(img, status, (20, 30), 0.7)
            _put(img, "no hand" if obs is None else f"{obs.handedness} hand", (20, 58), 0.55, GOOD if obs else BAD)
            cv2.imshow(window, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and recorder is None and not countdown_end:
                countdown_end = time.perf_counter() + COUNTDOWN_S
            elif key == ord("n") and recorder is None:
                current = (current + 1) % len(labels)
            elif key == ord("p") and recorder is None:
                current = (current - 1) % len(labels)
    finally:
        tracker.close()
        cam.stop()
        cv2.destroyAllWindows()
    print(f"done. totals: {counts}\nnext: python -m learning.train --data {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
