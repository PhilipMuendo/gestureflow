"""Dataset format for recorded, labelled hand-landmark data.

One recording = one JSON object per line (JSONL), so files can be appended to, concatenated
with ``cat`` and inspected with any text tool::

    {"version": 1,
     "kind": "static" | "sequence",
     "label": "fist",
     "session": "2026-10-06T14-02-11",
     "handedness": "Right",
     "aspect": 1.3333,
     "frames": [ {"t": 0.0,
                  "image": [[x, y, z] * 21],     # normalised frame coordinates
                  "world": [[x, y, z] * 21]},    # metric world coordinates
                 ... ]}

* ``static``   - a single hand pose (``frames`` has one entry).  This is what the optional
  static classifier trains on; recording tools write one sample per captured frame.
* ``sequence`` - a short clip (``frames`` holds the whole motion, e.g. a swipe) for dynamic
  gestures.  The format and the recorder support it so data can be collected today; the
  bundled trainer currently consumes static samples (see the README's future work).

Raw landmarks are stored rather than computed features, so the dataset stays valid when the
feature extractor changes: features are recomputed at training time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from config import FeatureConfig
from gestures.features import FEATURE_VECTOR_SIZE, extract_features
from tracking.types import NUM_LANDMARKS, HandObservation

FORMAT_VERSION = 1
KINDS = ("static", "sequence")


@dataclass(frozen=True)
class FrameRecord:
    t: float
    image: np.ndarray  # (21, 3)
    world: np.ndarray  # (21, 3)


@dataclass(frozen=True)
class Sample:
    label: str
    frames: tuple[FrameRecord, ...]
    handedness: str = "Right"
    kind: str = "static"
    session: str = ""
    aspect: float = 4.0 / 3.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown sample kind {self.kind!r}")
        if not self.frames:
            raise ValueError("a sample needs at least one frame")
        if self.kind == "static" and len(self.frames) != 1:
            raise ValueError("a static sample holds exactly one frame")

    def observations(self) -> list[HandObservation]:
        return [HandObservation(f.image, f.world, self.handedness, 1.0, f.t, self.aspect) for f in self.frames]


# --------------------------------------------------------------------------- #
# (de)serialisation
# --------------------------------------------------------------------------- #
def _round(a: np.ndarray) -> list:
    return np.round(a, 5).tolist()


def sample_to_json(sample: Sample) -> str:
    return json.dumps(
        {
            "version": FORMAT_VERSION,
            "kind": sample.kind,
            "label": sample.label,
            "session": sample.session,
            "handedness": sample.handedness,
            "aspect": round(sample.aspect, 5),
            "frames": [{"t": round(f.t, 4), "image": _round(f.image), "world": _round(f.world)} for f in sample.frames],
        },
        separators=(",", ":"),
    )


def sample_from_json(line: str) -> Sample:
    d = json.loads(line)
    if d.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported dataset version {d.get('version')!r} (expected {FORMAT_VERSION})")
    frames = []
    for f in d["frames"]:
        image, world = np.asarray(f["image"], float), np.asarray(f["world"], float)
        if image.shape != (NUM_LANDMARKS, 3) or world.shape != (NUM_LANDMARKS, 3):
            raise ValueError("every frame needs 21 x 3 image and world landmarks")
        frames.append(FrameRecord(float(f["t"]), image, world))
    return Sample(
        label=d["label"], frames=tuple(frames), handedness=d.get("handedness", "Right"),
        kind=d.get("kind", "static"), session=d.get("session", ""), aspect=float(d.get("aspect", 4 / 3)),
    )


def write_samples(path: Path, samples: Iterable[Sample], append: bool = True) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(sample_to_json(sample) + "\n")
            n += 1
    return n


def read_samples(paths: Sequence[Path] | Path) -> Iterator[Sample]:
    for path in [paths] if isinstance(paths, Path) else paths:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    yield sample_from_json(line)
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc


# --------------------------------------------------------------------------- #
# feature matrices
# --------------------------------------------------------------------------- #
def build_static_matrix(
    samples: Iterable[Sample], cfg: FeatureConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn static samples into ``(X, y, groups)``.

    ``groups`` carries the recording session of each row so the trainer can split by
    session (consecutive frames of one take are near-duplicates; a random split would leak).
    Rows whose landmarks are degenerate are skipped.
    """
    rows, labels, groups = [], [], []
    for sample in samples:
        if sample.kind != "static":
            continue
        for obs in sample.observations():
            try:
                f = extract_features(obs, cfg) if cfg is not None else extract_features(obs)
            except ValueError:
                continue
            rows.append(f.to_vector())
            labels.append(sample.label)
            groups.append(sample.session)
    if not rows:
        return np.zeros((0, FEATURE_VECTOR_SIZE)), np.array([], dtype=str), np.array([], dtype=str)
    return np.vstack(rows), np.array(labels), np.array(groups)
