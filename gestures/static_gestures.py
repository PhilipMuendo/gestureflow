"""Rule-based static gesture classification.

A *static* gesture is recognised from the hand shape in a single frame.  Instead
of brittle ``if finger_up and not finger_down`` rules, every gesture gets a
**soft score** in [0, 1] built from fuzzy memberships:

    extended(e) -> 1 when finger extension e is high
    curled(e)   -> 1 when finger extension e is low

and gestures combine those with ``min`` (fuzzy AND).  A hand that is *almost* a
fist therefore scores 0.7 rather than flipping a boolean, and the temporal
smoother downstream can use those scores directly.

The classifier is a small protocol so a learned model can replace it (see
``gestures/learned_classifier.py``).
"""

from __future__ import annotations

from typing import Protocol

from config import StaticGestureConfig
from gestures.features import HandFeatures, ramp
from gestures.types import STATIC_GESTURES, Gesture


class StaticClassifier(Protocol):
    def classify(self, features: HandFeatures) -> dict[Gesture, float]:
        """Return a score in [0, 1] for every static gesture."""
        ...


class RuleBasedClassifier:
    def __init__(self, cfg: StaticGestureConfig | None = None) -> None:
        self.cfg = cfg or StaticGestureConfig()

    # fuzzy memberships --------------------------------------------------------
    def extended(self, ext: float) -> float:
        return float(ramp(ext, self.cfg.extended_lo, self.cfg.extended_hi))

    def curled(self, ext: float) -> float:
        # descending ramp: 1 at/below curled_hi, 0 at/above curled_lo
        return float(ramp(ext, self.cfg.curled_lo, self.cfg.curled_hi))

    def pinch_score(self, features: HandFeatures) -> float:
        c = self.cfg
        closeness = float(ramp(features.pinch_distance, c.pinch_open, c.pinch_closed))
        # A tight fist also brings the thumb near the index finger.  Requiring the
        # index to be at least partly extended keeps fist and pinch apart.
        index_open = float(ramp(features.extension[1], c.pinch_index_min_ext_lo, c.pinch_index_min_ext_hi))
        return closeness * index_open

    # classification -----------------------------------------------------------
    def classify(self, features: HandFeatures) -> dict[Gesture, float]:
        _, index, middle, ring, pinky = features.extension
        pinch = self.pinch_score(features)

        scores = {
            Gesture.OPEN_PALM: min(self.extended(index), self.extended(middle), self.extended(ring), self.extended(pinky)),
            Gesture.POINT: min(self.extended(index), self.curled(middle), self.curled(ring), self.curled(pinky)),
            Gesture.TWO_FINGERS: min(self.extended(index), self.extended(middle), self.curled(ring), self.curled(pinky)),
            Gesture.FIST: min(self.curled(index), self.curled(middle), self.curled(ring), self.curled(pinky)),
            Gesture.PINCH: pinch,
        }
        # Pinch is the most specific pose: when it is clearly present it suppresses
        # the poses that a half-closed hand would otherwise also resemble.
        scores[Gesture.POINT] *= 1.0 - 0.8 * pinch
        scores[Gesture.FIST] *= 1.0 - 0.6 * pinch
        scores[Gesture.OPEN_PALM] *= 1.0 - 0.8 * pinch
        return {g: float(scores[g]) for g in STATIC_GESTURES}


def best_static(scores: dict[Gesture, float], min_score: float = 0.0) -> tuple[Gesture, float]:
    """Highest-scoring gesture, or (NONE, 0) when nothing reaches ``min_score``."""
    gesture, score = max(scores.items(), key=lambda kv: kv[1])
    return (gesture, score) if score >= min_score else (Gesture.NONE, 0.0)
