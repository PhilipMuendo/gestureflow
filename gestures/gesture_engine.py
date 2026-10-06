"""The gesture engine: landmarks in, a stable :class:`GestureState` out.

Pipeline for one frame::

    HandObservation
        │  extract_features()             geometry (gestures/features.py)
        ▼
    HandFeatures
        │  StaticClassifier.classify()    per-gesture scores in [0, 1]
        ▼
    raw scores ─► EMA ─► hysteresis + confirm time ─► stable static gesture
        │                                              │
        └────────────► DynamicGestureRecognizer ◄──────┘   (swipe, zoom, drag, rotate)
                                │
                                ▼
                          GestureState

Three mechanisms keep the output from flickering:

* **EMA** (exponential moving average) of each gesture's score removes single-frame
  noise.
* **Hysteresis** - a gesture is *entered* above ``enter_threshold`` but only
  *left* below the lower ``exit_threshold``, so a score hovering around one value
  cannot toggle back and forth.
* **Confirm time** - a challenger must out-score the incumbent continuously for a
  few frames before it takes over.

Brief tracking dropouts (``hand_lost_grace_s``) keep the last state alive so that a
single missed detection does not drop an object you are holding.
"""

from __future__ import annotations

import math
from dataclasses import replace
from collections.abc import Iterable

from config import GestureConfig
from gestures.dynamic_gestures import DynamicGestureRecognizer
from gestures.features import HandFeatures, extract_features
from gestures.static_gestures import RuleBasedClassifier, StaticClassifier
from gestures.types import STATIC_GESTURES, Gesture, GestureState, MotionState
from tracking.types import HandObservation

ALL_GESTURES: frozenset[Gesture] = frozenset(g for g in Gesture if g is not Gesture.NONE)


class GestureEngine:
    def __init__(
        self,
        cfg: GestureConfig | None = None,
        classifier: StaticClassifier | None = None,
        enabled: Iterable[Gesture] | None = None,
    ) -> None:
        self.cfg = cfg or GestureConfig()
        self.classifier: StaticClassifier = classifier or RuleBasedClassifier(self.cfg.static)
        self.enabled: frozenset[Gesture] = frozenset(enabled) if enabled is not None else ALL_GESTURES
        self._dynamic = DynamicGestureRecognizer(self.cfg.dynamic)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._dynamic.reset()
        self._scores: dict[Gesture, float] = {g: 0.0 for g in STATIC_GESTURES}
        self._stable = Gesture.NONE
        self._stable_since = 0.0
        self._pending = Gesture.NONE
        self._pending_since = 0.0
        self._last_seen = -math.inf
        self._last_state = GestureState()

    def set_enabled(self, enabled: Iterable[Gesture]) -> None:
        self.enabled = frozenset(enabled)

    # ---------------------------------------------------------------- process
    def process(self, obs: HandObservation | None, t: float) -> GestureState:
        features: HandFeatures | None = None
        if obs is not None:
            try:
                features = extract_features(obs, self.cfg.features)
            except ValueError:  # degenerate landmarks are treated as "no hand"
                features = None
        if features is None:
            return self._handle_missing(t)

        if t - self._last_seen > 0.12:
            # The hand re-appeared after a gap: its old motion history would
            # produce a bogus velocity spike across the jump.
            self._dynamic.reset()
        self._last_seen = t

        smoothed = self._smooth_scores(features)
        self._update_stable(smoothed, t)
        static_conf = smoothed.get(self._stable, 0.0) if self._stable is not Gesture.NONE else 0.0

        dyn = self._dynamic.update(features, t, self._stable)
        active = dyn.active if dyn.active in self.enabled else Gesture.NONE
        events = tuple(e for e in dyn.events if e in self.enabled)

        if events:
            gesture, confidence = events[0], 0.9
        elif active is not Gesture.NONE:
            gesture, confidence = active, dyn.confidence
        else:
            gesture, confidence = self._stable, static_conf
        if confidence < self.cfg.min_confidence:
            gesture, confidence = Gesture.NONE, confidence

        state = GestureState(
            timestamp=t,
            hand_present=True,
            gesture=gesture,
            confidence=float(confidence),
            static=self._stable,
            static_confidence=float(static_conf),
            dynamic=active,
            dynamic_confidence=float(dyn.confidence if active is not Gesture.NONE else 0.0),
            hold_s=t - self._stable_since if self._stable is not Gesture.NONE else 0.0,
            events=events,
            motion=dyn.motion,
            features=features,
            scores=dict(smoothed),
            handedness=features.handedness,
            tracking_score=features.score,
            enabled=self.enabled,
        )
        self._last_state = state
        return state

    # --------------------------------------------------------------- internals
    def _handle_missing(self, t: float) -> GestureState:
        grace = self.cfg.hand_lost_grace_s
        if self._last_state.hand_present and t - self._last_seen <= grace:
            # Brief dropout: freeze the last pose, report it as stale and with
            # zeroed motion so nothing jumps.
            fade = 1.0 - (t - self._last_seen) / grace
            self._last_state = replace(
                self._last_state,
                timestamp=t,
                stale=True,
                events=(),
                motion=MotionState(),
                confidence=self._last_state.confidence * (0.5 + 0.5 * fade),
                enabled=self.enabled,
            )
            return self._last_state
        if self._last_state.hand_present:
            self.reset()
        self._last_state = GestureState(timestamp=t, enabled=self.enabled)
        return self._last_state

    def _smooth_scores(self, features: HandFeatures) -> dict[Gesture, float]:
        raw = self.classifier.classify(features)
        alpha = self.cfg.score_ema_alpha
        for g in STATIC_GESTURES:
            score = raw.get(g, 0.0) if g in self.enabled else 0.0
            self._scores[g] = alpha * score + (1.0 - alpha) * self._scores[g]
        return self._scores

    def _confirm_time(self, gesture: Gesture) -> float:
        return self.cfg.pinch_confirm_time_s if gesture is Gesture.PINCH else self.cfg.confirm_time_s

    def _update_stable(self, scores: dict[Gesture, float], t: float) -> None:
        cfg = self.cfg
        challenger = max(STATIC_GESTURES, key=lambda g: scores[g])
        challenger_score = scores[challenger]
        incumbent_score = scores[self._stable] if self._stable is not Gesture.NONE else 0.0

        desired = self._stable
        if (
            challenger is not self._stable
            and challenger_score >= cfg.enter_threshold
            and (self._stable is Gesture.NONE or challenger_score >= incumbent_score + cfg.switch_margin)
        ):
            desired = challenger
        if desired is self._stable and self._stable is not Gesture.NONE and incumbent_score < cfg.exit_threshold:
            desired = Gesture.NONE

        if desired is self._stable:
            self._pending = Gesture.NONE
            return
        if desired is not self._pending:
            self._pending, self._pending_since = desired, t
        if t - self._pending_since >= self._confirm_time(desired):
            self._stable, self._stable_since = desired, t
            self._pending = Gesture.NONE
