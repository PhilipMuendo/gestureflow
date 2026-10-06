"""Optional learned static-gesture classifier (scikit-learn).

The rule-based classifier is the default and needs no data.  This module lets you train a
small model on *your own hand* (see ``learning/``) and plug it into the same interface:

    python app.py --classifier learned     # the model alone
    python app.py --classifier hybrid      # average of rules and model: both must agree

The model is a scikit-learn estimator saved with joblib.  Security note: joblib files are
pickles - only load models you trained yourself or otherwise trust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gestures.features import FEATURE_VECTOR_SIZE, HandFeatures
from gestures.static_gestures import StaticClassifier
from gestures.types import STATIC_GESTURES, Gesture

BUNDLE_FORMAT = "gestureflow-static-v1"


class LearnedModelError(RuntimeError):
    """The learned model is missing, unreadable or incompatible."""


class LearnedClassifier:
    def __init__(self, model: Any, labels: list[str]) -> None:
        self.model = model
        self.labels = list(labels)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1  # one-row predictions: thread start-up would cost more than the work
        self._index = {label: i for i, label in enumerate(self.labels)}

    @classmethod
    def load(cls, path: Path) -> "LearnedClassifier":
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise LearnedModelError("scikit-learn/joblib are required: pip install scikit-learn joblib") from exc
        if not path.exists():
            raise LearnedModelError(
                f"No learned model at {path}. Record data and train one first (see README: 'Learned classifier')."
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - corrupt / incompatible pickle
            raise LearnedModelError(f"Could not read {path}: {exc}") from exc
        if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
            raise LearnedModelError(f"{path} is not a GestureFlow static-classifier bundle.")
        if bundle.get("feature_size") != FEATURE_VECTOR_SIZE:
            raise LearnedModelError(
                f"{path} was trained with {bundle.get('feature_size')} features but this version of "
                f"GestureFlow produces {FEATURE_VECTOR_SIZE}. Re-train the model."
            )
        return cls(bundle["model"], bundle["labels"])

    def classify(self, features: HandFeatures) -> dict[Gesture, float]:
        proba = self.model.predict_proba(features.to_vector()[None, :])[0]
        scores = {g: 0.0 for g in STATIC_GESTURES}
        for label, p in zip(self.labels, proba):
            try:
                gesture = Gesture(label)
            except ValueError:
                continue  # unknown label (e.g. "none"): its probability simply is not credited to anyone
            if gesture in scores:
                scores[gesture] = float(p)
        return scores


class HybridClassifier:
    """Average of the rule-based and learned scores.

    A gesture only reaches the engine's entry threshold when *both* agree, which trades a
    little responsiveness for fewer false positives.
    """

    def __init__(self, rules: StaticClassifier, learned: StaticClassifier, learned_weight: float = 0.5) -> None:
        self.rules = rules
        self.learned = learned
        self.weight = learned_weight

    def classify(self, features: HandFeatures) -> dict[Gesture, float]:
        a, b = self.rules.classify(features), self.learned.classify(features)
        return {g: float((1.0 - self.weight) * a.get(g, 0.0) + self.weight * b.get(g, 0.0)) for g in STATIC_GESTURES}


__all__ = ["BUNDLE_FORMAT", "HybridClassifier", "LearnedClassifier", "LearnedModelError"]
