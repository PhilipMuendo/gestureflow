"""Train the optional learned static-gesture classifier.

    python -m learning.train --data "data/recordings/*.jsonl" --out models/static_classifier.joblib
    python -m learning.train --data data/synthetic.jsonl --model mlp

Pipeline: JSONL recordings -> features (recomputed from raw landmarks) -> scikit-learn model.

Evaluation is honest about leakage.  Consecutive frames of one take are near-duplicates, so a
random row split would report ~100% accuracy for any model.  Instead the hold-out set is made
of whole *recording sessions* (or, if there is only one session, the final part of each
label's frames in time order).
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gestures.features import FEATURE_VECTOR_SIZE
from gestures.learned_classifier import BUNDLE_FORMAT
from gestures.types import STATIC_GESTURES
from learning.dataset import build_static_matrix, read_samples

VALID_LABELS = {g.value for g in STATIC_GESTURES} | {"none"}


@dataclass(frozen=True)
class TrainReport:
    n_train: int
    n_test: int
    accuracy: float
    per_class: dict[str, tuple[float, float, int]]  # label -> (precision, recall, support)
    confusion: np.ndarray
    labels: list[str]
    split: str

    def format(self) -> str:
        lines = [f"hold-out split: {self.split}  (train {self.n_train}, test {self.n_test})", f"accuracy: {self.accuracy:.3f}", ""]
        lines.append(f"{'label':14s} {'precision':>9s} {'recall':>8s} {'support':>8s}")
        for label, (p, r, n) in self.per_class.items():
            lines.append(f"{label:14s} {p:9.2f} {r:8.2f} {n:8d}")
        lines += ["", "confusion matrix (rows = true, cols = predicted):", "              " + " ".join(f"{l[:7]:>7s}" for l in self.labels)]
        for label, row in zip(self.labels, self.confusion):
            lines.append(f"{label[:13]:13s} " + " ".join(f"{v:7d}" for v in row))
        return "\n".join(lines)


def _make_model(kind: str, seed: int):
    """Small models only: inference runs once per frame, so latency matters as much as
    accuracy (measured per-row ``predict_proba``: RF-30 ~2 ms, MLP ~0.5 ms, HistGB-60 ~5 ms,
    RF-120 ~9 ms on a laptop CPU)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "rf":
        return RandomForestClassifier(n_estimators=30, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)
    if kind == "gb":
        return HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, random_state=seed)
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=400, random_state=seed))
    raise ValueError(f"unknown model kind {kind!r} (use 'rf', 'gb' or 'mlp')")


def split_holdout(y: np.ndarray, groups: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Boolean train/test masks that never put near-duplicate frames on both sides."""
    sessions = np.unique(groups)
    if len(sessions) >= 3:
        rng = np.random.default_rng(seed)
        n_test = max(1, int(round(len(sessions) * test_fraction)))
        test_sessions = set(rng.choice(sessions, n_test, replace=False))
        test = np.array([g in test_sessions for g in groups])
        return ~test, test, f"{n_test} of {len(sessions)} sessions held out"
    test = np.zeros(len(y), bool)  # chronological tail of each label
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        test[idx[int(len(idx) * (1 - test_fraction)):]] = True
    return ~test, test, f"last {test_fraction:.0%} of every label (time order)"


def train(paths: list[Path], out: Path, kind: str = "rf", test_fraction: float = 0.25, seed: int = 0) -> TrainReport:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    X, y, groups = build_static_matrix(read_samples(paths))
    unknown = sorted(set(y) - VALID_LABELS)
    if unknown:
        keep = np.isin(y, list(VALID_LABELS))
        print(f"warning: ignoring samples with unknown labels {unknown}; valid labels are {sorted(VALID_LABELS)}")
        X, y, groups = X[keep], y[keep], groups[keep]
    if len(np.unique(y)) < 2:
        raise ValueError("need at least two different labels to train a classifier")
    assert X.shape[1] == FEATURE_VECTOR_SIZE

    train_mask, test_mask, split = split_holdout(y, groups, test_fraction, seed)
    if not train_mask.any() or not test_mask.any():
        raise ValueError("not enough data to hold out a test set; record more samples")

    model = _make_model(kind, seed).fit(X[train_mask], y[train_mask])
    pred = model.predict(X[test_mask])
    labels = sorted(np.unique(y))
    p, r, _, support = precision_recall_fscore_support(y[test_mask], pred, labels=labels, zero_division=0)
    report = TrainReport(
        n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        accuracy=float((pred == y[test_mask]).mean()),
        per_class={l: (float(a), float(b), int(c)) for l, a, b, c in zip(labels, p, r, support)},
        confusion=confusion_matrix(y[test_mask], pred, labels=labels), labels=labels, split=split,
    )

    final = _make_model(kind, seed).fit(X, y)  # deploy the model trained on everything
    import joblib

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"format": BUNDLE_FORMAT, "model": final, "labels": [str(c) for c in final.classes_],
         "feature_size": FEATURE_VECTOR_SIZE, "accuracy_holdout": report.accuracy, "n_samples": int(len(y))},
        out,
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="JSONL files or glob patterns")
    p.add_argument("--out", type=Path, default=Path("models/static_classifier.joblib"))
    p.add_argument("--model", choices=["rf", "gb", "mlp"], default="rf", help="rf = random forest (default), gb = gradient boosting, mlp = tiny neural net (fastest)")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    files = [Path(f) for pattern in args.data for f in sorted(glob.glob(pattern))]
    if not files:
        raise SystemExit(f"no files match {args.data}")
    report = train(files, args.out, args.model, args.test_fraction, args.seed)
    print(report.format())
    print(f"\nsaved model to {args.out}\nuse it with:  python app.py --classifier learned   (or hybrid)")


if __name__ == "__main__":
    main()
