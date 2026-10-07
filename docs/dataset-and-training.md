# Dataset format, recorder and trainer

The learned classifier is optional; the rule-based engine is the default. This document specifies the
data format and the pipeline so you can record, inspect, convert or extend it. (Usage summary: the
README's *Optional: learned classifier*.)

## 1. Dataset format (`learning/dataset.py`)

**JSON Lines**: one JSON object per line, so files can be appended to, concatenated with `cat`, diffed
and inspected with any text tool. Version `1`:

```json
{"version": 1,
 "kind": "static",
 "label": "fist",
 "session": "2026-10-06T14-02-11-fist-3",
 "handedness": "Right",
 "aspect": 1.33333,
 "frames": [
   {"t": 0.0,
    "image": [[x, y, z], ... 21 rows ...],
    "world": [[x, y, z], ... 21 rows ...]}
 ]}
```

| Field | Meaning |
|---|---|
| `version` | format version; readers reject anything but `1` |
| `kind` | `static` (exactly one frame) or `sequence` (a clip of N frames) |
| `label` | gesture label. For the static trainer: a `Gesture` value (`open_palm`, `point`, `pinch`, `fist`, `two_fingers`) or `none` |
| `session` | identifier of the *take*; the trainer holds out whole sessions |
| `handedness` | `Right` / `Left` as reported by MediaPipe (mirrored view) |
| `aspect` | frame width / height at recording time (needed to rebuild image-plane geometry) |
| `frames[].t` | seconds since the start of the take |
| `frames[].image` | 21x3 image-normalised landmarks |
| `frames[].world` | 21x3 metric world landmarks |

**Raw landmarks are stored, not features.** Features are recomputed at training time, so a dataset stays
valid when the feature extractor changes. `Sample.observations()` rebuilds `HandObservation`s, i.e. the
exact input the live engine sees. Coordinates are rounded to 5 decimals (~1 KB per frame).

Reader errors carry `path:line` (`ValueError: data.jsonl:17: ...`) and the reader rejects a frame whose
landmark arrays are not 21x3.

`build_static_matrix(samples)` returns `(X, y, groups)`: one row per static frame (sequences are skipped,
degenerate frames are skipped), with `groups` = the session of each row.

## 2. Recording (`learning/record.py`)

```bash
python -m learning.record                                  # six labels, 150 frames per take
python -m learning.record --labels pinch fist --frames 200
python -m learning.record --sequence --labels swipe_left swipe_right --seconds 1.2
```

Keys: `SPACE` starts a take (3-second countdown, then it records), `N`/`P` change label, `Q` quits. Each
take gets its own session id `"<run>-<label>-<n>"` so takes can be held out as a unit. Frames without a
hand are skipped; a take is **exactly** the requested length (`TakeRecorder` ignores input after it is
done). `TakeRecorder` is independent of the camera and window, which is what makes it unit-testable.

Good data: several takes per label; vary distance, angle and position; record both hands if you use
both; include a **`none`** class (relaxed and in-between shapes) so the model learns what is *not* a
gesture.

## 3. Synthetic data (`learning/synthesize.py`)

`generate(per_class, noise, seed)` samples each class from the parametric hand model with random curls,
spread, position, palm size, roll, tilt, handedness and landmark noise (plus a `none` class of random
half-open hands), grouped into sessions of 40. Use it to try the pipeline before recording, and in the
tests. It is trivially separable, so high accuracy on it proves the plumbing, not real-world quality.

## 4. Training (`learning/train.py`)

```bash
python -m learning.train --data "data/recordings/*.jsonl" --model rf --out models/static_classifier.joblib
```

Steps: read -> features (`HandFeatures.to_vector()`, 82 values - see
[gesture-recognition.md](gesture-recognition.md#14-learned-model-feature-vector)) -> hold-out split ->
fit -> report -> **refit on all data** for the deployed model.

**Honest evaluation.** Consecutive frames of one take are near-duplicates, so a random row split would
report ~100 % for any model. `split_holdout` instead:

* with ≥ 3 sessions: holds out whole sessions (`--test-fraction` of them, chosen by seed);
* otherwise: holds out the final fraction of every label's frames *in time order*.

The report lists accuracy, per-class precision / recall / support and a confusion matrix.

| `--model` | Estimator | Per-frame `predict_proba` (measured on a laptop CPU) |
|---|---|---|
| `rf` (default) | RandomForest, 30 trees, `min_samples_leaf=2`, balanced classes | ~2 ms |
| `mlp` | StandardScaler + MLP(64) | ~0.5 ms |
| `gb` | HistGradientBoosting, 60 iterations | ~5 ms |

Inference runs once per frame, so latency matters as much as accuracy; a 120-tree forest (~9 ms) was
rejected for that reason. Labels outside `VALID_LABELS` are ignored with a warning; at least two labels
are required.

## 5. The model bundle and loading (`gestures/learned_classifier.py`)

`joblib.dump` of a dict:

```python
{"format": "gestureflow-static-v1", "model": estimator, "labels": [...], "feature_size": 82,
 "accuracy_holdout": float, "n_samples": int}
```

`LearnedClassifier.load` raises `LearnedModelError` (never a traceback at the user) when the file is
missing, not a bundle, unreadable, or built with a different `feature_size` - re-train after changing
the feature vector. On load `n_jobs` is set to 1 (thread start-up would cost more than a one-row
prediction). `python app.py --classifier learned|hybrid` loads it; if loading fails the app prints the
reason and falls back to the rules.

> **Security:** joblib files are pickles. Only load models you trained yourself or otherwise trust.

## 6. Sequences

The format and the recorder capture `sequence` samples (motion clips) so dynamic-gesture data can be
collected today. The bundled trainer consumes `static` samples only; a temporal model (GRU / 1-D CNN, or a
forest over resampled trajectories) is future work. The natural integration point is a second recogniser
next to `DynamicGestureRecognizer` producing the same `DynamicResult`.
