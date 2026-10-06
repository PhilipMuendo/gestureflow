"""Optional learned-classifier pipeline: dataset format, training, loading, integration."""

import json

import joblib
import numpy as np
import pytest

from config import GestureConfig
from gestures.features import FEATURE_VECTOR_SIZE, extract_features
from gestures.gesture_engine import GestureEngine
from gestures.learned_classifier import BUNDLE_FORMAT, HybridClassifier, LearnedClassifier, LearnedModelError
from gestures.static_gestures import RuleBasedClassifier
from gestures.types import STATIC_GESTURES, Gesture
from learning import synthesize
from learning.dataset import (
    FORMAT_VERSION, FrameRecord, Sample, build_static_matrix, read_samples, sample_from_json, sample_to_json, write_samples,
)
from learning.record import TakeRecorder
from learning.train import split_holdout, train
from tests.helpers import run
from tracking.synthetic import make_observation


@pytest.fixture(scope="module")
def synthetic_samples():
    return synthesize.generate(per_class=120, noise=0.015, seed=1)  # 3 sessions per label


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory, synthetic_samples):
    d = tmp_path_factory.mktemp("model")
    data = d / "data.jsonl"
    write_samples(data, synthetic_samples, append=False)
    out = d / "model.joblib"
    report = train([data], out, kind="rf", test_fraction=0.25, seed=0)
    return out, report


# --------------------------------------------------------------------------- #
# dataset format
# --------------------------------------------------------------------------- #
def _sample(label="fist", kind="static", n=1):
    obs = make_observation("fist", noise=0.0)
    frames = tuple(FrameRecord(i / 30, obs.landmarks, obs.world) for i in range(n))
    return Sample(label, frames, "Right", kind, "s1", obs.aspect)


def test_sample_round_trips_through_json_losslessly_enough():
    s = _sample()
    back = sample_from_json(sample_to_json(s))
    assert back.label == s.label and back.kind == "static" and back.handedness == "Right"
    assert np.allclose(back.frames[0].world, s.frames[0].world, atol=1e-5)
    assert np.allclose(back.frames[0].image, s.frames[0].image, atol=1e-5)


def test_one_json_object_per_line_with_documented_fields(tmp_path):
    path = tmp_path / "d.jsonl"
    write_samples(path, [_sample(), _sample("point")])
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    d = json.loads(lines[0])
    assert d["version"] == FORMAT_VERSION and {"kind", "label", "session", "handedness", "aspect", "frames"} <= set(d)
    assert len(d["frames"][0]["image"]) == 21 and len(d["frames"][0]["world"]) == 21


def test_sequences_hold_many_frames_and_static_samples_exactly_one():
    seq = sample_from_json(sample_to_json(_sample("swipe_left", "sequence", n=12)))
    assert seq.kind == "sequence" and len(seq.frames) == 12 and len(seq.observations()) == 12
    with pytest.raises(ValueError):
        _sample(kind="static", n=2)
    with pytest.raises(ValueError):
        Sample("x", (), kind="static")


def test_appending_and_reading_back_multiple_files(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_samples(a, [_sample("fist")])
    write_samples(a, [_sample("point")])  # append
    write_samples(b, [_sample("pinch")])
    assert [s.label for s in read_samples([a, b])] == ["fist", "point", "pinch"]


def test_corrupt_lines_are_reported_with_file_and_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(sample_to_json(_sample()) + "\n{not json}\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(read_samples([path]))
    wrong = tmp_path / "ver.jsonl"
    wrong.write_text(json.dumps({"version": 99, "label": "x", "frames": []}) + "\n")
    with pytest.raises(ValueError, match="version"):
        list(read_samples([wrong]))


def test_feature_matrix_skips_sequences_and_degenerate_frames():
    degenerate = Sample("fist", (FrameRecord(0.0, np.zeros((21, 3)), np.zeros((21, 3))),), "Right", "static", "z", 1.33)
    X, y, groups = build_static_matrix([_sample("fist"), _sample("x", "sequence", 5), degenerate])
    assert X.shape == (1, FEATURE_VECTOR_SIZE) and list(y) == ["fist"] and list(groups) == ["s1"]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_synthetic_data_covers_all_classes_and_both_hands(synthetic_samples):
    labels = {s.label for s in synthetic_samples}
    assert labels == {g.value for g in STATIC_GESTURES} | {"none"}
    assert {s.handedness for s in synthetic_samples} == {"Left", "Right"}


def test_trained_model_generalises_to_held_out_sessions(trained_model):
    _, report = trained_model
    assert report.accuracy > 0.95
    assert report.n_train > 0 and report.n_test > 0
    assert "session" in report.split
    assert report.confusion.sum() == report.n_test


def test_holdout_split_never_mixes_a_session_across_train_and_test():
    groups = np.array([f"s{i // 10}" for i in range(100)])
    y = np.array(["a"] * 100)
    train_mask, test_mask, _ = split_holdout(y, groups, 0.3, seed=0)
    assert not (train_mask & test_mask).any() and (train_mask | test_mask).all()
    assert not set(groups[train_mask]) & set(groups[test_mask])


def test_single_session_falls_back_to_a_chronological_split():
    y = np.array(["a"] * 50 + ["b"] * 50)
    train_mask, test_mask, how = split_holdout(y, np.array(["only"] * 100), 0.2, seed=0)
    assert "time order" in how
    assert test_mask[40:50].all() and test_mask[90:].all() and not test_mask[:40].any()


def test_training_needs_two_labels(tmp_path):
    path = tmp_path / "one.jsonl"
    write_samples(path, [_sample("fist")] * 5)
    with pytest.raises(ValueError, match="two different labels"):
        train([path], tmp_path / "m.joblib")


def test_every_model_kind_trains(tmp_path, synthetic_samples):
    data = tmp_path / "d.jsonl"
    write_samples(data, synthetic_samples[:240] + synthetic_samples[-240:], append=False)
    for kind in ("rf", "gb", "mlp"):
        report = train([data], tmp_path / f"{kind}.joblib", kind=kind)
        assert report.accuracy > 0.85, kind


# --------------------------------------------------------------------------- #
# using the model
# --------------------------------------------------------------------------- #
def test_learned_classifier_recognises_unseen_poses(trained_model):
    clf = LearnedClassifier.load(trained_model[0])
    rng = np.random.default_rng(99)
    for preset, gesture in [("open_palm", Gesture.OPEN_PALM), ("point", Gesture.POINT), ("fist", Gesture.FIST),
                            ("two_fingers", Gesture.TWO_FINGERS), ("pinch", Gesture.PINCH)]:
        obs = make_observation(preset, roll_deg=float(rng.uniform(-60, 60)), palm_px=0.2, noise=0.015, rng=rng, handedness="Left")
        scores = clf.classify(extract_features(obs))
        assert set(scores) == set(STATIC_GESTURES)
        assert max(scores, key=scores.get) is gesture and scores[gesture] > 0.5


def test_engine_runs_on_the_learned_classifier(trained_model):
    engine = GestureEngine(GestureConfig(), LearnedClassifier.load(trained_model[0]))
    states = run(engine, 1.2, "two_fingers", noise=0.01)
    assert states[-1].static is Gesture.TWO_FINGERS


def test_hybrid_averages_rule_and_learned_scores(trained_model):
    rules, learned = RuleBasedClassifier(), LearnedClassifier.load(trained_model[0])
    hybrid = HybridClassifier(rules, learned, 0.5)
    f = extract_features(make_observation("point"))
    a, b, c = rules.classify(f), learned.classify(f), hybrid.classify(f)
    for g in STATIC_GESTURES:
        assert c[g] == pytest.approx(0.5 * a[g] + 0.5 * b[g])


@pytest.mark.parametrize("problem", ["missing", "garbage", "wrong_format", "wrong_size"])
def test_bad_model_files_raise_a_clear_error(tmp_path, problem):
    path = tmp_path / "m.joblib"
    if problem == "garbage":
        path.write_bytes(b"not a pickle")
    elif problem == "wrong_format":
        joblib.dump({"hello": 1}, path)
    elif problem == "wrong_size":
        joblib.dump({"format": BUNDLE_FORMAT, "model": object(), "labels": [], "feature_size": FEATURE_VECTOR_SIZE + 1}, path)
    with pytest.raises(LearnedModelError):
        LearnedClassifier.load(path)


# --------------------------------------------------------------------------- #
# recorder
# --------------------------------------------------------------------------- #
def _feed(rec, n, hand_every=1):
    for i in range(n):
        obs = make_observation("fist", timestamp=i / 30) if i % hand_every == 0 else None
        rec.feed(obs, i / 30)


def test_static_take_stops_after_the_requested_number_of_frames_and_skips_missing_hands():
    rec = TakeRecorder("fist", "static", frames=20, seconds=1.0, session="t1")
    _feed(rec, 60, hand_every=2)  # every other frame has no hand
    assert rec.done
    samples = rec.samples()
    assert len(samples) == 20 and all(s.kind == "static" and s.session == "t1" and s.label == "fist" for s in samples)


def test_sequence_take_is_one_clip_and_needs_enough_frames():
    rec = TakeRecorder("swipe_left", "sequence", frames=0, seconds=1.0, session="t2")
    _feed(rec, 40)
    assert rec.done
    (clip,) = rec.samples()
    assert clip.kind == "sequence" and len(clip.frames) >= 30 and clip.frames[0].t == 0.0
    empty = TakeRecorder("swipe_left", "sequence", frames=0, seconds=1.0, session="t3")
    _feed(empty, 40, hand_every=100)  # a single frame with a hand
    assert empty.samples() == []


def test_recorder_progress_is_monotonic():
    rec = TakeRecorder("fist", "static", frames=10, seconds=1.0, session="t")
    seen = []
    for i in range(12):
        rec.feed(make_observation("fist"), i / 30)
        seen.append(rec.progress)
    assert seen == sorted(seen) and seen[-1] == 1.0
