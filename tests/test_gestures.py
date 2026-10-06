"""Tests for gesture classification, temporal smoothing and dynamic gestures.

Everything is driven by synthetic landmarks and explicit timestamps - no webcam,
no wall clock.
"""

import numpy as np
import pytest

from config import GestureConfig
from gestures.features import extract_features
from gestures.gesture_engine import ALL_GESTURES, GestureEngine
from gestures.static_gestures import RuleBasedClassifier, best_static
from gestures.types import STATIC_GESTURES, Gesture
from tests.helpers import DT, lerp_pose, run, smoothstep
from tracking.synthetic import HandPose, make_observation
from tracking.types import HandObservation

PRESET_TO_GESTURE = {
    "open_palm": Gesture.OPEN_PALM,
    "point": Gesture.POINT,
    "pinch": Gesture.PINCH,
    "fist": Gesture.FIST,
    "two_fingers": Gesture.TWO_FINGERS,
}


def classify(pose, **kw) -> dict[Gesture, float]:
    return RuleBasedClassifier().classify(extract_features(make_observation(pose, **kw)))


# =========================================================================== #
# Static classification
# =========================================================================== #
@pytest.mark.parametrize("preset,gesture", PRESET_TO_GESTURE.items())
def test_each_preset_is_classified_correctly_with_margin(preset, gesture):
    scores = classify(preset)
    winner, score = best_static(scores)
    assert winner is gesture
    assert score > 0.85
    runner_up = sorted(scores.values())[-2]
    assert runner_up < 0.3  # unambiguous


@pytest.mark.parametrize("preset,gesture", PRESET_TO_GESTURE.items())
@pytest.mark.parametrize("hand", ["Right", "Left"])
@pytest.mark.parametrize("roll", [-100, 0, 70])
@pytest.mark.parametrize("palm_px", [0.1, 0.3])
def test_classification_is_invariant_to_hand_distance_rotation_and_side(preset, gesture, hand, roll, palm_px):
    winner, score = best_static(classify(preset, handedness=hand, roll_deg=roll, palm_px=palm_px, center=(0.3, 0.6)))
    assert winner is gesture and score > 0.8


@pytest.mark.parametrize("preset,gesture", PRESET_TO_GESTURE.items())
def test_classification_survives_landmark_noise(preset, gesture):
    wins = 0
    for seed in range(20):
        winner, _ = best_static(classify(preset, noise=0.03, rng=np.random.default_rng(seed)), min_score=0.5)
        wins += winner is gesture
    assert wins >= 18


def test_half_curled_hand_is_not_confidently_anything():
    pose = HandPose(curls=(0.5, 0.5, 0.5, 0.5), spread=0.2)
    _, score = best_static(classify(pose))
    assert score < 0.5


def test_fist_and_pinch_are_not_confused():
    assert classify("fist")[Gesture.PINCH] < 0.2
    assert classify("pinch")[Gesture.FIST] < 0.2
    assert classify("point")[Gesture.PINCH] < 0.1


def test_scores_are_in_unit_range_for_random_poses(rng):
    clf = RuleBasedClassifier()
    for _ in range(100):
        pose = HandPose(curls=tuple(rng.uniform(0, 1, 4)), spread=float(rng.uniform(0, 1)))
        obs = make_observation(pose, roll_deg=float(rng.uniform(-180, 180)), noise=0.02, rng=rng)
        scores = clf.classify(extract_features(obs))
        assert set(scores) == set(STATIC_GESTURES)
        assert all(0.0 <= s <= 1.0 for s in scores.values())


# =========================================================================== #
# Temporal smoothing / hysteresis
# =========================================================================== #
def test_steady_gesture_becomes_stable_with_high_confidence(engine):
    states = run(engine, 1.0, "open_palm")
    last = states[-1]
    assert last.hand_present and last.static is Gesture.OPEN_PALM
    assert last.gesture is Gesture.OPEN_PALM and last.confidence > 0.9
    assert last.hold_s > 0.7


def test_no_gesture_is_reported_before_confirm_time(engine):
    states = run(engine, 0.2, "fist")
    assert states[0].static is Gesture.NONE  # not instantly confident
    assert states[-1].static is Gesture.FIST


def test_single_frame_glitch_does_not_change_the_gesture(engine):
    def pose(u):
        return "fist" if abs(u - 0.5) < DT * 0.6 else "open_palm"  # exactly one fist frame

    states = run(engine, 1.0, pose)
    assert {s.static for s in states[10:]} == {Gesture.OPEN_PALM}


def test_clean_transition_changes_state_once(engine):
    run(engine, 0.6, "open_palm")
    states = run(engine, 0.8, "fist", start=0.6)
    seq = [s.static for s in states]
    changes = sum(a is not b for a, b in zip(seq, seq[1:]))
    assert seq[0] is Gesture.OPEN_PALM and seq[-1] is Gesture.FIST
    assert changes <= 2  # OPEN_PALM -> (NONE) -> FIST


def test_ambiguous_hand_does_not_flicker(engine):
    """A hand held at the boundary between two gestures, with sensor noise, must not
    toggle every frame."""
    pose = HandPose(curls=(0.02, 0.62, 0.62, 0.62), spread=0.1)  # between point and fist
    states = run(engine, 3.0, pose, noise=0.03, seed=3)
    seq = [s.static for s in states[15:]]
    changes = sum(a is not b for a, b in zip(seq, seq[1:]))
    assert changes <= 4


def test_hysteresis_keeps_gesture_while_score_dips_between_thresholds(engine):
    run(engine, 0.6, "point")
    # open the middle finger a little: POINT score drops but stays above the exit threshold
    pose = HandPose(curls=(0.02, 0.62, 0.95, 0.95), spread=0.1)
    states = run(engine, 0.6, pose, start=0.6)
    assert states[-1].static is Gesture.POINT


def test_brief_hand_dropout_is_bridged(engine):
    states = run(engine, 1.5, "pinch", present=lambda u: not (0.8 <= u < 0.95))
    dropped = [s for s in states if s.stale]
    assert dropped and all(s.hand_present and s.static is Gesture.PINCH for s in dropped)
    assert states[-1].static is Gesture.PINCH and not states[-1].stale
    assert all(s.motion.speed == 0.0 for s in dropped)


def test_long_dropout_resets_state_and_recovers(engine):
    states = run(engine, 2.0, "fist", present=lambda u: u < 0.8 or u >= 1.4)
    gone = [s for s in states if not s.hand_present]
    assert gone and all(s.gesture is Gesture.NONE for s in gone)
    assert states[-1].static is Gesture.FIST and states[-1].hand_present


def test_no_hand_at_all(engine):
    states = run(engine, 0.5, "open_palm", present=lambda u: False)
    assert all(not s.hand_present and s.gesture is Gesture.NONE and s.features is None for s in states)


def test_degenerate_observation_is_treated_as_no_hand(engine):
    obs = make_observation("open_palm")
    flat = HandObservation(obs.landmarks, np.zeros((21, 3)), "Right", 0.9, 0.0)
    state = engine.process(flat, 0.0)
    assert not state.hand_present


def test_disabled_gestures_are_never_reported():
    engine = GestureEngine(GestureConfig(), enabled=ALL_GESTURES - {Gesture.POINT})
    states = run(engine, 1.0, "point")
    assert all(s.static is not Gesture.POINT and s.gesture is not Gesture.POINT for s in states)
    engine.set_enabled(ALL_GESTURES)
    states = run(engine, 1.0, "point", start=1.0)
    assert states[-1].static is Gesture.POINT


def test_low_tracking_noise_does_not_break_pinch(engine):
    states = run(engine, 1.0, "pinch", noise=0.03, seed=11)
    assert sum(s.static is Gesture.PINCH for s in states[10:]) >= 0.9 * len(states[10:])


# =========================================================================== #
# Dynamic gestures
# =========================================================================== #
def _events(states):
    return [e for s in states for e in s.events]


def swipe_x(x0, x1, duration):
    return lambda u: (x0 + (x1 - x0) * smoothstep(u / duration), 0.55)


def test_fast_horizontal_motion_is_a_swipe_right(engine):
    run(engine, 0.6, "open_palm", center=(0.2, 0.55))
    states = run(engine, 0.6, "open_palm", start=0.6, center=swipe_x(0.2, 0.8, 0.35))
    assert _events(states) == [Gesture.SWIPE_RIGHT]


def test_fast_horizontal_motion_is_a_swipe_left(engine):
    run(engine, 0.6, "open_palm", center=(0.8, 0.55))
    states = run(engine, 0.6, "open_palm", start=0.6, center=swipe_x(0.8, 0.2, 0.35))
    assert _events(states) == [Gesture.SWIPE_LEFT]


def test_swipe_fires_only_once_per_stroke(engine):
    run(engine, 0.5, "open_palm", center=(0.1, 0.55))
    states = run(engine, 1.5, "open_palm", start=0.5, center=swipe_x(0.1, 0.9, 0.5))
    assert len(_events(states)) == 1


def test_slow_drift_is_not_a_swipe(engine):
    run(engine, 0.5, "open_palm", center=(0.2, 0.55))
    states = run(engine, 3.5, "open_palm", start=0.5, center=swipe_x(0.2, 0.8, 3.5))
    assert _events(states) == []


def test_vertical_motion_is_not_a_swipe(engine):
    run(engine, 0.5, "open_palm", center=(0.5, 0.2))
    states = run(engine, 0.6, "open_palm", start=0.5, center=lambda u: (0.5, 0.2 + 0.6 * smoothstep(u / 0.35)))
    assert _events(states) == []


def test_short_fast_jitter_is_not_a_swipe(engine):
    run(engine, 0.5, "open_palm")
    jitter = lambda u: (0.5 + 0.03 * np.sin(2 * np.pi * 6 * u), 0.55)  # noqa: E731
    states = run(engine, 2.0, "open_palm", start=0.5, center=jitter)
    assert _events(states) == []


def test_swipe_requires_an_open_hand(engine):
    for pose in ("fist", "pinch", "point"):
        engine.reset()
        run(engine, 0.6, pose, center=(0.2, 0.55))
        states = run(engine, 0.6, pose, start=0.6, center=swipe_x(0.2, 0.8, 0.35))
        assert _events(states) == [], pose


def test_return_stroke_right_after_a_swipe_is_ignored(engine):
    run(engine, 0.5, "open_palm", center=(0.2, 0.55))
    out = run(engine, 0.45, "open_palm", start=0.5, center=swipe_x(0.2, 0.8, 0.35))
    back = run(engine, 0.7, "open_palm", start=0.95, center=swipe_x(0.8, 0.2, 0.35))  # hand swings straight back
    assert _events(out) == [Gesture.SWIPE_RIGHT]
    assert _events(back) == []
    # a deliberate swipe after the hand has rested works again
    run(engine, 0.8, "open_palm", start=1.65, center=(0.2, 0.55))
    again = run(engine, 0.6, "open_palm", start=2.45, center=swipe_x(0.2, 0.8, 0.35))
    assert _events(again) == [Gesture.SWIPE_RIGHT]


def test_swipe_detection_is_distance_independent(engine):
    """A hand far from the camera (small in the image) must travel proportionally
    less on screen to count as a swipe."""
    palm_px = 0.10
    run(engine, 0.6, "open_palm", center=(0.3, 0.55), palm_px=palm_px)
    states = run(engine, 0.6, "open_palm", start=0.6, palm_px=palm_px, center=swipe_x(0.3, 0.55, 0.3))
    # travelled 0.25 of the frame width = 3.3 iso/0.1 -> 3.3 palm lengths
    assert _events(states) == [Gesture.SWIPE_RIGHT]


def _open_then(pose_end, seconds=0.5, hold=0.7):
    """Open hand held still, then morph to ``pose_end``."""
    return lambda u: lerp_pose("together", pose_end, smoothstep((u - hold) / seconds))


def test_spreading_fingers_zooms_in(engine):
    states = run(engine, 2.0, _open_then("spread"))
    zoom = [s for s in states if s.dynamic is Gesture.ZOOM_IN]
    assert zoom, "spreading the fingers should be recognised"
    assert not [s for s in states if s.dynamic is Gesture.ZOOM_OUT]
    assert sum(s.motion.spread_delta for s in zoom) > 0.1
    assert all(s.static is Gesture.OPEN_PALM for s in states[-10:])  # ends as a normal open palm


def test_closing_fingers_zooms_out(engine):
    pose = lambda u: lerp_pose("spread", "together", smoothstep((u - 0.7) / 0.5))  # noqa: E731
    states = run(engine, 2.0, pose)
    zoom = [s for s in states if s.dynamic is Gesture.ZOOM_OUT]
    assert zoom and not [s for s in states if s.dynamic is Gesture.ZOOM_IN]
    assert sum(s.motion.spread_delta for s in zoom) < -0.1


def test_zoom_stops_when_the_fingers_stop(engine):
    states = run(engine, 3.0, _open_then("spread"))
    assert states[-1].dynamic is Gesture.NONE


def test_still_open_hand_never_zooms_even_with_noise(engine):
    states = run(engine, 4.0, "open_palm", noise=0.03, seed=5)
    assert not [s for s in states if s.dynamic in (Gesture.ZOOM_IN, Gesture.ZOOM_OUT)]


def test_relaxing_a_fist_into_an_open_palm_is_not_a_zoom(engine):
    pose = lambda u: lerp_pose("fist", "together", smoothstep((u - 0.5) / 0.4))  # noqa: E731
    states = run(engine, 2.0, pose)
    assert not [s for s in states if s.dynamic in (Gesture.ZOOM_IN, Gesture.ZOOM_OUT)]


def test_pinch_and_twist_rotates(engine):
    run(engine, 0.6, "pinch", roll=0.0)
    states = run(engine, 1.6, "pinch", start=0.6, roll=lambda u: 70.0 * smoothstep(u / 0.9))
    rotating = [s for s in states if s.dynamic is Gesture.PINCH_ROTATE]
    assert rotating
    assert not [s for s in states if s.dynamic is Gesture.PINCH_DRAG]
    assert sum(s.motion.roll_delta_deg for s in states) == pytest.approx(70.0, abs=1.0)
    assert states[-1].dynamic is Gesture.NONE  # stops after the twist ends


def test_rotation_direction_is_signed(engine):
    run(engine, 0.6, "pinch")
    states = run(engine, 1.0, "pinch", start=0.6, roll=lambda u: -50.0 * smoothstep(u / 0.7))
    assert sum(s.motion.roll_delta_deg for s in states if s.dynamic is Gesture.PINCH_ROTATE) < -20


def test_pinch_and_move_drags(engine):
    run(engine, 0.6, "pinch", center=(0.3, 0.5))
    states = run(engine, 1.0, "pinch", start=0.6, center=lambda u: (0.3 + 0.3 * smoothstep(u / 0.6), 0.5))
    assert [s for s in states if s.dynamic is Gesture.PINCH_DRAG]
    assert not [s for s in states if s.dynamic is Gesture.PINCH_ROTATE]
    assert states[-1].dynamic is Gesture.NONE


def test_still_pinch_is_just_a_pinch(engine):
    states = run(engine, 2.0, "pinch", noise=0.02, seed=2)
    assert states[-1].static is Gesture.PINCH
    assert all(s.dynamic is Gesture.NONE for s in states)


def test_pinch_wobble_is_not_a_drag(engine):
    wobble = lambda u: (0.5 + 0.004 * np.sin(9 * u), 0.55)  # noqa: E731
    states = run(engine, 2.0, "pinch", center=wobble)
    assert all(s.dynamic is Gesture.NONE for s in states)


def test_dynamic_gestures_can_be_disabled():
    engine = GestureEngine(GestureConfig(), enabled=ALL_GESTURES - {Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT})
    run(engine, 0.6, "open_palm", center=(0.2, 0.55))
    states = run(engine, 0.6, "open_palm", start=0.6, center=swipe_x(0.2, 0.8, 0.35))
    assert _events(states) == []


def test_two_finger_scroll_reports_vertical_travel(engine):
    run(engine, 0.6, "two_fingers", center=(0.5, 0.7))
    states = run(engine, 0.6, "two_fingers", start=0.6, center=lambda u: (0.5, 0.7 - 0.3 * smoothstep(u / 0.5)))
    assert states[-1].static is Gesture.TWO_FINGERS
    total_up = sum(s.motion.scroll_delta for s in states)
    assert total_up == pytest.approx(0.3 / 0.22, rel=0.15)  # 0.3 frame heights / palm length


def test_scroll_is_zero_for_other_gestures(engine):
    states = run(engine, 1.2, "open_palm", center=lambda u: (0.5, 0.7 - 0.2 * u))
    assert all(s.motion.scroll_delta == 0.0 for s in states)


def test_engine_reset_clears_everything(engine):
    run(engine, 1.0, "fist")
    engine.reset()
    state = engine.process(None, 5.0)
    assert not state.hand_present and state.static is Gesture.NONE
