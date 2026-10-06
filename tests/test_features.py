"""Unit tests for the geometric feature layer.  All inputs are synthetic landmarks."""

import numpy as np
import pytest

from gestures.features import (
    extract_features,
    finger_joint_angles,
    joint_angle,
    normalize_landmarks,
    ramp,
    roll_angle_deg,
)
from tracking.synthetic import POSES, HandPose, make_observation, hand_points
from tracking.types import HandObservation, Landmark


# --------------------------------------------------------------------------- #
# math helpers
# --------------------------------------------------------------------------- #
def test_ramp_is_clamped_and_supports_descending_ranges():
    assert ramp(0.0, 0.2, 0.8) == 0.0
    assert ramp(1.0, 0.2, 0.8) == 1.0
    assert ramp(0.5, 0.2, 0.8) == pytest.approx(0.5)
    assert ramp(0.0, 0.8, 0.2) == 1.0  # descending: low input -> 1
    assert ramp(1.0, 0.8, 0.2) == 0.0


def test_joint_angle_known_geometry():
    o = np.zeros(3)
    assert joint_angle(np.array([1.0, 0, 0]), o, np.array([0, 1.0, 0])) == pytest.approx(90.0)
    assert joint_angle(np.array([1.0, 0, 0]), o, np.array([-1.0, 0, 0])) == pytest.approx(180.0)
    assert joint_angle(np.array([1.0, 0, 0]), o, np.array([1.0, 1.0, 0])) == pytest.approx(45.0)


def test_joint_angle_degenerate_is_straight_not_nan():
    p = np.ones(3)
    assert joint_angle(p, p, p) == 180.0


def test_roll_angle_matches_screen_rotation():
    # roll is relative to the hand's rest orientation; rotating clockwise by 30 deg adds 30.
    base = extract_features(make_observation("open_palm", roll_deg=0)).roll_deg
    for angle in (-60, -20, 25, 90):
        f = extract_features(make_observation("open_palm", roll_deg=angle))
        assert f.roll_deg - base == pytest.approx(angle, abs=1e-6)


def test_roll_angle_cardinal_directions():
    iso = np.zeros((21, 3))
    iso[Landmark.MIDDLE_MCP] = [0.0, -1.0, 0.0]  # straight up on screen (y grows downward)
    assert roll_angle_deg(iso) == pytest.approx(0.0)
    iso[Landmark.MIDDLE_MCP] = [1.0, 0.0, 0.0]  # pointing right
    assert roll_angle_deg(iso) == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_normalisation_puts_wrist_at_origin_with_unit_palm_length():
    obs = make_observation("open_palm", palm_px=0.3, center=(0.3, 0.4))
    norm = normalize_landmarks(obs.world)
    assert np.allclose(norm[Landmark.WRIST], 0.0)
    assert np.linalg.norm(norm[Landmark.MIDDLE_MCP]) == pytest.approx(1.0)
    # the middle knuckle lies on the palm "up" axis
    assert np.allclose(norm[Landmark.MIDDLE_MCP], [0.0, 1.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("pose", ["open_palm", "pinch", "fist", "point"])
def test_normalised_shape_is_invariant_to_rotation_and_tilt(pose):
    """The same hand shape must give the same normalised landmarks however the hand
    is rotated - that is the whole point of the palm frame."""
    reference = normalize_landmarks(make_observation(pose).world)
    for roll, yaw, pitch in [(35, 0, 0), (-80, 0, 0), (150, 0, 0), (10, 25, -15)]:
        rotated = normalize_landmarks(make_observation(pose, roll_deg=roll, yaw_deg=yaw, pitch_deg=pitch).world)
        assert np.allclose(reference, rotated, atol=1e-6)


def test_normalised_shape_is_invariant_to_distance_and_position():
    near = make_observation("pinch", palm_px=0.35, center=(0.3, 0.4))
    far = make_observation("pinch", palm_px=0.10, center=(0.8, 0.7))
    # world landmarks are metric, so distance does not change them; image landmarks do
    assert not np.allclose(near.landmarks, far.landmarks)
    assert np.allclose(normalize_landmarks(near.world), normalize_landmarks(far.world), atol=1e-9)


def test_left_hand_normalises_like_a_mirrored_right_hand():
    right = normalize_landmarks(make_observation("point", handedness="Right").world, "Right")
    left = normalize_landmarks(make_observation("point", handedness="Left").world, "Left")
    assert np.allclose(right, left, atol=1e-9)


def test_normalisation_rejects_degenerate_hand():
    with pytest.raises(ValueError):
        normalize_landmarks(np.zeros((21, 3)))


# --------------------------------------------------------------------------- #
# finger features
# --------------------------------------------------------------------------- #
def _ext(pose, **kw):
    return extract_features(make_observation(pose, **kw)).extension


def test_open_palm_has_all_fingers_extended():
    ext = _ext("open_palm")
    assert np.all(ext[1:] > 0.9)


def test_fist_has_all_fingers_curled():
    ext = _ext("fist")
    assert np.all(ext[1:] < 0.1)


def test_pointing_extends_only_the_index():
    ext = _ext("point")
    assert ext[1] > 0.9 and np.all(ext[2:] < 0.1)


def test_two_fingers_extends_index_and_middle():
    ext = _ext("two_fingers")
    assert ext[1] > 0.9 and ext[2] > 0.9 and ext[3] < 0.1 and ext[4] < 0.1


def test_extension_decreases_monotonically_with_curl():
    values = []
    for curl in np.linspace(0, 1, 11):
        pose = HandPose(curls=(curl, curl, curl, curl))
        values.append(extract_features(make_observation(pose)).extension[1:].mean())
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))
    assert values[0] > 0.9 and values[-1] < 0.1


def test_joint_angles_are_physically_sensible():
    straight = finger_joint_angles(hand_points(POSES["open_palm"]))
    curled = finger_joint_angles(hand_points(POSES["fist"]))
    # straight fingers: every joint near 180; curled fingers: clearly bent
    assert np.all(straight[1:] > 160)
    assert np.all(curled[1:, 1] < 95)  # PIP angle of a fist is acute-ish


def test_finger_extension_is_stable_under_noise(rng):
    clean = _ext("two_fingers")
    for seed in range(5):
        noisy = extract_features(make_observation("two_fingers", noise=0.02, rng=np.random.default_rng(seed))).extension
        assert np.abs(noisy[1:] - clean[1:]).max() < 0.2


@pytest.mark.parametrize("palm_px", [0.08, 0.15, 0.35])
@pytest.mark.parametrize("roll", [-90, 0, 120])
def test_extension_is_independent_of_distance_and_rotation(palm_px, roll):
    ref = _ext("pinch")
    got = _ext("pinch", palm_px=palm_px, roll_deg=roll, center=(0.2, 0.8))
    assert np.allclose(ref, got, atol=1e-6)


# --------------------------------------------------------------------------- #
# distances, spread, palm orientation
# --------------------------------------------------------------------------- #
def test_pinch_distance_separates_pinch_from_open_hand():
    pinch = extract_features(make_observation("pinch")).pinch_distance
    opened = extract_features(make_observation("open_palm")).pinch_distance
    assert pinch < 0.35 < 0.7 < opened


def test_pinch_distance_is_scale_invariant():
    a = extract_features(make_observation("pinch", palm_px=0.1)).pinch_distance
    b = extract_features(make_observation("pinch", palm_px=0.4)).pinch_distance
    assert a == pytest.approx(b, abs=1e-6)


def test_spread_orders_together_open_spread():
    together = extract_features(make_observation("together")).spread
    spread = extract_features(make_observation("spread")).spread
    assert spread > together + 10.0  # degrees of fan-out


def test_spread_is_independent_of_finger_flexion():
    """Bending the fingers must not change the measured fan-out (that is why the zoom
    gesture uses abduction angle rather than fingertip distance)."""
    base = extract_features(make_observation(HandPose(curls=(0, 0, 0, 0), spread=0.6))).spread
    for curl in (0.1, 0.2, 0.3):
        flexed = extract_features(make_observation(HandPose(curls=(curl,) * 4, spread=0.6))).spread
        assert flexed == pytest.approx(base, abs=1.0)


def test_spread_is_scale_and_rotation_invariant():
    ref = extract_features(make_observation("spread")).spread
    other = extract_features(make_observation("spread", palm_px=0.4, roll_deg=70, handedness="Left")).spread
    assert other == pytest.approx(ref, abs=1e-6)


def test_fingertip_distance_matrix_is_symmetric_with_zero_diagonal():
    d = extract_features(make_observation("open_palm")).fingertip_dists
    assert d.shape == (5, 5)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0.0)


@pytest.mark.parametrize("hand", ["Right", "Left"])
def test_palm_facing_camera_for_both_hands(hand):
    f = extract_features(make_observation("open_palm", handedness=hand))
    assert f.palm_facing == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(f.palm_normal, [0, 0, -1], atol=1e-6)  # out of the palm = towards the camera


def test_back_of_hand_facing_away():
    f = extract_features(make_observation("open_palm", yaw_deg=180))
    assert f.palm_facing == pytest.approx(-1.0, abs=1e-6)


def test_palm_yaw_tracks_rotation_about_vertical_axis():
    f = extract_features(make_observation("open_palm", yaw_deg=30))
    assert abs(f.palm_yaw_deg) == pytest.approx(30, abs=1.0)
    assert 0.8 < f.palm_facing < 0.9  # cos(30deg)


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def test_extract_features_positions_follow_the_image_landmarks():
    obs = make_observation("pinch", center=(0.25, 0.75))
    f = extract_features(obs)
    assert f.palm_center == pytest.approx([0.25, 0.75], abs=1e-6)
    thumb, index = obs.landmarks[Landmark.THUMB_TIP, :2], obs.landmarks[Landmark.INDEX_TIP, :2]
    assert np.allclose(f.pinch_point, (thumb + index) / 2)
    assert np.allclose(f.index_tip, index)


def test_palm_size_shrinks_with_distance():
    near = extract_features(make_observation("open_palm", palm_px=0.3)).palm_size
    far = extract_features(make_observation("open_palm", palm_px=0.1)).palm_size
    assert near == pytest.approx(0.3, rel=0.05) and far == pytest.approx(0.1, rel=0.05)


def test_feature_vector_has_fixed_length():
    from gestures.features import FEATURE_VECTOR_SIZE

    for pose in POSES:
        assert extract_features(make_observation(pose)).to_vector().shape == (FEATURE_VECTOR_SIZE,)


def test_extract_features_rejects_non_finite_landmarks():
    obs = make_observation("open_palm")
    bad = obs.world.copy()
    bad[5, 0] = np.nan
    with pytest.raises(ValueError):
        extract_features(HandObservation(obs.landmarks, bad, "Right", 0.9, 0.0))


def test_observation_validates_shape():
    with pytest.raises(ValueError):
        HandObservation(np.zeros((20, 3)), np.zeros((21, 3)), "Right", 1.0, 0.0)
