import numpy as np
import pytest

from tracking.filters import OneEuroFilter
from tracking.hand_tracker import gamma_lut, select_primary
from tracking.synthetic import make_observation


def _series(filter_, signal, dt=1 / 30):
    return np.array([filter_(np.array([[x]]), i * dt)[0, 0] for i, x in enumerate(signal)])


def test_first_sample_passes_through():
    f = OneEuroFilter(1.0, 0.0)
    assert np.allclose(f(np.array([1.0, 2.0]), 0.0), [1.0, 2.0])


def test_filter_reduces_jitter_on_a_still_signal():
    rng = np.random.default_rng(0)
    noisy = 0.5 + rng.normal(0, 0.01, 300)
    smooth = _series(OneEuroFilter(min_cutoff=1.0, beta=0.0), noisy)
    assert smooth[30:].std() < 0.5 * noisy[30:].std()


def test_beta_reduces_lag_on_fast_motion():
    ramp_signal = np.linspace(0, 3.0, 90)  # 1 unit/s
    slow = _series(OneEuroFilter(min_cutoff=0.8, beta=0.0), ramp_signal)
    adaptive = _series(OneEuroFilter(min_cutoff=0.8, beta=10.0), ramp_signal)
    assert abs(ramp_signal[-1] - adaptive[-1]) < abs(ramp_signal[-1] - slow[-1])


def test_filter_converges_after_a_step():
    out = _series(OneEuroFilter(1.0, 5.0), np.r_[np.zeros(30), np.ones(90)])
    assert out[-1] == pytest.approx(1.0, abs=0.01)


def test_duplicate_timestamp_is_ignored_and_reset_works():
    f = OneEuroFilter(1.0, 0.0)
    f(np.array([0.0]), 0.0)
    a = f(np.array([1.0]), 0.0)  # same timestamp -> keep previous estimate
    assert a[0] == 0.0
    f.reset()
    assert f(np.array([5.0]), 1.0)[0] == 5.0


def test_gamma_lut_brightens_dark_frames_only_up_to_a_limit():
    lut = gamma_lut(luma=30, target=110)
    assert lut[128] > 128 and lut[0] == 0 and lut[255] == 255
    assert np.all(np.diff(lut.astype(int)) >= 0)  # monotonic
    bright = gamma_lut(luma=200, target=110)  # never darken
    assert bright[128] == pytest.approx(128, abs=1)


def test_primary_prefers_dominant_hand_then_sticks_to_it():
    right = make_observation("open_palm", center=(0.7, 0.5), handedness="Right")
    left = make_observation("open_palm", center=(0.3, 0.5), handedness="Left", palm_px=0.3)  # larger palm
    assert select_primary([left, right], "Right") is right
    assert select_primary([left, right], "Left") is left
    assert select_primary([left, right], "Any") is left  # closest (largest) hand
    # stickiness: the previous wrist position wins over the dominant-hand preference
    assert select_primary([left, right], "Right", previous_wrist=left.wrist_xy) is left
    assert select_primary([], "Right") is None


def test_primary_falls_back_when_dominant_hand_is_absent():
    left = make_observation("open_palm", handedness="Left")
    assert select_primary([left], "Right") is left
