"""Shared pytest fixtures (nothing here touches a webcam)."""

import numpy as np
import pytest

from config import GestureConfig
from gestures.gesture_engine import GestureEngine


@pytest.fixture
def engine() -> GestureEngine:
    return GestureEngine(GestureConfig())


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
