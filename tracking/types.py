"""Plain data types shared by the tracking, gesture and visualisation layers.

Kept free of any MediaPipe / OpenCV import so the gesture code (and its tests)
can run on machines that only have NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

NUM_LANDMARKS = 21


class Landmark(IntEnum):
    """MediaPipe hand-landmark indices."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Landmark chains per finger: (base/knuckle, pip-like, dip-like, tip).  For the
# thumb the anatomical names differ (CMC, MCP, IP, TIP) but the topology is the same.
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
FINGERTIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (0, 5, 9, 13, 17)  # wrist + the four knuckles

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class HandObservation:
    """One tracked hand in one frame.

    Attributes:
        landmarks: (21, 3) image-normalised coordinates.  x and y are in [0, 1]
            relative to the (already mirrored) frame; z is depth relative to the
            wrist on roughly the same scale as x.
        world: (21, 3) metric 3D coordinates in metres, origin near the hand
            centre, axes aligned with the camera (x right, y down, z away).
            These do not change with the distance to the camera, which makes them
            the right input for shape features.
        handedness: "Left" or "Right" as seen by the user (frame is mirrored).
        score: detector confidence in [0, 1].
        timestamp: capture time in seconds (monotonic).
        aspect: frame width / height, needed to make x and y isotropic.
    """

    landmarks: np.ndarray
    world: np.ndarray
    handedness: str
    score: float
    timestamp: float
    aspect: float = 4.0 / 3.0

    def __post_init__(self) -> None:
        if self.landmarks.shape != (NUM_LANDMARKS, 3) or self.world.shape != (NUM_LANDMARKS, 3):
            raise ValueError("HandObservation expects (21, 3) landmark arrays")

    @property
    def image_iso(self) -> np.ndarray:
        """Image landmarks with x and z rescaled so one unit is the same length on
        every axis (units of frame height).  Needed before measuring angles or
        distances in image space."""
        return self.landmarks * np.array([self.aspect, 1.0, self.aspect])

    @property
    def wrist_xy(self) -> np.ndarray:
        return self.landmarks[Landmark.WRIST, :2]


@dataclass(frozen=True)
class TrackingResult:
    """Everything the tracker knows about one frame."""

    hands: tuple[HandObservation, ...] = ()
    primary: HandObservation | None = None
    inference_ms: float = 0.0
    brightness: float = 255.0
    low_light: bool = False
