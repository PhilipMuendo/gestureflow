"""Geometric feature extraction from 21 hand landmarks.

Pipeline position::

    HandObservation ──► extract_features() ──► HandFeatures ──► classifiers

Two coordinate systems are used on purpose:

* **World landmarks** (metres, camera-aligned, origin at the hand centre) are used
  for everything about the *shape* of the hand - joint angles, finger extension,
  fingertip distances, palm orientation.  They do not change when the hand moves
  closer to or farther from the camera, which is exactly the robustness we want.
* **Image landmarks** (normalised frame coordinates, x/z rescaled by the aspect
  ratio so axes are isotropic) are used for everything about *where the hand is
  and how it moves on screen* - cursor position, wrist roll, palm size in pixels.

Every length that ends up in a threshold is divided by the **palm length**
(wrist -> middle-finger knuckle) so the numbers are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import FeatureConfig
from tracking.types import FINGER_CHAINS, FINGERTIPS, PALM_POINTS, HandObservation, Landmark

_EPS = 1e-9
DEFAULT_FEATURE_CONFIG = FeatureConfig()


# --------------------------------------------------------------------------- #
# Small math helpers
# --------------------------------------------------------------------------- #
def ramp(x: float | np.ndarray, lo: float, hi: float) -> float | np.ndarray:
    """Linear membership: 0 at ``lo``, 1 at ``hi``, clamped.  ``lo > hi`` gives a
    descending ramp (1 at ``hi``, 0 at ``lo``)."""
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > _EPS else np.zeros_like(v)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at ``b`` of the path a-b-c, in degrees (180 = straight)."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < _EPS:
        return 180.0
    cos = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def palm_length(points: np.ndarray) -> float:
    """Wrist -> middle-finger knuckle: the unit of length for every normalised feature."""
    return float(np.linalg.norm(points[Landmark.MIDDLE_MCP] - points[Landmark.WRIST]))


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def palm_frame(points: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """Orthonormal basis attached to the palm, as a 3x3 matrix whose *columns* are
    the axes expressed in the input coordinate system.

        y  wrist -> middle knuckle   ("up" along the hand)
        x  index knuckle -> pinky knuckle, made orthogonal to y
        z  palm normal pointing OUT of the palm (towards what the palm faces)

    The sign of z depends on the hand: for a left hand the cross product x*y points
    out of the back of the hand, so it is negated.  After this correction a left
    hand looks exactly like a mirrored right hand, which lets a single classifier
    serve both.
    """
    y = unit(points[Landmark.MIDDLE_MCP] - points[Landmark.WRIST])
    across = points[Landmark.PINKY_MCP] - points[Landmark.INDEX_MCP]
    x = unit(across - np.dot(across, y) * y)
    sign = 1.0 if handedness == "Right" else -1.0
    z = sign * np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def normalize_landmarks(points: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """Express the landmarks in the palm frame, in units of palm lengths.

    * translation invariance : the wrist becomes the origin
    * scale invariance       : divide by the palm length
    * rotation invariance    : rotate into the palm frame

    A hand in the same *shape* gives (almost) the same output wherever it is,
    however large it appears and however it is rotated - the ideal input for any
    classifier.
    """
    length = palm_length(points)
    if length < _EPS:
        raise ValueError("degenerate hand: wrist and middle knuckle coincide")
    basis = palm_frame(points, handedness)
    return ((points - points[Landmark.WRIST]) @ basis) / length


# --------------------------------------------------------------------------- #
# Finger-level features
# --------------------------------------------------------------------------- #
def finger_joint_angles(points: np.ndarray) -> np.ndarray:
    """Three joint angles per finger in degrees, shape (5, 3), 180 = straight.

    Columns are (proximal, middle, distal):
        fingers : (MCP, PIP, DIP)      thumb : (CMC, MCP, IP)
    The proximal angle is measured against the wrist, so it captures how far the
    finger is folded at the knuckle.
    """
    wrist = points[Landmark.WRIST]
    angles = np.zeros((5, 3))
    for f, (base, mid, dist, tip) in enumerate(FINGER_CHAINS):
        angles[f, 0] = joint_angle(wrist, points[base], points[mid])
        angles[f, 1] = joint_angle(points[base], points[mid], points[dist])
        angles[f, 2] = joint_angle(points[mid], points[dist], points[tip])
    return angles


def finger_extension(points: np.ndarray, angles: np.ndarray, cfg: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> np.ndarray:
    """How extended each finger is, 0 (fully curled) .. 1 (straight), shape (5,).

    For the four fingers this blends three joint angles with a "reach" ratio
    |tip - wrist| / |knuckle - wrist|.  Angles are rotation invariant and precise
    when the finger is seen from the side; reach is more forgiving when the finger
    points at the camera and the angles become noisy.  Blending the two is more
    stable than either alone.

    The thumb is anatomically different - it barely "curls", it folds across the
    palm - so it uses its IP joint plus the distance from the tip to the index
    knuckle instead.
    """
    wrist = points[Landmark.WRIST]
    length = max(palm_length(points), _EPS)
    ext = np.zeros(5)

    for f in range(1, 5):
        base, _, _, tip = FINGER_CHAINS[f]
        e_mcp = ramp(angles[f, 0], cfg.mcp_curled_deg, cfg.mcp_straight_deg)
        e_pip = ramp(angles[f, 1], cfg.pip_curled_deg, cfg.pip_straight_deg)
        e_dip = ramp(angles[f, 2], cfg.dip_curled_deg, cfg.dip_straight_deg)
        reach = np.linalg.norm(points[tip] - wrist) / max(np.linalg.norm(points[base] - wrist), _EPS)
        e_reach = ramp(reach, cfg.reach_curled, cfg.reach_straight)
        total = cfg.w_mcp + cfg.w_pip + cfg.w_dip + cfg.w_reach
        ext[f] = (cfg.w_mcp * e_mcp + cfg.w_pip * e_pip + cfg.w_dip * e_dip + cfg.w_reach * e_reach) / total

    e_ip = ramp(angles[0, 2], cfg.thumb_ip_curled_deg, cfg.thumb_ip_straight_deg)
    tip_to_index = np.linalg.norm(points[Landmark.THUMB_TIP] - points[Landmark.INDEX_MCP]) / length
    e_out = ramp(tip_to_index, cfg.thumb_tip_folded, cfg.thumb_tip_extended)
    ext[0] = 0.4 * e_ip + 0.6 * e_out
    return ext


def fingertip_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Pairwise fingertip distances in palm lengths, shape (5, 5)."""
    tips = points[list(FINGERTIPS)]
    diff = tips[:, None, :] - tips[None, :, :]
    return np.linalg.norm(diff, axis=-1) / max(palm_length(points), _EPS)


def finger_spread(norm_points: np.ndarray) -> float:
    """How far the four fingers fan out sideways, in degrees (mean angle between
    neighbouring fingers).  ~0-8 deg with fingers pressed together, 15-25 deg splayed.

    Measured on the *proximal phalanges* (knuckle -> first joint) in the palm frame
    as the angle ``atan2(x, y)`` from the palm's long axis.  Bending a finger at the
    knuckle swings that bone towards the palm normal (z), not sideways, so this
    angle does not change when a finger flexes - unlike fingertip-to-fingertip
    distance, which changes whenever any finger bends.  That makes it a clean signal
    for the dynamic spread / close-fingers zoom gesture.
    """
    angles = []
    for base, pip, _, _ in FINGER_CHAINS[1:]:
        v = norm_points[pip] - norm_points[base]
        angles.append(np.degrees(np.arctan2(v[0], v[1])))
    return float((angles[3] - angles[0]) / 3.0)  # mean of the three adjacent differences


# --------------------------------------------------------------------------- #
# Palm orientation
# --------------------------------------------------------------------------- #
def palm_orientation(points: np.ndarray, handedness: str = "Right") -> tuple[np.ndarray, float, float, float]:
    """Return (normal, facing, yaw_deg, pitch_deg) for the palm.

    ``normal`` is the unit vector pointing out of the palm in camera axes
    (x right, y down, z away from the camera).  ``facing`` is +1 when the palm
    faces the camera squarely, 0 when edge-on, -1 when the back of the hand faces
    the camera.  Yaw/pitch describe how far the palm normal is turned sideways /
    up-down relative to the camera axis.
    """
    normal = palm_frame(points, handedness)[:, 2]
    facing = float(-normal[2])  # camera looks along +z, so facing the camera means normal.z < 0
    yaw = float(np.degrees(np.arctan2(normal[0], -normal[2])))
    pitch = float(np.degrees(np.arctan2(-normal[1], np.hypot(normal[0], normal[2]))))
    return normal, facing, yaw, pitch


def roll_angle_deg(image_iso: np.ndarray) -> float:
    """Wrist roll in the image plane: angle of the wrist -> middle-knuckle vector.
    0 = fingers pointing straight up on screen, positive = clockwise as displayed."""
    v = image_iso[Landmark.MIDDLE_MCP, :2] - image_iso[Landmark.WRIST, :2]
    return float(np.degrees(np.arctan2(v[0], -v[1])))


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HandFeatures:
    """Everything the classifiers need, computed once per frame."""

    # --- shape (from world landmarks, scale/rotation invariant) ---
    norm_points: np.ndarray  # (21, 3) palm frame, palm-length units
    joint_angles: np.ndarray  # (5, 3) degrees
    extension: np.ndarray  # (5,) 0 curled .. 1 straight   [thumb, index, middle, ring, pinky]
    fingertip_dists: np.ndarray  # (5, 5) palm lengths
    pinch_distance: float  # thumb tip <-> index tip, palm lengths
    spread: float  # finger fan-out, mean angle between neighbouring fingers (degrees)
    palm_normal: np.ndarray  # (3,) unit, out of the palm
    palm_facing: float  # +1 palm to camera .. -1 back of hand to camera
    palm_yaw_deg: float
    palm_pitch_deg: float

    # --- placement (from image landmarks) ---
    roll_deg: float  # wrist rotation in the image plane
    palm_center: np.ndarray  # (2,) normalised frame coords
    palm_center_iso: np.ndarray  # (2,) isotropic frame coords (units of frame height)
    palm_size: float  # palm length in the image, isotropic units
    index_tip: np.ndarray  # (2,) normalised frame coords
    pinch_point: np.ndarray  # (2,) midpoint of thumb tip and index tip, normalised
    two_finger_tip: np.ndarray  # (2,) midpoint of index and middle tips, normalised

    handedness: str
    score: float
    timestamp: float
    aspect: float = 4.0 / 3.0  # frame width / height (to convert normalised xy to isotropic)

    @property
    def four_finger_extension(self) -> np.ndarray:
        return self.extension[1:]

    @property
    def mean_extension(self) -> float:
        return float(self.extension[1:].mean())

    @property
    def min_extension(self) -> float:
        return float(self.extension[1:].min())

    def to_vector(self) -> np.ndarray:
        """Fixed-length numeric vector for the optional learned classifier."""
        return np.concatenate(
            [
                self.norm_points[1:].ravel(),  # 60 values (wrist is always the origin)
                self.extension,  # 5
                self.joint_angles.ravel() / 180.0,  # 15
                [self.pinch_distance, self.spread / 30.0],  # 2
            ]
        )


FEATURE_VECTOR_SIZE = 20 * 3 + 5 + 15 + 2


def extract_features(obs: HandObservation, cfg: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> HandFeatures:
    """Compute :class:`HandFeatures` for one observation.

    Raises:
        ValueError: if the landmarks are degenerate (NaN / zero-size hand).  The
            gesture engine treats that as "no hand in this frame".
    """
    world = obs.world
    if not np.all(np.isfinite(world)) or not np.all(np.isfinite(obs.landmarks)):
        raise ValueError("non-finite landmarks")

    norm = normalize_landmarks(world, obs.handedness)
    angles = finger_joint_angles(world)
    extension = finger_extension(world, angles, cfg)
    dists = fingertip_distance_matrix(world)
    normal, facing, yaw, pitch = palm_orientation(world, obs.handedness)

    iso = obs.image_iso
    image_palm = float(np.linalg.norm(iso[Landmark.MIDDLE_MCP] - iso[Landmark.WRIST]))
    if image_palm < _EPS:
        raise ValueError("degenerate hand: zero palm size in image")

    lm = obs.landmarks
    return HandFeatures(
        norm_points=norm,
        joint_angles=angles,
        extension=extension,
        fingertip_dists=dists,
        pinch_distance=float(dists[0, 1]),
        spread=finger_spread(norm),
        palm_normal=normal,
        palm_facing=facing,
        palm_yaw_deg=yaw,
        palm_pitch_deg=pitch,
        roll_deg=roll_angle_deg(iso),
        palm_center=lm[list(PALM_POINTS), :2].mean(axis=0),
        palm_center_iso=iso[list(PALM_POINTS), :2].mean(axis=0),
        palm_size=image_palm,
        index_tip=lm[Landmark.INDEX_TIP, :2].copy(),
        pinch_point=(lm[Landmark.THUMB_TIP, :2] + lm[Landmark.INDEX_TIP, :2]) / 2.0,
        two_finger_tip=(lm[Landmark.INDEX_TIP, :2] + lm[Landmark.MIDDLE_TIP, :2]) / 2.0,
        handedness=obs.handedness,
        score=obs.score,
        timestamp=obs.timestamp,
        aspect=obs.aspect,
    )
