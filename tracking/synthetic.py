"""A parametric 21-landmark hand model.

Used for three things, none of which needs a webcam:

1. **Unit tests** - build a hand with known finger curls and check that the
   feature extractor and classifiers agree with the ground truth.
2. **Simulation mode** (``python app.py --simulate``) - a scripted hand drives the
   whole application, which is handy for demos, CI and screenshots.
3. **Training-pipeline tests** for the optional learned classifier.

The model is simple forward kinematics: every finger is a 3-bone chain whose
bones rotate about the knuckle axis, plus a thumb described by explicit yaw/pitch
angles per bone.  It is not anatomically perfect, but its proportions are close
to MediaPipe's world landmarks, which is what the thresholds were calibrated on.

Coordinate convention (identical to MediaPipe's world landmarks, mirrored/selfie
view): x right, y down, z away from the camera, wrist at the origin, fingers
pointing up (-y), palm facing the camera, **right hand** (thumb on the left).
A left hand is the mirror image (x negated).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tracking.types import NUM_LANDMARKS, HandObservation

# --- anatomy (metres) --------------------------------------------------------
_MCP = np.array(
    [
        [-0.030, -0.085, 0.0],  # index
        [-0.010, -0.090, 0.0],  # middle
        [0.010, -0.085, 0.0],  # ring
        [0.028, -0.072, 0.0],  # pinky
    ]
)
_BONES = np.array(  # proximal, middle, distal phalanx lengths
    [
        [0.043, 0.025, 0.020],
        [0.047, 0.028, 0.022],
        [0.043, 0.026, 0.021],
        [0.034, 0.019, 0.018],
    ]
)
_REST_SPLAY_DEG = np.array([-20.0, 0.0, 16.0, 34.0])  # per-finger splay when spread == 1
_FLEX_AT_FULL_CURL_DEG = np.array([85.0, 105.0, 75.0])  # MCP, PIP, DIP
_THUMB_CMC = np.array([-0.025, -0.025, -0.004])
_THUMB_BONES = np.array([0.040, 0.033, 0.027])

# Thumb pose = 3 yaw angles (in-plane, 0 = up, negative = away from the palm)
# followed by 3 pitch angles (towards the camera), degrees, per bone.
THUMB_EXTENDED = (-38.0, -32.0, -30.0, 8.0, 6.0, 4.0)
THUMB_FOLDED = (-20.0, 30.0, 75.0, 35.0, 55.0, 60.0)


@dataclass(frozen=True)
class HandPose:
    """Finger configuration of the synthetic hand."""

    curls: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.1)  # index..pinky, 0 straight, 1 fist
    spread: float = 0.2  # 0 fingers together .. 1 fully splayed
    thumb: tuple[float, float, float, float, float, float] = THUMB_EXTENDED

    def with_(self, **kwargs) -> "HandPose":
        return replace(self, **kwargs)

    def lerp(self, other: "HandPose", u: float) -> "HandPose":
        u = float(np.clip(u, 0.0, 1.0))
        mix = lambda a, b: tuple(float(x + (y - x) * u) for x, y in zip(a, b))  # noqa: E731
        return HandPose(
            curls=mix(self.curls, other.curls),  # type: ignore[arg-type]
            spread=self.spread + (other.spread - self.spread) * u,
            thumb=mix(self.thumb, other.thumb),  # type: ignore[arg-type]
        )


# Thumb that touches the index fingertip when the index is bent to curl 0.45.
# Found by a small numerical search so the geometry is physically consistent.
THUMB_PINCH = (-21.0, 10.8, 51.9, 29.4, 40.3, 68.6)

POSES: dict[str, HandPose] = {
    "open_palm": HandPose(curls=(0.03, 0.03, 0.03, 0.05), spread=0.35, thumb=THUMB_EXTENDED),
    "point": HandPose(curls=(0.02, 0.92, 0.95, 0.95), spread=0.1, thumb=THUMB_FOLDED),
    "pinch": HandPose(curls=(0.45, 0.12, 0.10, 0.10), spread=0.2, thumb=THUMB_PINCH),
    "fist": HandPose(curls=(0.95, 0.97, 0.97, 0.97), spread=0.0, thumb=THUMB_FOLDED),
    "two_fingers": HandPose(curls=(0.02, 0.02, 0.95, 0.95), spread=0.6, thumb=THUMB_FOLDED),
    "spread": HandPose(curls=(0.0, 0.0, 0.0, 0.0), spread=1.0, thumb=THUMB_EXTENDED),
    "together": HandPose(curls=(0.03, 0.03, 0.03, 0.05), spread=0.0, thumb=THUMB_EXTENDED),
}


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def _bone_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit vector for a bone: yaw rotates in the palm plane (0 = up, + = towards +x),
    pitch lifts it towards the camera (-z)."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])


def hand_points(pose: HandPose) -> np.ndarray:
    """(21, 3) landmark positions of a right hand in the canonical frame (metres)."""
    pts = np.zeros((NUM_LANDMARKS, 3))

    # thumb: landmarks 1..4
    yaws, pitches = pose.thumb[:3], pose.thumb[3:]
    pos = _THUMB_CMC.copy()
    pts[1] = pos
    for i in range(3):
        pos = pos + _THUMB_BONES[i] * _bone_direction(yaws[i], pitches[i])
        pts[2 + i] = pos

    # fingers: landmarks 5.. in groups of four (MCP, PIP, DIP, TIP)
    for f in range(4):
        curl = float(np.clip(pose.curls[f], 0.0, 1.0))
        phi = np.radians(_REST_SPLAY_DEG[f] * pose.spread)
        up = np.array([np.sin(phi), -np.cos(phi), 0.0])  # finger direction in the palm plane
        lateral = np.array([np.cos(phi), np.sin(phi), 0.0])  # knuckle (flexion) axis
        pos = _MCP[f].copy()
        base = 5 + 4 * f
        pts[base] = pos
        flex = 0.0
        for j in range(3):
            flex += np.radians(_FLEX_AT_FULL_CURL_DEG[j] * curl)
            # Rodrigues rotation of `up` about `lateral`, bending towards the camera (-z)
            direction = up * np.cos(flex) + np.cross(lateral, up) * np.sin(flex)
            pos = pos + _BONES[f, j] * direction
            pts[base + 1 + j] = pos
    return pts


def _rotation(roll_deg: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Roll about the camera axis (clockwise on screen), yaw about the vertical axis,
    pitch about the horizontal axis."""
    r, y, p = np.radians([roll_deg, yaw_deg, pitch_deg])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return rz @ ry @ rx


def make_observation(
    pose: HandPose | str,
    *,
    center: tuple[float, float] = (0.5, 0.55),
    palm_px: float = 0.22,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    handedness: str = "Right",
    aspect: float = 4.0 / 3.0,
    timestamp: float = 0.0,
    score: float = 0.98,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> HandObservation:
    """Render a synthetic hand as a :class:`HandObservation`.

    Args:
        pose: finger configuration or the name of a preset in :data:`POSES`.
        center: where the palm centre appears, normalised frame coordinates.
        palm_px: palm length (wrist->middle knuckle) in the image, as a fraction of
            frame height.  Smaller = farther from the camera.
        roll_deg: rotation about the camera axis (clockwise positive), pivoting at
            the wrist.  ``yaw_deg``/``pitch_deg`` tilt the palm out of the image plane.
        noise: relative landmark noise (0.01 ~ 1% of the palm length) applied to
            both world and image landmarks.
    """
    if isinstance(pose, str):
        pose = POSES[pose]
    pts = hand_points(pose)
    if handedness == "Left":
        pts = pts * np.array([-1.0, 1.0, 1.0])  # mirror image of the right hand
    pts = pts @ _rotation(roll_deg, yaw_deg, pitch_deg).T

    palm_len = float(np.linalg.norm(pts[9] - pts[0]))
    palm_center = pts[[0, 5, 9, 13, 17]].mean(axis=0)

    world = pts - palm_center  # origin at the hand centre, like MediaPipe
    scale = palm_px / palm_len  # metres -> frame-height units
    iso = (pts - palm_center) * scale
    iso[:, 0] += center[0] * aspect
    iso[:, 1] += center[1]

    if noise > 0.0:
        rng = rng or np.random.default_rng(0)
        world = world + rng.normal(0.0, noise * palm_len, world.shape)
        iso = iso + rng.normal(0.0, noise * palm_px, iso.shape)

    landmarks = iso / np.array([aspect, 1.0, aspect])  # back to normalised coordinates
    return HandObservation(landmarks, world, handedness, score, timestamp, aspect)
