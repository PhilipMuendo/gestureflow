"""Central configuration for GestureFlow.

Every tunable number in the project lives here, grouped into small frozen
dataclasses.  Nothing in this module is mutable global state: components receive
the config they need through their constructors, and the command line builds a
modified copy with :func:`dataclasses.replace`.

Distances that describe *motion* are expressed in "palm lengths" (the 3D
distance from the wrist to the middle-finger knuckle).  That makes thresholds
independent of how far the hand is from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# --------------------------------------------------------------------------- #
# Camera / tracking
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraConfig:
    index: int = 0  # camera index passed to OpenCV
    width: int = 640  # requested capture width (the camera may deliver something else)
    height: int = 480  # requested capture height
    fps: int = 30  # requested capture frame rate
    mirror: bool = True  # selfie view: moving right on screen == moving right in reality
    fourcc: str = "MJPG"  # MJPG avoids the 5-15 FPS cap of raw YUV on many USB webcams
    read_timeout_s: float = 0.5  # how long the app waits for a frame before reporting the camera as stalled
    reopen_after_failures: int = 30  # consecutive failed reads before re-opening the device


@dataclass(frozen=True)
class TrackingConfig:
    model_path: Path = PROJECT_ROOT / "models" / "hand_landmarker.task"  # where the MediaPipe hand-landmark model is stored (downloaded on first run)
    model_url: str = HAND_MODEL_URL  # download location of the model
    # MediaPipe re-runs its (expensive) palm detector on *every* frame while it is looking for
    # a second hand, which roughly doubles inference time.  One hand is the fast default;
    # two-hand tracking (dominant-hand selection) is opt-in: --max-hands 2, or the M key.
    max_hands: int = 1
    min_detection_confidence: float = 0.6  # palm-detector confidence needed to start tracking a hand
    min_presence_confidence: float = 0.5  # landmark-model confidence that a hand is present
    min_tracking_confidence: float = 0.5  # confidence below which tracking is dropped and detection re-runs
    dominant_hand: str = "Right"  # "Right", "Left" or "Any" - who controls the scene
    primary_stickiness: float = 0.25  # max wrist jump (frame fraction) to keep the same primary hand

    # One-Euro landmark smoothing (see tracking/filters.py).  Lower min_cutoff =
    # steadier when still; higher beta = less lag when moving fast.
    smoothing_enabled: bool = True
    image_min_cutoff: float = 1.5  # One-Euro min cutoff (Hz) for image landmarks
    image_beta: float = 8.0  # One-Euro speed coefficient for image landmarks
    world_min_cutoff: float = 2.0  # One-Euro min cutoff (Hz) for world landmarks
    world_beta: float = 30.0  # One-Euro speed coefficient for world landmarks (their units are metres)
    filter_reset_after_s: float = 0.3  # forget filter history after this long without the hand

    # Low-light handling
    low_light_luma: float = 70.0  # mean luma (0-255) below which we warn + brighten
    enhance_low_light: bool = True  # gamma-brighten dark frames before detection (display is unchanged)
    target_luma: float = 110.0  # gamma correction aims for this mean luma


# --------------------------------------------------------------------------- #
# Gesture recognition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureConfig:
    """Calibration of the geometric features (joint angles are in degrees)."""

    # A finger joint is "fully curled" at *_curled_deg and "straight" at *_straight_deg.
    pip_curled_deg: float = 85.0
    pip_straight_deg: float = 165.0  # PIP joint angle (deg) counted as straight
    dip_curled_deg: float = 115.0  # DIP joint angle counted as fully curled
    dip_straight_deg: float = 168.0  # DIP joint angle counted as straight
    mcp_curled_deg: float = 105.0  # knuckle (MCP) angle counted as fully curled
    mcp_straight_deg: float = 172.0  # knuckle (MCP) angle counted as straight
    # |tip - wrist| / |knuckle - wrist| : ~1.0 when curled, ~1.9 when straight
    reach_curled: float = 1.15
    reach_straight: float = 1.80  # reach ratio of a straight finger
    # Weights of the three joint angles and the reach ratio in finger extension
    w_pip: float = 0.35
    w_dip: float = 0.15  # weight of the DIP angle in finger extension
    w_mcp: float = 0.15  # weight of the knuckle angle in finger extension
    w_reach: float = 0.35  # weight of the reach ratio in finger extension
    # Thumb: distance from tip to index knuckle (palm lengths) when folded / extended
    thumb_tip_folded: float = 0.45
    thumb_tip_extended: float = 0.90  # tip-to-index-knuckle distance (palm lengths) when the thumb is out
    thumb_ip_curled_deg: float = 120.0  # thumb IP angle counted as curled
    thumb_ip_straight_deg: float = 172.0  # thumb IP angle counted as straight


@dataclass(frozen=True)
class StaticGestureConfig:
    # extension (0..1) -> "extended" / "curled" membership ramps
    extended_lo: float = 0.45
    extended_hi: float = 0.78  # extension at/above which a finger is fully 'extended'
    curled_hi: float = 0.28  # at or below -> fully curled
    curled_lo: float = 0.58  # at or above -> not curled
    # pinch: normalised thumb-tip <-> index-tip distance (palm lengths)
    pinch_closed: float = 0.30  # <= this -> score 1
    pinch_open: float = 0.65  # >= this -> score 0
    pinch_index_min_ext_lo: float = 0.08  # index must not be fully curled into the palm
    pinch_index_min_ext_hi: float = 0.30  # index extension at which the pinch factor reaches 1


@dataclass(frozen=True)
class DynamicGestureConfig:
    history_s: float = 1.2  # seconds of hand samples kept for dynamic recognition

    # swipe: palm travels fast and far, mostly horizontally, with an open hand
    swipe_start_speed: float = 4.5  # palm lengths / s to begin tracking a stroke
    swipe_end_speed: float = 2.0  # stroke is over when speed falls below this
    swipe_min_distance: float = 2.0  # palm lengths of net travel
    swipe_max_duration_s: float = 0.8  # a swipe stroke must finish within this time
    swipe_horizontal_ratio: float = 2.2  # |dx| must exceed this * |dy|
    swipe_min_straightness: float = 0.75  # net / path length
    swipe_cooldown_s: float = 0.8  # refractory time after a swipe
    swipe_return_penalty: float = 1.4  # opposite-direction swipe right after one needs this * distance
    swipe_return_window_s: float = 1.0  # how long after a swipe the opposite direction counts as the 'return stroke'
    swipe_min_open_extension: float = 0.55  # mean 4-finger extension required

    # spread / close fingers (zoom)
    zoom_window_s: float = 0.30
    zoom_enter_rate: float = 20.0  # change of finger fan-out, degrees per second
    zoom_exit_rate: float = 9.0  # spread rate (deg/s) below which zoom ends
    zoom_hold_s: float = 0.25  # zoom label persists this long after the last qualifying motion
    zoom_max_extension_drift: float = 0.10  # finger shape must be steady: only the spacing changes
    zoom_min_open_extension: float = 0.75  # every finger nearly straight for the whole window

    # rotate while pinching (wrist roll in the image plane)
    rotate_window_s: float = 0.35
    rotate_enter_rate: float = 35.0  # degrees / s
    rotate_exit_rate: float = 12.0  # roll rate (deg/s) below which rotation ends
    rotate_min_total_deg: float = 9.0  # minimum total roll within the window to count as rotation
    rotate_max_translation: float = 1.0  # palm lengths over the window; more means "drag"
    rotate_hold_s: float = 0.20  # rotate label persists this long after the last qualifying motion

    # drag while pinching
    drag_deadzone: float = 0.35  # palm lengths of pinch-point travel since the pinch began
    drag_enter_speed: float = 0.6  # palm lengths / s
    drag_hold_s: float = 0.25  # drag label persists this long after the last qualifying motion


@dataclass(frozen=True)
class GestureConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)  # feature calibration
    static: StaticGestureConfig = field(default_factory=StaticGestureConfig)  # static-gesture memberships
    dynamic: DynamicGestureConfig = field(default_factory=DynamicGestureConfig)  # dynamic-gesture thresholds

    # Temporal smoothing of per-gesture scores
    score_ema_alpha: float = 0.5
    enter_threshold: float = 0.60  # smoothed score a challenger needs
    exit_threshold: float = 0.35  # incumbent is dropped when it falls below this
    switch_margin: float = 0.08  # challenger must beat the incumbent by this
    confirm_time_s: float = 0.09  # challenger must persist this long
    pinch_confirm_time_s: float = 0.08  # long enough to ignore the thumb sweeping past while a fist opens/closes
    hand_lost_grace_s: float = 0.30  # keep the last gesture alive through brief dropouts
    min_confidence: float = 0.40  # below this the reported gesture is NONE


# --------------------------------------------------------------------------- #
# Interaction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InteractionConfig:
    # Camera region that maps to the full scene.  Using ~80% of the frame means
    # you can reach the screen corners without leaving the camera's view.
    active_margin: float = 0.10
    cursor_min_cutoff: float = 1.2  # One-Euro min cutoff (Hz) of the on-screen cursor
    cursor_beta: float = 6.0  # One-Euro speed coefficient of the on-screen cursor
    max_cursor_step: float = 0.35  # clamp per-frame cursor jumps (noise / re-acquisition)

    pick_slop: float = 1.45  # hit radius = object radius * slop
    grab_slop: float = 2.2  # pinch may grab a little further away than hover
    select_dwell_s: float = 0.30  # point at an object this long to select it

    fist_toggle_hold_s: float = 0.55  # hold a fist this long to toggle standby
    fist_after_pinch_lockout_s: float = 0.4  # a fist right after a pinch is a release, not a toggle
    palm_reset_hold_s: float = 2.0  # hold a steady open palm this long to reset the scene
    palm_steady_speed: float = 1.0  # palm lengths / s: moving faster cancels the reset timer
    paused_time_scale: float = 0.12  # ambient scene speed while paused (open palm)

    scroll_step: float = 0.9  # palm lengths of vertical two-finger travel per object switch
    zoom_gain: float = 0.05  # scale multiplier = exp(gain * change in finger fan-out in degrees)
    rotate_gain: float = 1.0  # multiplier on wrist-roll degrees applied to the object
    min_scale: float = 0.45  # smallest allowed object scale
    max_scale: float = 3.2  # largest allowed object scale
    min_camera_zoom: float = 0.6  # smallest whole-scene zoom
    max_camera_zoom: float = 2.0  # largest whole-scene zoom
    throw_velocity_gain: float = 0.9  # multiplier on the pinch velocity when releasing (throwing) an object
    command_display_s: float = 1.1  # how long the latest command stays in the readout


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DisplayConfig:
    window_name: str = "GestureFlow"  # OpenCV window title
    window_width: int = 1280  # initial window size; the layout follows the live window size
    window_height: int = 720  # initial window height (the window is resizable)
    max_render_width: int = 1600  # render resolution cap; larger windows are scaled up by OpenCV
    show_panel: bool = True  # show the side panel at start
    video_contrast: float = 1.07  # gentle contrast lift for the camera feed (1.0 = untouched)
    transition_s: float = 0.5  # scene transition duration
    scene_seed: int = 7  # seed for the (deterministic) scene layouts


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)  # camera settings
    tracking: TrackingConfig = field(default_factory=TrackingConfig)  # tracking settings
    gestures: GestureConfig = field(default_factory=GestureConfig)  # gesture settings
    interaction: InteractionConfig = field(default_factory=InteractionConfig)  # interaction settings
    display: DisplayConfig = field(default_factory=DisplayConfig)  # display settings

    classifier: str = "rules"  # "rules" | "learned" | "hybrid"
    learned_model_path: Path = PROJECT_ROOT / "models" / "static_classifier.joblib"  # where the optional learned classifier is loaded from
    simulate: bool = False  # drive the app with a scripted synthetic hand (no webcam)
    headless: bool = False  # no window (used for smoke tests / CI)
    threaded: bool = True  # track on a worker thread (overlaps inference with rendering)
    max_frames: int = 0  # 0 = run until quit
    save_frame: str = ""  # write the final composed frame to this path
