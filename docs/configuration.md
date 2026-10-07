# Configuration reference

> Generated from `config.py` by `tools/gen_config_docs.py` - do not edit by hand.
> `pytest` fails if this file is out of date.

All tunable numbers live in `config.py` as **frozen dataclasses**. Nothing is mutable global
state: components receive the config they need through their constructors and the command line
builds a modified copy with `dataclasses.replace`.

Conventions:

* Motion distances are in **palm lengths** (3D distance from the wrist to the middle-finger
  knuckle) so thresholds do not depend on how far the hand is from the camera.
* Joint angles are in degrees; times in seconds; screen positions are normalised to 0-1.
* "Extension" is a 0-1 score per finger (0 = fully curled, 1 = straight), see
  [gesture-recognition.md](gesture-recognition.md).


## Camera - `CameraConfig`

Frame capture (`camera/capture.py`).

| Field | Default | Meaning |
|---|---|---|
| `index` | `0` | camera index passed to OpenCV |
| `width` | `640` | requested capture width (the camera may deliver something else) |
| `height` | `480` | requested capture height |
| `fps` | `30` | requested capture frame rate |
| `mirror` | `True` | selfie view: moving right on screen == moving right in reality |
| `fourcc` | `'MJPG'` | MJPG avoids the 5-15 FPS cap of raw YUV on many USB webcams |
| `read_timeout_s` | `0.5` | how long the app waits for a frame before reporting the camera as stalled |
| `reopen_after_failures` | `30` | consecutive failed reads before re-opening the device |

## Tracking - `TrackingConfig`

MediaPipe, landmark smoothing, low-light handling (`tracking/`).

| Field | Default | Meaning |
|---|---|---|
| `model_path` | `PROJECT_ROOT / 'models' / 'hand_landmarker.task'` | where the MediaPipe hand-landmark model is stored (downloaded on first run) |
| `model_url` | `HAND_MODEL_URL` | download location of the model |
| `max_hands` | `1` | MediaPipe re-runs its (expensive) palm detector on *every* frame while it is looking for a second hand, which roughly doubles inference time.  One hand is the fast default; two-hand tracking (dominant-hand selection) is opt-in: --max-hands 2, or the M key. |
| `min_detection_confidence` | `0.6` | palm-detector confidence needed to start tracking a hand |
| `min_presence_confidence` | `0.5` | landmark-model confidence that a hand is present |
| `min_tracking_confidence` | `0.5` | confidence below which tracking is dropped and detection re-runs |
| `dominant_hand` | `'Right'` | "Right", "Left" or "Any" - who controls the scene |
| `primary_stickiness` | `0.25` | max wrist jump (frame fraction) to keep the same primary hand |
| `smoothing_enabled` | `True` | One-Euro landmark smoothing (see tracking/filters.py).  Lower min_cutoff = steadier when still; higher beta = less lag when moving fast. |
| `image_min_cutoff` | `1.5` | One-Euro min cutoff (Hz) for image landmarks |
| `image_beta` | `8.0` | One-Euro speed coefficient for image landmarks |
| `world_min_cutoff` | `2.0` | One-Euro min cutoff (Hz) for world landmarks |
| `world_beta` | `30.0` | One-Euro speed coefficient for world landmarks (their units are metres) |
| `filter_reset_after_s` | `0.3` | forget filter history after this long without the hand |
| `low_light_luma` | `70.0` | mean luma (0-255) below which we warn + brighten |
| `enhance_low_light` | `True` | gamma-brighten dark frames before detection (display is unchanged) |
| `target_luma` | `110.0` | gamma correction aims for this mean luma |

## Feature calibration - `FeatureConfig`

How joint angles and distances map to finger extension (`gestures/features.py`).

Calibration of the geometric features (joint angles are in degrees).

| Field | Default | Meaning |
|---|---|---|
| `pip_curled_deg` | `85.0` | A finger joint is "fully curled" at *_curled_deg and "straight" at *_straight_deg. |
| `pip_straight_deg` | `165.0` | PIP joint angle (deg) counted as straight |
| `dip_curled_deg` | `115.0` | DIP joint angle counted as fully curled |
| `dip_straight_deg` | `168.0` | DIP joint angle counted as straight |
| `mcp_curled_deg` | `105.0` | knuckle (MCP) angle counted as fully curled |
| `mcp_straight_deg` | `172.0` | knuckle (MCP) angle counted as straight |
| `reach_curled` | `1.15` | \|tip - wrist\| / \|knuckle - wrist\| : ~1.0 when curled, ~1.9 when straight |
| `reach_straight` | `1.8` | reach ratio of a straight finger |
| `w_pip` | `0.35` | Weights of the three joint angles and the reach ratio in finger extension |
| `w_dip` | `0.15` | weight of the DIP angle in finger extension |
| `w_mcp` | `0.15` | weight of the knuckle angle in finger extension |
| `w_reach` | `0.35` | weight of the reach ratio in finger extension |
| `thumb_tip_folded` | `0.45` | Thumb: distance from tip to index knuckle (palm lengths) when folded / extended |
| `thumb_tip_extended` | `0.9` | tip-to-index-knuckle distance (palm lengths) when the thumb is out |
| `thumb_ip_curled_deg` | `120.0` | thumb IP angle counted as curled |
| `thumb_ip_straight_deg` | `172.0` | thumb IP angle counted as straight |

## Static gestures - `StaticGestureConfig`

Fuzzy membership ranges of the rule-based classifier (`gestures/static_gestures.py`).

| Field | Default | Meaning |
|---|---|---|
| `extended_lo` | `0.45` | extension (0..1) -> "extended" / "curled" membership ramps |
| `extended_hi` | `0.78` | extension at/above which a finger is fully 'extended' |
| `curled_hi` | `0.28` | at or below -> fully curled |
| `curled_lo` | `0.58` | at or above -> not curled |
| `pinch_closed` | `0.3` | <= this -> score 1 |
| `pinch_open` | `0.65` | >= this -> score 0 |
| `pinch_index_min_ext_lo` | `0.08` | index must not be fully curled into the palm |
| `pinch_index_min_ext_hi` | `0.3` | index extension at which the pinch factor reaches 1 |

## Dynamic gestures - `DynamicGestureConfig`

Multi-frame recognisers (`gestures/dynamic_gestures.py`).

| Field | Default | Meaning |
|---|---|---|
| `history_s` | `1.2` | seconds of hand samples kept for dynamic recognition |
| `swipe_start_speed` | `4.5` | palm lengths / s to begin tracking a stroke |
| `swipe_end_speed` | `2.0` | stroke is over when speed falls below this |
| `swipe_min_distance` | `2.0` | palm lengths of net travel |
| `swipe_max_duration_s` | `0.8` | a swipe stroke must finish within this time |
| `swipe_horizontal_ratio` | `2.2` | \|dx\| must exceed this * \|dy\| |
| `swipe_min_straightness` | `0.75` | net / path length |
| `swipe_cooldown_s` | `0.8` | refractory time after a swipe |
| `swipe_return_penalty` | `1.4` | opposite-direction swipe right after one needs this * distance |
| `swipe_return_window_s` | `1.0` | how long after a swipe the opposite direction counts as the 'return stroke' |
| `swipe_min_open_extension` | `0.55` | mean 4-finger extension required |
| `zoom_window_s` | `0.3` | spread / close fingers (zoom) |
| `zoom_enter_rate` | `20.0` | change of finger fan-out, degrees per second |
| `zoom_exit_rate` | `9.0` | spread rate (deg/s) below which zoom ends |
| `zoom_hold_s` | `0.25` | zoom label persists this long after the last qualifying motion |
| `zoom_max_extension_drift` | `0.1` | finger shape must be steady: only the spacing changes |
| `zoom_min_open_extension` | `0.75` | every finger nearly straight for the whole window |
| `rotate_window_s` | `0.35` | rotate while pinching (wrist roll in the image plane) |
| `rotate_enter_rate` | `35.0` | degrees / s |
| `rotate_exit_rate` | `12.0` | roll rate (deg/s) below which rotation ends |
| `rotate_min_total_deg` | `9.0` | minimum total roll within the window to count as rotation |
| `rotate_max_translation` | `1.0` | palm lengths over the window; more means "drag" |
| `rotate_hold_s` | `0.2` | rotate label persists this long after the last qualifying motion |
| `drag_deadzone` | `0.35` | palm lengths of pinch-point travel since the pinch began |
| `drag_enter_speed` | `0.6` | palm lengths / s |
| `drag_hold_s` | `0.25` | drag label persists this long after the last qualifying motion |

## Gesture engine - `GestureConfig`

Temporal smoothing and hysteresis (`gestures/gesture_engine.py`); nests the three groups above.

| Field | Default | Meaning |
|---|---|---|
| `features` | `FeatureConfig()` | feature calibration |
| `static` | `StaticGestureConfig()` | static-gesture memberships |
| `dynamic` | `DynamicGestureConfig()` | dynamic-gesture thresholds |
| `score_ema_alpha` | `0.5` | Temporal smoothing of per-gesture scores |
| `enter_threshold` | `0.6` | smoothed score a challenger needs |
| `exit_threshold` | `0.35` | incumbent is dropped when it falls below this |
| `switch_margin` | `0.08` | challenger must beat the incumbent by this |
| `confirm_time_s` | `0.09` | challenger must persist this long |
| `pinch_confirm_time_s` | `0.08` | long enough to ignore the thumb sweeping past while a fist opens/closes |
| `hand_lost_grace_s` | `0.3` | keep the last gesture alive through brief dropouts |
| `min_confidence` | `0.4` | below this the reported gesture is NONE |

## Interaction - `InteractionConfig`

What gestures *do* (`interaction/controller.py`) and the scene's limits.

| Field | Default | Meaning |
|---|---|---|
| `active_margin` | `0.1` | Camera region that maps to the full scene.  Using ~80% of the frame means you can reach the screen corners without leaving the camera's view. |
| `cursor_min_cutoff` | `1.2` | One-Euro min cutoff (Hz) of the on-screen cursor |
| `cursor_beta` | `6.0` | One-Euro speed coefficient of the on-screen cursor |
| `max_cursor_step` | `0.35` | clamp per-frame cursor jumps (noise / re-acquisition) |
| `pick_slop` | `1.45` | hit radius = object radius * slop |
| `grab_slop` | `2.2` | pinch may grab a little further away than hover |
| `select_dwell_s` | `0.3` | point at an object this long to select it |
| `fist_toggle_hold_s` | `0.55` | hold a fist this long to toggle standby |
| `fist_after_pinch_lockout_s` | `0.4` | a fist right after a pinch is a release, not a toggle |
| `palm_reset_hold_s` | `2.0` | hold a steady open palm this long to reset the scene |
| `palm_steady_speed` | `1.0` | palm lengths / s: moving faster cancels the reset timer |
| `paused_time_scale` | `0.12` | ambient scene speed while paused (open palm) |
| `scroll_step` | `0.9` | palm lengths of vertical two-finger travel per object switch |
| `zoom_gain` | `0.05` | scale multiplier = exp(gain * change in finger fan-out in degrees) |
| `rotate_gain` | `1.0` | multiplier on wrist-roll degrees applied to the object |
| `min_scale` | `0.45` | smallest allowed object scale |
| `max_scale` | `3.2` | largest allowed object scale |
| `min_camera_zoom` | `0.6` | smallest whole-scene zoom |
| `max_camera_zoom` | `2.0` | largest whole-scene zoom |
| `throw_velocity_gain` | `0.9` | multiplier on the pinch velocity when releasing (throwing) an object |
| `command_display_s` | `1.1` | how long the latest command stays in the readout |

## Display - `DisplayConfig`

Window and rendering (`visualization/`).

| Field | Default | Meaning |
|---|---|---|
| `window_name` | `'GestureFlow'` | OpenCV window title |
| `window_width` | `1280` | initial window size; the layout follows the live window size |
| `window_height` | `720` | initial window height (the window is resizable) |
| `max_render_width` | `1600` | render resolution cap; larger windows are scaled up by OpenCV |
| `show_panel` | `True` | show the side panel at start |
| `video_contrast` | `1.07` | gentle contrast lift for the camera feed (1.0 = untouched) |
| `transition_s` | `0.5` | scene transition duration |
| `scene_seed` | `7` | seed for the (deterministic) scene layouts |

## Application - `AppConfig`

Top level; nests all of the above plus run options (`app.py`).

| Field | Default | Meaning |
|---|---|---|
| `camera` | `CameraConfig()` | camera settings |
| `tracking` | `TrackingConfig()` | tracking settings |
| `gestures` | `GestureConfig()` | gesture settings |
| `interaction` | `InteractionConfig()` | interaction settings |
| `display` | `DisplayConfig()` | display settings |
| `classifier` | `'rules'` | "rules" \| "learned" \| "hybrid" |
| `learned_model_path` | `PROJECT_ROOT / 'models' / 'static_classifier.joblib'` | where the optional learned classifier is loaded from |
| `simulate` | `False` | drive the app with a scripted synthetic hand (no webcam) |
| `headless` | `False` | no window (used for smoke tests / CI) |
| `threaded` | `True` | track on a worker thread (overlaps inference with rendering) |
| `max_frames` | `0` | 0 = run until quit |
| `save_frame` | `''` | write the final composed frame to this path |
