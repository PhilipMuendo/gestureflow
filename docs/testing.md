# Testing

```bash
pip install -r requirements-dev.txt
pytest                         # the whole suite; none of it needs a webcam
pytest tests/test_gestures.py  # one file
pytest -k "swipe or zoom"      # by keyword
ruff check .                   # lint (the codebase is clean)
```

`pytest.ini` puts the project root on the path (the packages are top-level) and runs quietly. The suite
takes a few minutes on a busy machine, mostly the learning tests (they train real models) and the
render-at-many-sizes tests.

## 1. Strategy

**Hardware is replaced by ground truth.** Every test builds landmarks with the **synthetic hand model**
(`tracking/synthetic.py`) and feeds them through the real code with explicit timestamps. The model gives
known finger curls, spread, roll, distance and noise, so "this should be an open palm" is a statement of
fact, not a guess about a camera. Nothing reads the wall clock, so tests are deterministic.

### The synthetic hand model

* `HandPose(curls=(index, middle, ring, pinky), spread, thumb)` - finger curls 0-1, spread 0-1, and the
  thumb as three yaw and three pitch angles per bone.
* `hand_points(pose)` - forward kinematics of a right hand in MediaPipe world axes (x right, y down, z
  away, wrist at the origin, palm to the camera); each finger is a three-bone chain bending towards the
  camera. Proportions are approximate human ones (metres).
* `POSES` - presets: `open_palm`, `point`, `pinch`, `fist`, `two_fingers`, `spread`, `together`. The pinch
  thumb was found by a numerical search so the tips genuinely touch.
* `make_observation(pose, center, palm_px, roll_deg, yaw_deg, pitch_deg, handedness, aspect, timestamp,
  noise, rng)` renders it as a `HandObservation`: rotation about the wrist, mirroring for a left hand,
  orthographic projection to image coordinates at a chosen apparent size, optional Gaussian noise.
* `HandPose.lerp` morphs between poses; `tracking/simulated.py` chains such morphs into scripted
  performances (`GestureScript`), the basis of `--simulate` and the end-to-end test.

## 2. What is tested

| File | Covers |
|---|---|
| `test_features.py` | math helpers; normalisation invariances (translation, scale, rotation, handedness); finger extension (monotonic in curl, stable under noise); pinch distance; spread incl. flexion invariance; palm orientation; roll; degenerate input |
| `test_gestures.py` | classification of every gesture under distance / rotation / side / noise; hysteresis, flicker, dropouts, disabled gestures; swipe (slow, vertical, jitter, return stroke, distance independence), zoom (incl. fist -> palm), rotate, drag, scroll |
| `test_interaction.py` | the controller against a fake scene: select dwell, grab / drag / throw / rotate, pause and reset, standby, two-finger cycling, swipes, zoom, hand loss |
| `test_scene.py` | physics and bounds, command handling and clamping, reset, pause, scene transitions, aspect changes, absence of decorative effects |
| `test_ui_layout.py` | responsive layout at many sizes, canvas share, panel collapse, per-axis camera->view mapping, typography, icons, landmark highlighting, gesture families, the quiet empty state |
| `test_learning.py` | dataset round-trip and errors, training of every model kind, the honest split, model-loading errors, hybrid scores, the recorder |
| `test_pipeline_and_app.py` | the worker-thread pipeline (pairing, dropping, stalls, crashes), camera and model-download failure paths, CLI -> config, graceful exits |
| `test_simulation.py` | the whole demo through engine -> controller -> scene -> renderer; every view renders; mouse and keyboard; screenshots |
| `test_filters.py` | One-Euro filter, gamma LUT, primary-hand selection |
| `test_docs.py` | the generated config reference is current; links and anchors resolve; identifiers named in the docs still exist |

## 3. Helpers (`tests/helpers.py`, `tests/conftest.py`)

| Helper | Purpose |
|---|---|
| `run(engine, duration, pose, center=…, roll=…, noise=…, present=…)` | feed a synthetic timeline at 30 FPS into a `GestureEngine`; `pose`, `center`, `roll` may be functions of the local time `u` |
| `aim_center(pose, scene_xy, aim, roll)` | the palm centre that puts the index tip / pinch point on a scene position |
| `FakeScene` | a minimal `SceneQuery` that *applies* selection / move commands (so tests model the real feedback loop) |
| `drive(engine, controller, states)` | feed engine states through the controller, applying its commands to the scene as the app does |
| `smoothstep`, `lerp_pose` | build natural-looking motion and pose morphs |
| fixtures `engine`, `rng` | a default `GestureEngine`; a seeded generator |

## 4. Writing a good gesture test

* State the **negative** cases as carefully as the positive ones: a recogniser is only trustworthy if the
  neighbouring gestures and the transitions between them do *not* trigger it.
* Assert on **behaviour at the interface** (events, commands, state) rather than on internal scores, so
  retuning a threshold does not rewrite the tests.
* Make the motion **physically plausible**: use `smoothstep` profiles, morph poses over realistic durations,
  and add `noise=` so the test would catch a recogniser that only works on perfect data.
* If a test fails after a threshold change, decide whether the *behaviour* or the *test* is wrong, and say
  which in the commit.

## 5. What the tests cannot cover

* **MediaPipe on a live hand**, and the OS window itself, need hardware. They are exercised manually; the
  real-camera pipeline has been run end-to-end (capture -> MediaPipe -> engine -> render) and the static
  classifier checked on real MediaPipe landmarks from sample photographs.
* **Dynamic-gesture thresholds** are validated on synthetic trajectories, not on a population of users.
* **Absolute performance.** Timings depend on the machine; see the measuring advice in
  [architecture.md](architecture.md#7-performance-architecture). No test asserts a frame rate.
* **Visual quality.** Tests assert that every view renders a valid, non-empty canvas of the right size and
  that specific elements stay small; the *look* is reviewed from screenshots
  (`python tools/make_screenshots.py`).
