# Extending GestureFlow

Recipes for the common changes. The layering rules in [architecture.md](architecture.md) tell you
where each change belongs; the tests listed with each recipe are the safety net.

## Add a configuration value

1. Add a field (with a trailing `# comment`) to the right dataclass in `config.py`.
2. `python tools/gen_config_docs.py` - regenerates [configuration.md](configuration.md). `pytest` fails if
   you forget (`tests/test_docs.py`).
3. Use it through the constructor-injected config object; never read module-level state.

## Add a static gesture (e.g. "thumbs up")

1. **Vocabulary** (`gestures/types.py`): add the member to `Gesture`, to `STATIC_GESTURES`, to
   `LEGEND_ORDER` and `GESTURE_INFO` (label + one-line action).
2. **Score** (`gestures/static_gestures.py`): add a fuzzy rule to `RuleBasedClassifier.classify` built from
   `extended()`, `curled()` and the other features (use `ramp` for new ones). Return it in the dict - the
   engine loops over `STATIC_GESTURES`, so smoothing/hysteresis come for free. If it can be confused with an
   existing pose, add a suppression term like pinch has.
3. **Meaning** (`interaction/controller.py`): add a branch to `update()` that emits commands.
4. **UI** (`visualization/ui_state.py`): every gesture must belong to exactly one `GestureGroup`
   (`tests/test_ui_layout.py` checks this); add an icon in `visualization/icons.py` if it needs a new row.
   Optionally add its landmarks to `highlighted_landmarks` in `visualization/renderer.py`.
5. **Synthetic pose** (`tracking/synthetic.py`): add a `POSES` preset so it can be tested; extend
   `PRESET_TO_GESTURE` in `tests/test_gestures.py` (the parametrised tests then check invariance to distance,
   rotation, side and noise automatically).
6. **Learned model**: `VALID_LABELS` derives from `STATIC_GESTURES`; re-record and re-train.

## Add a dynamic gesture

1. Add the `Gesture` member to `DYNAMIC_GESTURES` (+ info, legend, UI group).
2. Implement the recogniser in `gestures/dynamic_gestures.py` as another `_update_*` step fed by the sample
   history (positions in palm lengths, time from the caller). Report it through `DynamicResult`
   (`active` for a continuous gesture, `events` for a one-shot). Think about *false positives first*: which
   transitions between ordinary poses could produce the same signal? (The zoom recogniser's requirement of a
   steady finger shape exists because of exactly that.)
3. The engine already masks disabled dynamic gestures and picks `events > active dynamic > static`.
4. Controller branch, as above.
5. Tests: a positive synthetic trajectory, and **negatives** - slow motion, the wrong axis, noise, and the
   transitions between neighbouring poses (see `test_gestures.py`). Run the full demo
   (`tests/test_simulation.py`) to see that nothing fires spuriously elsewhere.

## Add a command

1. `CommandType` member (`interaction/commands.py`) and, if needed, a field on `Command`.
2. Handle it in `Scene.apply` (`visualization/scene.py`) or `SceneManager.apply` if it spans scenes.
3. Emit it from the controller; surface it in the UI (toast in `GestureFlowApp._apply`, readout via the
   command `label`).
4. Test the scene effect in `test_scene.py` and the emission in `test_interaction.py`.

## Add a shape or a scene

* **Shape**: add a `Mesh(vertices, edges)` to `MESHES` in `visualization/shapes.py` (vertices inside the unit
  sphere, edges as index pairs). `test_meshes_are_well_formed` validates it.
* **Scene**: append a `ScenePreset(name, shapes, tint, background)` to `SCENE_PRESETS`. Layouts are generated
  from the seed; the pager, scene selector and swipes pick it up automatically.

## Add a classifier

Implement `StaticClassifier` (`classify(features) -> dict[Gesture, float]`, scores in [0, 1], one entry per
static gesture) and pass it to `GestureEngine(cfg, classifier)`. To expose it on the command line, extend
`build_classifier` and the `--classifier` choices in `app.py`.

## Add a UI element

* **On the canvas**: draw it in `Hud.draw_view` (or the renderer for things tied to scene objects), using
  `layout.u()` for sizes, `theme` colours and `typeface.text`. Ask: does it communicate tracking,
  interaction or state? If not, do not add it.
* **In the side panel**: draw it in `Hud._render_sidebar`, append a `Hitbox` with a new action tuple, handle
  the action in `GestureFlowApp._on_mouse`, and add whatever it depends on to the cache key in
  `Hud.draw_sidebar` (otherwise the cached image will not update).
* Check it at several window sizes (`tests/test_ui_layout.py` renders ten) and regenerate the screenshots with
  `python tools/make_screenshots.py`.

## Tune a threshold

1. Run `python app.py --debug` (or press `D`). Watch the finger-extension bars, pinch distance, spread and the
   five classifier scores while you perform the gesture and while you perform its likely confusers.
2. Change the field in `config.py` (see [configuration.md](configuration.md) for what it means). Prefer the
   smallest change; thresholds interact (e.g. `enter_threshold` / `exit_threshold` / `confirm_time_s`).
3. Re-run `pytest`; the synthetic tests encode the intended behaviour, and the simulation test guards against
   spurious actions. If you changed behaviour deliberately, update the test, not just the number.
4. Better still: record your own data and train the learned classifier.

## Run on a new platform

Only the camera backend choice is platform-specific (`camera/capture.py: _backends`). On Windows
Media Foundation is preferred (DirectShow caps many webcams at ~15 FPS in OpenCV) and the
`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` environment variable - which must be set **before the first
`import cv2`** (hence `camera/__init__.py` and the top of `app.py`) - avoids a ~8 s start-up.
