# GestureFlow technical documentation

The top-level [README](../README.md) explains what GestureFlow is and how to run it. These
documents explain **how it works** and **how to change it**.

| Document | Read it to learn... |
|---|---|
| [architecture.md](architecture.md) | the layers, data types, threads, coordinate systems, error handling, module map |
| [gesture-recognition.md](gesture-recognition.md) | the math of the features, the static classifier, temporal smoothing, every dynamic recogniser |
| [interaction-and-scene.md](interaction-and-scene.md) | how gestures become commands, the controller state machine, scene physics |
| [ui-and-rendering.md](ui-and-rendering.md) | layout, design tokens, typography, the render pipeline, feedback cues, hit-testing |
| [dataset-and-training.md](dataset-and-training.md) | the recording format, the recorder, the trainer, the model bundle |
| [extending.md](extending.md) | recipes: add a gesture, a command, a scene, a classifier, a UI element |
| [testing.md](testing.md) | the test strategy, the synthetic hand model, helpers, what is *not* covered |
| [configuration.md](configuration.md) | every tunable, its default and meaning (generated from `config.py`) |

## Ten-second summary

```
Frame ─► HandObservation ─► HandFeatures ─► GestureState ─► Command ─► Scene ─► pixels
camera/    tracking/         gestures/        gestures/      interaction/   visualization/
```

* Each arrow is a small, immutable, typed hand-off; each stage can be tested without the others.
* **Time always comes from frame timestamps**, never the wall clock, so every stage is
  deterministic and unit-testable.
* **Every threshold is a field in `config.py`** and every motion distance is in *palm lengths*,
  so behaviour does not depend on how far the hand is from the camera.
* A **synthetic hand model** (`tracking/synthetic.py`) generates landmarks with known ground
  truth. It powers the tests, `--simulate`, and the training-pipeline tests.
