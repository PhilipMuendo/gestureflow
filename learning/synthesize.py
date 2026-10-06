"""Generate a synthetic labelled dataset from the parametric hand model.

Useful for trying the training pipeline before recording real data, and for the tests::

    python -m learning.synthesize --out data/synthetic.jsonl --per-class 300

Each class is sampled with randomised finger curl, spread, hand position, size, roll, tilt,
handedness and landmark noise, so a classifier trained on it has to be genuinely
rotation / scale / side invariant to score well.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from learning.dataset import FrameRecord, Sample, write_samples
from tracking.synthetic import POSES, HandPose, make_observation

CLASS_TO_PRESET = {
    "open_palm": "open_palm",
    "point": "point",
    "pinch": "pinch",
    "fist": "fist",
    "two_fingers": "two_fingers",
}


def _jitter(pose: HandPose, rng: np.random.Generator, amount: float) -> HandPose:
    curls = tuple(float(np.clip(c + rng.normal(0, amount), 0.0, 1.0)) for c in pose.curls)
    thumb = tuple(float(v + rng.normal(0, 6.0 * amount / 0.08)) for v in pose.thumb)
    return HandPose(curls=curls, spread=float(np.clip(pose.spread + rng.normal(0, 0.15), 0, 1)), thumb=thumb)  # type: ignore[arg-type]


def _random_pose(rng: np.random.Generator) -> HandPose:
    """A random half-open hand, labelled 'none' (the catch-all class)."""
    curls = tuple(float(c) for c in rng.uniform(0.3, 0.7, 4))
    return HandPose(curls=curls, spread=float(rng.uniform(0, 1)), thumb=POSES["open_palm"].thumb)  # type: ignore[arg-type]


def generate(per_class: int = 200, noise: float = 0.015, seed: int = 0, session: str = "synthetic") -> list[Sample]:
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    labels = list(CLASS_TO_PRESET) + ["none"]
    for label in labels:
        for i in range(per_class):
            pose = _random_pose(rng) if label == "none" else _jitter(POSES[CLASS_TO_PRESET[label]], rng, 0.06)
            hand = "Right" if rng.random() < 0.5 else "Left"
            obs = make_observation(
                pose,
                center=(float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.25, 0.75))),
                palm_px=float(rng.uniform(0.1, 0.35)),
                roll_deg=float(rng.uniform(-90, 90)),
                yaw_deg=float(rng.normal(0, 12)),
                pitch_deg=float(rng.normal(0, 12)),
                handedness=hand,
                noise=noise,
                rng=rng,
            )
            samples.append(
                Sample(label, (FrameRecord(0.0, obs.landmarks, obs.world),), hand, "static", f"{session}-{i // 40}", obs.aspect)
            )
    return samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/synthetic.jsonl"))
    p.add_argument("--per-class", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.015)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    n = write_samples(args.out, generate(args.per_class, args.noise, args.seed), append=False)
    print(f"wrote {n} samples to {args.out}")


if __name__ == "__main__":
    main()
