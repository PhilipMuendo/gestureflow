"""Download the MediaPipe hand-landmark model ahead of time.

    python tools/download_model.py

GestureFlow downloads it automatically on first run; use this script to prepare an
offline machine or to see the download happen separately from the first launch.
(Run from the project root.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TrackingConfig  # noqa: E402
from tracking.hand_tracker import TrackerError, ensure_model  # noqa: E402


def main() -> int:
    cfg = TrackingConfig()
    try:
        path = ensure_model(cfg.model_path, cfg.model_url)
    except TrackerError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"model ready: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
