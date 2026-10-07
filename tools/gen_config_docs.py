"""Generate docs/configuration.md from config.py.

    python tools/gen_config_docs.py            # rewrite docs/configuration.md
    python tools/gen_config_docs.py --check    # exit 1 if the file is out of date (used by the tests)

The reference is derived from the dataclass fields, their defaults and the comments next to
them, so it cannot drift from the code.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.py"
OUTPUT = ROOT / "docs" / "configuration.md"

INTRO = """# Configuration reference

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
"""

SECTIONS = {
    "CameraConfig": ("Camera", "Frame capture (`camera/capture.py`)."),
    "TrackingConfig": ("Tracking", "MediaPipe, landmark smoothing, low-light handling (`tracking/`)."),
    "FeatureConfig": ("Feature calibration", "How joint angles and distances map to finger extension (`gestures/features.py`)."),
    "StaticGestureConfig": ("Static gestures", "Fuzzy membership ranges of the rule-based classifier (`gestures/static_gestures.py`)."),
    "DynamicGestureConfig": ("Dynamic gestures", "Multi-frame recognisers (`gestures/dynamic_gestures.py`)."),
    "GestureConfig": ("Gesture engine", "Temporal smoothing and hysteresis (`gestures/gesture_engine.py`); nests the three groups above."),
    "InteractionConfig": ("Interaction", "What gestures *do* (`interaction/controller.py`) and the scene's limits."),
    "DisplayConfig": ("Display", "Window and rendering (`visualization/`)."),
    "AppConfig": ("Application", "Top level; nests all of the above plus run options (`app.py`)."),
}


def _comment_for(lines: list[str], lineno: int) -> str:
    """Trailing comment on the field's line, else the contiguous comment block above it."""
    line = lines[lineno - 1]
    trailing = re.search(r"\s#\s?(.*)$", line)
    if trailing:
        return trailing.group(1).strip()
    block: list[str] = []
    i = lineno - 2
    while i >= 0 and lines[i].strip().startswith("#"):
        text = lines[i].strip().lstrip("#").strip()
        if not set(text) <= {"-", " "}:  # skip ---- separators
            block.insert(0, text)
        i -= 1
    return " ".join(block)


def _default(node: ast.expr | None) -> str:
    if node is None:
        return "-"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "field":
        for kw in node.keywords:
            if kw.arg == "default_factory":
                return f"`{ast.unparse(kw.value)}()`"
    return f"`{ast.unparse(node)}`"


def render() -> str:
    source = CONFIG.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    out = [INTRO]
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in SECTIONS):
        title, blurb = SECTIONS[cls.name]
        out.append(f"\n## {title} - `{cls.name}`\n\n{blurb}\n")
        doc = ast.get_docstring(cls)
        if doc:
            out.append(doc + "\n")
        out.append("| Field | Default | Meaning |\n|---|---|---|")
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                comment = _comment_for(lines, node.lineno).replace("|", "\\|")
                out.append(f"| `{node.target.id}` | {_default(node.value)} | {comment} |")
    return "\n".join(out) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("docs/configuration.md is out of date: run `python tools/gen_config_docs.py`", file=sys.stderr)
            return 1
        return 0
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
