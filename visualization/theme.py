"""Design tokens for the interface.  One accent, neutral surfaces, small radii.

Colours are BGR.  The accent is used sparingly and only for *state*: tracking, the current
gesture, selection.  Everything else is grey on near-black, with hierarchy from size and
weight rather than colour or borders.
"""

from __future__ import annotations

Color = tuple[int, int, int]

# surfaces
CANVAS: Color = (13, 12, 11)  # window background / empty states
SURFACE: Color = (19, 18, 17)  # sidebar, top bar
SURFACE_RAISED: Color = (29, 28, 26)  # hovered / selected rows
HAIRLINE: Color = (42, 40, 38)  # 1 px separators, used only where they aid hierarchy

# text
TEXT: Color = (238, 236, 233)
TEXT_2: Color = (158, 153, 147)  # secondary
TEXT_3: Color = (108, 104, 99)  # tertiary / disabled

# state colours (the accent is the only chromatic colour in normal use)
ACCENT: Color = (255, 194, 76)  # #4cc2ff
WARN: Color = (86, 172, 226)  # muted amber: tracking lost, low light
DANGER: Color = (104, 104, 226)  # camera problems only

# geometry
RADIUS_PANEL = 6
RADIUS_CONTROL = 3
TOPBAR_HEIGHT = 44
SIDEBAR_MIN = 280
SIDEBAR_FRACTION = 0.26
SIDEBAR_MAX = 360
INSET = 16  # distance of overlays from the viewport edge

# motion (seconds)
FADE = 0.2
