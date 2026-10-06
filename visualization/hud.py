"""Heads-up display: top bar, canvas overlays, side panel, help and debug.

Information hierarchy (most to least prominent):

    1. the hand and the scene           (renderer - the HUD stays out of the way)
    2. the current gesture              (compact readout, bottom-left of the canvas)
    3. tracking state                   (top-bar status + one line in the readout)
    4. the selected object              (label next to the object, drawn by the renderer)
    5. controls                         (side panel)
    6. debug information                (hidden until ``D``)

Hierarchy comes from size, weight and grey-level; the single accent colour marks *state*
(tracking, the active gesture, selection).  All sizes scale with the window (``layout.scale``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gestures.types import GESTURE_INFO, LEGEND_ORDER, STATIC_GESTURES, Gesture, GestureState
from interaction.controller import InteractionState
from tracking.types import TrackingResult
from visualization import draw, theme, typeface
from visualization.icons import draw_icon
from visualization.scene import SceneManager
from visualization.theme import ACCENT, HAIRLINE, SURFACE, SURFACE_RAISED, TEXT, TEXT_2, TEXT_3, WARN
from visualization.ui_state import GESTURE_GROUPS, FrameStats, Hitbox, UIState, ViewMode

KEY_HELP = (
    ("Q", "Quit"),
    ("R", "Reset scene"),
    ("1  2", "Gesture / visualization mode"),
    ("H", "Toggle this help"),
    ("D", "Debug overlay"),
    ("G", "Toggle side panel"),
    ("N  P", "Next / previous scene"),
    ("Tab", "Next object"),
    ("M", "One / two-hand tracking"),
    ("S", "Save screenshot"),
)


@dataclass(frozen=True)
class Layout:
    """Pixel layout of the window, derived from its size (never hard-coded)."""

    width: int
    height: int
    top: int  # top-bar height
    sidebar: int  # side-panel width (0 = hidden)
    scale: float  # UI scale factor

    @property
    def view_w(self) -> int:
        return self.width - self.sidebar

    @property
    def view_h(self) -> int:
        return self.height - self.top

    @property
    def view_aspect(self) -> float:
        return self.view_w / max(self.view_h, 1)

    def u(self, px: float) -> int:
        """Scale a design-pixel size to this window."""
        return max(1, int(round(px * self.scale)))

    @staticmethod
    def compute(width: int, height: int, show_panel: bool) -> "Layout":
        scale = min(max(min(width / 1280.0, height / 720.0), 0.85), 1.35)
        top = int(round(theme.TOPBAR_HEIGHT * scale))
        sidebar = 0
        if show_panel and width >= 760:  # collapse the panel on very narrow windows: the canvas is the product
            wanted = max(width * theme.SIDEBAR_FRACTION, theme.SIDEBAR_MIN * scale)
            sidebar = int(min(wanted, theme.SIDEBAR_MAX * scale, width * 0.35))
        return Layout(width, height, top, sidebar, scale)


def _wrap(text: str, size: int, weight: str, max_w: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if typeface.measure(trial, size, weight)[0] <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


class Hud:
    def __init__(self) -> None:
        self._held_event: tuple[Gesture, float] = (Gesture.NONE, -1e9)
        self._side_key: tuple | None = None
        self._side_cache: tuple[np.ndarray, list[Hitbox]] | None = None

    # ================================================================== top bar
    def draw_topbar(self, bar: np.ndarray, lay: Layout, ui: UIState, stats: FrameStats, g: GestureState, interaction: InteractionState) -> None:
        h, w, u = bar.shape[0], bar.shape[1], lay.u
        bar[:] = SURFACE
        cv2.line(bar, (0, h - 1), (w, h - 1), HAIRLINE, 1)
        cy = h // 2
        x = u(18)
        # app mark: a small viewfinder (a ring with a centre dot)
        cv2.circle(bar, (x + u(5), cy), u(5), TEXT, 1, draw.AA)
        cv2.circle(bar, (x + u(5), cy), 1, TEXT, -1, draw.AA)
        x += u(18)
        x += typeface.text(bar, "GestureFlow", x, cy, u(14), TEXT, "semibold") + u(14)
        cv2.line(bar, (x, cy - u(8)), (x, cy + u(8)), HAIRLINE, 1)
        x += u(14)
        typeface.text(bar, ui.mode.value, x, cy, u(12), TEXT_2)

        # right side: tracking status, then FPS
        status, color = self._status(g, interaction, stats)
        right = w - u(18)
        sw = typeface.text(bar, status, right, cy, u(12), TEXT if color == ACCENT else TEXT_2, "medium", "r")
        cv2.circle(bar, (right - sw - u(12), cy), u(3), color, -1, draw.AA)
        fx = right - sw - u(30)
        cv2.line(bar, (fx, cy - u(8)), (fx, cy + u(8)), HAIRLINE, 1)
        fx -= u(14)
        unit_w = typeface.text(bar, "FPS", fx, cy, u(11), TEXT_3, "regular", "r")
        value_w = typeface.text(bar, f"{stats.fps:.0f}", fx - unit_w - u(6), cy, u(12), TEXT, "medium", "r")
        if stats.simulated:
            typeface.text(bar, "Simulation", fx - unit_w - value_w - u(26), cy, u(11), TEXT_3, "regular", "r")

    @staticmethod
    def _status(g: GestureState, interaction: InteractionState, stats: FrameStats) -> tuple[str, theme.Color]:
        if stats.camera_status != "ok":
            return "Camera not responding", theme.DANGER
        if not interaction.active:
            return "Standby", TEXT_3
        if not g.hand_present:
            return "Waiting for hand", TEXT_3
        if g.stale:
            return "Hand lost", WARN
        return "Tracking", ACCENT

    # ============================================================ canvas overlays
    def draw_view(
        self, view: np.ndarray, lay: Layout, tracking: TrackingResult, g: GestureState, interaction: InteractionState,
        ui: UIState, stats: FrameStats, t: float,
    ) -> None:
        vh, vw, u = view.shape[0], view.shape[1], lay.u
        if not interaction.active:  # standby: quiet the whole canvas
            cv2.convertScaleAbs(view, dst=view, alpha=0.55)
            typeface.text(view, "Standby", vw // 2, int(vh * 0.14), u(18), TEXT, "medium", "c")
            typeface.text(view, "Hold a fist to resume", vw // 2, int(vh * 0.14) + u(24), u(12), TEXT_2, "regular", "c")
        elif not g.hand_present and stats.camera_status == "ok":
            self._empty_state(view, lay)

        self._readout(view, lay, g, interaction, t)
        if ui.show_debug:
            self._debug(view, lay, tracking, g, stats)
        self._toast(view, lay, ui, t)
        if stats.camera_status != "ok":
            typeface.text(view, "Camera not responding - reconnecting", vw // 2, vh // 2, u(13), theme.DANGER, "medium", "c")
        if ui.show_help:
            self._help(view, lay)

    @staticmethod
    def _empty_state(view: np.ndarray, lay: Layout) -> None:
        """'Waiting for hand': a small viewfinder mark and one quiet line, gone the moment a hand appears."""
        u = lay.u
        cx, cy = view.shape[1] // 2, view.shape[0] // 2 - u(6)
        s, arm = u(15), u(7)
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = cx + sx * s, cy + sy * s
                cv2.line(view, (x, y), (x - sx * arm, y), TEXT_2, 1, draw.AA)
                cv2.line(view, (x, y), (x, y - sy * arm), TEXT_2, 1, draw.AA)
        cv2.circle(view, (cx, cy), 2, TEXT_2, -1, draw.AA)
        typeface.text(view, "Waiting for hand", cx, cy + u(36), u(13), TEXT_2, "regular", "c")

    def _readout(self, view: np.ndarray, lay: Layout, g: GestureState, interaction: InteractionState, t: float) -> None:
        """Compact, translucent tracking + gesture readout in the bottom-left corner."""
        u, vh = lay.u, view.shape[0]
        inset = u(theme.INSET)
        if g.events:  # one-shot gestures (swipes) stay readable for a moment
            self._held_event = (g.events[0], t)
        gesture = g.gesture
        if gesture is Gesture.NONE and t - self._held_event[1] < 0.8:
            gesture = self._held_event[0]
        if not g.hand_present:
            return  # the top bar and the empty state already say so; no extra card

        w, h = u(232), u(86)
        x, y = inset, vh - h - inset
        draw.glass(view, x, y, w, h, SURFACE, 0.80, theme.RADIUS_PANEL, HAIRLINE)
        px = x + u(14)
        typeface.text(view, "HAND TRACKING", px, y + u(18), u(10), TEXT_3, "semibold", tracking=1)
        typeface.text(view, f"{g.tracking_score * 100:.0f}%", x + w - u(14), y + u(18), u(12), TEXT, "medium", "r")
        bar_w = w - u(28)
        draw.fill(view, px, y + u(28), bar_w, 2, HAIRLINE)
        draw.fill(view, px, y + u(28), int(bar_w * min(max(g.tracking_score, 0.0), 1.0)), 2, ACCENT)

        info = GESTURE_INFO[gesture]
        active = gesture is not Gesture.NONE
        label_w = typeface.text(view, "GESTURE", px, y + u(46), u(10), TEXT_3, "semibold", tracking=1)
        if active:
            cv2.circle(view, (px + label_w + u(9), y + u(46)), u(3), ACCENT, -1, draw.AA)
        typeface.text(view, info.label if active else "No gesture", px, y + u(66), u(17), TEXT if active else TEXT_3, "semibold")
        if interaction.status:
            typeface.text(view, interaction.status.title(), x + w - u(14), y + u(66), u(12), ACCENT, "medium", "r")
        elif active:
            typeface.text(view, info.action.split("|")[0].strip(), x + w - u(14), y + u(66), u(11), TEXT_3, "regular", "r")

    @staticmethod
    def _toast(view: np.ndarray, lay: Layout, ui: UIState, t: float) -> None:
        if t >= ui.toast_until or not ui.toast:
            return
        u = lay.u
        k = min(1.0, (t - ui.toast_start) / theme.FADE, (ui.toast_until - t) / theme.FADE)
        if k <= 0.04:
            return
        label = ui.toast + (f"  ·  {ui.toast_sub}" if ui.toast_sub else "")
        tw, _ = typeface.measure(label, u(12), "medium")
        w, h = tw + u(28), u(30)
        x, y = (view.shape[1] - w) // 2, u(theme.INSET)
        draw.glass(view, x, y, w, h, SURFACE, 0.82 * k, theme.RADIUS_CONTROL + 1, HAIRLINE if k > 0.9 else None)
        typeface.text(view, label, x + w // 2, y + h // 2, u(12), TEXT, "medium", "c", alpha=k)

    @staticmethod
    def _help(view: np.ndarray, lay: Layout) -> None:
        u = lay.u
        vh, vw = view.shape[:2]
        w, h = min(u(640), vw - u(32)), min(u(420), vh - u(32))
        x, y = (vw - w) // 2, (vh - h) // 2
        draw.glass(view, x, y, w, h, SURFACE, 0.94, theme.RADIUS_PANEL + 2, HAIRLINE, blur=10)
        typeface.text(view, "Help", x + u(24), y + u(30), u(16), TEXT, "semibold")
        col2 = x + w // 2 + u(8)
        typeface.text(view, "KEYBOARD", x + u(24), y + u(62), u(10), TEXT_3, "semibold", tracking=1)
        for i, (key, desc) in enumerate(KEY_HELP):
            yy = y + u(88) + i * u(25)
            typeface.text(view, key, x + u(24), yy, u(12), TEXT, "medium")
            typeface.text(view, desc, x + u(84), yy, u(12), TEXT_2)
        typeface.text(view, "GESTURES", col2, y + u(62), u(10), TEXT_3, "semibold", tracking=1)
        for i, gesture in enumerate(LEGEND_ORDER):
            yy = y + u(88) + i * u(25)
            info = GESTURE_INFO[gesture]
            typeface.text(view, info.label, col2, yy, u(12), TEXT, "medium")
            typeface.text(view, info.action, col2 + u(112), yy, u(11), TEXT_3)
        typeface.text(view, "Good front lighting and a plain background help.  Keep your hand 30-80 cm from the camera.", x + u(24), y + h - u(22), u(11), TEXT_3)

    # ===================================================================== debug
    @staticmethod
    def _debug(view: np.ndarray, lay: Layout, tracking: TrackingResult, g: GestureState, stats: FrameStats) -> None:
        """Dense technical readout; only shown on demand (D)."""
        u = lay.u
        x, y = u(theme.INSET), u(theme.INSET)
        w, h = u(268), u(452)
        draw.glass(view, x, y, w, h, SURFACE, 0.86, theme.RADIUS_PANEL, HAIRLINE)
        row, size = u(16), u(11)
        cy = y + u(22)

        def heading(label: str) -> None:
            nonlocal cy
            typeface.text(view, label, x + u(14), cy, u(10), TEXT_3, "semibold", tracking=1)
            cy += row + u(2)

        def line(label: str, value: str, color: theme.Color = TEXT) -> None:
            nonlocal cy
            typeface.text(view, label, x + u(14), cy, size, TEXT_2)
            typeface.text(view, value, x + w - u(14), cy, size, color, "medium", "r")
            cy += row

        heading("PERFORMANCE")
        line("frame rate", f"{stats.fps:.1f} fps")
        line("tracking", f"{stats.track_ms:.1f} ms")
        line("gestures + interaction", f"{stats.gesture_ms:.2f} ms")
        line("render", f"{stats.render_ms:.1f} ms")
        cy += u(6)
        heading("TRACKING")
        line("hands in view", f"{len(tracking.hands)}")
        line("hand / confidence", f"{g.handedness or '-'}  {g.tracking_score:.2f}" if g.hand_present else "-")
        line("brightness", f"{tracking.brightness:.0f}", WARN if tracking.low_light else TEXT)
        f = g.features
        if f is not None and g.hand_present:
            line("index tip (x, y)", f"{f.index_tip[0]:.3f}, {f.index_tip[1]:.3f}")
            line("pinch point (x, y)", f"{f.pinch_point[0]:.3f}, {f.pinch_point[1]:.3f}")
            cy += u(6)
            heading("FEATURES")
            for i, name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                typeface.text(view, name, x + u(14), cy, size, TEXT_2)
                bx, bw = x + u(84), w - u(84) - u(52)
                draw.fill(view, bx, cy - 1, bw, 3, HAIRLINE)
                draw.fill(view, bx, cy - 1, int(bw * float(f.extension[i])), 3, TEXT_2)
                typeface.text(view, f"{f.extension[i]:.2f}", x + w - u(14), cy, size, TEXT, "regular", "r")
                cy += row - u(2)
            line("pinch distance", f"{f.pinch_distance:.2f}")
            line("finger spread", f"{f.spread:.1f}°")
            line("roll / facing", f"{f.roll_deg:.0f}°  /  {f.palm_facing:+.2f}")
            line("speed", f"{g.motion.speed:.2f} palm/s")
        cy += u(6)
        heading("CLASSIFIER")
        for gesture in STATIC_GESTURES:
            score = g.scores.get(gesture, 0.0) if g.hand_present else 0.0
            typeface.text(view, GESTURE_INFO[gesture].label, x + u(14), cy, size, TEXT if gesture is g.static else TEXT_2)
            bx, bw = x + u(104), w - u(104) - u(52)
            draw.fill(view, bx, cy - 1, bw, 3, HAIRLINE)
            draw.fill(view, bx, cy - 1, int(bw * min(score, 1.0)), 3, ACCENT if gesture is g.static else TEXT_3)
            typeface.text(view, f"{score:.2f}", x + w - u(14), cy, size, TEXT, "regular", "r")
            cy += row - u(2)

    # ==================================================================== sidebar
    def draw_sidebar(self, side: np.ndarray, lay: Layout, g: GestureState, scenes: SceneManager, ui: UIState) -> list[Hitbox]:
        """Draw the side panel into ``side`` and return its hit boxes (relative to ``side``).

        Static between state changes, so it is cached and re-rendered only when something
        visible in it changes.
        """
        active = frozenset(
            gr.id for gr in GESTURE_GROUPS if g.hand_present and any(m in (g.gesture, g.dynamic, g.static) for m in gr.members)
        )
        key = (side.shape, ui.enabled, ui.mode, scenes.index, active, ui.selected_group)
        if key != self._side_key or self._side_cache is None:
            self._side_cache = self._render_sidebar(side.shape[1], side.shape[0], lay, active, scenes, ui)
            self._side_key = key
        base, boxes = self._side_cache
        np.copyto(side, base)
        return boxes

    @staticmethod
    def _render_sidebar(w: int, h: int, lay: Layout, active: frozenset[str], scenes: SceneManager, ui: UIState) -> tuple[np.ndarray, list[Hitbox]]:
        u = lay.u
        img = np.empty((h, w, 3), np.uint8)
        img[:] = SURFACE
        cv2.line(img, (0, 0), (0, h), HAIRLINE, 1)
        boxes: list[Hitbox] = []
        pad = u(18)

        # ---- header
        typeface.text(img, "GESTURES", pad, u(30), u(11), TEXT_2, "semibold", tracking=1)
        n_on = sum(1 for gesture in LEGEND_ORDER if gesture in ui.enabled)
        typeface.text(img, f"{n_on} of {len(LEGEND_ORDER)} enabled", w - pad, u(30), u(11), TEXT_3, "regular", "r")
        typeface.text(img, "Active gestures", pad, u(48), u(12), TEXT_3)

        # ---- rows: one per gesture family, a status dot instead of a toggle switch
        compact = h < u(560)
        row_h = u(30) if compact else u(34)
        y0 = u(62)
        for i, grp in enumerate(GESTURE_GROUPS):
            y = y0 + i * row_h
            cy = y + row_h // 2
            enabled = bool(ui.enabled & set(grp.members))
            live = grp.id in active
            if live:
                draw.fill(img, u(8), y, w - u(16), row_h - 2, TEXT, 0.06, theme.RADIUS_CONTROL)
                draw.fill(img, u(8), y + u(7), 2, row_h - u(14), ACCENT)
            elif ui.selected_group == grp.id:
                draw.fill(img, u(8), y, w - u(16), row_h - 2, SURFACE_RAISED, 1.0, theme.RADIUS_CONTROL)
            name_color = TEXT if live else TEXT_2 if enabled else TEXT_3
            draw_icon(img, grp.icon, pad + u(8), cy, u(16), ACCENT if live else (TEXT_2 if enabled else TEXT_3))
            typeface.text(img, grp.name, pad + u(28), cy, u(13), name_color, "medium" if live else "regular")
            typeface.text(img, grp.action, w - pad - u(22), cy, u(12), TEXT_2 if live else TEXT_3, "regular", "r")
            dot = (w - pad - u(4), cy)
            if live:
                cv2.circle(img, dot, u(3), ACCENT, -1, draw.AA)
            elif enabled:
                cv2.circle(img, dot, u(3), TEXT_3, -1, draw.AA)
            else:
                cv2.circle(img, dot, u(3), TEXT_3, 1, draw.AA)
            boxes.append(Hitbox(w - pad - u(16), y, w, y + row_h, ("group_toggle", grp.id)))
            boxes.append(Hitbox(0, y, w - pad - u(16), y + row_h, ("group_select", grp.id)))

        # ---- details, only for the selected row
        y = y0 + len(GESTURE_GROUPS) * row_h + u(10)
        sel = next((gr for gr in GESTURE_GROUPS if gr.id == ui.selected_group), None)
        reserved = u(150) if compact else u(176)  # the bottom controls
        if sel is not None and y + u(70) < h - reserved:
            cv2.line(img, (pad, y), (w - pad, y), HAIRLINE, 1)
            y += u(22)
            typeface.text(img, sel.name, pad, y, u(14), TEXT, "semibold")
            y += u(8)
            for line in _wrap(sel.how, u(12), "regular", w - 2 * pad):
                y += u(17)
                typeface.text(img, line, pad, y, u(12), TEXT_2)
            y += u(26)
            for member in sel.members:
                if y + u(24) > h - reserved:
                    break
                info = GESTURE_INFO[member]
                on = member in ui.enabled
                cv2.circle(img, (pad + u(4), y), u(3), TEXT_2 if on else TEXT_3, -1 if on else 1, draw.AA)
                typeface.text(img, info.label, pad + u(16), y, u(12), TEXT if on else TEXT_3, "medium")
                typeface.text(img, info.action.split("|")[0].strip(), w - pad, y, u(11), TEXT_3, "regular", "r")
                boxes.append(Hitbox(0, y - u(11), w, y + u(11), ("member_toggle", member)))
                y += u(24)
        elif sel is None:
            typeface.text(img, "Select a gesture for details", pad, y + u(14), u(11), TEXT_3)

        # ---- bottom block, anchored to the bottom: view mode, scene selector, key hints
        by = h - u(20)
        typeface.text(img, "H help   D debug   Q quit", pad, by, u(11), TEXT_3)
        by -= u(40)
        seg_w, seg_h = w - 2 * pad, u(28)
        draw.outline(img, pad, by, seg_w, seg_h, HAIRLINE, theme.RADIUS_CONTROL)
        for i, (mode, label) in enumerate(((ViewMode.GESTURE, "Gesture"), (ViewMode.VISUALIZATION, "Visualize"))):
            sx = pad + i * (seg_w // 2)
            if ui.mode is mode:
                draw.fill(img, sx + 2, by + 2, seg_w // 2 - 3, seg_h - 4, TEXT, 0.10, theme.RADIUS_CONTROL - 1)
            typeface.text(img, label, sx + seg_w // 4, by + seg_h // 2, u(12), TEXT if ui.mode is mode else TEXT_3, "medium", "c")
            boxes.append(Hitbox(sx, by, sx + seg_w // 2, by + seg_h, ("mode", mode)))
        typeface.text(img, "VIEW", pad, by - u(12), u(10), TEXT_3, "semibold", tracking=1)
        by -= u(66)

        typeface.text(img, "SCENE", pad, by - u(12), u(10), TEXT_3, "semibold", tracking=1)
        typeface.text(img, f"{scenes.index + 1:02d} / {len(scenes.scenes):02d}", w - pad, by - u(12), u(11), TEXT_3, "regular", "r")
        cy = by + u(12)
        typeface.text(img, scenes.current.preset.name, pad, cy, u(15), TEXT, "medium")
        r = u(5)
        for sign, cx in ((-1, w - pad - u(40)), (1, w - pad - u(10))):
            chevron = np.array([[cx - sign * r // 2, cy - r], [cx + sign * r // 2, cy], [cx - sign * r // 2, cy + r]], np.int32)
            cv2.polylines(img, [chevron], False, TEXT_2, 1, draw.AA)
            boxes.append(Hitbox(cx - u(14), cy - u(14), cx + u(14), cy + u(14), ("scene", sign)))
        return img, boxes
