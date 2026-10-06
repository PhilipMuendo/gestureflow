"""Frame composition: top bar, camera/scene canvas, side panel.

Layout (computed from the live window size - nothing is hard-coded)::

    ┌──────────────────────────────────────────────┐
    │ top bar                                      │
    ├───────────────────────────────┬──────────────┤
    │                               │              │
    │   canvas  (camera + scene)    │  side panel  │
    │   ~74 % of the width          │  ~26 %       │
    │                               │              │
    └───────────────────────────────┴──────────────┘

Everything is drawn straight into one preallocated canvas with OpenCV.  The camera feed is
shown as it is (a gentle contrast lift, no colour grading) *cropped to cover* the canvas, so
there are no letterbox bars and the hand stays where the cursor says it is.  The scene and
the hand skeleton are thin, neutral strokes on top; the single accent colour appears only
where it carries state (the tracked landmarks of the active gesture, the selected object,
the pointer).  Feedback is short-lived and small: nothing glows, pulses or emits particles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from config import DisplayConfig
from gestures.types import Gesture, GestureState
from interaction.controller import InteractionState, Phase
from interaction.mapping import camera_to_view, view_margins
from tracking.types import HAND_CONNECTIONS, HandObservation, TrackingResult
from visualization import draw, theme, typeface
from visualization.draw import AA
from visualization.hud import Hud, Layout
from visualization.scene import Scene, SceneManager, SceneObject, ease_in_out
from visualization.shapes import MESHES
from visualization.theme import ACCENT, HAIRLINE, TEXT, TEXT_2, TEXT_3, Color
from visualization.ui_state import FrameStats, Hitbox, UIState, ViewMode

SKELETON = np.array(HAND_CONNECTIONS, dtype=np.int32)
FINGERTIPS = (4, 8, 12, 16, 20)
HAND_COLOR: Color = (226, 224, 222)
HAND_COLOR_SECONDARY: Color = (120, 118, 116)
PINCH_FAMILY = (Gesture.PINCH, Gesture.PINCH_DRAG, Gesture.PINCH_ROTATE)
ZOOM_FAMILY = (Gesture.ZOOM_IN, Gesture.ZOOM_OUT)


def rotation_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def highlighted_landmarks(gesture: Gesture) -> tuple[set[int], list[tuple[int, int]]]:
    """Landmarks (and the bones between them) that *are* the gesture.

    Lighting only these - the thumb and index for a pinch, the index finger for a point -
    shows what the classifier is looking at without lighting up the whole hand.
    """
    if gesture in PINCH_FAMILY:
        return {4, 8}, [(4, 8)]
    if gesture is Gesture.POINT:
        return {5, 6, 7, 8}, [(5, 6), (6, 7), (7, 8)]
    if gesture is Gesture.TWO_FINGERS:
        return {5, 6, 7, 8, 9, 10, 11, 12}, [(5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12)]
    if gesture is Gesture.OPEN_PALM or gesture in ZOOM_FAMILY:
        return set(FINGERTIPS), []
    if gesture is Gesture.FIST:
        return {5, 9, 13, 17}, [(5, 9), (9, 13), (13, 17)]
    return set(), []


@dataclass(frozen=True)
class _Projected:
    obj: SceneObject
    center: tuple[int, int]
    radius: float
    front: np.ndarray  # (n, 2, 2) near-side edges
    back: np.ndarray  # (m, 2, 2) far-side edges
    k: float  # brightness


class Renderer:
    def __init__(self, cfg: DisplayConfig, camera_aspect: float, base_margin: float, width: int | None = None, height: int | None = None) -> None:
        self.cfg = cfg
        self.camera_aspect = camera_aspect
        self.base_margin = base_margin
        self.hud = Hud()
        self.hitboxes: list[Hitbox] = []
        self.layout_version = 0  # bumped whenever the layout (and so the camera->view mapping) changes
        self._size = (width or cfg.window_width, height or cfg.window_height)
        self._show_panel = cfg.show_panel
        self._lut = np.clip((np.arange(256) - 128.0) * cfg.video_contrast + 128.0 + 2.0, 0, 255).astype(np.uint8)
        self._swipe_fx: tuple[int, float] | None = None  # (direction, start time)
        self._rotate_fx: tuple[int, float, float] | None = None  # (object id, start roll, last active time)
        self._apply_layout(Layout.compute(*self._size, self._show_panel))

    # ----------------------------------------------------------------- layout
    def _apply_layout(self, lay: Layout) -> None:
        self.layout = lay
        self.margin = view_margins(lay.view_aspect, self.camera_aspect, self.base_margin)
        vw, vh = lay.view_w, lay.view_h
        self._canvas = np.zeros((lay.height, lay.width, 3), np.uint8)
        self._video = np.zeros((vh, vw, 3), np.uint8)
        ys = np.linspace(-1, 1, vh)[:, None]
        xs = np.linspace(-1, 1, vw)[None, :]
        shade = 46.0 * np.clip((xs**2 + ys**2) / 2.0, 0, 1) ** 1.4  # a very soft corner falloff, nothing more
        self._vignette = np.repeat(shade.astype(np.uint8)[:, :, None], 3, axis=2)
        self._backgrounds: dict[int, np.ndarray] = {}
        self.layout_version += 1

    def ensure_size(self, width: int, height: int, show_panel: bool) -> None:
        """Adopt a new window size / panel visibility (cheap when nothing changed)."""
        self._size, self._show_panel = (width, height), show_panel
        lay = Layout.compute(width, height, show_panel)
        if lay != self.layout:
            self._apply_layout(lay)

    @property
    def size(self) -> tuple[int, int]:
        """Requested render size (the layout may reserve less of it for the canvas)."""
        return self._size

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.layout.width, self.layout.height

    @property
    def view_aspect(self) -> float:
        return self.layout.view_aspect

    # ----------------------------------------------------------------- render
    def render(
        self,
        frame: np.ndarray | None,
        tracking: TrackingResult,
        gesture: GestureState,
        interaction: InteractionState,
        scenes: SceneManager,
        ui: UIState,
        stats: FrameStats,
        t: float,
    ) -> np.ndarray:
        self.ensure_size(*self._size, ui.show_panel)
        lay = self.layout
        canvas = self._canvas
        vw, top = lay.view_w, lay.top
        view = canvas[top:, :vw]  # the canvas is a window into the frame buffer: no final copy
        gesture_mode = ui.mode is ViewMode.GESTURE

        self._base(frame, scenes, gesture_mode, view)
        labels: list[tuple[str, int, int]] = []
        self._draw_scenes(view, scenes, interaction, labels, t, gesture_mode)
        for hand in tracking.hands:
            self._draw_hand(view, hand, hand is tracking.primary, gesture, interaction.active, gesture_mode)
        self._draw_feedback(view, interaction, gesture, scenes.current, t)
        for text, x, y in labels:
            typeface.text(view, text, x, y, lay.u(11), TEXT, "medium", "c")
        if ui.show_debug:
            self._draw_debug_landmarks(view, tracking)
        if not gesture_mode:
            self._draw_pip(view, frame, tracking)

        self.hud.draw_view(view, lay, tracking, gesture, interaction, ui, stats, t)
        self.hud.draw_topbar(canvas[:top], lay, ui, stats, gesture, interaction)
        self.hitboxes = []
        if lay.sidebar:
            boxes = self.hud.draw_sidebar(canvas[top:, vw:], lay, gesture, scenes, ui)
            self.hitboxes = [Hitbox(b.x0 + vw, b.y0 + top, b.x1 + vw, b.y1 + top, b.action) for b in boxes]
        return canvas

    # ------------------------------------------------------------------- base
    def _background(self, scenes: SceneManager) -> np.ndarray:
        key = scenes.index
        if key not in self._backgrounds:
            lay = self.layout
            base = np.array(scenes.current.preset.background, float)
            t = np.linspace(0, 1, lay.view_h)[:, None, None]
            gradient = np.repeat(base * (1.25 - 0.45 * t), lay.view_w, axis=1)
            self._backgrounds[key] = np.clip(gradient - self._vignette * 0.5, 0, 255).astype(np.uint8)
        return self._backgrounds[key]

    def _base(self, frame: np.ndarray | None, scenes: SceneManager, gesture_mode: bool, out: np.ndarray) -> None:
        if gesture_mode and frame is not None:
            h, w = frame.shape[:2]
            mx, my = self.margin  # show exactly the part of the camera that maps onto the canvas (cover crop)
            crop = frame[int(h * my) : int(h * (1 - my)), int(w * mx) : int(w * (1 - mx))]
            cv2.resize(crop, (self.layout.view_w, self.layout.view_h), dst=self._video, interpolation=cv2.INTER_LINEAR)
            cv2.LUT(self._video, self._lut, dst=self._video)
            cv2.subtract(self._video, self._vignette, dst=out)
        else:
            np.copyto(out, self._background(scenes))

    # ----------------------------------------------------------------- scenes
    def _draw_scenes(self, view: np.ndarray, scenes: SceneManager, interaction: InteractionState, labels: list, t: float, gesture_mode: bool) -> None:
        tr = scenes.transition
        vw = self.layout.view_w
        if tr is not None:
            e = ease_in_out(tr.progress)
            if scenes.previous is not None:
                self._draw_scene(view, scenes.previous, -tr.direction * e * vw * 0.25, 1.0 - e, interaction, labels, t, False, gesture_mode)
            self._draw_scene(view, scenes.current, tr.direction * (1.0 - e) * vw * 0.25, e, interaction, labels, t, True, gesture_mode)
        else:
            self._draw_scene(view, scenes.current, 0.0, 1.0, interaction, labels, t, True, gesture_mode)

    def _draw_scene(self, view: np.ndarray, scene: Scene, dx: float, alpha: float, interaction: InteractionState,
                    labels: list, t: float, is_current: bool, gesture_mode: bool) -> None:
        projected = [p for o in sorted(scene.objects, key=lambda o: -o.depth)  # far objects first
                     if (p := self._project(scene, o, dx, alpha)) is not None]
        if gesture_mode and projected:  # one shared, faint under-stroke keeps thin lines legible over any video
            front = [p.front for p in projected if len(p.front)]
            if front:
                self._shadow(view, np.concatenate(front), 3, 0.35)
        for p in projected:
            self._stroke(view, p)
            if is_current:
                self._annotate(view, scene, p, interaction, labels, t)

    def _project(self, scene: Scene, o: SceneObject, dx: float, alpha: float) -> "_Projected | None":
        """Screen-space geometry and brightness of one object (None if off-screen)."""
        vw, vh = self.layout.view_w, self.layout.view_h
        sp = scene.screen_pos(o)
        cx, cy = sp[0] * vw + dx, sp[1] * vh
        radius = scene.screen_radius(o) * vh
        if radius < 2 or not (-radius * 2 < cx < vw + radius * 2):
            return None
        intro = min(1.0, max(0.0, o.appear + 0.15))
        level = 0.66 + 0.14 * o.hover_level + 0.20 * o.select_level
        k = alpha * intro * level * (1.0 - 0.25 * o.depth)
        mesh = MESHES[o.shape]
        v = mesh.vertices @ rotation_matrix(o.angles[0], o.angles[1], o.angles[2] + o.user_roll).T
        xy = np.array([cx, cy]) + v[:, :2] * radius / (1.0 + 0.22 * v[:, 2:3])
        seg = xy[mesh.edges].astype(np.int32)
        front = v[mesh.edges, 2].mean(axis=1) < 0.0
        return _Projected(o, (int(cx), int(cy)), radius, seg[front], seg[~front], k)

    @staticmethod
    def _stroke(view: np.ndarray, p: "_Projected") -> None:
        o = p.obj
        if len(p.back):
            cv2.polylines(view, p.back, False, draw.scale_color(o.color, p.k * 0.45), 1, AA)
        if len(p.front):
            cv2.polylines(view, p.front, False, draw.scale_color(o.color, p.k), 1, AA)

    def _annotate(self, view: np.ndarray, scene: Scene, p: "_Projected", interaction: InteractionState, labels: list, t: float) -> None:
        """Selection / hover brackets, the object label and the rotation arc."""
        lay = self.layout
        vw, vh = lay.view_w, lay.view_h
        o, c, radius = p.obj, p.center, p.radius
        s = int(radius * 1.32)
        length = max(7, int(s * 0.34))
        if o.select_level > 0.04:  # selection / grab: thin corner brackets
            sel = draw.mix_color(theme.TEXT_3, ACCENT, min(1.0, o.select_level))
            draw.brackets(view, c[0] - s, c[1] - s, c[0] + s, c[1] + s, sel, length, 2 if o.grab_level > 0.5 else 1)
        elif o.hover_level > 0.04:
            draw.brackets(view, c[0] - s, c[1] - s, c[0] + s, c[1] + s, draw.scale_color(TEXT_2, o.hover_level), length, 1)
        if o.select_level > 0.5:
            labels.append((f"{o.shape.capitalize()}  ·  {o.scale:.1f}×", c[0], c[1] + s + lay.u(16)))

        # rotation: a thin arc around the object showing how far it has been turned
        if interaction.phase is Phase.ROTATING and interaction.grabbed_id == o.id:
            if self._rotate_fx is None or self._rotate_fx[0] != o.id:
                self._rotate_fx = (o.id, o.user_roll_target, t)
            self._rotate_fx = (o.id, self._rotate_fx[1], t)
        if self._rotate_fx is not None and self._rotate_fx[0] == o.id:
            _, start, last = self._rotate_fx
            fade = 1.0 - (t - last) / 0.35
            if fade <= 0:
                self._rotate_fx = None
            else:
                sweep = float(np.clip(math.degrees(o.user_roll_target - start), -330, 330))
                r = int(s * 1.30)
                pad = r + 6
                x0, y0 = max(0, c[0] - pad), max(0, c[1] - pad)
                x1, y1 = min(vw, c[0] + pad), min(vh, c[1] + pad)
                if x1 > x0 and y1 > y0 and abs(sweep) > 2:
                    roi = view[y0:y1, x0:x1]
                    layer = roi.copy()
                    centre = (c[0] - x0, c[1] - y0)
                    draw.arc(layer, centre, r, -90 + min(0, sweep), -90 + max(0, sweep), ACCENT, 2)
                    end = math.radians(-90 + sweep)
                    tip = (int(centre[0] + r * math.cos(end)), int(centre[1] + r * math.sin(end)))
                    direction = 1 if sweep > 0 else -1
                    heading = math.atan2(direction * math.cos(end), -direction * math.sin(end))  # tangent at the arc's end
                    for wing in (-0.5, 0.5):  # arrow head: two short strokes trailing back along the tangent
                        a = heading + math.pi + wing
                        cv2.line(layer, tip, (int(tip[0] + 8 * math.cos(a)), int(tip[1] + 8 * math.sin(a))), ACCENT, 2, AA)
                    cv2.addWeighted(layer, fade, roi, 1.0 - fade, 0, dst=roi)

    @staticmethod
    def _shadow(view: np.ndarray, segs: np.ndarray, width: int, opacity: float) -> None:
        """Darken the pixels under a set of line segments by ``opacity`` (blended inside their bounding box)."""
        vh, vw = view.shape[:2]
        pad = width + 2
        x0, y0 = max(0, int(segs[..., 0].min()) - pad), max(0, int(segs[..., 1].min()) - pad)
        x1, y1 = min(vw, int(segs[..., 0].max()) + pad), min(vh, int(segs[..., 1].max()) + pad)
        if x1 <= x0 or y1 <= y0:
            return
        roi = view[y0:y1, x0:x1]
        shade = roi.copy()
        cv2.polylines(shade, segs - np.array([x0, y0], np.int32), False, (0, 0, 0), width, AA)
        cv2.addWeighted(shade, opacity, roi, 1.0 - opacity, 0, dst=roi)

    # ------------------------------------------------------------ hand overlay
    def _hand_points(self, hand: HandObservation) -> np.ndarray:
        vw, vh = self.layout.view_w, self.layout.view_h
        return (camera_to_view(hand.landmarks[:, :2], self.margin) * (vw, vh)).astype(np.int32)

    def _draw_hand(self, view: np.ndarray, hand: HandObservation, primary: bool, g: GestureState, active: bool, gesture_mode: bool) -> None:
        """Thin skeleton + small joints; only the landmarks that define the current gesture
        are drawn in the accent colour."""
        pts = self._hand_points(hand)
        vh, vw = view.shape[:2]
        base = HAND_COLOR if primary else HAND_COLOR_SECONDARY
        if not active:
            base = draw.scale_color(base, 0.55)
        which = Gesture.NONE
        if primary and active and g.hand_present:
            which = g.dynamic if g.dynamic is not Gesture.NONE else g.static
        hl_points, hl_segs = highlighted_landmarks(which)

        pad = 10
        x0, y0 = max(0, int(pts[:, 0].min()) - pad), max(0, int(pts[:, 1].min()) - pad)
        x1, y1 = min(vw, int(pts[:, 0].max()) + pad), min(vh, int(pts[:, 1].max()) + pad)
        if x1 <= x0 or y1 <= y0:
            return
        segs = pts[SKELETON]
        if gesture_mode and primary:  # soft shadow so the skeleton reads on bright video
            self._shadow(view, segs, 3, 0.45)

        cv2.polylines(view, segs, False, base, 1, AA)
        for i, p in enumerate(pts):
            cv2.circle(view, (int(p[0]), int(p[1])), 3 if i in FINGERTIPS else 2, base, -1, AA)
        for a, b in hl_segs:
            cv2.line(view, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), ACCENT, 2, AA)
        for i in hl_points:
            cv2.circle(view, (int(pts[i][0]), int(pts[i][1])), 4, ACCENT, -1, AA)

    def _draw_pip(self, view: np.ndarray, frame: np.ndarray | None, tracking: TrackingResult) -> None:
        """Camera picture-in-picture (visualisation mode)."""
        if frame is None:
            return
        lay = self.layout
        w = max(lay.u(168), int(lay.view_w * 0.2))
        h = int(w * frame.shape[0] / frame.shape[1])
        x0, y0 = lay.view_w - w - lay.u(theme.INSET), lay.view_h - h - lay.u(theme.INSET)
        if x0 < 0 or y0 < 0:
            return
        thumb = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        for hand in tracking.hands:
            pts = (hand.landmarks[:, :2] * (w, h)).astype(np.int32)
            cv2.polylines(thumb, pts[SKELETON], False, ACCENT if hand is tracking.primary else TEXT_3, 1, AA)
        view[y0 : y0 + h, x0 : x0 + w] = cv2.addWeighted(view[y0 : y0 + h, x0 : x0 + w], 0.1, thumb, 0.9, 0)
        draw.outline(view, x0, y0, w, h, HAIRLINE, 0)

    # ------------------------------------------------------- interaction feedback
    def _draw_feedback(self, view: np.ndarray, interaction: InteractionState, g: GestureState, scene: Scene, t: float) -> None:
        """Small, brief cues: pointer reticle, pinch point, progress arcs, swipe direction."""
        lay = self.layout
        vw, vh = lay.view_w, lay.view_h
        if g.events:
            for e in g.events:
                if e in (Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT):
                    self._swipe_fx = (-1 if e is Gesture.SWIPE_LEFT else 1, t)
        self._draw_swipe(view, t)

        if not g.hand_present:
            return
        dim = 1.0 if interaction.active else 0.4
        if interaction.pinching and interaction.pinch_point is not None:
            p = (int(interaction.pinch_point[0] * vw), int(interaction.pinch_point[1] * vh))
            grabbed = interaction.grabbed_id is not None
            cv2.circle(view, p, lay.u(11 if grabbed else 9), draw.scale_color(ACCENT, dim), 2 if grabbed else 1, AA)
        elif interaction.cursor is not None and interaction.phase is Phase.POINTING:
            c = (int(interaction.cursor[0] * vw), int(interaction.cursor[1] * vh))
            draw.reticle(view, c, draw.scale_color(ACCENT, dim), lay.u(9))
            if interaction.dwell_progress < 0.999:  # a full ring means 'selected' and is shown by the brackets instead
                draw.progress_arc(view, c, lay.u(15), interaction.dwell_progress, TEXT, 2)

        if g.features is not None:  # hold-to-act progress, a thin ring on the palm
            palm = camera_to_view(g.features.palm_center, self.margin) * (vw, vh)
            pc = (int(palm[0]), int(palm[1]))
            if interaction.fist_progress > 0.02:
                draw.progress_arc(view, pc, lay.u(30), interaction.fist_progress, ACCENT, 2)
            if interaction.reset_progress > 0.02:
                draw.progress_arc(view, pc, lay.u(30), interaction.reset_progress, TEXT, 2)

    def _draw_swipe(self, view: np.ndarray, t: float) -> None:
        if self._swipe_fx is None:
            return
        direction, start = self._swipe_fx
        age = t - start
        if age > 0.5:
            self._swipe_fx = None
            return
        fade = 1.0 - age / 0.5
        u = self.layout.u
        vw, vh = self.layout.view_w, self.layout.view_h
        cx, cy = vw // 2 + int(direction * age * u(60)), vh // 2
        pad = u(40)
        x0, y0, x1, y1 = max(0, cx - pad), max(0, cy - pad), min(vw, cx + pad), min(vh, cy + pad)
        roi = view[y0:y1, x0:x1]
        layer = roi.copy()
        for i in range(3):
            ox = cx - x0 + direction * (i - 1) * u(13)
            oy = cy - y0
            chevron = np.array([[ox - direction * u(5), oy - u(8)], [ox + direction * u(5), oy], [ox - direction * u(5), oy + u(8)]], np.int32)
            cv2.polylines(layer, [chevron], False, TEXT, 2, AA)
        cv2.addWeighted(layer, 0.85 * fade, roi, 1.0 - 0.85 * fade, 0, dst=roi)

    # ------------------------------------------------------------------ debug
    def _draw_debug_landmarks(self, view: np.ndarray, tracking: TrackingResult) -> None:
        u = self.layout.u
        for hand in tracking.hands:
            pts = self._hand_points(hand)
            x0, y0 = pts.min(axis=0) - 14
            x1, y1 = pts.max(axis=0) + 14
            primary = hand is tracking.primary
            draw.brackets(view, int(x0), int(y0), int(x1), int(y1), ACCENT if primary else TEXT_3, u(12), 1)
            typeface.text(view, f"{hand.handedness}  {hand.score:.2f}", int(x0), int(y0) - u(8), u(11), ACCENT if primary else TEXT_3, "medium")
            for i, p in enumerate(pts):
                typeface.text(view, str(i), int(p[0]) + u(9), int(p[1]) - u(2), u(9), TEXT_2)
