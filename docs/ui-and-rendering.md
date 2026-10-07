# UI and rendering

Everything is drawn with **OpenCV into one preallocated NumPy canvas**; text is rasterised with
Pillow and blended in. There is no GUI toolkit and no GPU requirement.

## 1. Design principles

The hand and the scene are the product; every other element must justify itself by communicating
*tracking, interaction or system state*.

| # | Priority | Where it lives |
|---|---|---|
| 1 | The hand / the scene | the canvas (~74 % of the window) |
| 2 | The current gesture | `Hud._readout` - a compact translucent card, bottom-left |
| 3 | Tracking state | status dot in the top bar; confidence line in the readout |
| 4 | The selected object | corner brackets + a one-line label, drawn by the renderer |
| 5 | Controls | the side panel |
| 6 | Debug information | hidden until `D` (`Hud._debug`, `Renderer._draw_debug_landmarks`) |

Hierarchy comes from **size, weight and grey-level**, not from colour or borders. There is **one
accent colour**, used only for state (tracking, the active gesture, selection). There are no glows,
particles, trails, pulses or looping animations; motion is limited to 150-250 ms eases.

## 2. Design tokens (`visualization/theme.py`)

| Token | Value (BGR) | Use |
|---|---|---|
| `CANVAS` | (13, 12, 11) | window background |
| `SURFACE` | (19, 18, 17) | top bar, side panel, glass panels |
| `SURFACE_RAISED` | (29, 28, 26) | selected row |
| `HAIRLINE` | (42, 40, 38) | 1 px separators, only where they help hierarchy |
| `TEXT`, `TEXT_2`, `TEXT_3` | soft white / secondary / tertiary grey | text levels |
| `ACCENT` | (255, 194, 76) = #4cc2ff | state only |
| `WARN`, `DANGER` | muted amber / red | tracking lost / camera problems only |
| `RADIUS_PANEL`, `RADIUS_CONTROL` | 6, 3 px | corner radii |
| `TOPBAR_HEIGHT`, `SIDEBAR_*`, `INSET`, `FADE` | 44, 280-360 (26 %), 16, 0.2 s | geometry and motion |

## 3. Responsive layout (`visualization/hud.py: Layout`)

`Layout.compute(width, height, show_panel)` derives everything from the window size:

```
scale   = clip(min(width/1280, height/720), 0.85, 1.35)       # UI scale; u(px) = round(px·scale)
top     = round(TOPBAR_HEIGHT · scale)
sidebar = 0  if the panel is hidden or width < 760            # collapse: the canvas is the product
        = min( max(0.26·width, 280·scale), 360·scale, 0.35·width )
canvas  = (width - sidebar) × (height - top)
```

Properties checked by the tests at ten window sizes: the canvas gets ≥ 65 % of the width, the top bar is a
slim strip, the side panel never drops below a readable width, and the canvas is exactly the remainder.

**The window drives the render size.** Each ~6 frames the app asks `cv2.getWindowImageRect` for the live
window size, caps it to `DisplayConfig.max_render_width` (rendering above ~1600 px wide only costs time;
OpenCV scales the result up) and calls `Renderer.ensure_size`, so text stays crisp and positions are
never tied to a screenshot size. A size change reallocates the buffers, recomputes the camera->view
margins and bumps `layout_version`; `GestureFlowApp._sync_layout` then updates the controller's margin and
the scene's aspect. `--window WxH` only sets the *initial* size.

## 4. Typography and icons

* **Fonts** (`visualization/typeface.py`): Inter Regular / Medium / SemiBold, bundled in
  `assets/fonts/` (SIL Open Font License, `OFL.txt`). Fallbacks: Segoe UI / SF / Roboto / DejaVu / Arial,
  then Pillow's default. OpenCV's Hershey fonts are not used for the UI.
* **Rasterise once, blend cheaply.** `_mask(text, weight, size, tracking)` renders an 8-bit alpha mask
  with Pillow (cached, 1024 entries); `blend_mask` alpha-blends a colour through it into just the text's
  region of the frame. `text(img, s, x, y, size, color, weight, anchor, alpha, tracking)` positions by the
  text's *vertical centre* so rows align without baseline arithmetic.
* **Scale.** Sizes are design pixels passed through `layout.u()`. Hierarchy: app name 14 semibold,
  gesture name 17 semibold, body 12-13, labels 10-11 (uppercase labels use 1 px tracking).
* **Icons** (`visualization/icons.py`): a few polylines on a unit square per gesture family, rasterised
  4x supersampled and area-downscaled to an alpha mask (cached per size), then tinted like text.

## 5. The render pipeline (`Renderer.render`)

One call per frame, always in this order, into `canvas` (a `(H, W, 3)` buffer; the viewport, top bar and
side panel are *views* into it, so there is no final copy):

| # | Step | Details |
|---|---|---|
| 1 | **Layout** | `ensure_size(…, ui.show_panel)` (no-op if unchanged) |
| 2 | **Base layer** | *Gesture mode*: crop the camera to the active region (cover), resize into a scratch buffer, apply a gentle contrast LUT `clip((x-128)·c + 128 + 2)` (`video_contrast`, no colour grading), subtract a soft corner vignette (max 46 levels). *Visualization mode*: a per-scene dark vertical gradient (cached). |
| 3 | **Scene objects** | project every object (rotation matrix -> perspective-ish divide -> screen edges), then **one shared faint shadow pass** (gesture mode), then 1 px strokes (far edges dimmer), then per-object annotations. Transitions draw the outgoing scene sliding out and the incoming one in, brightness eased. |
| 4 | **Hands** | per hand: a soft shadow (primary, gesture mode), skeleton polylines, small joints (fingertips larger), then the **gesture-defining landmarks in the accent colour**. Secondary hands are dimmed; in standby everything is dimmed. |
| 5 | **Feedback cues** | see below |
| 6 | **Labels** | the selected object's `"Name · scale×"` |
| 7 | **Debug landmarks** (only with `D`) | brackets, handedness, landmark indices |
| 8 | **Picture-in-picture** (visualization mode) | the camera at 20 % of the canvas width, bottom-right, hairline border |
| 9 | **HUD over the canvas** | standby or empty-state text, the readout, the debug panel, the toast, the camera warning, the help sheet |
| 10 | **Top bar**, then **side panel** | the side panel image is cached and only re-rendered when something visible in it changes |

### Gesture-defining landmarks (`highlighted_landmarks`)

| Gesture | Accent landmarks / bones |
|---|---|
| Pinch, drag, rotate | thumb tip + index tip and the bone between them |
| Point | index finger (MCP-PIP-DIP-TIP) |
| Two fingers | index and middle fingers |
| Open palm, zoom | the five fingertips |
| Fist | the four knuckles and the bones between them |

This shows what the classifier is looking at without lighting up the whole hand.

### Feedback cues (all small and brief)

| Cue | When | Drawing |
|---|---|---|
| Reticle + dwell ring | pointing | thin ring with a gap-cut crosshair and centre dot; a 2 px arc fills over `select_dwell_s` and disappears when the selection brackets appear |
| Pinch ring | pinching | ring at the pinch point (thicker once an object is held) |
| Selection / hover brackets | selected / hovered object | four corner brackets (accent / grey), 2 px while grabbed |
| Rotation arc | while rotating, +0.35 s fade | thin accent arc outside the brackets from the starting roll to the current one, with an arrowhead tangent to it |
| Swipe chevrons | a swipe event, 0.5 s | three small chevrons moving in the swipe direction, fading |
| Progress rings | holding a fist / a steady open palm | thin ring on the palm (accent / white) |

The rotation arc and the swipe chevrons keep a little presentation-only state inside the renderer
(start time / start roll); nothing reads it back.

## 6. HUD pieces

* **Top bar** - app mark + name, the mode name, then (right) FPS and a status dot with one word:
  *Tracking*, *Waiting for hand*, *Hand lost*, *Standby*, *Camera not responding*.
* **Empty state** - four corner marks and "Waiting for hand"; gone the instant a hand is tracked. When no
  hand is present the readout is not drawn at all.
* **Readout** - "HAND TRACKING" + confidence + a 2 px bar; "GESTURE" + name; the latest command in the
  accent colour (`InteractionState.status`, shown for `command_display_s`). One-shot gestures (a swipe)
  stay readable for 0.8 s.
* **Standby** - the canvas is dimmed and "Standby / Hold a fist to resume" appears above the hand.
* **Toast** - a small chip at the top of the canvas, fading in/out over `FADE`.
* **Debug panel** - performance, tracking, features, classifier scores, index tip and pinch point
  coordinates; dense by design.

## 7. The side panel and hit-testing

Rows are **gesture families** (`ui_state.GESTURE_GROUPS`): Open palm, Point, Pinch (pinch + drag +
rotate), Fist, Two fingers, Swipe (left + right), Zoom (in + out). Each row: an icon, a name, a one-word
effect and a status dot (accent = active now, filled grey = enabled, hollow = disabled). The row of a
currently detected gesture is subtly highlighted with an accent bar. Selecting a row opens details (how
to perform it, and a per-gesture switch for each member). Below: a *Gesture | Visualize* segmented
control, a scene selector (`01 / 03`, name, ‹ ›) and key hints. On short windows the rows tighten and the
details are skipped rather than overlapping the bottom controls.

`Hud.draw_sidebar` returns `Hitbox(x0, y0, x1, y1, action)` rectangles; the renderer offsets them to
canvas coordinates and exposes `renderer.hitboxes`. `GestureFlowApp._on_mouse` dispatches:

| Action | Effect |
|---|---|
| `("group_select", id)` | open / close the row's details |
| `("group_toggle", id)` | switch the whole family on (or off if any member is on); `engine.set_enabled(...)` |
| `("member_toggle", gesture)` | switch one gesture; `engine.set_enabled(...)` |
| `("mode", ViewMode)` | gesture / visualization mode |
| `("scene", ±1)` | previous / next scene |

OpenCV reports mouse positions in *image* coordinates even when the window is scaled, so no conversion
is needed.

## 8. UI state (`visualization/ui_state.py`)

`UIState` (owned by the app): `mode`, `show_help`, `show_debug`, `show_panel`, `selected_group`, `enabled`
(the set of switched-on gestures) and the current toast. `FrameStats` carries fps and per-stage timings
for the top bar and the debug panel. Keyboard shortcuts are handled in `GestureFlowApp._on_key` (`Q R 1 2 H
D G N P Tab M F S`).
