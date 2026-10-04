# rendering.md: Match Visualization Guide

## Game plane contract

**The game is strictly 2D.** Every entity — players, ball, passes, shots — has only an `(x, y)` position. There is no height (z) component anywhere: not in physics, not in mechanics, not in observations. Passes and shots travel along the ground plane. The renderer must faithfully represent this: do not add height, arc, or elevation to any object.

The chosen renderer is a **top-down 2D Pygame view** (`scripts/pygame_render.py`). It is the only renderer supported. The old matplotlib renderer (`live_render.py`) remains as a lightweight fallback but is no longer the primary tool.

---

## 1. What the Old Renderer Had

`scripts/live_render.py` (matplotlib) draws:
- Green pitch rectangle, centre line, goal boxes
- Players as circles (blue team 0, red team 1), player numbers inside
- Ball as a black circle
- Faint `×` for anchor positions
- Vision circle for player 0 of team 0 only
- Gold ring for the playmaker
- Title bar with step, score, formations

**Missing:** pass trajectory, shot indicator, event flashes, vision for all players, formation overlay, keyboard controls, speed > 30 fps.

---

## 2. Pygame Renderer (`scripts/pygame_render.py`)

### 2.1 Window and coordinate mapping

Pitch is 100 × 60 m. Window is 1080 × 720 px with a 50 px margin on all sides.

```
MARGIN   = 50       # px around pitch
WIN_W    = 1080
WIN_H    = 720
PITCH_PX_W = WIN_W - 2 * MARGIN   # 980 px
PITCH_PX_H = WIN_H - 2 * MARGIN   # 620 px
```

```python
def w2s(wx, wy, L=100.0, W=60.0):
    """World coords (m) → screen coords (px). y is flipped (screen y goes downward)."""
    sx = MARGIN + int(wx / L * PITCH_PX_W)
    sy = MARGIN + int((1.0 - wy / W) * PITCH_PX_H)
    return sx, sy

def w2r(metres, L=100.0):
    """World length (m) → screen pixels."""
    return max(1, int(metres / L * PITCH_PX_W))
```

### 2.2 What to Draw — Complete Checklist

**Pitch layer (drawn once per episode onto a background Surface):**
- [ ] Dark green fill (`#2d5a27`)
- [ ] Lighter green stripes alternating every 10 m along the length (subtle, `alpha ≈ 20`)
- [ ] White pitch border (2 px)
- [ ] Centre line (1 px white)
- [ ] Centre circle: radius ≈ 9 m → `w2r(9)`
- [ ] Goal boxes at both ends: width `GOAL_W = 12` m, depth 3 m, white outline
- [ ] Shooting zone arc: arc of radius `35` m centered on each goal mouth (`D_SHOOT_MAX`)
- [ ] Anchor positions as faint `+` (cross, alpha 60) in team colour — drawn when `show_anchors=True`

**Players (every frame):**
- [ ] Filled circle, radius `w2r(1.5)` px, team colour (blue `#2b6cb0` / red `#c53030`)
- [ ] White border 2 px normally, gold 3 px for playmaker
- [ ] Thicker border (4 px) when this player holds the ball
- [ ] Frozen state: grey fill `#888`
- [ ] Player number as white text, centered, font size 11
- [ ] Slot label (D/M/F) as tiny coloured text below/above the circle

**Ball (every frame):**
- [ ] White circle, radius `w2r(0.9)`, outlined black 1 px
- [ ] When `ball_state == "held"`: draw a thin line from holder to ball position (they are the same point, so this just confirms possession visually — skip)
- [ ] When `ball_state == "flight"`: see pass/shot section below

**Pass in flight:**
- [ ] Dotted white line from `flight["start"]` to `flight["end"]`
- [ ] Ball moves along this line at `flight["k"] / flight["n"]` fraction
- [ ] Draw a small arrow at the ball's current position pointing toward the end

**Shot in flight:**
- [ ] Same as pass but line is red/orange
- [ ] Goal-mouth indicator: a small coloured bar on the near goal-line showing where the shot is aimed

**Shot probability indicator (when any player holds the ball in the shoot zone):**
- [ ] Small bar at bottom-right of screen: `"Shoot p: 0.62  ████████░░"`
- [ ] Colour: green if `p > 0.5`, yellow if `0.25 < p <= 0.5`, red if `p <= 0.25`
- [ ] Only shown for the player currently holding the ball

**Vision radius display (toggle `V`):**
- [ ] For each of OUR 5 players (team 0): semi-transparent circle, team colour, alpha 25
- [ ] Overlap between players makes combined coverage brighter (natural alpha blend)

**Formation ghost overlay (toggle `F`):**
- [ ] Show anchor positions of all 5 formations as ghost circles (outline only, small, labelled)
- [ ] Current formation anchors shown solid; others shown faded
- [ ] Useful for seeing what formation the opponent might be in

**HUD (top bar, always visible):**
- [ ] Score: `"OUR  2 – 1  OPP"` centered, large font
- [ ] Step and time: `"t=480 / 1200  (40%)"` right-aligned
- [ ] Our formation and opponent's last known formation: left-aligned
- [ ] Possession dot: coloured circle (blue=ours, red=opp, grey=loose)
- [ ] Playmaker indicator: `"★ PM: player 3"` if sigma > 0

**Event flashes (fade out over 40 frames):**
- [ ] Goal: full-screen overlay `"GOAL!"` in scoring team's colour, alpha fades 200→0
- [ ] Tackle success: yellow spark (4 lines) at tackle position, fades over 20 frames
- [ ] Interception: orange `"INT"` text at interception point, fades 20 frames
- [ ] Pass completed: brief green dot at receiver, 15 frames
- [ ] Turnover (pass intercepted by opponent): red `"LOST"` at passer position, 15 frames

### 2.3 Keyboard Controls

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `→` | Step one frame while paused |
| `↑` / `↓` | Increase / decrease FPS (range 1–60) |
| `V` | Toggle vision radius display |
| `A` | Toggle anchor markers |
| `F` | Toggle formation ghost overlay |
| `R` | Restart from seed 0 |
| `1`–`5` | Highlight player 1–5 of our team |
| `Q` / `Esc` | Quit |

### 2.4 Main Loop Structure

```python
def run(env, agent=None, fps=30, seed=0):
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Football Agent — Top-down View")
    clock = pygame.time.Clock()

    obs, _ = env.reset(seed=seed)
    bg = build_background(env)    # static pitch surface, rebuilt on reset

    paused = False
    show_vision = False
    show_anchors = True
    show_formation_ghost = False
    flashes = []                  # list of active flash animations

    while True:
        # --- events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                handle_key(event.key, ...)

        # --- step ---
        if not paused and not env.done:
            actions = agent.act(obs) if agent else None
            obs, _, done, info = env.step(actions)
            for ev in info["events"]:
                flashes.append(make_flash(ev, env))

        # --- draw ---
        screen.blit(bg, (0, 0))
        if show_anchors:   draw_anchors(screen, env)
        if show_formation_ghost: draw_formation_ghosts(screen, env)
        draw_flight(screen, env)
        draw_players(screen, env)
        draw_ball(screen, env)
        if show_vision:    draw_vision(screen, env)
        draw_shot_indicator(screen, env)
        draw_flashes(screen, flashes)
        draw_hud(screen, env, fps)

        clock.tick(fps)
        pygame.display.flip()
```

---

## 3. Shot Probability Display

When the ball holder is inside `D_SHOOT_MAX = 35` m of the opponent goal, show this in the HUD:

```python
from football.mechanics import shot_probability
from football.geometry import to_frame, dist

def draw_shot_indicator(screen, env, font):
    if env.holder is None:
        return
    ht, hi = env.holder
    me_w = env.pos[ht][hi]
    me_tf = to_frame(ht, me_w, env.L, env.W)
    goal_tf = np.array([env.L, env.W / 2])
    if dist(me_tf, goal_tf) > env.cfg.mechanics.shoot_max:
        return
    defs_w = [env.pos[1 - ht][j] for j in range(5)]
    defs_tf = [to_frame(ht, d, env.L, env.W) for d in defs_w]
    p = shot_probability(me_tf, goal_tf, defs_tf,
                         env.control(ht, hi),
                         env.cfg.mechanics.p_max,
                         env.cfg.mechanics.d0,
                         env.cfg.mechanics.shot_block_scale)
    # Draw bar
    bar_x, bar_y = WIN_W - 200, WIN_H - 40
    bar_w = 150
    colour = (60, 180, 60) if p > 0.5 else (220, 180, 0) if p > 0.25 else (200, 60, 60)
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_w, 18))
    pygame.draw.rect(screen, colour, (bar_x, bar_y, int(bar_w * p), 18))
    label = font.render(f"Shoot p: {p:.2f}", True, (255, 255, 255))
    screen.blit(label, (bar_x, bar_y - 18))
```

---

## 4. Pass Trajectory (Ground-Level)

Since the game is 2D (no height), the pass travels along a straight line on the ground. Render it as a dashed white line from release point to destination, with the ball dot moving along it.

```python
def draw_flight(screen, env):
    if env.ball_state != "flight" or env.flight is None:
        return
    f = env.flight
    ht = f["passer"][0]
    is_shot = False  # shots go through _shoot not _pass, ball_state becomes loose immediately
    start = w2s(*f["start"])
    end   = w2s(*f["end"])
    colour = (255, 255, 255)
    # Dashed line
    draw_dashed_line(screen, colour, start, end, dash=8, gap=6)
    # Ball position
    t = min(1.0, f["k"] / max(1, f["n"]))
    bx = int(start[0] + (end[0] - start[0]) * t)
    by = int(start[1] + (end[1] - start[1]) * t)
    pygame.draw.circle(screen, (255, 255, 255), (bx, by), w2r(0.9))
    pygame.draw.circle(screen, (0, 0, 0), (bx, by), w2r(0.9), 1)

def draw_dashed_line(surface, colour, start, end, dash=8, gap=5):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1, int((dx**2 + dy**2) ** 0.5))
    ux, uy = dx / length, dy / length
    pos = 0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        x0 = int(start[0] + ux * pos)
        y0 = int(start[1] + uy * pos)
        x1 = int(start[0] + ux * min(pos + seg, length))
        y1 = int(start[1] + uy * min(pos + seg, length))
        if drawing:
            pygame.draw.line(surface, colour, (x0, y0), (x1, y1), 2)
        pos += seg
        drawing = not drawing
```

---

## 5. Vision Coverage Overlay

```python
def draw_vision(screen, env):
    for i in range(5):
        rho = env.vision_radius(0, i)
        if not np.isfinite(rho):
            continue
        p = env.pos[0][i]
        cx, cy = w2s(*p)
        r_px = w2r(rho)
        surf = pygame.Surface((r_px * 2, r_px * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (43, 108, 176, 30), (r_px, r_px), r_px)
        pygame.draw.circle(surf, (43, 108, 176, 80), (r_px, r_px), r_px, 1)
        screen.blit(surf, (cx - r_px, cy - r_px))
```

---

## 6. Running the Renderer

```bash
cd football_adapt
.venv/bin/python -m scripts.pygame_render --mode scripted --fps 20
.venv/bin/python -m scripts.pygame_render --mode random --sigma 3 --opp-mode SCHEDULED --fps 15
```

Flags:

| Flag | Default | Effect |
|---|---|---|
| `--mode` | `scripted` | `scripted` = both teams heuristic; `random` = our team random |
| `--T` | 1200 | Match length in steps |
| `--fps` | 20 | Starting FPS |
| `--radius` | cfg default | Override vision radius |
| `--sigma` | 0 | Playmaker player number (1–5) |
| `--opp-mode` | `NONE` | Formation switching mode |
| `--seed` | 0 | Match seed |
| `--no-anchors` | off | Disable anchor markers on startup |

---

## 7. File Locations

| File | Purpose |
|---|---|
| `football_adapt/scripts/live_render.py` | Legacy matplotlib renderer (fallback) |
| `football_adapt/scripts/pygame_render.py` | **Primary renderer — top-down Pygame 2D** |
| `football_adapt/scripts/watch_match.py` | Replay / GIF export (uses pygame renderer) |
