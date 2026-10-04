# rendering.md: Match Visualization Guide

This document covers how to render a match: what is currently implemented, what is missing, and how to build a proper game-quality 2D/3D renderer. An AI agent implementing the renderer should read this document completely before writing any code.

---

## 1. What Exists Now

The current renderer lives in `football_adapt/scripts/live_render.py`. It uses **matplotlib** and is intentionally minimal.

**What it draws:**
- Green pitch rectangle with centre line and goal posts
- Players as filled circles (blue = team 0, red = team 1)
- Player numbers as text inside circles
- Ball as a small black circle
- Anchor positions as faint `×` markers
- Vision radius circle for player 0 of team 0 (only when `show_vision=True`)
- Gold ring around the playmaker if `sigma > 0`
- Title bar: step `t`, score, formation names

**How to run:**
```bash
cd football_adapt
python -m scripts.live_render --mode scripted --T 400 --fps 10
python -m scripts.live_render --mode random --radius 15 --sigma 3 --opp-mode SCHEDULED
```

**Flags:**
| Flag | Default | Effect |
|---|---|---|
| `--mode` | `scripted` | `scripted` = both teams heuristic; `random` = our team random |
| `--T` | 400 | Number of steps |
| `--fps` | 10 | Render speed |
| `--radius` | default cfg | Override vision radius |
| `--sigma` | 0 | Playmaker index (1–5) |
| `--opp-mode` | `NONE` | Formation switching mode |
| `--seed` | 0 | Match seed |

**Limitations:**
- No trajectory arcs for passes or shots
- No vision overlays for all players
- No formation labels or formation geometry
- No event annotations (goal, tackle, interception)
- No speed control (pause/step) or scrubbing
- Not a real-time game UI — matplotlib is not designed for interactive rendering
- No depth or 3D perspective

---

## 2. Renderer Architecture (Any Implementation)

All rendering variants should follow this interface so they are interchangeable:

```python
class MatchRenderer:
    def reset(self, env: FootballEnv) -> None:
        """Called once per episode to set up the scene."""

    def step(self, env: FootballEnv, events: list) -> None:
        """Called every step. Draw the current state plus any events from this step."""

    def close(self) -> None:
        """Clean up (close window, release GPU, etc.)."""
```

The renderer reads from the environment's public attributes:
- `env.pos[team][i]` — player positions (world frame)
- `env.ball_pos` — ball position
- `env.anchors[team][i]` — anchor positions
- `env.holder` — who holds the ball (`(team, i)` or `None`)
- `env.ball_state` — `"held"`, `"loose"`, `"flight"`
- `env.flight` — flight dict: `start`, `end`, `k`, `n` (current step / total steps)
- `env.sigma_idx` — playmaker index (0-based) or `None`
- `env.score`, `env.t`, `env.formation` — match info
- `env.vision_radius(team, i)` — vision radius per player
- `env.cfg` — config access

**Never** read hidden state (e.g. opponent's undisclosed formation string during inference testing).

---

## 3. Improved 2D Renderer (Pygame)

### 3.1 Why Pygame

Pygame runs at 60 fps, handles keyboard input natively, draws with hardware acceleration, and is pure Python. It is the best step-up from matplotlib for this use case.

Install: `pip install pygame`

### 3.2 Coordinate Mapping

The pitch is 100 × 60 m. Map to a window of at least 900 × 600 px with a margin:

```python
MARGIN = 40     # px
WIN_W, WIN_H = 1000, 680

def world_to_screen(wx, wy, L=100.0, W=60.0):
    sx = MARGIN + int(wx / L * (WIN_W - 2 * MARGIN))
    sy = MARGIN + int((1 - wy / W) * (WIN_H - 2 * MARGIN))  # y flipped (screen y goes down)
    return sx, sy
```

### 3.3 What to Draw (complete checklist)

**Pitch:**
- [ ] Green pitch rectangle with 4 px white border
- [ ] Centre circle (radius ≈ 9 m in world units → scale to px)
- [ ] Centre line
- [ ] Goal posts at both ends (width `GOAL_W = 12` m, depth 2 m)
- [ ] Penalty arcs or shooting zone arc (radius `D_SHOOT_MAX = 35` m from goal centre)
- [ ] Formation anchor positions as faint `+` markers (only when `show_anchors=True`)

**Players:**
- [ ] Circle for each player, radius ≈ 1.4 m (scale to px)
- [ ] Team colour fill (blue / red) with white stroke
- [ ] Bold stroke when holding ball
- [ ] Gold ring for playmaker
- [ ] Player number inside circle (white text)
- [ ] Frozen state: grey fill or striped pattern
- [ ] Slot label (D/M/F) as a tiny subscript near the player

**Vision radii (optional, toggle with `V` key):**
- [ ] Dashed circle around each player of team 0 (our side)
- [ ] Fill the vision circle with a transparent colour (alpha 10–20%)
- [ ] Overlap between our players makes combined coverage visible

**Ball:**
- [ ] Black circle, radius ≈ 0.8 m
- [ ] White pentagon texture (draw 5 curved patches) for realistic football look

**Pass / shot in flight:**
- [ ] Dotted line from passer to target while ball is in flight
- [ ] Ball moves along the line
- [ ] When a shot is in flight: draw a goal-line indicator (see Section 4)

**Events (flash animations, fade after 20 frames):**
- [ ] Goal: large flash overlay ("GOAL!" text, team colour)
- [ ] Tackle: small spark effect at tackle position
- [ ] Interception: yellow star at interception point
- [ ] Pass completed: brief green arc
- [ ] Turnover (pass intercepted): brief red arc

**HUD (top bar):**
- [ ] Score: "Our Team 2 – 1 Opponent"
- [ ] Step counter and time fraction
- [ ] Both formations
- [ ] Possession indicator (coloured dot)
- [ ] Playmaker indicator if sigma > 0

**Controls (keyboard):**
- [ ] `SPACE` — pause / resume
- [ ] `→` — step one frame (while paused)
- [ ] `+`/`-` — increase/decrease fps (1–60)
- [ ] `V` — toggle vision radius display
- [ ] `A` — toggle anchor display
- [ ] `F` — toggle formation overlay (ghost positions for both formations)
- [ ] `R` — restart from seed 0
- [ ] `Q` / `Esc` — quit

### 3.4 Pygame Main Loop Sketch

```python
import pygame, sys

def run_pygame(env, agent=None, fps=30):
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    paused = False
    show_vision = False

    obs, _ = env.reset(seed=0)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_v: show_vision = not show_vision
                # ... other keys

        if not paused and not env.done:
            actions = agent.act(obs) if agent else None
            obs, _, done, info = env.step(actions)
            draw(screen, env, info["events"], show_vision)

        clock.tick(fps)
        pygame.display.flip()
```

---

## 4. Shot Trajectory Visualization

When a shot is taken (`SHOOT` action), show the trajectory in real-time.

### 4.1 What to Draw

**On the pitch:**
1. A curved arc from the shooter to the goal (parabolic in 2.5D if using perspective, straight line in flat 2D).
2. An "aim indicator" cone showing the shot direction and spread. The half-angle of the cone corresponds to the shot miss spread parameter.
3. The ball moves along the arc at `V_PASS` speed.
4. If it scores: ball crosses the goal line, flash effect.
5. If it misses: ball veers off to a point near the goal line. Draw the deviation visually.

**Goal-mouth indicator (always visible when in shooting zone):**
- Draw the goal mouth as a vertical bar on the right side of the screen (or at the goal end) — a "goal target" showing where the ball would go based on current aim.
- Show a coloured segment indicating the predicted scoring probability (green = high p, red = low).
- This appears while any player of either team is in the shooting zone and holds the ball.

### 4.2 Probability Display

When the carrier is in the shoot zone, display:
```
Shoot prob: 0.62  ████████░░
```
Compute it: `shot_probability(pos, goal, visible_defenders, control, ...)` from `mechanics.py`.

### 4.3 Implementation (Pygame)

```python
from football.mechanics import shot_probability
from football.geometry import to_frame

def draw_shot_indicator(screen, env):
    if env.holder is None:
        return
    ht, hi = env.holder
    me_w = env.pos[ht][hi]
    me = to_frame(ht, me_w, env.L, env.W)
    goal = np.array([env.L, env.W / 2])
    d = dist(me, goal)
    if d > env.cfg.mechanics.shoot_max:
        return
    defs = [env.pos[1 - ht][j] for j in range(5)]
    p = shot_probability(me, goal, defs, env.control(ht, hi),
                         env.cfg.mechanics.p_max, env.cfg.mechanics.d0,
                         env.cfg.mechanics.shot_block_scale)
    # draw probability bar, aim cone, etc.
```

---

## 5. 3D Perspective Renderer

A full 3D renderer significantly improves visual communication, especially for presentations and demos. This section describes two feasible approaches.

### 5.1 Approach A: Pygame + Isometric Projection (Recommended — Pure Python)

Use a **2.5D isometric projection** rather than full 3D. This looks like a classic football management game (Football Manager, Sensible Soccer). It is achievable in pure Python/Pygame without a 3D engine.

**Projection formula:**

```python
def world_to_iso(wx, wy, wz=0.0, tile_w=20, tile_h=10):
    # wx, wy: pitch coords (0..L, 0..W); wz: height above pitch
    sx = (wx - wy) * (tile_w / 2)
    sy = (wx + wy) * (tile_h / 2) - wz * tile_h
    return int(SCREEN_CX + sx), int(SCREEN_CY + sy)
```

**What changes in 3D:**
- Pitch is a diamond shape (isometric rectangle)
- Goal posts have height (draw vertical lines + crossbar)
- The ball has a height component: `wz > 0` during flight (parabolic arc)
- Players cast shadows on the pitch
- Shot arc is a visible parabola above the pitch

**Ball height during flight:**
```python
def ball_height(flight_dict):
    k = flight_dict["k"]
    n = flight_dict["n"]
    t = k / n                        # 0 → 1 during flight
    return 4.0 * t * (1 - t) * MAX_HEIGHT   # parabola peaking at midpoint
```

Suggested `MAX_HEIGHT`: 3.0 m for passes, 5.0 m for shots.

**Draw order (painter's algorithm):**
1. Pitch (back to front in isometric order)
2. Anchor markers
3. Player shadows (flat ellipses on pitch)
4. Players (back row first, front row last to handle overlap correctly)
5. Ball
6. Pass/shot arc
7. HUD

### 5.2 Approach B: Three.js Web Renderer (Best Visual Quality)

Use a web server that streams game state as JSON and renders it in the browser with Three.js.

**Architecture:**
```
Python env  →  WebSocket server  →  Browser (Three.js scene)
```

**Setup:**
```bash
pip install websockets
# client: Three.js loaded from CDN in browser
```

**Python side** (emit state each step):
```python
import asyncio, json, websockets

async def broadcast_state(env, ws):
    state = {
        "t": env.t, "score": env.score,
        "players": [{"team": t, "i": i, "pos": env.pos[t][i].tolist(),
                     "has_ball": env.holder == (t, i)}
                    for t in (0,1) for i in range(5)],
        "ball": env.ball_pos.tolist(),
        "ball_state": env.ball_state,
        "sigma": env.sigma_idx,
    }
    await ws.send(json.dumps(state))
```

**Three.js scene** (browser side):
- Flat green pitch plane
- Cylinder players with team colour materials
- Sphere ball
- Actual 3D parabolic shot trajectory (TubeGeometry along a bezier curve)
- Spot lights casting player shadows
- Orbit controls for camera rotation (zoom in/out, rotate around pitch)
- Stats panel in HTML overlay

**Bezier arc for shot trajectory:**
```javascript
const start = new THREE.Vector3(ball.x, 0, ball.z);
const end   = new THREE.Vector3(goal.x, 0, goal.z);
const mid   = start.clone().add(end).multiplyScalar(0.5);
mid.y = 5.0;  // peak height
const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
const points = curve.getPoints(50);
const geometry = new THREE.BufferGeometry().setFromPoints(points);
```

**Goal post geometry (Three.js):**
```javascript
function makeGoalPost(x, W) {
    const group = new THREE.Group();
    const postMat = new THREE.MeshStandardMaterial({color: 0xffffff});
    const postGeo = new THREE.CylinderGeometry(0.12, 0.12, 3.0, 8);
    // left post, right post, crossbar
    const lp = new THREE.Mesh(postGeo, postMat);
    lp.position.set(x, 1.5, W/2 - 6);
    group.add(lp);
    // ... right post and crossbar similarly
    return group;
}
```

### 5.3 Approach C: Panda3D or Ursina (Pure Python 3D)

For a self-contained Python 3D renderer without a web browser:

```bash
pip install ursina
```

Ursina is simpler than Panda3D and suits rapid prototyping. Create `Entity` objects for pitch, players, and ball. Use `camera.position` and `camera.rotation` to set a top-down or angled view. Ball parabola via updating `ball.y` each frame.

This is the best option if you need a **standalone executable** with real 3D and no browser.

---

## 6. Best Practices

### 6.1 Decouple Physics from Rendering

The environment steps at whatever speed the training loop requires (potentially thousands of steps per second). The renderer should operate independently:

```python
# Good: renderer reads env state after step
env.step(actions)
renderer.step(env, events)

# Bad: renderer is called inside env.step()
```

For training, pass `renderer=None` and skip all rendering calls entirely.

### 6.2 Event-Driven Animations

The step returns `info["events"]` — a list of dicts like `{"type": "goal", "team": 0}`. Consume this list in the renderer to trigger one-shot animations. Don't derive events from state deltas.

```python
for ev in info["events"]:
    if ev["type"] == "goal":
        renderer.trigger_goal_flash(ev["team"])
    elif ev["type"] == "tackle":
        renderer.trigger_tackle_spark(ev.get("by"))
```

### 6.3 Camera Modes

Offer at least two camera modes:

| Mode | Description | Best for |
|---|---|---|
| Top-down orthographic | Flat 2D view, full pitch visible | Debugging, formation analysis |
| Isometric | 2.5D angled view, full pitch | Demo and presentation |
| Follow ball | Camera tracks ball, ¾ angle | Gameplay feel |
| Follow player | Camera tracks a specific player | Vision radius debugging |

Switch with a key (e.g. `C`).

### 6.4 Formation Overlay

A **ghost overlay** (toggle with `F`) shows the anchor positions of both formations as transparent player-shaped ghosts. This helps compare the current opponent positions against each possible formation — exactly what the belief filter is doing.

```python
def draw_formation_ghost(screen, env, formation_name, team, alpha=60):
    anchors = env.book.anchors_team_frame(formation_name)  # (5, 2) team frame
    for i, a_tf in enumerate(anchors):
        a_w = to_frame(team, a_tf, env.L, env.W)   # back to world
        sx, sy = world_to_screen(*a_w)
        ghost = pygame.Surface((20, 20), pygame.SRCALPHA)
        color = (*TEAM_COLORS[team], alpha)
        pygame.draw.circle(ghost, color, (10, 10), 10)
        screen.blit(ghost, (sx - 10, sy - 10))
```

### 6.5 Replay Mode

Save match state to a list of dicts during a match, then replay at any speed:

```python
replay_buffer = []
while not env.done:
    obs, r, done, info = env.step(actions)
    replay_buffer.append({
        "pos": env.pos.copy(),
        "ball": env.ball_pos.copy(),
        "holder": env.holder,
        "events": info["events"],
        "score": list(env.score),
        "t": env.t,
    })
# Then replay_buffer[t] gives the state at step t
```

### 6.6 Vision Coverage Map

A useful debug view: a heatmap overlay showing **how much of the pitch is currently visible** to our team. Build it once per step:

```python
import numpy as np

def vision_coverage(env, team=0, resolution=50):
    xs = np.linspace(0, env.L, resolution)
    ys = np.linspace(0, env.W, resolution)
    covered = np.zeros((resolution, resolution), bool)
    for i in range(5):
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                d = dist(env.pos[team][i], np.array([x, y]))
                if d <= env.vision_radius(team, i):
                    covered[yi, xi] = True
    return covered
```

Render as a green-tinted overlay. Red areas = blind spots.

### 6.7 Performance Notes

- For matplotlib: `ax.clear()` is slow. Use `set_data` on existing artists to update them instead of redrawing.
- For Pygame: draw the static pitch background to a Surface once on reset; blit it each frame instead of redrawing rectangles.
- For Three.js: update object positions via `object.position.set(...)` rather than creating new objects each frame.
- Vision coverage map is expensive at high resolution. Compute at 20×12 or 25×15 and upscale.

---

## 7. Implementation Roadmap

| Phase | What to build | Effort |
|---|---|---|
| 1 (immediate) | Improved matplotlib: pass trajectory dotted line, shot probability bar | 1 day |
| 2 | Pygame 2D: full checklist in Section 3, all events, keyboard controls | 2–3 days |
| 3 | Pygame isometric: 2.5D with ball height, shadow, goal posts | 3–5 days |
| 4 | Replay mode + formation overlay + vision heatmap | 1–2 days |
| 5 | Three.js 3D web renderer OR Ursina standalone | 3–5 days |

Start with Phase 2 (Pygame 2D). It covers all the essential visual communication and is the fastest to implement correctly. The isometric view (Phase 3) is worthwhile for the project report and demos.

---

## 8. File Locations

| File | Purpose |
|---|---|
| `football_adapt/scripts/live_render.py` | Current matplotlib renderer |
| `football_adapt/scripts/watch_match.py` | Skeleton for replay/watch script |
| *(to create)* `football_adapt/scripts/pygame_render.py` | Pygame 2D renderer |
| *(to create)* `football_adapt/scripts/iso_render.py` | Isometric renderer |
| *(to create)* `football_adapt/scripts/web_render/` | Three.js renderer (server + HTML) |
| *(to create)* `football_adapt/rendering/` | Shared rendering utilities (coordinate transforms, colours, event flash manager) |
