# codebase.md: Codebase Reference

Complete map of every file, module, and public API in the project. An AI agent should read this document to understand where things live and how to import them before reading or editing any code.

---

## 1. Repository Layout

```
football-agent/
├── docs/                          # Project documentation (start here)
│   ├── game.md                    # Full game specification with calibrated values
│   ├── opponent.md                # Scripted opponent team reference
│   ├── rendering.md               # Rendering guide (current + advanced)
│   ├── codebase.md                # This file
│   ├── training.md                # Training protocol and MAPPO setup
│   ├── agents.md                  # Agent variants reference
│   └── experiments.md             # Experiment conditions and evaluation protocol
│
├── football_adapt/                # Main Python package
│   ├── football/                  # Core game engine (no ML dependencies)
│   │   ├── __init__.py            # Re-exports: FootballEnv, load_config, ActionSpace
│   │   ├── env.py                 # FootballEnv: the 5-a-side game
│   │   ├── actions.py             # Action, ActionSpace, Kind enum
│   │   ├── mechanics.py           # Pure physics functions (shared by env + opponent)
│   │   ├── observation.py         # build_observation(), flatten(), global_state()
│   │   ├── formations.py          # FormationBook, Formation, slot_types_from_name()
│   │   ├── geometry.py            # dist(), unit(), clip_norm(), move_toward(), to_frame()
│   │   └── config.py              # load_config(), Cfg wrapper
│   │
│   ├── opponent/                  # Scripted opponent team
│   │   ├── __init__.py
│   │   ├── controller.py          # ScriptedTeam: full heuristic decision logic
│   │   └── formation_manager.py   # FormationManager: switching logic
│   │
│   ├── agents/                    # Learning agent implementations
│   │   ├── __init__.py            # Exports: RandomAgent
│   │   ├── random_agent.py        # RandomAgent: uniform random over valid actions
│   │   ├── belief_filter.py       # BeliefFilter: Bayesian formation inference (skeleton)
│   │   └── mappo.py               # MAPPO actor/critic (skeleton)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py               # Training loop (skeleton)
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_smoke.py          # Smoke tests
│   │
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── live_render.py         # Matplotlib live renderer
│   │   ├── phase0_checks.py       # Phase 0 calibration checks
│   │   └── watch_match.py         # Match replay script (skeleton)
│   │
│   ├── configs/
│   │   └── default.yaml           # All calibrated parameters (source of truth)
│   │
│   └── requirements.txt           # Python dependencies
│
├── default.yaml                   # (legacy copy at root — use configs/default.yaml)
├── controller.py                  # (legacy — see football_adapt/opponent/controller.py)
├── env.py                         # (legacy — see football_adapt/football/env.py)
└── README.md
```

**Always use the `football_adapt/` package.** The root-level `.py` files are legacy and may be out of date.

---

## 2. Module Reference: `football/`

### `football/env.py` — `FootballEnv`

The main simulation. Stateful. One instance per episode.

```python
from football import FootballEnv, load_config

env = FootballEnv(cfg=None, scripted_ours=False)
```

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `cfg` | `Cfg` or `None` | `None` → loads `default.yaml` | Config object |
| `scripted_ours` | `bool` | `False` | If True, team 0 is also driven by ScriptedTeam |

**Key methods:**

```python
obs, info = env.reset(
    seed=None,              # int or None (random)
    our_formation=None,     # str, e.g. "2-2-1" (default from cfg)
    opp_formation=None,     # str
    sigma=None,             # int 0–5 (0 = no playmaker)
    zeta=None,              # int 0–2
    radius=None,            # float, our team's vision radius
    radius_opp=None,        # float, opponent's vision radius
    start_team=None,        # int 0 or 1, who starts with ball
    our_style=None,         # str, play style: "normal" | "aggressive" | "defensive" (default from cfg)
    opp_style=None,         # str
)
# Returns:
#   obs: {"vec": (5, 79), "mask": (5, 23), "state": (51,)}
#   info: see below

obs, reward, done, info = env.step(
    our_actions=None        # dict {i: int} or None (if scripted_ours=True)
)
# Returns:
#   obs: same structure as reset
#   reward: float (team reward)
#   done: bool
#   info: dict (see below)
```

**`info` dict keys:**

| Key | Type | Content |
|---|---|---|
| `t` | int | Current step |
| `score` | tuple | (team0_goals, team1_goals) |
| `g` | str | Opponent's current formation |
| `f` | str | Our formation |
| `sigma` | int | Playmaker player number (1–5) or 0 |
| `events` | list | Events this step (pass, tackle, shot, goal, switch, pickup) |
| `switch` | dict or None | Switch event if one happened this step |
| `reward_parts` | dict | {"team": float, "shaping": float, "terminal": float} |
| `opp_anchor_dist` | float | Mean distance of opponent players from anchors (for Phase 0 check) |
| `final_score` | tuple or None | Final score when `done=True` |

**Public attributes read by the renderer:**

```python
env.pos          # np.ndarray (2, 5, 2) world positions
env.anchors      # np.ndarray (2, 5, 2) world anchor positions
env.ball_pos     # np.ndarray (2,)
env.ball_state   # str: "held", "loose", "flight"
env.flight       # dict or None: {start, end, n, k, passer, designated, intended, predicted}
env.holder       # tuple (team, i) or None
env.score        # list [int, int]
env.t            # int
env.done         # bool
env.sigma_idx    # int (0-based) or None
env.formation    # list [str, str]
env.frozen       # np.ndarray (2, 5) int — steps remaining frozen
env.cfg          # Cfg object
env.L, env.W     # float: pitch dimensions
env.space        # ActionSpace
env.book         # FormationBook
env.event_log    # list of all events this episode
env.stats        # dict of per-episode statistics
env.opp_anchor_dist  # list of per-step mean anchor distances
```

**Key methods:**
```python
env.vision_radius(team, idx)  # float, inf for playmaker
env.control(team, idx)        # float, control rating 0..1
env.set_shaping_weight(w)     # anneal lambda during training
```

---

### `football/actions.py` — `ActionSpace`, `Action`, `Kind`

```python
from football.actions import ActionSpace, Action, Kind, STAY

space = ActionSpace(n_directions=8)
space.n         # 23 (total actions)
space.D         # 8 (MOVE_DIR count)
space.decode(idx)           # int → Action
space.index_of(action)      # Action → int or None (for MOVE_TO macros)
space.index_of_kind(kind)   # Kind → first matching index

# Action kinds:
Kind.MOVE_DIR   # arg = direction index 0..7
Kind.STAY
Kind.HOLD_SHAPE
Kind.GO_TO_BALL
Kind.PASS       # arg = teammate index 0..3
Kind.DRIBBLE_GOAL
Kind.SHOOT
Kind.TACKLE
Kind.SHADOW     # arg = opponent index 0..4
Kind.MOVE_TO    # arg = np.array([x, y]) in TEAM FRAME (macro, not in learner's set)
```

**Indexed action layout** (with D=8):
```
Indices 0–7:   MOVE_DIR (8 directions, evenly spaced 0..2π)
Index  8:      STAY
Index  9:      HOLD_SHAPE
Index 10:      GO_TO_BALL
Index 11–14:   PASS_0, PASS_1, PASS_2, PASS_3 (teammates in ascending index order)
Index 15:      DRIBBLE_GOAL
Index 16:      SHOOT
Index 17:      TACKLE
Index 18–22:   SHADOW_0, SHADOW_1, SHADOW_2, SHADOW_3, SHADOW_4
```

---

### `football/mechanics.py` — Pure Physics

```python
from football.mechanics import control_rating, shot_probability, pass_interception, tackle_probability

control_rating(pos, anchor, zeta, c_min, l_ctrl)
# → float [c_min, 1.0]

shot_probability(shooter, goal_center, defenders, c, p_max, d0, scale)
# defenders: list of position arrays
# → float [0, p_max]

pass_interception(passer, target, defenders, v_pass, v_player, margin)
# defenders: list of (key, position) tuples — key is the player index
# → (key, q, slack) of best interceptor, or None

tackle_probability(c_def, c_carrier)
# → float [0, 1]
```

**These are pure functions** — no state, no randomness. They can be called by both the environment and the opponent controller for identical physics.

---

### `football/observation.py` — Observation Building

```python
from football.observation import build_observation, flatten, global_state, flat_dim, Obs
from football.observation import OWN, OTHER, LOOSE  # possession constants: 0, 1, 2

obs = build_observation(env, team, idx)   # → Obs dataclass
vec = flatten(env, obs)                   # → np.ndarray (79,)
state = global_state(env)                 # → np.ndarray (51,)
dim = flat_dim(env)                       # → int (79)
```

**`Obs` dataclass fields:**

```python
obs.team          # int 0 or 1
obs.idx           # int 0–4 (0-based player index)
obs.pos           # (2,) team frame
obs.anchor        # (2,) team frame, or None if zeta=0
obs.control       # float
obs.has_ball      # bool
obs.frozen        # bool
obs.teammate_ids  # list of 4 ints (ascending, self excluded)
obs.teammate_pos  # (4, 2) team frame, relative-to-player NOT applied here (absolute)
obs.teammate_has_ball  # (4,) bool
obs.opp_pos       # (5, 2) team frame, zeros where not visible
obs.opp_visible   # (5,) bool
obs.ball_pos      # (2,) team frame, zeros if not visible
obs.ball_visible  # bool
obs.possession    # int: OWN=0, OTHER=1, LOOSE=2
obs.score_diff    # int (own - other)
obs.t             # int
obs.t_frac        # float
obs.formation     # str
obs.slot_type     # str: "D", "M", or "F"
obs.radius        # float (vision radius for this player)
obs.mask          # (23,) bool — valid actions
obs.prev_action   # int (-1 if none/macro)
```

**Note on positions in `flatten()`:** teammate and opponent positions in the flat vector are **relative to the player** (i.e. `obs.teammate_pos[k] - obs.pos`) and **normalized by pitch size**. In the raw `Obs`, positions are absolute (team frame).

---

### `football/formations.py` — Formations

```python
from football.formations import FormationBook, Formation, slot_types_from_name

book = FormationBook(cfg)
book["2-2-1"]                    # → Formation dataclass
book.names                       # list of formation name strings
book.index("2-2-1")              # → int (0-based index in names list)
book.anchors_team_frame("2-2-1") # → (5, 2) array in world units, team frame
book.kickoff_slot("2-2-1")       # → int (0-based index of first forward)

slot_types_from_name("2-2-1")    # → ("D", "D", "M", "M", "F")
```

---

### `football/geometry.py` — Geometric Utilities

```python
from football.geometry import dist, unit, clip_norm, move_toward, rotate, to_frame, dir_to_frame

dist(a, b)                       # Euclidean distance between 2D points
unit(v)                          # Unit vector (returns zero if |v| == 0)
clip_norm(v, max_len)            # Clip vector to max_len
move_toward(pos, target, speed)  # New position moving toward target at most `speed`
rotate(v, angle)                 # Rotate 2D vector by angle (radians)

to_frame(team, pos_world, L, W)  # World → team frame (mirrors x for team 1)
dir_to_frame(team, dir_world)    # Direction world → team frame (mirrors x only for team 1)
```

**Team frame convention:**
- Team 0: team frame = world frame (no transform)
- Team 1: team frame mirrors the x-axis: `x_team = L - x_world`, `y_team = y_world`
- Both teams attack toward +x in their own team frame

---

### `football/config.py` — Configuration

```python
from football import load_config

cfg = load_config()                          # loads configs/default.yaml
cfg = load_config(path="path/to/custom.yaml")
cfg = load_config(overrides={"vision.radius": 15.0, "time.T": 600})

cfg.pitch.length         # 100.0
cfg.opponent.zone.r_zone # 12.0
cfg["pass"]              # also works for keys that shadow Python builtins
cfg.to_dict()            # → raw dict (deep copy)
```

---

## 3. Module Reference: `opponent/`

### `opponent/controller.py` — `ScriptedTeam`

```python
from opponent.controller import ScriptedTeam

team = ScriptedTeam(cfg, rng, team=1, formation="2-2-1", playmaker=None, mode_override=None)
# rng: np.random.default_rng() with its own seed
# team: 0 or 1
# playmaker: 0-based index of the playmaker player, or None
# mode_override: override cfg.opponent.switching.mode (e.g. "NONE" to disable switching)

actions = team.act(t, obs_by_player)          # {i: Action}, obs_by_player: {i: Obs}
new_f = team.update_formation(t, own_goals, other_goals)  # str or None
```

### `opponent/formation_manager.py` — `FormationManager`

```python
from opponent.formation_manager import FormationManager

fm = FormationManager(cfg, rng, formation="2-2-1", formation_names=book.names, mode_override=None)
new_f = fm.update(t, own_goals, other_goals)  # str or None; increments dwell
fm.reset("1-3-1")                             # reset to a new formation
fm.formation                                  # current formation name
```

---

## 4. Module Reference: `agents/`

### `agents/random_agent.py` — `RandomAgent`

```python
from agents import RandomAgent

agent = RandomAgent(seed=42)
actions = agent.act(obs)
# obs: {"vec": (5, 79), "mask": (5, 23), "state": (51,)}
# Returns: {i: int} for i in 0..4 — uniform random over valid (masked) actions
```

### `agents/belief_filter.py` — `BeliefFilter` (skeleton)

```python
from agents.belief_filter import BeliefFilter

bf = BeliefFilter(cfg, n_players=5)
bf.reset()
belief = bf.update(obs_by_player)
# belief: np.ndarray (5,) — probability over 5 formations (same order as book.names)
# bf.argmax() → formation name string
```

The belief filter is documented in full in `agents.md`.

### `agents/mappo.py` — MAPPO (skeleton)

See `training.md` for the full training protocol and implementation guide.

---

## 5. Module Reference: `scripts/`

### `scripts/live_render.py` — Matplotlib Renderer

```bash
cd football_adapt
python -m scripts.live_render --mode scripted --T 400 --fps 10 --seed 0
python -m scripts.live_render --mode random --sigma 3 --opp-mode SCHEDULED
```

### `scripts/phase0_checks.py` — Phase 0 Calibration

```bash
python -m scripts.phase0_checks
```

Runs checks A–F from `game.md` Section 14. Prints pass/fail and logs results to `phase0_results.json`.

---

## 6. Import Paths

All imports should use absolute paths from within `football_adapt/`. When running scripts, `cd football_adapt` first, or add `football_adapt/` to `PYTHONPATH`.

```python
# Core imports
from football import FootballEnv, load_config
from football.actions import ActionSpace, Action, Kind, STAY
from football.mechanics import control_rating, shot_probability, pass_interception
from football.observation import build_observation, flatten, global_state, OWN, OTHER, LOOSE
from football.formations import FormationBook
from football.geometry import dist, to_frame

# Opponent
from opponent.controller import ScriptedTeam
from opponent.formation_manager import FormationManager

# Agents
from agents import RandomAgent
from agents.belief_filter import BeliefFilter
```

The environment imports `ScriptedTeam` lazily (`from opponent.controller import ScriptedTeam`) inside `env.reset()` to avoid circular imports.

---

## 7. Data Flow: One Step

```
env.step(our_actions)
│
├─ 1. build_observation(env, team, i) × 10          # observation.py
│       └─ compute_mask(env, team, i, partial)
│
├─ 2. ScriptedTeam.act(t, obs[1])                   # opponent/controller.py
│       └─ FormationManager.update(t, ...)           # opponent/formation_manager.py
│
├─ 3. Resolve tackles                               # env.py using mechanics.tackle_probability
│
├─ 4. Resolve pass/shot                             # env.py using mechanics.*
│       ├─ _pass: mechanics.pass_interception + error
│       └─ _shoot: mechanics.shot_probability
│
├─ 5. Move players: _target() → move_toward()       # geometry.move_toward
│
├─ 6. Advance flight, _pickup                       # env.py
│
├─ 7. Goal check, kickoff
│
├─ 8. update_formation (opponent)
│
└─ 9. Reward, logging, return (obs_out, reward, done, info)
```

---

## 8. Random Streams

Each run uses **three independent seeded streams**:

| Stream | Created by | Used for |
|---|---|---|
| Environment | `np.random.default_rng(s_env)` from `SeedSequence(seed)` | Tackle resolution, shot outcomes, pass error, loose ball pickup ties |
| Opponent | `np.random.default_rng(s_opp)` | Scripted team's random actions (`eps_opp`) and RANDOM switching |
| Our team | `np.random.default_rng(s_ours)` | If `scripted_ours=True`, same stream as opponent but independently derived |

All three are derived from the episode seed via `np.random.SeedSequence(seed).spawn(3)`. Pass `seed=42` to `env.reset()` for reproducibility.

---

## 9. Configuration System

All tunable parameters live in `configs/default.yaml`. Override any value:

```python
cfg = load_config(overrides={
    "vision.radius": 12.0,
    "time.T": 600,
    "opponent.switching.mode": "SCHEDULED",
    "opponent.playmaker.sigma": 3,
})
env = FootballEnv(cfg)
```

Dotted key notation (`"a.b.c"`) sets nested YAML values. The `Cfg` wrapper allows attribute-style access at any depth. Use `cfg.to_dict()` to serialize back to a dict for logging.

**Play styles.** `styles.presets.<name>` is a partial config that is merged over the config for one team (`apply_style(cfg, name)` returns that team's view). The environment keeps one view per team in `env.team_cfg` and hands it to that team's `ScriptedTeam`. A preset may only set `opponent.*` and `formations.phases.*`; unknown keys and anything else are rejected when the config is loaded. See `opponent.md`, Section 8.

---

## 10. Testing

```bash
cd football_adapt
python -m pytest tests/
```

`tests/test_smoke.py` checks:
- Environment resets without error
- One full episode runs to completion
- Observation shapes and mask shapes are correct
- Scripted vs scripted produces a non-zero score
- Same seed gives identical results (determinism)
