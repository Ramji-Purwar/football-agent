# opponent.md: The Scripted Opponent Team

This document covers the rule-based opponent team (`ScriptedTeam` in `football_adapt/opponent/controller.py`). Read `game.md` first — the opponent plays by the same rules and physics, and all calibrated constants are listed there.

**Also usable for our team:** `ScriptedTeam` can drive our side too (set `scripted_ours=True` in `FootballEnv`). This makes the Phase 0 heuristic-vs-heuristic baseline straightforward.

---

## 1. Purpose and Design Constraints

The opponent is a **fixed, hand-written team that does not learn**. It must satisfy:

| Requirement | Why |
|---|---|
| Holds its formation (players near anchors) | Q2 requires the formation to be readable from player positions |
| Vision-limited (each player sees only within `R_VISION`) | Prevents information advantage that would break Q1/Q2 |
| Can switch formation mid-match (3 modes) | Q2 |
| One player can be a playmaker (infinite vision) | Q3 |
| Worthwhile opponent (random policy loses clearly) | Learning results must be meaningful |
| Reproducible (same seed → same match) | Paired evaluation and debugging |
| Same physics as our team (no teleporting) | Fairness |

**Must NOT be:** man-marking (pulls players from anchors, hides formation), learning, or omniscient (except the playmaker).

---

## 2. Inputs and Outputs

- **Input to each player:** the `Obs` dataclass built by `football.observation.build_observation()` — the **same function** used for our players, called with that player's own vision radius.
- **Output:** one `Action` per player per step.
- **Team frame:** the environment mirrors coordinates before the opponent receives them. The opponent code runs as if attacking toward +x, identical to our team code.
- **Control rating:** the opponent always uses structure level `zeta=2`. Effectiveness drops when a player is far from its anchor.

**Critical rule:** the opponent controller is called with the `obs_by_player` dict only — never with the full environment state.

---

## 3. Architecture

```
ScriptedTeam
 ├─ FormationManager     — decides the current formation g_t
 └─ acts on obs dict     — runs priority list for each player
```

**Priority list** per player (first applicable rule wins):

```
1. Transition walk    (just switched formation, not near new anchor, no ball)
2. Has ball           → attack routine (Section 6)
3. Teammate has ball  → off-ball support (Section 7)
4. Ball is loose      → loose-ball rule (Section 5.3)
5. Opponent holds ball → zonal defence (Section 5)
```

---

## 4. The Formation Manager (`opponent/formation_manager.py`)

### 4.1 State

| Variable | Meaning |
|---|---|
| `formation` | Current formation name string |
| `dwell` | Steps since last switch (starts at `10^9` so first switch is never blocked) |
| `in_transition` | Whether players are currently walking to new anchors (tracked in `ScriptedTeam`) |
| `transition_steps` | Steps elapsed since the last switch |

### 4.2 Switching Modes

Set via `opponent.switching.mode` in the config. The minimum dwell between switches is `dwell_min = 150` steps.

| Mode | Rule |
|---|---|
| `NONE` | Formation never changes (default; used in Q1, Q3) |
| `SCHEDULED` | Switch once at step `t_switch` (default 600) to `target` formation (default `"1-2-2"`) |
| `RANDOM` | After `dwell_min`, switch each step with probability `h_switch` (default 0.005) to a random other formation |
| `SCORE_REACTIVE` | Every `check_interval` (50) steps: if trailing by `>= trail_thresh` (1) → switch to `f_att`; if leading by `>= lead_thresh` (1) → switch to `f_def`; otherwise stay |

Default `score_reactive` config: `f_att = "1-2-2"`, `f_def = "3-1-1"`.

### 4.3 What Happens at a Switch

1. `FormationManager.update()` returns the new formation name.
2. `ScriptedTeam` sets `in_transition = True`, `transition_steps = 0`.
3. Players **walk** to new anchors at speed `V_PLAYER`. No teleporting.
4. Transition ends when all players are within `eps_arrive = 1.0` m of their anchors, or after `transition_max = 60` steps.
5. A switch event is logged: `{t, type="switch", team, from_, to}`.

During transition, players are far from new anchors → control rating is low → natural cost of switching, and the visible signal our team can detect.

---

## 5. Defence: Zonal

### 5.1 Zonal Target Rule

Player `j` at position `p`, anchor `a`, zone radius `R_ZONE = 12.0`:

```
if ball visible AND ||ball - anchor|| <= R_ZONE AND I am the presser:
    target = ball                                     # press
else if ball visible:
    target = anchor + OMEGA * clip(ball - anchor, R_ZONE)  # small ball-side drift
else:
    target = anchor                                   # hold home position
```

`OMEGA = 0.4` keeps the drift to 40% of the zone radius at most.

### 5.2 Pressing Arbitration

Only **one presser** per step. Among players whose zone contains the ball **and** who see the ball, the one **nearest to the ball** is the presser (ties broken by player index).

### 5.3 Loose Ball

If I see the ball and I am the **nearest** team player to it: `GO_TO_BALL`. Otherwise use the zonal rule.

### 5.4 Tackling

If I am the presser and the opponent carrier is within `D_TACKLE = 2.0` m: `TACKLE`.

### 5.5 Interception (not a decision)

Resolved automatically by pass geometry in `mechanics.pass_interception()`. A player is eligible only if the ball was within its vision radius when the pass was played. The playmaker is eligible for every pass.

---

## 6. Attack: Ball Carrier

Decision order (first that succeeds wins):

### 6.1 Shoot

If within `D_SHOOT_MAX = 35.0` m of goal: estimate `p = shot_probability(pos, goal, visible_defenders, ...)`. If `p >= shoot_min = 0.3`: `SHOOT`.

### 6.2 Pass

For each of 4 teammates `k`, compute a pass score:

```
score(k) = w_fwd * forward_gain(k)
         + w_open * openness(k)
         - w_dist * distance_to_k / PITCH_L
         + hub_bonus if k is the playmaker
```

Parameters: `w_fwd=2.0`, `w_open=0.5`, `w_dist=0.3`, `hub_bonus=0.3`, `open_max=15.0`.

- `forward_gain(k)`: teammate `k`'s x-position minus carrier's (normalized). Can be negative.
- `openness(k)`: min distance from `k` to nearest **visible** defender, clipped to `open_max`, then divided by `open_max`.

A pass is **allowed** only if `pass_interception()` finds no visible defender able to intercept.

Choose the allowed pass with highest score if that score `>= pass_min = 0.3`.

### 6.3 Dribble

If no visible defender is within `d_danger = 6.0` m ahead of the carrier: `DRIBBLE_GOAL`.

### 6.4 Recycle

Otherwise: pass to the **safest** allowed teammate (highest `openness`, ignoring `pass_min` and `w_fwd`). If no safe pass: `STAY`.

---

## 7. Off-Ball Attacking Support

When a teammate has the ball:

```
target = anchor + shift
```

| Slot type | Shift |
|---|---|
| F (forward) | toward opponent goal by up to `adv = 8.0` m |
| M (midfielder) | toward ball carrier by up to `mid = 6.0` m |
| D (defender) | toward opponent goal by `def = 3.0` m |

`shift` length is capped at `r_support = 10.0` m. Then `MOVE_TO(target)`.

---

## 8. The Playmaker Variant (Q3)

When `sigma > 0`, opponent player at index `sigma_idx = sigma - 1` (0-based) is the playmaker.

| Aspect | Normal player | Playmaker |
|---|---|---|
| Vision radius | `R_VISION = 25.0` | infinity |
| Zonal drift | only when ball inside radius | always (ball always visible) |
| Pass safety check | only visible defenders | all defenders |
| Interception eligibility | only if ball was inside radius | all passes |
| Hub bonus | none | teammates add `hub_bonus = 0.3` to pass score when passing to him |

Everything else (anchor, zone, speed, control rating) is identical. The advantage is **information only**.

The playmaker is rendered with a gold ring in the matplotlib renderer.

---

## 9. Code Reference

### Key Files

| File | Role |
|---|---|
| `opponent/controller.py` | `ScriptedTeam` class — full decision logic |
| `opponent/formation_manager.py` | `FormationManager` class — switching logic |
| `opponent/__init__.py` | Exports |

### How the Environment Calls It

```python
# Inside FootballEnv.reset():
self.controllers[1] = ScriptedTeam(cfg, rng, team=1, formation=..., playmaker=sigma_idx)

# Inside FootballEnv.step():
raw[1] = self.controllers[1].act(self.t, obs[1])  # obs[1] is dict {i: Obs}
new_f = self.controllers[1].update_formation(self.t, score[1], score[0])
```

`update_formation()` must be called **after** the step resolves goals, so the score it sees is already updated.

### Driving Our Team with the Scripted Controller

```python
env = FootballEnv(cfg, scripted_ours=True)
# Both teams are scripted. Useful for Phase 0 heuristic-vs-heuristic matches.
obs, info = env.reset(seed=42)
while not env.done:
    obs, r, done, info = env.step()  # no our_actions needed
```

---

## 10. Logging

All events are appended to `env.event_log` and returned in `info["events"]` each step.

| Event type | Fields | Used for |
|---|---|---|
| `"pass"` | passer, receiver, predicted | Behaviour stats, Q3 analysis |
| `"pass_result"` | passer, result, by | Completion/interception rates |
| `"tackle"` | by, from_, success | Behaviour stats |
| `"shot"` | by, p, scored | Behaviour stats |
| `"goal"` | team, score | Score tracking |
| `"pickup"` | by | Loose ball analysis |
| `"switch"` | team, from_, to, t | Detection delay (Q2) |

Stats accumulated per match are in `env.stats` (passes, completions, tackles, shots, goals, possession steps, shadow steps, etc.). Also `env.opp_anchor_dist` tracks mean opponent anchor distance each step for Phase 0 check A.

---

## 11. Phase 0 Checks

Run from `scripts/phase0_checks.py`. Must all pass before training.

| Check | Method | Pass condition |
|---|---|---|
| A. Formation readability | Plot `env.opp_anchor_dist` over a match | Mean dist << inter-formation anchor dist |
| B. Vision matters | Heuristic R=25 vs same team R=12 over 100 matches | R=25 team wins clearly |
| C. Playmaker works | Heuristic+playmaker vs heuristic over 100 matches | Playmaker team wins clearly |
| D. Balance | Heuristic vs heuristic over 100 matches | ≈ 50% win rate each side |
| E. Determinism | Same seed and config, two runs | Identical event logs |
| F. Formation win matrix | All 5×5 formation pairs, 100 matches each | Matrix logged, no formation wins 100% |

**Calibration order** if checks fail: zone params (`R_ZONE`, `OMEGA`) first → attack params → support params → playmaker (`hub_bonus`).

---

## 12. Common Mistakes

- **Man-marking by accident:** target must always be a point defined by `anchor` and `ball`, never by one of our player positions.
- **Using true ball position when ball is invisible:** pass `None` for ball to the zonal rule when `ball_visible` is False.
- **Multiple pressers at once:** the arbitration in `_team_context()` must run before any player decisions.
- **Opponent seeing more than its radius:** always use `o.opp_visible[k]` to gate what the opponent knows about our players.
- **Instant formation switches:** players walk to new anchors; the environment sets new anchors but players move at `V_PLAYER`.
- **Mixing random streams:** `ScriptedTeam` takes its own `rng` (a `np.random.default_rng`) seeded independently from the environment and policy.
- **Hardcoding constants:** always read from `cfg.*`.

---

## 13. Summary

The scripted opponent holds its formation via zonal defence (one presser allowed; everyone else drifts slightly toward the ball). In attack the carrier shoots, passes to the best safe teammate, dribbles, or recycles, using only what it can see. Off-ball teammates make small capped runs. The formation can switch mid-match in three ways — players walk to new anchors. One player can be a playmaker with full-pitch vision whose pass decisions and interception eligibility cover the whole pitch, and who attracts extra passes from teammates via a hub bonus. The controller is called with observation dictionaries only, never with the full environment state.
