# game.md: The 5-a-Side Football Game and the Learning Agent

This document explains the whole project in detail: what the game is, how every mechanic works, what the learning agent sees and does, what we vary in experiments, and what we measure. Read this first. The companion file `opponent.md` covers the rule-based opponent team. See `codebase.md` for the code map and `training.md` for the training protocol.

**Parameter status:** all numeric constants are now filled in from `configs/default.yaml`. The source of truth is always the config file; this document reflects the current calibrated defaults.

---

## 1. The Big Picture

We build a small 2D football game and train a team of AI players to play it.

- Two teams of **5 outfield players**. No goalkeepers.
- **Our team** is controlled by **one shared neural network** (parameter-shared MAPPO). All five players run the same network, each on its own local view.
- The **opponent team** is not learned. It follows hand-written rules (see `opponent.md`).
- Every player has **limited vision**: it sees opponents and the ball only within a radius around itself.
- Teams play in **formations** (e.g. 2-2-1). The opponent may **change formation mid-match**, and one opponent player can be a **playmaker** who sees the whole pitch.

**Project theme: structure versus information.** When players cannot see everything, how much can a formation (structure) compensate, how well can the team infer what the opponent is doing, and can it adapt to an opponent with extra information?

### 1.1 The Three Research Questions

| # | Question | What happens in the game |
|---|---|---|
| **Q1** | Can formation structure compensate for limited vision? | Three structure levels tested at several vision radii. Does structure help more when vision is small? Does the best formation change with vision? |
| **Q2** | How fast can the team infer an opponent formation switch, and how much does vision matter? | Opponent switches mid-match unannounced. Compare memoryless, recurrent, belief-filter, and oracle agents |
| **Q3** | Can the team adapt to an opponent playmaker? | One opponent sees the whole pitch. Our team is not told who. Measure cost, recovery, and identification |

### 1.2 Known and Unknown to Our Team

| Known | Unknown |
|---|---|
| Rules of the game | Opponent's current formation (must be inferred) |
| Our own formation and anchors | When the opponent switches formation |
| All 4 teammates' positions (always) | Whether the opponent has a playmaker, and who |
| Score and possession state | How the opponent decides |
| Opponent players inside our vision radius | Opponent players outside our vision radius |

**Oracle variants** are given the hidden information and exist only as upper-bound comparisons.

---

## 2. Pitch, Coordinates and Time

| Parameter | Value | Notes |
|---|---|---|
| `PITCH_L` | 100.0 m | Length axis (+x direction) |
| `PITCH_W` | 60.0 m | Width axis (+y direction) |
| `GOAL_W` | 12.0 m | Goal mouth, centered on each end line |
| `T` | 1200 steps | Match length |
| `T_HALF` | 600 steps | Halftime marker (for scheduled switches only; no side swap) |

- **Pitch:** a bounded 2D rectangle. The entire game is played in a single 2D plane — every entity (players, ball, passes, shots) has only an `(x, y)` position. There is no height (z) component anywhere in the physics, observations, or mechanics. Passes and shots travel along the ground.
- **Goals:** one at each end of the length axis, width `GOAL_W`, centered. No goalkeeper.
- **Team frame:** each team always attacks toward **+x in its own frame**. The environment stores world coordinates and calls `geometry.to_frame()` before giving a player its observation. The opponent controller receives mirrored coordinates, so its code is identical to ours. No halftime side swap is needed.
- **One step:** all 10 players choose one action simultaneously, then the world updates once.
- **Outcome:** after `T` steps, more goals wins. Win = 1, draw = 0.5, loss = 0.

---

## 3. True Game State

What the environment stores at step `t`:

| Variable | Meaning |
|---|---|
| `pos[team][i]` | World position of player `i` on `team` (both 0-indexed) |
| `ball_pos` | Ball position (world frame) |
| `holder` | `(team, i)` tuple, or `None` if loose or in flight |
| `ball_state` | `"held"`, `"loose"`, or `"flight"` |
| `flight` | Flight state dict while ball is in the air |
| `formation[team]` | Formation name string (e.g. `"2-2-1"`) |
| `anchors[team][i]` | World position of player `i`'s anchor |
| `sigma_idx` | 0-based index of the playmaker, or `None` |
| `score[team]` | Goals scored |
| `t` | Current step |
| `frozen[team][i]` | Steps remaining in tackle recovery (0 = free) |

Players are 0-indexed in code (`i = 0..4`). Opponent numbers are consistent per player index, so our agents can compare an opponent's position against the anchor of each possible formation slot.

---

## 4. Formations

A formation gives each of the 5 players a **slot type** (D, M, or F) and a **home position** (anchor). Slot order (index 0 to 4) is always D-first, then M, then F.

| Formation | Slot types (indices 0–4) | Anchor fractions (team frame, [x, y]) |
|---|---|---|
| `2-2-1` | D, D, M, M, F | [0.25,0.30], [0.25,0.70], [0.50,0.30], [0.50,0.70], [0.75,0.50] |
| `2-1-2` | D, D, M, F, F | [0.25,0.30], [0.25,0.70], [0.50,0.50], [0.75,0.30], [0.75,0.70] |
| `1-3-1` | D, M, M, M, F | [0.25,0.50], [0.50,0.20], [0.50,0.50], [0.50,0.80], [0.75,0.50] |
| `1-2-2` | D, M, M, F, F | [0.25,0.50], [0.50,0.30], [0.50,0.70], [0.75,0.30], [0.75,0.70] |
| `3-1-1` | D, D, D, M, F | [0.25,0.20], [0.25,0.50], [0.25,0.80], [0.50,0.50], [0.75,0.50] |

Anchor fractions are multiplied by `[PITCH_L, PITCH_W]` to get world units. The first forward slot (the forward with the lowest index) takes the kickoff.

- **Formations are not enforced.** A player may leave its anchor. How much this costs depends on structure level (Section 5).
- **Kickoff slot:** the first `"F"` slot index in the formation's slot list.
- **Formation switch (opponent only):** anchors update instantly but players **walk** to new anchors at normal speed. Nobody teleports.
- **After a goal:** all players return to anchors in the current formation for kickoff.

---

## 5. Structure Levels and the Control Rating (Q1 knob)

For **our team** only:

| `zeta` | Name | Effect |
|---|---|---|
| 0 | No structure | Formation only sets kickoff positions. No anchor in observation, no `HOLD_SHAPE` action |
| 1 | Structure as information | Agent sees its anchor and can use `HOLD_SHAPE`. Leaving the anchor is free |
| 2 | Structure as constraint | As level 1, plus effectiveness depends on distance from anchor |

**Control rating** (level 2 only; = 1.0 for levels 0 and 1):

```
c(x, anchor) = C_MIN + (1 - C_MIN) * exp( -||x - anchor||^2 / L_CTRL^2 )
```

| Parameter | Value |
|---|---|
| `C_MIN` | 0.5 |
| `L_CTRL` | 20.0 m |

The control rating multiplies tackle probability, pass accuracy (noise scale), and shot success. **The opponent always uses level 2.**

---

## 6. Vision

| Parameter | Value |
|---|---|
| `R_VISION` | 25.0 m (normal players) |
| Playmaker | `infinity` (sees the whole pitch) |

- A point is visible to a player if its distance is `<= R_VISION`.
- A player **always** knows: all 4 teammates' positions, possession state, score, and time.
- A player sees opponents and the ball **only** within its radius.
- Hidden entries in the observation are **zero** and the visibility flag is 0 — so the network can distinguish "not seen" from "seen at origin".
- **Interception requires sight:** a defender can only intercept a pass if the ball was inside its vision radius when the pass was played.

---

## 7. Actions

Total action count with `D=8` directions: **23 actions**.

| Index range | Action | Mask condition |
|---|---|---|
| 0 – 7 | `MOVE_DIR_d` (8 evenly spaced directions) | always valid |
| 8 | `STAY` | always valid |
| 9 | `HOLD_SHAPE` | masked when `zeta = 0` |
| 10 | `GO_TO_BALL` | masked when ball not visible |
| 11 – 14 | `PASS_k` (4 teammates, ascending by player index) | masked when not holding ball |
| 15 | `DRIBBLE_GOAL` | masked when not holding ball |
| 16 | `SHOOT` | masked when not holding ball or outside shoot zone |
| 17 | `TACKLE` | masked when no opponent carrier within `D_TACKLE` |
| 18 – 22 | `SHADOW_j` (5 opponent indices, 0-based) | masked when opponent `j` not visible |

Notes:
- **`SHADOW_j`** is for our learned team only. The opponent does not use it.
- `HOLD_SHAPE`, `GO_TO_BALL`, `DRIBBLE_GOAL`, `SHADOW_j` are macros built on `MOVE_TO(point)`.
- `PASS_k` where `k=0..3` maps to the 4 teammates in ascending player index order (self excluded).
- Invalid actions are masked: the policy cannot select them, but if the raw integer arrives, the env replaces it with `STAY`.
- `MOVE_TO` (macro used by scripted controllers) is **not** in the learner's indexed action set.
- `CLEAR` (macro used by scripted controllers, also not in the learner's set) is a clearance: the carrier kicks the ball up to `mechanics.clear.dist` = 40 m toward a point (default: straight upfield), with an aiming error of `mechanics.clear.spread` = 0.2 rad. The ball goes over everyone, so it cannot be intercepted or cut out in flight. It lands loose and whoever reaches it first picks it up.

---

## 8. Mechanics in Detail

### 8.1 Possession and Loose Balls

| Parameter | Value |
|---|---|
| `D_PICKUP` | 1.5 m |

- A holder carries the ball: ball position = player position.
- **Loose ball:** the player who can reach the ball resting point first takes possession. Ties broken by seeded random draw. A player picks up if within `D_PICKUP`.

### 8.2 Passing

| Parameter | Value |
|---|---|
| `V_PASS` | 3.0 m/step |
| `S_PASS` | 0.15 (angular error scale) |
| `INTERCEPT_MARGIN` | 0.5 steps |

1. Passer chooses `PASS_k`. Target = position of teammate `k` at release.
2. Angular error added: `error ~ Normal(0, S_PASS² * (1 - c_passer))`. Larger error when control rating is lower.
3. **Interception check** for each defender `d` that **was inside its radius when the pass was played**:
   - `q` = closest point on pass segment to `d`
   - `t_ball = ||q - passer|| / V_PASS`
   - `t_def = ||q - d|| / V_PLAYER`
   - Intercepts if `t_def <= t_ball + INTERCEPT_MARGIN`
4. Ball flies for `ceil(||passer - end|| / V_PASS)` steps. Result is decided at release.
5. If receiver is within `D_PICKUP` when ball arrives, they take possession; otherwise ball is loose.

### 8.3 Dribbling

| Parameter | Value |
|---|---|
| `V_DRIBBLE` | 0.75 m/step |
| `V_PLAYER` | 1.0 m/step |

Carrier moves at `V_DRIBBLE` (slower than free movement `V_PLAYER`) while holding the ball.

**Cut-out in flight** (`mechanics.flight_cut`): on every step of a pass, an opponent who is not frozen and stands within `radius` m of the stretch the ball covers that step takes the ball, even if he was not picked as the interceptor at the kick. The first `free` m of the pass are exempt. `radius: 0` turns it off (interception is then decided only at the kick).

### 8.4 Tackling

| Parameter | Value |
|---|---|
| `D_TACKLE` | 2.0 m |
| `T_TACKLE_RECOVER` | 3 steps |

- A defender within `D_TACKLE` of the carrier may choose `TACKLE`.
- Success probability: `p_win = c_defender / (c_defender + c_carrier)`.
- Success: defender takes possession. Failure: defender is frozen for `T_TACKLE_RECOVER` steps.
- Multiple tackle attempts in one step are resolved in seeded random order until one succeeds.
- **Steal chain limit:** at most `max_steal_chain = 2` successful tackles in a row (A takes from B, B takes back). After that `TACKLE` is masked out for both teams until the ball is passed, shot or picked up loose, or until `steal_lock = 15` steps pass. This stops two players trading the ball on the spot.

### 8.5 Shooting

**Sure goal** (`mechanics.sure_goal`): a shot taken from inside the goal area (the small box: within `depth` m of the goal line and `half_width` m either side of the goal centre) always scores, whatever the distance, defenders or control rating. There is no goalkeeper. `enabled: false` uses the normal formula everywhere.

**Missed shot = goal kick** (`mechanics.goal_kick`): the defending player nearest to his own goal is placed on the goal-kick spot (`x` m in front of his goal centre) with the ball, a `goal_kick` event is logged, and nobody may tackle him for `protect` steps. With `clear_box: true`, no player of the other team may be inside the kicker's penalty box (`box`: 16.5 m deep, 22.5 m either side of the goal centre) when the kick is taken: those inside are placed on the nearest edge of the box, and none can enter until the kicker plays the ball or carries it out of the box. With `enabled: false` the ball is left loose near the goal line as before.

| Parameter | Value |
|---|---|
| `D_SHOOT_MAX` | 35.0 m |
| `P_MAX` | 0.8 |
| `D0` | 30.0 m (decay constant) |
| `SHOT_BLOCK_SCALE` | 4.0 m |
| `SHOT_MISS_SPREAD` | 6.0 m |

Scoring probability:
```
p = P_MAX * exp(-d / D0) * openness(shooter, goal, defenders) * c_shooter
```
where `openness` = product over defenders of `1 - exp(-(dist_to_shot_line / SHOT_BLOCK_SCALE)²)`.

- The shooter must be within `D_SHOOT_MAX` of the goal center (in team frame).
- **Goal:** score updated, kickoff. **Miss:** ball becomes loose near goal line with random y-offset ~ `Normal(0, SHOT_MISS_SPREAD²)`, clipped to pitch.

### 8.6 Out of Bounds

Ball is clamped to the pitch. No throw-ins or corners. Only a ball that crosses the end line inside the goal mouth (width `GOAL_W`, centered) counts as a goal.

### 8.7 Kickoff

After a goal, all players return to their formation anchors. The team that **conceded** starts with the ball (given to the first forward by slot index). Team that scored, kicks off too — wait, actually: the team that **conceded** gets the ball (standard football rule). The ball is placed at the kickoff player's position.

### 8.8 One Step of the Game Loop (exact order)

```
1. Build observations for all 10 players (with visibility, masks)
2. Collect actions: our policy (or scripted) for team 0, scripted opponent for team 1
3. Resolve tackles (seeded random order if multiple)
4. Resolve the ball holder's pass or shot (create flight or apply shot result)
5. Move all non-acting, non-frozen players toward their targets
6. Advance ball in flight (if any); resolve loose-ball pickups
7. Check for goals; if scored, update score and run kickoff
8. Update opponent formation (FormationManager.update)
9. Compute rewards, log events, advance t
10. Episode ends when t >= T
```

---

## 9. Rewards

```
R_t = R_team + lambda_t * phi(state, actions) + [t == T] * ETA * sign(goal_difference)
```

| Component | Value | Meaning |
|---|---|---|
| `R_team` | +1 / -1 | Goal scored / conceded |
| `ETA` | 1.0 | End-of-match outcome bonus |
| `w_pass` | +0.02 | Completed pass (scaled by forward advancement) |
| `w_tackle` | +0.02 | Successful tackle by our team |
| `w_shot` | +0.02 | Shot by our team |
| `w_turnover` | -0.02 | Turnover (pass intercepted by opponent) |
| `shape_cap` | 5.0 | Maximum total `|shaping|` per episode |
| `shaping_weight` | 1.0 → 0 | Annealed during training |

All five players receive the **same team reward**.

**Reward-hacking watch:** always log shaping return and win rate together. If shaping rises while win rate is flat, reduce `lambda` or reweight the pass reward by forward distance or defenders beaten.

---

## 10. Observation Vectors

Each player gets its own observation vector and action mask. Positions are **relative to the player** (team frame).

**Flat observation vector** (total **79 features** with D=8, 5 formations):

| Block | Content | Size |
|---|---|---|
| Self position | (x, y) normalized by pitch | 2 |
| Previous action | one-hot (23 actions + "none" slot) | 24 |
| State flags | has_ball, frozen | 2 |
| Slot one-hot | which of 5 player slots am I | 5 |
| Slot type | D / M / F | 3 |
| Anchor offset + control | (dx, dy) to anchor / pitch, control rating. **zeros if `zeta=0`** | 3 |
| Own formation | one-hot over 5 formations | 5 |
| Teammates (4) | (dx, dy) relative + has_ball each | 12 |
| Opponents (5) | (dx, dy) relative + visibility flag (zeros if not visible) | 15 |
| Ball | (dx, dy) relative + visibility flag | 3 |
| Match info | possession one-hot (OWN/OTHER/LOOSE), score diff clipped/3, t/T | 5 |
| **Total** | | **79** |

**Global state** (for centralized critic, 51 features):
All 10 positions (normalized), ball position, holder one-hot (11 options including None), 2× formation one-hot (5), sigma one-hot (6), score diff, t/T.

**Extra inputs for comparison variants** (appended to the observation):
- Belief-augmented: belief vector `b` over 5 formations (5 floats)
- Oracle Q2: true opponent formation as one-hot (5 floats)
- Oracle Q3: playmaker index as one-hot (6 floats, 0 = none)

---

## 11. Experiment Conditions

| Knob | Values |
|---|---|
| Our formation `f` | all 5 formations |
| Structure level `zeta` | 0, 1, 2 (Q1); fixed at 2 for Q2/Q3 |
| Vision radius `R_VISION` | small (≈12), medium (≈25, default), large (≈40), full | 
| Opponent initial formation `g0` | from the 5 formations |
| Opponent switching mode | NONE, SCHEDULED, RANDOM, SCORE_REACTIVE |
| Playmaker index `sigma` | 0 (none), or a D/M/F slot index |
| Agent variant | random, heuristic, memoryless, recurrent, belief-augmented, oracle |
| Training seeds | 5 |
| Evaluation matches per condition/seed | 100 |

Every run must be reproducible from a **config file + seed**. Log the config with the results.

---

## 12. The Learning Agent

### 12.1 Algorithm

**Parameter-shared MAPPO.** One actor network shared across all 5 players. One centralized critic that sees the full state. The actor is called once per player per step.

- Generalized Advantage Estimation (GAE)
- Clipped PPO objective
- Multiple epochs per batch
- Separate seeded random streams for env, opponent, and policy

See `training.md` for hyperparameters and the training protocol.

### 12.2 Actor Variants

| Variant | Architecture | Used for |
|---|---|---|
| Memoryless | MLP on current observation | Lower baseline, Q1 |
| Recurrent | MLP → GRU → action head | Main learner Q2, Q3 |
| Belief-augmented | Memoryless MLP with belief appended | Strong non-learning baseline, Q2 |
| Oracle Q2 | Memoryless MLP + true formation one-hot | Upper bound Q2 |
| Oracle Q3 | Memoryless MLP + playmaker one-hot | Upper bound Q3 |

**Training the GRU:** train on sequences (chunks of consecutive steps). Reset hidden state at episode start. Maintain per-player hidden states throughout a sequence.

### 12.3 The Bayesian Belief Filter (non-learning baseline for Q2)

Each player maintains a probability distribution `b_i(g)` over the 5 opponent formations:

```
predict:  b~(g)  = (1 - H) * b_prev(g) + H / |F|
update:   b(g)  ∝ b~(g) * ∏_{j visible} Normal( y_j ; anchor_g(j), S_NOISE² * I )
```

- `H` = assumed formation switch probability per step (TBD, tune in Phase 0)
- `S_NOISE` = noise absorbing how far opponents drift from anchors (TBD)
- Only visible opponent players contribute to the update
- Inferred formation = argmax of `b`

See `agents.md` for the implementation reference.

### 12.4 Probes (measuring what the network learned)

A **probe** is a small classifier trained on the recurrent policy's hidden state `z`, **without changing the policy**:
- **Formation probe:** predicts `g` from `z` (Q2)
- **Playmaker probe:** predicts `sigma` from `z` (Q3)

Train probes after the policy is trained, using ground truth from `info["g"]` and `info["sigma"]`.

### 12.5 Training Curriculum

1. **Sanity:** random policy must lose clearly. Check reward wiring.
2. **Easy:** one formation, full vision, no switching, no playmaker.
3. **Widen:** all formations → smaller radii → structure levels (Q1).
4. **Switching opponents** (Q2) and **playmaker** (Q3).

---

## 13. What We Measure

See `experiments.md` for the full evaluation protocol. Key metrics:

| Metric | Question |
|---|---|
| Win score `W` (wins + 0.5×draws) / matches | All |
| Goal difference | All |
| Structure benefit `B_zeta` = W(zeta) - W(zeta=0) at same f, radius | Q1 |
| Best formation `f*(radius)` | Q1 |
| Regret vs oracle | Q2, Q3 |
| Detection delay | Q2 |
| Adaptation lag | Q2 |
| Playmaker cost | Q3 |
| Recovery ratio | Q3 |
| Shadow rate | Q3 |
| Behaviour stats (possession, pass completion, shots, spread) | All |

Report mean ± std over 5 training seeds with **paired evaluation** seeds.

---

## 14. Phase 0 Checks

Run these before any training. All must pass.

| Check | Pass condition |
|---|---|
| A. Formation readability | Mean opponent anchor distance << mean inter-formation anchor distance |
| B. Vision matters | Heuristic team with R=25 beats same team with R=12 |
| C. Playmaker works | Heuristic team with sigma beats same team without |
| D. Balance and determinism | Heuristic vs heuristic ≈ 50%, same seed = same match |
| E. Formation win matrix | Run all 5×5 formation pairs with heuristic, store matrix |
| F. Random policy loses | Random policy clearly loses to scripted opponent |

If checks B–C fail, tune constants before any training. See `opponent.md` Section 14 for calibration order.

---

## 15. Common Mistakes to Avoid

- Giving the opponent controller more information than its vision radius allows.
- Forgetting to mask actions (agent picks impossible moves).
- Implementing the game loop in a different order from Section 8.8.
- Using one random number stream for everything (breaks reproducibility).
- Logging only win rate (always log shaping return too).
- Reporting only a single seed.
- Confusing 0-based player indices in code with the 1-based "player numbers" described informally.

---

## 16. Calibrated Parameter Reference

All values from `configs/default.yaml` at project head:

```yaml
pitch:   {length: 100.0, width: 60.0, goal_width: 12.0}
time:    {T: 1200, T_half: 600}
speeds:  {player: 1.0, dribble: 0.75, pass: 3.0}
vision:  {radius: 25.0}
actions: {n_directions: 8}

mechanics:
  tackle_dist: 2.0
  tackle_recover: 3
  pickup: 1.5
  intercept_margin: 0.5
  shadow_dist: 4.0
  shoot_max: 35.0
  p_max: 0.8
  d0: 30.0
  shot_block_scale: 4.0
  shot_miss_spread: 6.0
  s_pass: 0.15

control:   {c_min: 0.5, l_ctrl: 20.0}
structure: {zeta: 2}

reward:
  eta: 1.0
  w_pass: 0.02
  w_tackle: 0.02
  w_shot: 0.02
  w_turnover: -0.02
  shape_cap: 5.0
  shaping_weight: 1.0

formations:
  ours: "2-2-1"
  opponent: "2-2-1"

opponent:
  zone:   {r_zone: 12.0, omega: 0.4, eps_arrive: 1.0, transition_max: 60}
  switching: {mode: NONE, dwell_min: 150}
  attack:  {shoot_min: 0.3, w_fwd: 2.0, w_open: 0.5, w_dist: 0.3,
            pass_min: 0.3, d_danger: 6.0, hub_bonus: 0.3, open_max: 15.0}
  support: {adv: 8.0, mid: 6.0, def: 3.0, r_support: 10.0}
  randomness: {eps_opp: 0.0}
  playmaker: {sigma: 0}
```

**Belief filter parameters** (still TBD; set before Q2 experiments):

| Parameter | Meaning | Suggested starting point |
|---|---|---|
| `H` | Per-step switch probability | 0.001–0.01 |
| `S_NOISE` | Opponent anchor noise | 3–8 m |

---

## 17. Glossary

| Term | Meaning |
|---|---|
| Anchor | Home position of a formation slot (world coords) |
| Slot | A position in a formation (D, M, or F) with a 0-based player index |
| Structure level `zeta` | How much formation structure our agent gets (0, 1, 2) |
| Control rating | Effectiveness multiplier that drops as a player strays from anchor |
| Vision radius | How far a player can see opponents and the ball |
| Playmaker | Opponent player with full-pitch vision |
| Oracle | Agent given hidden information; upper bound only |
| Belief | Probability distribution over opponent formations |
| Probe | Small classifier read off the network's hidden state |
| Shadow | Stay between an opponent and our own goal |
| CTDE | Centralized training, decentralized execution |
| Paired evaluation | Different policies tested on the same random seeds |
| Team frame | Coordinate frame where the team attacks toward +x |
