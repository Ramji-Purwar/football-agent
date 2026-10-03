# game.md: The 5-a-Side Football Game and the Learning Agent

This document explains the whole project scenario in detail: what the game is, how every mechanic works, what the learning agent sees and does, what we vary in experiments, and what we measure. It is written so that anyone on the team can read it, understand the full picture, and start coding. The companion file `opponent.md` explains how to build the opponent team.

**Rule for numbers:** no numeric constant is fixed yet. Every tunable value is written as a named parameter (for example `V_PLAYER`, `R_VISION`). They are all listed in Section 17 and are decided in Phase 0 (calibration), or taken from the reference papers where they exist.

---

## 1. The Big Picture

We build a small football game and train a team of AI players to play it.

- Two teams of **5 outfield players**. **No goalkeepers.**
- **Our team** is controlled by **one shared neural network**. All five players run the same network, each on its own local view of the game.
- The **opponent team** is **not learned**. It follows hand-written rules (see `opponent.md`).
- Every player has **limited vision**: it sees opponents and the ball only within a radius around itself.
- Teams play in **formations**, such as 2-2-1. The opponent may **change formation mid-match**, and one opponent player may be a **playmaker** who sees the whole pitch.

**The project theme is structure versus information.** When players cannot see everything, how much can a formation (structure) make up for it, how well can the team infer what the opponent is doing (inference), and can it adapt to an opponent who has extra information?

### 1.1 The three research questions, in game terms

| # | Question | What happens in the game |
|---|---|---|
| **Q1** | Can formation structure compensate for limited vision? | We give our team three levels of formation structure and test them at different vision radii. We see if structure helps more when vision is small, and if the best formation changes with vision |
| **Q2** | How fast can the team infer that the opponent changed formation, and how much does vision matter? | The opponent switches formation mid-match. Our team is not told. We compare four kinds of agent (memoryless, recurrent, belief-filter, oracle) |
| **Q3** | Can the team adapt to an opponent playmaker? | One opponent player sees the entire pitch. Our team is not told who. We measure how much we lose, how much an adaptive team recovers, and whether it can identify him |

### 1.2 Known and unknown to our team

| Known to our team | Unknown to our team |
|---|---|
| Rules of the game | The opponent's current formation (inferred from positions) |
| Our own formation and our own anchors | When the opponent switches formation |
| Positions of our 4 teammates (always) | Whether the opponent has a playmaker, and who he is |
| Score and possession state | How the opponent decides (its rules) |
| Opponent players that are inside our vision radius | Opponent players outside our vision radius |

The **oracle** variants of our agent are allowed to see the hidden information. They exist only as upper-bound comparisons.

---

## 2. Pitch, Coordinates and Time

- **Pitch:** a bounded 2D rectangle of size `PITCH_L` by `PITCH_W`. The ball moves on the ground only (no aerial passes).
- **Goals:** one at each end of the length axis, with mouth width `GOAL_W`, centered on the end line. **No goalkeeper.**
- **Team frame (important for coding):** each team always attacks toward **+x in its own frame**. The environment stores world coordinates and converts. The opponent controller receives mirrored coordinates, so its code is identical to ours. This also removes the need for halftime side swaps.
- **Time:** the match lasts `T` steps. One step is one tick of the game: all players choose an action, then the world updates once.
- **Halftime marker:** step `T_HALF` is a marker used by the scheduled formation switch. There is no side swap. If we follow the benchmark paper (Song et al., 2023), `T = 3000` and halftime is at step 1501. This is optional, since our environment is a custom one.
- **Outcome:** after `T` steps, more goals wins, equal goals is a draw. For scoring, a win counts 1, a draw 0.5, a loss 0.

---

## 3. Entities and State

The true game state at step `t` (what the environment stores):

| Variable | Meaning |
|---|---|
| `x[i]` | Position of our player `i` (i = 1..5) |
| `y[j]` | Position of opponent player `j` (j = 1..5) |
| `ball` | Ball position |
| `holder` | Who holds the ball: one of our players, one of the opponent's, or nobody (loose or in flight) |
| `g` | Opponent's current formation (**hidden from our team**) |
| `f` | Our formation (fixed during a match) |
| `sigma` | Index of the opponent playmaker, 0 if none (**hidden**) |
| `score` | Goals for each team |
| `t` | Current step |
| `last_seen[...]`, `cooldowns` | Internal timers (tackle recovery, pass reception delay) |

Players are identified by **number** (1 to 5). Opponent numbers are visible to our team, so our players can tell which opponent is which. This is what lets our team compare an opponent's position with the anchor of that slot in each possible formation.

---

## 4. Formations

A formation gives each of the 5 players a **slot** (defender D, midfielder M, or forward F) and a **home position** called an **anchor**.

Slot order (player 1 to 5) for each formation. Always keep this order, so player numbers are consistent:

| Formation | Slot types for players 1 to 5 |
|---|---|
| 2-2-1 | D, D, M, M, F |
| 2-1-2 | D, D, M, F, F |
| 1-3-1 | D, M, M, M, F |
| 1-2-2 | D, M, M, F, F |
| 3-1-1 | D, D, D, M, F |

- **Anchor coordinates** are stored in the config as fractions of the pitch (x along the length from own goal to opponent goal, y across the width). **They are TBD**: decide them in Phase 0, then freeze them.
- The final set of formations may change. Start with three (for example 2-2-1, 1-3-1, 3-1-1) and add more if time allows.
- **Formations are not enforced.** A player may leave its anchor. How much it matters depends on the structure level (Section 5).
- **Formation switch (opponent only):** the anchors change at once and the players **walk** to the new anchors at normal speed. Nobody teleports. During the walk, players are far from their new anchors, so they are less effective. That is the natural cost of switching, and the visible sign of a switch.
- After a goal, all players return to their current anchors for a kickoff.

---

## 5. Structure Levels and the Control Rating (the Q1 knob)

For **our team** we control how much formation structure the agent gets. This is the structure level `zeta`:

| `zeta` | Name | What changes |
|---|---|---|
| 0 | No structure | Formation only sets the kickoff positions. The agent does **not** observe its anchor and has **no** `HOLD_SHAPE` action |
| 1 | Structure as information | The agent observes its own anchor and has `HOLD_SHAPE`. Leaving the anchor is free |
| 2 | Structure as constraint | As level 1, plus effectiveness depends on distance from the anchor (control rating below) |

**Control rating** (level 2 only; it is 1 for levels 0 and 1):

```
c(x, anchor) = C_MIN + (1 - C_MIN) * exp( - ||x - anchor||^2 / L_CTRL^2 )
```

`C_MIN` is the minimum rating and `L_CTRL` a distance scale. The control rating multiplies:

- **tackle success** (Section 8.4),
- **pass accuracy** (Section 8.2),
- **shot success** (Section 8.5).

So a player far from its anchor plays worse. That creates the tension between "chase the ball" and "hold your shape". **The opponent always uses level 2.**

---

## 6. Vision

Vision is the central information limit of the whole project.

- **Visibility indicator:** a point `z` is visible to a player at position `x` with radius `rho` if `||z - x|| <= rho`.
- **Normal players** (all of ours, and the opponent's non-playmakers) use `rho = R_VISION`.
- **The opponent playmaker** (player `sigma`, Q3 only) uses `rho = infinity`.
- What a player sees: **opponents and the ball**, only inside its radius.
- What a player always knows: its **teammates' positions**, whether it holds the ball, the **possession state** (ours / opponent's / loose), the **score**, and the time. (This is a modelling simplification and is stated in the report.)
- **Interception needs sight:** a defender can only try to intercept a pass if the ball was inside its vision radius when the pass was played (Section 8.2).
- Hidden entries are set to **zero** in the observation vector and a **visibility flag** is set to 0, so the network can tell "not seen" from "seen at the origin".

---

## 6.1 Why This Setup Works (design reasoning)

Read this once so the design does not look arbitrary.

- **The opponent holds its formation** (zonal, not man-marking). Man-marking would pull opponent players away from their anchors to chase our players, hiding the formation, and Q2 would become unanswerable. Holding formation keeps the opponent's shape readable from where its players stand.
- **Vision still matters** without man-marking. Our players that cannot see defenders cannot find free teammates or safe passing lanes, and the opponent also cannot press or intercept what is outside its own radius.
- **The playmaker** has a real effect because interception needs sight and his attacking decisions use the whole pitch.

---

## 7. Actions

Every player chooses **one discrete action per step**. The action set is the same for every player. Invalid actions are **masked** (the network cannot choose them), using an action-mask vector given with each observation.

| Index group | Action | Meaning | Mask (invalid when) |
|---|---|---|---|
| 0 to `D-1` | `MOVE_d` | Move up to `V_PLAYER` in direction `d` (there are `D` evenly spaced directions, `D` is TBD). With the ball, the ball moves with the player | never |
| next | `STAY` | Do not move | never |
| next | `HOLD_SHAPE` | Move toward your own anchor | `zeta = 0` |
| next | `GO_TO_BALL` | Move toward the ball | ball not visible |
| next 4 | `PASS_k` | Pass to teammate `k` (4 teammates, fixed order by player number) | you do not hold the ball |
| next | `DRIBBLE_GOAL` | Move toward the opponent's goal with the ball (a macro for the best direction) | you do not hold the ball |
| next | `SHOOT` | Shoot at goal | not holding the ball, or outside the shooting zone |
| next | `TACKLE` | Try to win the ball from the opponent carrier | no opponent carrier within `D_TACKLE` |
| next 5 | `SHADOW_j` | Move toward the point at distance `D_SHADOW` from opponent `j`, on the line from `j` to **our own goal** (stay between him and our goal) | opponent `j` not visible |

Notes:

- **`SHADOW_j` is used only by our learned team.** It is our concrete way of "adapting" to a playmaker: work out who he is and shadow him. The opponent heuristic does not use it.
- Keep the action indexing in **one place** in the code (an enum), because the mask and the policy both depend on it.
- `HOLD_SHAPE`, `GO_TO_BALL`, `DRIBBLE_GOAL` and `SHADOW_j` are all **macros** built on one low-level routine, `MOVE_TO(point)`, which moves the player toward a point by at most `V_PLAYER` per step. The opponent controller uses the same `MOVE_TO` routine (see `opponent.md`), so both teams obey the same movement rules.
- Total number of actions = `D + 1 + 1 + 1 + 4 + 1 + 1 + 1 + 5`.

---

## 8. Mechanics in Detail

### 8.1 Possession and loose balls

- A player who **holds** the ball carries it: the ball position equals the player's position.
- If nobody holds the ball (after a failed pass, a missed shot, a failed tackle that knocks it loose), the ball is **loose**. The ball decelerates and stops. The player who can reach its resting point first (by time to reach) takes possession, with ties broken by a seeded random draw. A player takes the ball if it is within `D_PICKUP`.
- A ball **in flight** (pass or shot) is not held by anyone until it arrives.

### 8.2 Passing

1. The passer chooses `PASS_k`. The target point is the **position of teammate `k` at the moment of release**.
2. A small random **angular error** is added. Its size grows as the passer's control rating drops: `error ~ Normal(0, S_PASS^2 * (1 - c_passer))`.
3. **Interception check.** For each opposing player `d` that was **allowed to react** (the ball was inside `d`'s vision radius at release):
   - `q` = the closest point on the pass segment to `d`,
   - `t_ball(q)` = `||q - passer|| / V_PASS`,
   - `t_def(q)` = `||q - d|| / V_PLAYER`,
   - `d` intercepts if `t_def(q) <= t_ball(q) + INTERCEPT_MARGIN`.
   If several defenders qualify, the one with the **smallest `t_def - t_ball`** wins.
4. **Outcome.** If intercepted, the ball goes to the interceptor. Otherwise it travels to the target. If the receiver is within `D_PICKUP` of the target when the ball arrives, he takes possession, otherwise the ball is loose there.
5. **Timing.** The ball is "in flight" for `ceil(t_arrive)` steps, moving along the segment. The result is decided at release (it is deterministic given the random error), so no moving-target physics is needed.

### 8.3 Dribbling

The carrier moves at `V_DRIBBLE` (`<= V_PLAYER`) while holding the ball.

### 8.4 Tackling

- A defender within `D_TACKLE` of the carrier may choose `TACKLE`.
- Success probability: `p_win = c_defender / (c_defender + c_carrier)`. With equal ratings it is 0.5.
- **Success:** possession goes to the defender. **Failure:** the carrier keeps the ball and the defender is frozen for `T_TACKLE_RECOVER` steps.
- If several defenders tackle in the same step, resolve them in a **seeded random order** until one succeeds.

### 8.5 Shooting

- A shot is only allowed inside the **shooting zone** (distance to the opponent's goal at most `D_SHOOT_MAX`).
- Scoring probability: `p = P_MAX * exp(-d / D0) * openness * c_shooter`, where `d` is the distance to the goal center and `openness` is the measure of how clear the shot line is of nearby defenders (decreasing with their proximity to the line).
- With no goalkeeper, the only contest is the defenders' proximity to the shot line.
- **Goal:** the match score is updated and play restarts from kickoff (Section 4). **Miss:** the ball becomes loose near the goal line, at a point with random spread.

### 8.6 Out of bounds

The ball is clamped to the pitch (it stops or is returned in play). Only a ball inside the goal mouth that crosses the end line counts as a goal. This is a simplification: no throw-ins or corners.

### 8.7 Kickoff

After a goal, everyone returns to their anchors in their current formation. The team that conceded gets the ball (given to the player of a fixed kickoff slot, such as the first forward).

### 8.8 One step of the game loop (the exact order to implement)

1. Compute each player's **observation** (with visibility) and action mask. Opponent players use the **same function**, with their own radius.
2. Collect all 10 actions: ours from the policy, the opponent's from its rules.
3. Resolve **tackles**.
4. Resolve **passes and shots** (create the flight or the shot result).
5. Move **players** (movement, walking to anchors, shadowing). The ball moves with its carrier, or along its flight path.
6. Resolve **loose-ball pickups**.
7. Check for **goals**. If a goal happened, run the kickoff.
8. Update **formation switches** for the opponent (Section 4 in `opponent.md`).
9. Compute **rewards** and **logs**.
10. Advance `t`. The episode ends at `T`.

---

## 9. Rewards (what the learning agent is trained on)

All five of our players receive the **same team reward** each step:

```
R_t = R_team + lambda_t * phi(state, actions) + [t == T] * ETA * sign(goal_difference)
```

- **`R_team`**: +1 when we score, -1 when we concede.
- **`phi` (shaping, individual-level)**: small rewards for completed passes (weighted by how much they advance the ball), successful tackles, shots on target, and small penalties for turnovers.
- **`lambda_t`**: the shaping weight, **annealed toward 0** during training.
- **`ETA`**: the end-of-match outcome bonus. It is larger than any single shaping event so that winning stays the top priority.
- **Cap:** total shaping reward per episode is capped at `C_SHAPE`.

**Reward-hacking watch.** PPO maximizes total reward. If shaping is earned many times per match (for example endless safe sideways passes), it can outweigh the match outcome and the team can look good on shaping while not winning. So during training **always log shaping return and win rate together**. If shaping rises while win rate is flat, reduce `lambda`, or reweight (for example scale the pass reward by distance or the number of defenders beaten), then retrain.

---

## 10. What the Agent Sees (Observation Vectors)

Each player gets its **own** observation vector and action mask. The same network is applied to each one. Positions are given **relative to the player** and normalized by the pitch size.

| Block | Contents | Size note |
|---|---|---|
| Self | position, previous action (one-hot), has-ball flag, **slot one-hot** (5), slot-type one-hot (D, M, F) | fixed |
| Anchor | own anchor (x, y) and current control rating. **Zeros if `zeta = 0`** | fixed |
| Own formation | one-hot of `f` | `|F|` |
| Teammates (4, fixed order by player number) | relative position (dx, dy), has-ball flag | always visible |
| Opponents (5, fixed order by opponent number) | relative position (dx, dy), **visibility flag**. Zeros if not visible | 5 x 3 |
| Ball | relative position, **visibility flag**. Zeros if not visible | fixed |
| Match info | possession state (ours / opponent / loose), score difference (clipped), time fraction `t/T` | fixed |

Extra inputs for the **comparison variants** (never in the plain agent):

- **Belief-augmented:** the belief vector `b_i` over the opponent formation (Section 12.3).
- **Oracle (Q2):** the true opponent formation `g` as a one-hot.
- **Oracle (Q3):** the playmaker index `sigma` as a one-hot.

**Critic input (training only, never given at execution):** the full state: all 10 positions, ball, holder, `g` as a one-hot, `sigma` as a one-hot, score, and time. This is called centralized training with decentralized execution.

---

## 11. Experiment Conditions (what we vary)

| Knob | Values | Used in |
|---|---|---|
| Our formation `f` | start with three formations | all |
| Structure level `zeta` | 0, 1, 2 | Q1 (Q2 and Q3 use 2) |
| Vision radius `R_VISION` | three levels, plus full vision as a reference | all |
| Opponent initial formation `g0` | from the formation set | all |
| Opponent switching mode `G` | none, scheduled, random, score-reactive | Q2 (others use none) |
| Playmaker index `sigma` | 0 (none), or a player at a defender, midfielder or forward slot | Q3 |
| Agent variant | random, heuristic, memoryless, recurrent (GRU), belief-augmented, oracle | all |
| Training seeds | 5 (provisional) | all |
| Evaluation matches per condition and seed | 100 (provisional) | all |

Every run must be reproducible from a **config file plus a seed**. Log the config with the results.

---

## 12. The Learning Agent (for the coders)

### 12.1 Algorithm

- **Parameter-shared PPO** (MAPPO style). One actor network and one centralized critic. The actor is called once per player per step.
- Use generalized advantage estimation (GAE), the clipped PPO objective and several epochs per batch. Hyperparameters are TBD and are tuned during the first runs.
- Our opponent is fixed, so the environment is stationary during training. There is no self-play.

### 12.2 Actor variants

| Variant | Description | Used for |
|---|---|---|
| Memoryless | MLP on the current observation | Lower baseline in Q2 |
| Recurrent | MLP then **GRU**, then action head. The hidden state carries what the player saw earlier | Main learner in Q2 and Q3 |
| Belief-augmented | Memoryless MLP with the Bayesian belief appended to its input | Strong non-learned-inference baseline in Q2 |
| Oracle | Memoryless MLP with the true `g` (Q2) or `sigma` (Q3) appended | Upper bound |

**Training with the GRU:** train on sequences (chunks of consecutive steps), reset the hidden state at episode start, and keep hidden states per player.

### 12.3 The Bayesian belief filter (a non-learning baseline for Q2)

Each player keeps a probability `b_i(g)` over the possible opponent formations. Each step:

```
predict:  b~(g)  = (1 - H) * b_prev(g) + H / |F|             # H = assumed switch probability per step
update:   b(g)  ∝ b~(g) * product over visible opponents j of  Normal( y_j ; anchor_g(j), S_NOISE^2 * I )
```

Only opponents the player currently **sees** contribute to the update. Normalize `b` after the update. `S_NOISE` absorbs how far the opponent's pressing and drifting pull players from their anchors. The inferred formation is the argmax of `b`.

### 12.4 The probes (for measuring what the network learned)

A **probe** is a small classifier trained on the recurrent policy's hidden state `z`, **without changing the policy**:

- **Formation probe:** predicts the opponent formation `g` from `z` (Q2).
- **Playmaker probe:** predicts the playmaker index `sigma` from `z` (Q3).

Train probes after the policy is trained, using the hidden truth from the environment's `info`.

### 12.5 Training stages (curriculum)

1. **Sanity runs:** a random policy must lose clearly. Check the reward wiring.
2. **Easy setting:** one formation, the largest vision radius, no switching, no playmaker. The learned team should clearly beat random and approach the heuristic.
3. **Widen:** all formations, then smaller radii, then structure levels (Q1).
4. **Switching opponents** (Q2) and the **playmaker** (Q3).
Mark each widening point on the learning curves.

### 12.6 Code interface (suggested)

```python
env = FootballEnv(config)                  # config holds every parameter (Section 17)
obs, info = env.reset(seed=seed)           # obs: dict {player_id: vector}, plus masks
obs, reward, done, info = env.step(actions)  # actions: dict {player_id: int}
# reward: one shared scalar for the team (plus a dict of shaping parts for logging)
# info: hidden truth and events:
#   info["g"], info["sigma"], info["score"], info["switch_events"], info["events"]
```

- `info["events"]` is a list of logged events: passes (completed or intercepted), tackles, shots, goals, formation switches, shadow actions. **All metrics are computed from this log**, not from ad hoc counters.
- The **opponent controller lives inside the environment** and uses the same observation function as our players, so it cannot see more than its radius allows (except the playmaker).

---

## 13. What We Measure

| Metric | Definition | Question |
|---|---|---|
| Win score `W` | wins plus half of draws, over matches | all |
| Goal difference | goals for minus goals against | all |
| Structure benefit `B_zeta(f,r)` | `W` at structure level `zeta` minus `W` at level 0, same formation and radius | Q1 |
| Best formation `f*(r)` | the formation with the highest `W` at radius `r` | Q1 |
| Regret vs oracle | `W(oracle) - W(agent)` | Q2, Q3 |
| Detection delay | steps after a switch until the inferred formation equals the true one (belief filter or probe) | Q2 |
| Adaptation lag | steps after a switch until the rolling goal difference recovers to its pre-switch level | Q2 |
| Playmaker cost `E(sigma)` | `W` without playmaker minus `W` with playmaker, for the unaware team | Q3 |
| Recovery ratio | fraction of that cost the adaptive team wins back | Q3 |
| Identification accuracy and delay | probe accuracy for `sigma`, and steps until it is stably right | Q3 |
| Shadow rate | share of steps in which some player of ours shadows opponent `sigma` | Q3 |
| Behaviour stats | possession share, passes and completion, shots, team spread | Q1 to Q3 |

Report mean and standard deviation over training seeds. Use the same evaluation seeds for every policy so comparisons are **paired**.

---

## 14. Phase 0 Checks (before any serious training)

1. **Random loses:** a random policy loses clearly to the heuristic opponent.
2. **Balance:** heuristic vs heuristic is roughly balanced.
3. **Vision matters:** the same heuristic team with a small radius loses to itself with a large radius.
4. **Formation is readable:** opponent players stay close to their anchors (measure the average distance), otherwise Q2 is not well posed.
5. **Playmaker works:** a heuristic team with a full-vision playmaker beats the same team without one.
6. **Formation win matrix:** run formation vs formation matches with the heuristic and store the matrix. It is a benchmark for Q1 and gives the best-response formation.
7. **Determinism:** the same seed and config give the same match.

If checks 3 to 5 fail, tune the opponent or constants before training. Otherwise the experiments will show nothing.

---

## 15. Priorities (if time runs short)

1. **Phase 0 and Q1** are the core.
2. **Q2** comes next.
3. **Q3** comes last. Cut the number of tested playmaker positions first, then the number of formations.

---

## 16. Common Mistakes to Avoid

- Giving the opponent controller more information than a player with the same radius would have (it will invalidate Q2 and Q3).
- Forgetting to mask actions, which makes the agent pick impossible moves.
- Letting the environment mutate state in a different order from Section 8.8, which changes results.
- Using one random number stream for everything. Use separate seeded streams for the environment, the opponent and the policy so runs are reproducible.
- Looking at win rate only. Always log shaping return too.
- Reporting only a single seed.

---

## 17. Parameter List (all TBD, fill in during Phase 0)

| Parameter | Meaning |
|---|---|
| `PITCH_L`, `PITCH_W`, `GOAL_W` | Pitch and goal size |
| `T`, `T_HALF` | Match length and halftime marker |
| `V_PLAYER`, `V_DRIBBLE`, `V_PASS` | Speeds |
| `R_VISION` | Vision radius (several levels) |
| `D` | Number of movement directions |
| `D_TACKLE`, `T_TACKLE_RECOVER` | Tackle distance and recovery time |
| `D_PICKUP`, `INTERCEPT_MARGIN` | Ball pickup distance and interception margin |
| `D_SHADOW` | Shadowing distance |
| `D_SHOOT_MAX`, `P_MAX`, `D0` | Shooting zone and probability constants |
| `S_PASS` | Pass error scale |
| `C_MIN`, `L_CTRL` | Control rating constants |
| `ETA`, `LAMBDA` schedule, `C_SHAPE` | Reward constants |
| `H`, `S_NOISE` | Belief filter constants |
| Anchor coordinates per formation | Fractions of the pitch |
| Opponent constants | See `opponent.md` |
| PPO hyperparameters, sequence length, number of parallel environments | Learning |

A config skeleton:

```yaml
pitch: {length: null, width: null, goal_width: null}      # TBD
time: {T: null, T_half: null}                              # TBD
speeds: {player: null, dribble: null, pass: null}          # TBD
vision: {radii: null}                                      # TBD (several levels)
mechanics: {tackle_dist: null, tackle_recover: null, pickup: null,
            intercept_margin: null, shadow_dist: null,
            shoot_max: null, p_max: null, d0: null, s_pass: null}
control: {c_min: null, l_ctrl: null}
reward: {eta: null, lambda_schedule: null, shape_cap: null}
belief: {hazard: null, noise: null}
formations: {anchors: null}                                # TBD per formation
seeds: {train: null, eval: null}
```

---

## 18. Glossary

| Term | Meaning |
|---|---|
| Anchor | The home position of a formation slot |
| Slot | A position in a formation (D, M or F) with a player number |
| Structure level `zeta` | How much formation structure our agent gets (0, 1, 2) |
| Control rating | A multiplier that drops as a player strays from its anchor |
| Vision radius | How far a player can see opponents and the ball |
| Playmaker | An opponent player with full-pitch vision |
| Oracle | An agent given the hidden information. An upper bound only |
| Belief | A probability distribution over the opponent's formation |
| Probe | A small classifier read off the network's hidden state |
| Shadow | Stay between an opponent and our own goal |
| CTDE | Centralized training, decentralized execution |
| Paired evaluation | Different policies tested on the same random seeds |