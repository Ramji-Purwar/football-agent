# opponent.md: How to Build the Opponent Team

This document explains how to build the opponent team for the 5-a-side football game described in `game.md`. Read `game.md` first, especially Sections 2 to 8, since the opponent plays by the same rules and physics as our team.

**Rule for numbers:** no numeric constant is fixed yet. Tunable values are named parameters (for example `R_ZONE`, `SHOOT_MIN`), listed in Section 12, and set during Phase 0 calibration (Section 13).

---

## 1. Purpose and Design Requirements

The opponent is a **fixed, hand-written team**. It does not learn. Its job is to be a meaningful, readable opponent for our learning team.

It must satisfy these requirements. Each one exists because an experiment depends on it:

| # | Requirement | Why it matters |
|---|---|---|
| 1 | **It holds its formation.** Players stay near their anchors | Q2 asks our team to infer the opponent's formation from where its players stand. If opponents wander far from their anchors, the formation cannot be read |
| 2 | **It is vision-limited.** Each player decides only from what it can see within its radius | Otherwise the opponent has an information advantage our team does not, which breaks Q1 and Q2 |
| 3 | **It can switch formation mid-match** in three ways | Q2 |
| 4 | **One player can be a playmaker** with full-pitch vision | Q3 |
| 5 | **It is a worthwhile opponent.** A random policy must lose clearly, and it must be balanced against itself | Otherwise learning results mean nothing |
| 6 | **It is reproducible.** Same seed and config give the same match | Paired evaluation and debugging |
| 7 | **It plays by the same physics** as our team, with no teleporting and no extra abilities | Fairness |

**What it must NOT be:** man-marking (this pulls players away from their anchors and hides the formation), learning, or omniscient (except the playmaker).

---

## 2. What the Opponent Is (Inputs and Outputs)

- The opponent controls its 5 players. Each opponent player `j` (j = 1..5) acts every step, like our players.
- **Input to each player:** the **same observation function** our players use (see `game.md` Section 10), computed from that player's own position with its own vision radius `rho_j`:
  - `rho_j = R_VISION` for normal players,
  - `rho_j = infinity` for the playmaker (player `sigma`, if any).
- **Output:** one action per player per step. The opponent uses the same actions as our team, with macros of the same kind (move toward a point, go to the ball, hold shape), but **not** `SHADOW_j`.
- **Team frame:** the opponent code works as if it attacks toward +x. The environment mirrors the coordinates before the opponent sees them, so there is no separate "left side" code.
- **Control rating:** the opponent always uses structure level 2. Its effectiveness drops when a player is far from its anchor, exactly as for our level-2 team.

**Important rule for the code:** the opponent controller must be called with the **observation dictionary only**, not with the full environment state. This is the easiest way to make sure it never uses information it should not have.

---

## 3. Architecture

The opponent has two layers.

```
OpponentTeam
 ├─ FormationManager     (Section 4)   decides the current formation g_t and its anchors
 └─ PlayerController x5  (Sections 5-9) decides each player's action from its own observation
```

Every step, for each player `j`:

1. The **FormationManager** has already set the current anchors (it runs once per step, before the player decisions).
2. The **PlayerController** runs a priority list. The first rule that applies decides the action:

```
1. Transition rule      if the team has just switched formation and I am not yet near my new anchor,
                        and I do not hold the ball:         walk to my new anchor
2. I hold the ball:                                            attack routine with the ball   (Section 6)
3. A teammate holds the ball:                                  off-ball attacking support     (Section 7)
4. The ball is loose:                                          loose-ball rule                (Section 5.3)
5. Our (the learning team's) side holds the ball:              zonal defence                  (Section 5)
```

Because the rules are ordered, you can build and test them one at a time (Section 13).

---

## 4. The Formation Manager

This section is referenced by `game.md` (Section 8.8, step 8).

### 4.1 State

| Variable | Meaning |
|---|---|
| `g` | Current opponent formation |
| `anchors` | Anchors of the current formation, one per player number |
| `dwell` | Steps since the last switch |
| `in_transition` | True while players are walking to new anchors |

The formation set and the anchor coordinates are the same configuration used for our team (`game.md` Section 4). Slots keep their **player numbers**: after a switch, player `j` simply gets a new anchor for slot `j`.

### 4.2 Switching modes

The mode is set in the config. In every mode the team must **respect a minimum dwell time `DWELL_MIN`** between switches, so that the formation is stable long enough for Q2 to measure detection.

| Mode | Rule |
|---|---|
| `NONE` | The formation never changes. Used in Q1 and Q3 |
| `SCHEDULED` | Switch once at step `T_SWITCH` (for example the halftime marker) to a configured target formation `G_TARGET` |
| `RANDOM` | After `DWELL_MIN`, switch with a per-step probability `H_SWITCH` to a uniformly random **different** formation |
| `SCORE_REACTIVE` | Every `CHECK_INTERVAL` steps, compute `d = opponent goals - our goals`. If `d <= -TRAIL_THRESH`, switch to the attacking formation `F_ATT`. If `d >= LEAD_THRESH`, switch to the defensive formation `F_DEF`. Otherwise stay. Respect `DWELL_MIN` |

- `F_ATT` and `F_DEF` come from a small mapping (config). Attacking formations have more forwards (for example 1-2-2, 2-1-2), defensive ones have more defenders (for example 3-1-1, 2-2-1). Decide the mapping in Phase 0.
- In `SCORE_REACTIVE` mode the score is public, so a good learning team can partly **predict** a switch from the score. This is intended: it is the realistic mode.

### 4.3 What happens at a switch

1. Set `g` to the new formation and replace the anchors.
2. Set `in_transition = True`, `dwell = 0`.
3. Players **walk** to their new anchors at normal speed (rule 1 of the priority list). **No teleporting.**
4. `in_transition` becomes False when all players are within `EPS_ARRIVE` of their anchors, or after `TRANSITION_MAX` steps, whichever comes first.
5. **Log a switch event:** `{t, from_formation, to_formation}`. This is the ground truth for detection delay and adaptation lag.

During the transition the players are far from their new anchors, so their control rating is low. That is the natural cost of switching, and also the visible sign of a switch that our team can use.

---

## 5. Defence: Zonal and Formation-Holding

This is how the opponent defends, and it is the **core design choice** that keeps the formation readable.

### 5.1 The rule

Each player `j` has a **home position** `anchor_g(j)` and a **zone** around it of radius `R_ZONE`. Let `u_j` be whether player `j` **sees the ball** (within its radius). Its desired position `p_j` is:

```
if u_j and ball is inside my zone (||ball - anchor|| <= R_ZONE) and I am the PRESSER:
        p_j = ball                                    # press the ball
elif u_j:
        p_j = anchor + OMEGA * clip(ball - anchor, R_ZONE)   # small drift toward the ball side
else:
        p_j = anchor                                  # hold the home position
```

- `clip(v, R)` limits the length of vector `v` to `R`.
- `OMEGA` is in `[0, 1)`, so the drift is always a fraction of the zone radius.
- The player then moves toward `p_j` with the macro `MOVE_TO(p_j)` at speed `V_PLAYER`. Stop when within `EPS_ARRIVE`, to avoid jittering around the target.

### 5.2 Pressing arbitration (only one presser)

Several players' zones can overlap. To avoid the whole team swarming the ball (which would destroy the formation), only **one presser at a time** is allowed:

- Among the opponent players whose zone contains the ball **and** who see it, the one **nearest to the ball** is the presser.
- Everyone else uses the drift rule.
- Break ties by player number (so it is deterministic).

### 5.3 Loose ball

When the ball is loose and not in flight:

- If I see the ball and I am the **nearest opponent player to it** (teammates' positions are always known), then `GO_TO_BALL`.
- Otherwise follow the zonal rule (5.1).

### 5.4 Tackling

If an opponent player is the presser and our carrier is within `D_TACKLE`, choose `TACKLE`. The success probability and the consequences are in `game.md` Section 8.4.

### 5.5 Interception

Interception is **not a decision**. It is resolved by the pass-interception geometry in `game.md` Section 8.2, for every opponent player who was inside the vision radius when the pass was played. So you do **not** write interception code in the opponent. Limited vision automatically reduces the opponent's interceptions, and the playmaker (infinite radius) is eligible everywhere.

### 5.6 Why it works

Because `R_ZONE` and `OMEGA` are small, no player ever strays far from its anchor, so the team's positions stay informative about the formation. The Phase 0 check in Section 13 verifies this numerically.

---

## 6. Attack: The Player With the Ball

The ball carrier `j` uses only what it can **see** (its visible opponents and the ball), plus the always-known positions of its teammates. Use this decision order and take the **first rule that succeeds**:

### 6.1 Shoot

If the carrier is inside the shooting zone, estimate the scoring probability with the formula of `game.md` Section 8.5, using only **visible** defenders to compute `openness`. If the estimate is at least `SHOOT_MIN`, then `SHOOT`.

### 6.2 Pass

For each of the 4 teammates `k`, compute a **pass score**:

```
score(k) =   W_FWD  * forward_gain(k)
           + W_OPEN * openness(k)
           - W_DIST * distance(k)
           + HUB_BONUS * [k is the playmaker]
```

- `forward_gain(k)`: how much closer to the opponent goal teammate `k` is than the carrier (can be negative).
- `openness(k)`: the distance from teammate `k` to the nearest **visible** defender, clipped to a maximum, so more open is better.
- `distance(k)`: the pass length.
- `HUB_BONUS` applies only when a playmaker exists (Section 9). It is 0 otherwise.

A pass is **allowed** only if the predicted interception check (the same formula as in `game.md` Section 8.2) finds **no visible defender** able to intercept. Defenders the carrier cannot see do **not** count: this is how limited vision makes the opponent's passing riskier, and it is intended.

Choose the allowed pass with the highest score, if that score is at least `PASS_MIN`.

### 6.3 Dribble

If no rule above fired and **no visible defender** is within `D_DANGER` ahead of the carrier, `DRIBBLE_GOAL`.

### 6.4 Recycle

Otherwise (the carrier is pressed and nothing is safe), pass to the **safest** teammate even if it is backward or sideways, using the same allowed check but ignoring `PASS_MIN` and `W_FWD`. If no pass is allowed, hold the ball (`STAY`).

### 6.5 Randomness

By default the opponent is **deterministic**. If we later need opponent diversity, add an optional probability `EPS_OPP` of choosing a random allowed action, drawn from the opponent's own random stream. Keep it off for the main experiments.

---

## 7. Off-Ball Attacking Support (a teammate has the ball)

The goal is to give the carrier passing options **without leaving the formation**:

```
target = anchor + shift
```

- **Forwards:** shift toward the opponent goal by up to `SUPPORT_ADV`.
- **Midfielders:** shift toward the ball carrier by up to `SUPPORT_MID`.
- **Defenders:** hold the anchor (shift = 0), maybe a small step up the pitch of at most `SUPPORT_DEF`.
- **Cap:** the length of `shift` is at most `R_SUPPORT`, with `R_SUPPORT <= R_ZONE`. This is what keeps the formation readable even when attacking.

Then `MOVE_TO(target)`.

---

## 8. Using Vision (How to Avoid Cheating)

The opponent must not use information beyond what a player at its position and radius could know. Concretely:

- Build one function `build_observation(player, state)` that is used for **both** our players and opponent players. It takes the player's radius as an argument.
- The opponent controller only receives the output of that function.
- What an opponent player always knows (same as ours): its teammates' positions, possession state, score and time.
- What it knows only inside its radius: our players and the ball.
- If a player does not see the ball, it **does not know where the ball is**. Do not let it use the true ball position "just for movement".
- It cannot see our formation or intentions. It never receives hidden state of our agent.

---

## 9. The Playmaker Variant (Q3)

When the config sets `sigma` > 0, opponent player `sigma` is the playmaker. **Only these things change:**

| Aspect | Normal player | Playmaker |
|---|---|---|
| Vision radius | `R_VISION` | infinity (always sees all players and the ball) |
| Zonal rule | drift only if the ball is within radius | the ball is always seen, so the drift and pressing rules always apply |
| Pass evaluation (his own passes) | only visible defenders count | **all** defenders count, so his passes are safer and better chosen |
| Interception | only if the ball was within `R_VISION` when the pass was played | eligible for every pass on the pitch |
| Pass target for his teammates | no bonus | teammates add `HUB_BONUS` when passing to him (if the pass is allowed), so he sees more of the ball |

- **Why `HUB_BONUS`:** it makes the playmaker involved in many attacks, so his influence is real. It also makes him partly identifiable from his behaviour. Set `HUB_BONUS = 0` to test a "silent" playmaker.
- Everything else (anchor, zone, speed, tackling, control rating) is **identical** to a normal player. The advantage is information only.
- **Which slot:** the slot type of `sigma` (defender, midfielder or forward) is chosen by the experiment config.
- **Phase 0 check:** a team with a playmaker must beat the same team without one, otherwise the Q3 experiment has nothing to measure (Section 13).

---

## 10. Code Structure (suggested)

```python
class OpponentTeam:
    def __init__(self, cfg, rng):            # rng: the opponent's OWN random stream
        self.cfg = cfg; self.formation_manager = FormationManager(cfg, rng)

    def reset(self, g0, sigma):              # initial formation and playmaker index (0 = none)
        ...

    def update_formation(self, t, score):    # called once per step, before decisions
        """returns a switch event dict or None; also updates anchors and transition state"""

    def act(self, t, obs_by_player):         # obs_by_player: {j: observation built with that player's radius}
        return {j: self.decide(j, obs_by_player[j]) for j in range(1, 6)}

    def decide(self, j, obs):                # the priority list of Section 3
        if self.in_transition and not obs.has_ball and not near_anchor(obs):
            return MOVE_TO(self.anchor(j))
        if obs.has_ball:
            return self.attack_with_ball(j, obs)          # Section 6
        if obs.teammate_has_ball:
            return self.support(j, obs)                   # Section 7
        if obs.ball_loose:
            return self.loose_ball(j, obs)                # Section 5.3
        return self.zonal(j, obs)                         # Section 5
```

Helper functions to write and **unit-test separately**:

| Function | What it does |
|---|---|
| `zonal_target(anchor, ball, sees_ball, is_presser)` | Section 5.1 |
| `choose_presser(obs_of_all_opponents)` | Section 5.2 |
| `shot_estimate(carrier, visible_defenders)` | Section 6.1 |
| `pass_candidates(carrier, teammates, visible_defenders)` | Section 6.2 |
| `would_be_intercepted(passer, target, visible_defenders)` | The interception geometry (reuse the environment's function) |
| `support_target(anchor, slot_type, carrier, ball)` | Section 7 |
| `FormationManager.step(t, score)` | Section 4 |

**Reuse the environment's physics functions** (interception check, shot probability, control rating) instead of rewriting them. The opponent should call the same code, restricted to visible information.

**Random streams:** the opponent has its own seeded random stream, separate from the environment and the learning policy.

---

## 11. Logging (needed for the metrics)

The opponent (or the environment on its behalf) must log:

| Event | Fields | Used for |
|---|---|---|
| Formation switch | `t`, from, to, mode | Detection delay, adaptation lag (Q2) |
| Pass | `t`, passer, receiver, completed or intercepted by whom | Behaviour stats, playmaker analysis (Q3) |
| Playmaker involvement | passes received and made by `sigma`, interceptions by `sigma` | Q3 identification |
| Anchor distance | each step, mean distance of each opponent player to its anchor | Formation readability check (Section 13) |
| Tackles, shots, goals | standard | All |

---

## 12. Parameters (all TBD, decide in Phase 0)

| Parameter | Meaning |
|---|---|
| `R_ZONE` | Zone radius around the anchor |
| `OMEGA` | Weight of the ball-side drift, in [0, 1) |
| `EPS_ARRIVE`, `TRANSITION_MAX` | Arrival tolerance and maximum transition length |
| `DWELL_MIN` | Minimum steps between formation switches |
| `T_SWITCH`, `G_TARGET` | Scheduled switch time and target |
| `H_SWITCH` | Per-step switch probability in `RANDOM` mode |
| `CHECK_INTERVAL`, `TRAIL_THRESH`, `LEAD_THRESH` | `SCORE_REACTIVE` mode |
| `F_ATT`, `F_DEF` | Attacking and defensive formation choices |
| `SHOOT_MIN` | Minimum scoring estimate to shoot |
| `W_FWD`, `W_OPEN`, `W_DIST`, `PASS_MIN` | Pass scoring weights and threshold |
| `HUB_BONUS` | Pass bonus toward the playmaker |
| `D_DANGER` | Distance within which a visible defender blocks dribbling |
| `SUPPORT_ADV`, `SUPPORT_MID`, `SUPPORT_DEF`, `R_SUPPORT` | Off-ball attacking shifts and cap |
| `EPS_OPP` | Optional randomness (default off) |

A config skeleton:

```yaml
opponent:
  zone: {r_zone: null, omega: null, eps_arrive: null, transition_max: null}     # TBD
  switching:
    mode: NONE                  # NONE | SCHEDULED | RANDOM | SCORE_REACTIVE
    dwell_min: null
    scheduled: {t_switch: null, target: null}
    random: {h_switch: null}
    score_reactive: {check_interval: null, trail_thresh: null, lead_thresh: null,
                     f_att: null, f_def: null}
  attack: {shoot_min: null, w_fwd: null, w_open: null, w_dist: null, pass_min: null,
           d_danger: null, hub_bonus: null}
  support: {adv: null, mid: null, def: null, r_support: null}
  randomness: {eps_opp: 0.0}    # default off
  playmaker: {sigma: 0}         # 0 = none; otherwise the player number
```

---

## 13. Build Order and Phase 0 Calibration

Build and test in this order. Do not move on until each step passes its test.

| Step | Build | Test |
|---|---|---|
| 1 | Zonal defence and hold-anchor only (no attack) | Players stay near anchors. Plot positions over a match |
| 2 | Loose-ball rule and pressing arbitration | Only one player presses. No swarming. Formation stays readable |
| 3 | Attack with the ball: dribble only, then shoot, then pass, then recycle | The team scores sometimes against a passive opponent |
| 4 | Off-ball support | The carrier usually has an open teammate |
| 5 | Heuristic vs heuristic matches | Results are roughly balanced (no side advantage) |
| 6 | Random policy vs the heuristic | The random policy **loses clearly** |
| 7 | Formation switch modes, one at a time | Switch events are logged. Players walk (no teleport). Transitions end |
| 8 | Playmaker variant | See check C below |

**Phase 0 checks (all must pass before training our agent):**

- **A. Formation readability.** The average distance of opponent players from their anchors must be **much smaller** than the average distance between the corresponding anchors of two different formations. Otherwise the formations cannot be told apart and Q2 is not well posed. If this fails, lower `R_ZONE`, `OMEGA` and `R_SUPPORT`.
- **B. Vision matters.** The same heuristic team with a small radius loses to itself with a large radius. If not, vision is not affecting play and Q1 will show nothing, so adjust the constants (for example the interception and pass-safety rules).
- **C. Playmaker works.** A heuristic team with a playmaker beats the same team without one. If not, increase the playmaker's effect (for example `HUB_BONUS`), or check that his information is really being used in Sections 6 and 9.
- **D. Balance and determinism.** Heuristic vs heuristic is roughly balanced, and the same seed gives the same match.
- **E. Formation win matrix.** Run formation vs formation matches with the heuristic and store the matrix (it is the benchmark for Q1 and feeds the `SCORE_REACTIVE` mapping).

Calibration order for constants: movement and zone (`R_ZONE`, `OMEGA`) first, then attack (`SHOOT_MIN`, pass weights), then support, then playmaker (`HUB_BONUS`). Freeze the values in the config and record them in the report.

---

## 14. Common Mistakes

- **Man-marking by accident.** If a player follows one of our players around, it will leave its zone. Always make the target a point defined by the **anchor** and the **ball**, never by one of our players.
- **Using the true ball position when the ball is not visible.**
- **Letting several players press at once** (destroys the formation).
- **Letting the opponent see more than its radius** when computing openness or pass safety (it gives the opponent a free advantage and invalidates Q2 and Q3).
- **Instant formation switches** (teleporting). Always walk to the new anchors.
- **Making the opponent too strong or too weak.** A strong opponent makes every learning curve flat, a weak one makes Q2 and Q3 invisible.
- **Mixing random streams.** Keep one separate seeded stream for the opponent.
- **Hardcoding constants** in the code instead of reading them from the config.

---

## 15. Summary (one paragraph for the team)

The opponent is a rule-based team that **keeps its formation**: each player stays near its anchor, presses the ball only when it enters its zone (one presser at a time), and otherwise drifts only slightly toward the ball. In attack, the carrier shoots, passes to the best safe teammate, dribbles, or recycles, using only what it can see. Off-ball teammates make small, capped runs. The team can switch formation in three ways (scheduled, random, score-reactive) by walking to new anchors, and it can have one playmaker with full-pitch vision whose passes are better informed and who receives extra passes from his teammates. It plays by the same physics and with the same information rules as our team, except for the playmaker's wider vision.