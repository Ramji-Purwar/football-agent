# agents.md: Agent Variants Reference

All agent variants used in the project, with their architecture, inputs, outputs, and when to use them. Read `game.md` (Section 10) for the observation format and `training.md` for the training protocol.

---

## 1. Overview

| Variant | Memory | Extra Input | Used for |
|---|---|---|---|
| Random | None | None | Lower bound, sanity check |
| Heuristic (Scripted) | None (rule-based) | None | Phase 0 baseline, both teams |
| Memoryless MLP | None | None | Q1 main agent, lower baseline in Q2 |
| Recurrent (GRU) | GRU hidden state | None | Main learner Q2, Q3 |
| Belief-augmented | None | Belief vector (5 floats) | Strong baseline Q2 |
| Oracle Q2 | None | True formation (5 floats) | Upper bound Q2 |
| Oracle Q3 | None | Playmaker index (6 floats) | Upper bound Q3 |

All learned variants use **parameter-shared MAPPO**: all 5 players run the same network weights. See `training.md` for the training algorithm.

---

## 2. Random Agent

**File:** `agents/random_agent.py`

**Architecture:** samples uniformly from valid (unmasked) actions.

```python
from agents import RandomAgent

agent = RandomAgent(seed=42)
actions = agent.act(obs)
# obs: {"vec": (5, 79), "mask": (5, 23), "state": (51,)}
# → {i: int} for i in 0..4
```

**Expected performance:** clearly loses to the heuristic opponent. This is a Phase 0 check: if it doesn't lose clearly, something is wrong with the reward or mechanics setup.

---

## 3. Heuristic (Scripted) Agent

**File:** `opponent/controller.py` (`ScriptedTeam` with `team=0`, `mode_override="NONE"`)

Used to drive our own side for Phase 0 checks (heuristic vs heuristic). Activate with:

```python
env = FootballEnv(cfg, scripted_ours=True)
```

See `opponent.md` for the full decision logic. When driving our side, `SHADOW_j` actions are not available (the scripted team doesn't use them).

**Expected performance:** roughly balanced against itself (≈50% win rate heuristic vs heuristic). Clearly beats the random agent.

---

## 4. Memoryless MLP Agent

**Architecture:**
```
Input: obs_vec (79,)
   → Linear(79, 256) + ReLU
   → Linear(256, 256) + ReLU
   → Action head: Linear(256, 23)  [logits, then masked softmax]
   → Value head: Linear(256, 1)    [for critic, sees full state (51,)]
```

**Input dim:** 79 (obs vector from `flatten(env, obs)`)
**Output:** action logit vector (23,); mask out invalid actions before sampling

**Used for:** Q1 (main agent at all three structure levels), lower baseline in Q2/Q3.

**Call interface:**
```python
# Actor (called 5 times per step, once per player)
logits = actor(obs_vec)           # (23,)
logits[~mask] = -1e9              # mask invalid actions
action = Categorical(logits=logits).sample()

# Critic (called once per step, central)
value = critic(global_state)      # (1,)
```

---

## 5. Recurrent Agent (GRU)

**Architecture:**
```
Input: obs_vec (79,)
   → Linear(79, 256) + ReLU        # encoder
   → GRU(256, 256)                 # hidden state h_t, shape (1, 256)
   → Action head: Linear(256, 23)  # logits
   → Value head: (see critic)
```

**Hidden state:** `h[team][player]` is a `(1, 1, 256)` tensor, one per player. Reset to zeros at episode start. Kept across steps within an episode.

**Used for:** main learner in Q2 and Q3. The hidden state accumulates what each player has seen, enabling inference of the opponent formation (Q2) and playmaker identification (Q3).

**Training:** sequences of length `SEQ_LEN` (default 32 steps). See `training.md` Section 3.

**Call interface:**
```python
# Actor step
logits, h_new = actor(obs_vec.unsqueeze(0), h_prev)  # add seq dim
action = sample_masked(logits, mask)
h = h_new

# At episode reset:
h = torch.zeros(1, 1, 256)
```

**Probes** (trained post-hoc on frozen policy):
```python
# Formation probe (Q2)
pred_formation = LinearProbe(h, n_classes=5)  # one per player, aggregated

# Playmaker probe (Q3)
pred_sigma = LinearProbe(h, n_classes=6)      # 6 = none + 5 player slots
```

---

## 6. Belief-Augmented Agent

**Architecture:** same MLP as the memoryless agent, but the input is extended with a **belief vector** computed externally by the Bayesian belief filter.

```
Input: concat(obs_vec (79,), belief (5,)) = (84,)
   → Linear(84, 256) + ReLU
   → Linear(256, 256) + ReLU
   → Action head: Linear(256, 23)
```

The belief vector is computed per player (each player has their own independent belief). This is a **non-learning inference baseline**: the formation inference is explicit and Bayesian, not learned.

**Used for:** strong non-learning baseline in Q2. If the recurrent agent does not beat this, its inference is not better than Bayes.

---

## 7. Oracle Agents

**Oracle Q2 (true formation):**
```
Input: concat(obs_vec (79,), formation_onehot (5,)) = (84,)
```

**Oracle Q3 (true playmaker):**
```
Input: concat(obs_vec (79,), sigma_onehot (6,)) = (85,)
   # 6 classes: index 0 = no playmaker, indices 1–5 = player 1–5
```

These agents are given the hidden information directly. They are **upper bounds only** and cannot be deployed in a real match. They bound the regret of the learning agents: if the learner's win rate approaches the oracle's, inference is near-optimal.

---

## 8. Belief Filter: Implementation Reference

**File:** `agents/belief_filter.py`

### 8.1 What It Does

Maintains a probability distribution `b_i(g)` over the 5 opponent formations, updated each step as each player gets new observations.

### 8.2 Update Equations

```
predict:  b~(g)  = (1 - H) * b_prev(g) + H / |F|
update:   b(g)  ∝ b~(g) * ∏_{j visible} Normal( opp_pos_j ; anchor_g(j), S_NOISE² * I )
normalize: b(g) = b(g) / Σ_g b(g)
```

Where:
- `H` = per-step hazard rate (assumed switch probability). Start with `H = 0.003` (≈ 1 expected switch per 333 steps in a 1200-step match)
- `S_NOISE` = noise absorbing how far opponents drift from anchors. Start with `S_NOISE = 5.0` m
- `|F| = 5` (number of formations)
- `anchor_g(j)` = anchor of player `j` under formation `g`, in team frame world coords

**Important:** only **visible** opponents `j` (where `obs.opp_visible[j]` is True) contribute to the likelihood product. Non-visible opponents contribute a factor of 1.0 (no information update).

### 8.3 Implementation Skeleton

```python
import numpy as np
from football.geometry import to_frame

class BeliefFilter:
    def __init__(self, cfg, book, team=0):
        self.cfg = cfg
        self.book = book
        self.team = team
        self.n_formations = len(book.names)
        self.H = 0.003        # hazard rate, tune in Phase 0
        self.S_NOISE = 5.0    # noise, tune in Phase 0
        self.b = None         # (5, 5) — belief per player per formation
        self.L, self.W = cfg.pitch.length, cfg.pitch.width

    def reset(self):
        self.b = np.ones((5, self.n_formations)) / self.n_formations

    def update(self, obs_by_player: dict) -> np.ndarray:
        """
        obs_by_player: {i: Obs} for i in 0..4.
        Returns aggregated belief (5,) — probability over formations.
        """
        for i, obs in obs_by_player.items():
            # predict step
            self.b[i] = (1 - self.H) * self.b[i] + self.H / self.n_formations

            # update step — for each visible opponent
            for j in range(5):
                if not obs.opp_visible[j]:
                    continue
                observed_pos = obs.opp_pos[j]  # team frame, absolute
                for f_idx, f_name in enumerate(self.book.names):
                    anchor_tf = self.book.anchors_team_frame(f_name)[j]
                    delta = observed_pos - anchor_tf
                    log_lik = -0.5 * np.dot(delta, delta) / (self.S_NOISE ** 2)
                    self.b[i, f_idx] *= np.exp(log_lik)

            # normalize
            s = self.b[i].sum()
            if s > 0:
                self.b[i] /= s
            else:
                self.b[i] = np.ones(self.n_formations) / self.n_formations

        # aggregate over all players (geometric mean, then renormalize)
        agg = np.exp(np.mean(np.log(self.b + 1e-12), axis=0))
        return agg / agg.sum()

    def argmax_formation(self) -> str:
        agg = self.update_last  # or call update() first
        return self.book.names[np.argmax(agg)]
```

### 8.4 Tuning `H` and `S_NOISE`

| Parameter | Effect if too high | Effect if too low |
|---|---|---|
| `H` | Belief resets too fast; slow detection on a non-switching opponent | Belief never updates; slow to detect a real switch |
| `S_NOISE` | All formations look equally likely; no discrimination | Belief collapses too fast on noisy opponents |

Tune against Phase 0 results: use `env.opp_anchor_dist` to estimate the typical anchor deviation (that gives `S_NOISE`), and use `DWELL_MIN = 150` to calibrate `H` (you want `H * 150 ≈ 0.3–0.5` so the belief loses most of its pre-switch confidence by the time a switch is "allowed").

---

## 9. Action Masking

All agents **must** apply the action mask before sampling. The mask is a `(23,)` bool array per player in `obs["mask"][i]`.

```python
# PyTorch way
mask = torch.BoolTensor(obs["mask"][i])  # (23,)
logits = actor(obs["vec"][i])             # (23,)
logits[~mask] = float("-inf")
action = torch.distributions.Categorical(logits=logits).sample().item()

# NumPy way (for heuristic/scripted)
valid = np.flatnonzero(obs["mask"][i])
action = int(np.random.choice(valid))
```

If the environment receives an invalid action anyway, it replaces it with `STAY` and increments `env.stats["invalid_actions"][team]`. Track this during training: a high invalid action count means the mask is not being applied correctly.

---

## 10. Multi-Agent Coordination

**Parameter sharing:** all 5 players run the **same** network. There is no communication between players during a step (decentralized execution). Players coordinate implicitly through the formation and through their observable shared world state.

**Critic:** uses `global_state(env)` (51-dim), which includes all positions, ball, true formation, and playmaker index. This is centralized training with decentralized execution (CTDE).

**Observation independence:** each player's observation is computed independently. The policy is called 5 times per step, once per player. In a GRU agent, each player maintains its own hidden state.

**No explicit communication:** if you want to add communication (e.g. CommNet or QMIX), it must be through the observation vector — the environment does not support passing messages between agents.
