# training.md: Training Protocol and MAPPO Implementation

This document covers the full training pipeline: the MAPPO algorithm, GRU sequence handling, reward engineering, curriculum, hyperparameters, and what to log. Read `game.md` (Sections 9–12) and `agents.md` first.

---

## 1. Algorithm: Parameter-Shared MAPPO

We use **Multi-Agent PPO (MAPPO)** with full parameter sharing: all 5 of our players run the **same** actor network and share the **same** centralized critic.

**Key design choices:**
- Centralized training, decentralized execution (CTDE)
- Actor input: per-player observation vector (79-dim)
- Critic input: full global state (51-dim) — sees everything including hidden truth
- One shared actor update per training batch (gradients from all 5 players)
- One shared critic update per training batch

**Why parameter sharing works here:**
- All players have the same action set
- Each player's observation is in its own team frame (so the network sees a normalized, symmetric view)
- The slot one-hot in the observation tells the network which player it is, enabling role-specific behavior

---

## 2. Network Architectures

### 2.1 Memoryless Actor (Q1)

```python
import torch.nn as nn

class MemorylessActor(nn.Module):
    def __init__(self, obs_dim=79, n_actions=23, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, obs):            # obs: (..., 79) → logits: (..., 23)
        return self.net(obs)
```

### 2.2 Recurrent Actor (Q2, Q3)

```python
class RecurrentActor(nn.Module):
    def __init__(self, obs_dim=79, n_actions=23, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU()
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs, h=None):
        # obs: (batch, seq, 79) or (batch, 1, 79) at inference
        enc = self.encoder(obs)         # (batch, seq, hidden)
        out, h_new = self.gru(enc, h)   # out: (batch, seq, hidden), h_new: (1, batch, hidden)
        logits = self.head(out)         # (batch, seq, 23)
        return logits, h_new
```

### 2.3 Centralized Critic

```python
class Critic(nn.Module):
    def __init__(self, state_dim=51, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):          # state: (..., 51) → value: (..., 1)
        return self.net(state)
```

### 2.4 Belief-Augmented Actor

Same as `MemorylessActor` but `obs_dim = 79 + 5 = 84`. Prepend the belief vector to `obs` before calling.

---

## 3. GRU Sequence Handling (Recurrent Agent)

### 3.1 Episode and Sequence Structure

- Each episode has `T = 1200` steps.
- Train on **sequences of length `SEQ_LEN = 32`** (consecutive steps within an episode).
- At episode start: reset hidden state `h = torch.zeros(1, 1, hidden)` for each of the 5 players.
- Within an episode: propagate `h` step by step (detach from the gradient graph at sequence boundaries).

### 3.2 Rollout Buffer for Recurrent PPO

Store these per step, per player:

| Field | Shape per step | Notes |
|---|---|---|
| `obs_vec` | `(5, 79)` | Observation for each player |
| `mask` | `(5, 23)` | Action mask |
| `action` | `(5,)` | int action taken |
| `log_prob` | `(5,)` | log π(a|o) |
| `value` | `(5,)` | critic estimate (same global state for all) |
| `reward` | `(1,)` | shared team reward |
| `done` | `bool` | episode done flag |
| `hidden` | `(5, 1, 1, 256)` | hidden state BEFORE this step |
| `global_state` | `(51,)` | for critic |

After `ROLLOUT_LEN` steps (e.g. 1200 = full episode or shorter), compute GAE and update.

### 3.3 Sequence Batch Construction

```python
def make_sequence_batch(buffer, seq_len=32):
    """Slice buffer into sequences of length seq_len, keeping hidden states at sequence starts."""
    N = len(buffer)
    sequences = []
    for start in range(0, N - seq_len + 1, seq_len):
        seq = {k: buffer[k][start:start+seq_len] for k in buffer}
        seq["h0"] = buffer["hidden"][start]  # initial hidden for this sequence
        sequences.append(seq)
    return sequences
```

### 3.4 Gradient through Time

```python
# Forward pass for a sequence batch:
logits, h = actor(obs_batch, h0)          # obs_batch: (B, seq_len, 79)
# logits: (B, seq_len, 23)
# Do NOT backprop through h across sequence boundaries (detach h0)
h0 = h0.detach()
```

---

## 4. PPO Update

### 4.1 GAE Computation

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    last_adv = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages.insert(0, last_adv)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns
```

### 4.2 PPO Loss

```python
def ppo_loss(logits, actions, old_log_probs, advantages, masks,
             clip_eps=0.2, entropy_coef=0.01):
    dist = MaskedCategorical(logits, masks)
    log_probs = dist.log_prob(actions)
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy = dist.entropy().mean()
    return policy_loss - entropy_coef * entropy

def value_loss(values, returns, clip_eps=0.2):
    return F.mse_loss(values, returns)
    # or clipped version:
    # v_clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    # return torch.max(F.mse_loss(values, returns), F.mse_loss(v_clipped, returns)).mean()
```

### 4.3 Masked Categorical

The action mask must be applied before computing the distribution:

```python
class MaskedCategorical:
    def __init__(self, logits, mask):
        logits = logits.clone()
        logits[~mask] = float("-inf")
        self.dist = torch.distributions.Categorical(logits=logits)

    def sample(self):
        return self.dist.sample()

    def log_prob(self, action):
        return self.dist.log_prob(action)

    def entropy(self):
        return self.dist.entropy()
```

---

## 5. Hyperparameter Reference

| Parameter | Value | Notes |
|---|---|---|
| `hidden_dim` | 256 | MLP and GRU hidden size |
| `lr_actor` | 3e-4 | Adam optimizer |
| `lr_critic` | 1e-3 | Adam optimizer (higher than actor) |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clip range |
| `n_epochs` | 10 | Update epochs per rollout |
| `batch_size` | 512 | Minibatch size (steps × players) |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `SEQ_LEN` | 32 | GRU sequence chunk length |
| `ROLLOUT_LEN` | 1200 | Steps per rollout (full episode) |
| `n_envs` | 8–16 | Parallel environments |
| `total_steps` | 10M–50M | Total environment steps |
| `shaping_anneal_end` | 5M steps | Steps at which `lambda_t → 0` |

These are starting points. Tune during the first easy-setting runs (one formation, full vision).

---

## 6. Shaping Anneal Schedule

The shaping weight `lambda_t` is annealed from 1.0 to 0.0 over training:

```python
def shaping_weight(step, anneal_end=5_000_000):
    return max(0.0, 1.0 - step / anneal_end)

# Each rollout:
w = shaping_weight(total_steps_so_far)
env.set_shaping_weight(w)
```

The `shape_cap = 5.0` per episode hard-caps total shaping regardless of `lambda_t`.

---

## 7. Training Curriculum

Run these stages in order. Mark the transition step on all learning curves.

### Stage 1: Sanity Check

Config: `T=300`, single formation `"2-2-1"`, full vision (`radius=100`), no switching, no playmaker, `zeta=2`.

Expected: random policy gets near-zero win rate within 100k steps. If not, the reward signal is broken.

### Stage 2: Easy Setting

Config: `T=1200`, single formation, `radius=25` (default), `zeta=2`.

Expected: trained policy clearly beats random (win rate > 80%) and approaches heuristic baseline (win rate > 40% against heuristic) within 2M steps.

**Check:** if the policy does not beat random after 2M steps:
- Check that masks are applied correctly (high `invalid_actions` count = mask bug)
- Check that shaping is being received (`reward_parts["shaping"]` should be positive)
- Check that the critic is seeing global state (not per-player obs)

### Stage 3: Q1 — Vision and Structure

Config: all 5 formations, three vision radii (`12, 25, 40`), all structure levels (`zeta=0,1,2`).

Run a separate training for each `(formation, radius, zeta)` combination. Each run: 5M steps.

### Stage 4: Q2 — Formation Switching

Config: `zeta=2`, `radius=25`, opponent switching modes: SCHEDULED, RANDOM, SCORE_REACTIVE.

Use the recurrent actor (GRU). Compare all 4 agent variants (memoryless, recurrent, belief-augmented, oracle). Each run: 10M steps.

### Stage 5: Q3 — Playmaker

Config: `zeta=2`, `radius=25`, `sigma` = D slot / M slot / F slot (3 runs per agent variant).

Use the recurrent actor. Each run: 10M steps.

---

## 8. What to Log

Log these every `EVAL_INTERVAL = 100_000` training steps:

### Per-Episode Metrics

```python
log = {
    "step": total_steps,
    "win_rate": wins / n_eval_matches,
    "draw_rate": draws / n_eval_matches,
    "goal_diff": mean goal difference per match,
    "shaping_return": mean total shaping per episode,
    "goal_diff_over_time": [score after each 100 steps, averaged over matches],
    # Behaviour stats from env.stats:
    "pass_completion": passes_completed / passes,
    "shots_per_match": shots / n_eval_matches,
    "possession_share": possession_steps[0] / T,
    "invalid_action_rate": invalid_actions[0] / (T * 5),
    # Opponent anchor distance:
    "opp_anchor_dist": mean of env.opp_anchor_dist per episode,
}
```

### Q2-Specific Metrics

```python
# From env.event_log, for each switch event:
detection_delay = steps from switch until inferred_formation == true_formation
adaptation_lag = steps until rolling goal_diff recovers to pre-switch level

log["detection_delay_mean"] = ...
log["adaptation_lag_mean"] = ...
log["regret_vs_oracle"] = W_oracle - W_recurrent
```

### Q3-Specific Metrics

```python
log["playmaker_cost"] = W_no_playmaker - W_with_playmaker (for unaware agent)
log["recovery_ratio"] = (W_adaptive - W_unaware) / playmaker_cost
log["shadow_rate"] = shadow_steps_on_sigma / shadow_steps_total
log["probe_formation_acc"] = accuracy of formation probe
log["probe_sigma_acc"] = accuracy of playmaker probe
```

### Config Logging

Save the full config dict with every run:

```python
import json, datetime
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"runs/{run_id}/config.json", "w") as f:
    json.dump(cfg.to_dict(), f, indent=2)
```

---

## 9. Parallel Environments

Use multiple environments in parallel to increase sample throughput:

```python
from multiprocessing import Pool

def make_env(seed):
    env = FootballEnv(cfg)
    obs, _ = env.reset(seed=seed)
    return env, obs

# Vectorized step (simplified)
class VecEnv:
    def __init__(self, n_envs, cfg):
        self.envs = [FootballEnv(cfg) for _ in range(n_envs)]

    def reset(self, seeds):
        return [env.reset(seed=s) for env, s in zip(self.envs, seeds)]

    def step(self, all_actions):
        return [env.step(a) for env, a in zip(self.envs, all_actions)]
```

For GRU agents, maintain separate hidden states per environment and per player: `h[env_idx][player_idx]`.

---

## 10. Evaluation Protocol

**Evaluation runs:** 100 matches per condition, separate from training environments.

**Paired evaluation:** use the same 100 seeds for every agent variant within a condition, so comparisons are paired (reduces variance from environment randomness).

```python
EVAL_SEEDS = list(range(10_000, 10_100))  # fixed for all evaluations

def evaluate(agent, cfg, n_matches=100):
    results = []
    for seed in EVAL_SEEDS[:n_matches]:
        env = FootballEnv(cfg)
        obs, _ = env.reset(seed=seed)
        h = {i: torch.zeros(1, 1, 256) for i in range(5)}  # if GRU
        while not env.done:
            actions = {}
            for i in range(5):
                logit, h[i] = agent.actor(torch.tensor(obs["vec"][i]), h[i])
                mask = torch.BoolTensor(obs["mask"][i])
                logit[~mask] = float("-inf")
                actions[i] = Categorical(logits=logit).sample().item()
            obs, _, _, info = env.step(actions)
        results.append({"score": info["final_score"], "stats": env.stats})
    return results
```

**Win score:** `W = (wins + 0.5 * draws) / n_matches`. Report `mean ± std` over 5 training seeds.

---

## 11. Common Training Issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Win rate stuck at ~0 | Action mask not applied | Check `invalid_action_rate`; apply mask before sampling |
| High shaping, flat win rate | Shaping reward hacked | Reduce `shaping_weight`; add distance scaling to pass reward |
| GRU not learning Q2 | Sequences too short | Increase `SEQ_LEN` to 64 or 128 |
| Q2 belief filter beats recurrent | Hidden state not used | Check that hidden state is propagated (not reset each step) |
| NaN loss | Learning rate too high | Reduce `lr_actor`, add gradient clipping |
| Training very slow | `n_envs = 1` | Increase parallel envs to 8–16 |
| Formation probe near chance | Policy not learning to infer | Inspect hidden state activation patterns; check `opp_anchor_dist` (if too high, formation is not readable) |
