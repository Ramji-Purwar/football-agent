# training.md: Training Protocol and MAPPO Implementation

This document covers the full training pipeline: the MAPPO algorithm, GRU sequence handling, reward engineering, curriculum, hyperparameters, and what to log. Read `game.md` (Sections 9–12) and `agents.md` first.

---

## 0. The Trainer (`training/train.py`)

Sections 1 to 6 describe the method. This section is how to run the code that implements it. Run everything from `football_adapt/`. It needs `torch` (the CPU build is enough: the networks are small and the time goes into the game simulation).

```bash
# Stage 1: short matches, full vision
python -m training.train --name sanity --T 300 --radius 100 --total-steps 600000 --workers --opp-eps-start 0.5
# Stage 2: full matches, default vision
python -m training.train --name easy --total-steps 2000000 --workers --opp-eps-start 0.5
# a grid of runs, one core each (here the Q1 grid with 3 seeds)
python -m training.sweep --jobs 28 --formations 2-2-1,2-1-2,1-3-1,1-2-2,3-1-1 --radii 12,25,40,120 --zetas 0,1,2 \
       --seeds 0,1,2 -- --total-steps 5000000 --opp-eps-start 0.5
# rule-based start (Section 7): copy the hand-written team, then train against the full-strength opponent
python -m training.pretrain --out runs/warm.pt --set actions.context_mask=true
python -m training.train --name warm --workers --init runs/warm.pt --critic-warmup 10 --entropy-coef 0.001 \
       --shaping-anneal-end 1000000000 --eval-matches 50 --set actions.context_mask=true
```

| What | How |
|---|---|
| Output | `runs/<name>/`: `config.json`, `log.jsonl` (one line per update and per evaluation), `ckpt.pt` and, when the run is finished, `result.json` |
| Resume | Run the same command again. It continues from `ckpt.pt`; a finished run is skipped. Use this when a node expires |
| One fast run | `--workers`: each of the `--n-envs` games (default 8) gets its own process |
| Many runs at once | `training.sweep --jobs N`: one process per run, `N` at a time. Everything after `--` goes to every run |
| The setting | `--our-formation`, `--radius`, `--zeta` are fixed for a run. `--opp-formations`, `--opp-styles`, `--sigmas` take comma lists, and one value is drawn at every kick-off. `--switching` sets the opponent's switching mode, `--T` the match length |
| Any other config value | `--set key=value`, for example `--set reward.w_progress=0.005` (can be repeated) |
| Actor | `--actor mlp` (memoryless) or `--actor gru`. The belief-augmented and oracle variants are not written yet |
| Easy start | `--opp-eps-start 0.5`, see Section 7 |
| Rule-based start | `training.pretrain` writes a file, `--init` starts from it, see Section 7 |
| Evaluation | Every `--eval-interval` steps, `--eval-matches` matches on fixed seeds against the opponent at full strength; `--final-eval-matches` at the end |
| Watching a trained team | `python -m scripts.pygame_render --ckpt runs/<name>/ckpt.pt` (it is the blue team). Add `--opp-eps 0.45` for the weakened opponent it trains against. `scripts.watch_match` takes the same two flags and writes a GIF |
| Using a trained team in code | `MAPPOAgent.load("runs/<name>/ckpt.pt")` in `agents/mappo.py` has the same `act(obs)` as `RandomAgent` |

Speed: one core plays about 375 game steps per second. With `--workers` on 8 games a 12-core laptop reached about 650 to 830 steps per second for the MLP actor. The GRU update costs more (about 3.6 s per 2048 game steps on one core against 6 s of simulation).

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
| `n_epochs` | 5 | Update epochs per rollout (`--epochs`). 10 doubles the time spent in the update |
| `batch_size` | 512 | Minibatch size (steps × players) |
| `entropy_coef` | 0.01 | Entropy bonus coefficient. 0.001 with the rule-based start (Section 7) |
| `value_coef` | 0.5 | Value loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `SEQ_LEN` | 32 | GRU sequence chunk length |
| `ROLLOUT_LEN` | 256 | Steps per game between updates (`--rollout-len`): 2048 game steps per update with 8 games |
| `n_envs` | 8 | Games played side by side (`--n-envs`) |
| `total_steps` | 10M–50M | Total environment steps |
| `shaping_anneal_end` | 5M steps | Steps at which `lambda_t → 0` |

These are the defaults of `training/train.py`, and each is a flag there. They are starting points. Tune during the first easy-setting runs (one formation, full vision).

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

With the easy start (Section 7) the trainer does not start this anneal at step 0. The weight stays at 1.0 while the opponent is still weakened, and the anneal over `--shaping-anneal-end` steps starts at the step where the opponent first reaches full strength. The reason is a run in which the weight had fallen to 0.6 while the opponent was still 40% random: the attack faded (13 shots a match down to 4). Two runs restarted from the same checkpoint with the weight held at 1.0 kept shooting 6 to 11 times a match.

The shaping terms are listed in `game.md`, Section 9. Two of them matter for getting a team to attack at all: `w_progress` pays for moving the ball toward the opponent's goal (without it a new policy never reaches the shooting zone), and `w_pass` is 0 because a bonus per completed pass was farmed by passing back and forth.

---

## 7. Training Curriculum

Run these stages in order. Mark the transition step on all learning curves.

### Easy start: a weaker opponent first (`--opp-eps-start`)

Against the scripted team at full strength a new policy does not learn to attack. It never gets near the goal, so it never sees a shot or a goal, and the best thing it finds is to keep the ball away from the opponent. The trainer therefore starts against a weakened opponent and makes it stronger as our team improves.

- `--opp-eps-start 0.5`: each opponent player acts at random on half of its steps (`opponent.randomness.eps_opp`, see `opponent.md`).
- After every 50 training matches the trainer looks at our win score (wins + draws / 2) over those matches. At 0.6 or more the randomness drops by 0.05, down to the config value (0 = full strength). At 0.4 or less it goes back up by 0.05, up to the start value. In between it stays. The four numbers are `EPS_STEP`, `EPS_WINDOW`, `EPS_HARDER` and `EPS_EASIER` at the top of `training/train.py`.
- Evaluations are always against the opponent at full strength, so the evaluation lines are the honest measure. Every training line in `log.jsonl` carries `opp_eps`: how far it has fallen shows how far a run has come.

What the first runs showed (one seed each, 2-2-1 against 2-2-1, memoryless actor, a laptop):

| Run | What happened |
|---|---|
| Full-strength opponent from step 0, 600k steps, 300-step matches, full vision (six variants of the reward and rules) | No wins and almost no shots in any of them. The team keeps the ball deep in its own half, where the scripted team did not come for it at the time (it does now: `opponent.md`, 5.9) |
| Opponent 25% random, same setting | The same: no attack in 600k steps |
| Opponent 50% random, with `w_progress` | Attacks from about 250k steps. At 600k: 60% wins against that opponent, one goal per 300-step match, the ball in their third 43% of the time |
| Opponent 50% random, `w_progress` off | No attack in 600k steps: one shot in 40 evaluation matches |
| Opponent 50% random, full matches, 25 m vision, `w_pass = 0.02` | Pass farming: 286 passes a match, no shots |
| The same with `w_pass = 0` | Shots rise from 0 to about 10 a match by 600k steps, close to one goal a match. Repeated against the scripted team that hunts a parked ball (`opponent.md`, 5.9): 12 shots and 0.9 goals a match at 600k, and no hiding (the ball in our third 21% of the time) |
| The attacking team, then the opponent made stronger on a fixed timetable (50% to 0 in 500k steps) | The attack faded as fast as the opponent improved. No wins at full strength |
| The attacking teams, then the opponent made stronger only as they got ahead (the rule above), 600k more steps | Full matches: from 50% down to 40% random. Short matches: from 35% to between 25% and 30%. Against the full-strength opponent both still lose every match (about 0-7 over a full match) |
| The full-match team continued to 3M steps, shaping annealed from step 0 | It reached 35% random at 1.6M steps with 10 to 13 shots a match. Then the attack faded (about 4 shots a match, the ball in their third 8% of the time) and the opponent was eased back to 45%. No win against the full-strength opponent at any evaluation (0-7 at the end) |
| The same from the 1.2M checkpoint with the shaping weight held at 1.0, two seeds, to 2.4M steps | The attack did not fade (6 to 11 shots a match throughout). The opponent still did not get stronger: one run moved between 35% and 40% random, the other slid back to 50%. No win against the full-strength opponent (between 0-6 and 0-7) |
| Open passes only, and receiving automatic (`game.md`, Section 8.2), 600k steps | Before these two the teams above made under one pass a match: half of their passes were intercepted and most of the rest landed where nobody was. With them: 15 to 19 passes a match, about 90% completed, and the opponent could be strengthened from 50% to 15% random in 540k steps. Against the full-strength opponent: 3 wins and 2 draws in 20 matches, 2.4 goals for and 5.2 against a match. These are the first wins at full strength |

So the easy start gets a team that attacks, and the trainer keeps the opponent at a level the team can just beat. Without passing, no team competed with the full-strength scripted team, and training longer did not get there: every long run stalled at an opponent that is 35% to 50% random. Keeping the shaping weight up keeps the attack alive (which is why the anneal now waits, Section 6), but it did not lift that ceiling. The carrier dribbled alone. What lifted it was making passing work (the last row): the team still loses more than it wins at full strength, but it now scores there and wins some matches.

### Rule-based start: copy a hand-written team first (`--init`)

The easy start gets a team that knows what to do with the ball and little else. Three of its teams were measured after 1M steps, over 28 full matches each against the full-strength opponent:

- **The four players off the ball move as a random team's do.** Of the ground such a player covers, 27 to 33% takes him anywhere (distance gained over distance run, in windows of 20 steps). For a random policy it is 22 to 25%, for the scripted team 73 to 76%.
- **The weakened opponent does not attack.** Against a sound defence the opponent at 50% random takes 0.0 shots a match, at 15% random 0.8, at full strength 3.5. While it is weak there is nothing to defend against, and the latest runs all stalled at an opponent that is 10 to 25% random.
- **The signal is weak.** Goals are nearly all of the reward. A critic fitted afresh on matches of a trained team explains 12% of the goal return in matches it has not seen, and one update (2048 steps) holds about 7 goals.

So a new policy is not asked to find a whole team by itself. It starts as a copy of one:

1. **The rules** (`training/rules.py`). A team in about 25 lines, with the learner's own observation, mask and 23 actions. Without the ball the two players nearest to it go for it (`GO_TO_BALL`, `TACKLE` when in range) and the others `HOLD_SHAPE`. The carrier shoots at a chance of 30% or better. If not, he plays the open pass that gains the most ground (5 m or more), and if there is none, `DRIBBLE_GOAL`. Against the full-strength opponent: 0.9 goals for and 2.0 against a match (42 matches).
2. **The copy** (`training/pretrain.py`). The rules play 96 matches against the scripted team. 15% of the choices are swapped for a random one, so that the rules are also seen in the situations a learner's mistakes lead to. The actor is trained to make the rules' choices, and the critic to predict the return of that play. It takes about 2 minutes, and the actor then makes the rules' choice on 97.5% of the steps kept aside.
3. **PPO from there** (`training.train --init runs/warm.pt`), against the full-strength opponent from step 0: no `--opp-eps-start`.

The settings that go with it (the commands are in Section 0):

| Setting | What it does |
|---|---|
| `--entropy-coef 0.001` | With the default 0.01 the copy dissolves: the entropy climbs from 0.06 to 1.25 in 600k steps, and the team ends no better than the copy (table below) |
| `--critic-warmup 10` | For the first 10 updates only the critic learns, so the copied actor is not moved before the critic can judge its play |
| `--set actions.context_mask=true`, for `pretrain` and `train` alike | Off the ball a player chooses among `STAY`, `HOLD_SHAPE`, `GO_TO_BALL`, `TACKLE` and `SHADOW_j`, not among the 8 directions (`game.md`, Section 7). The trainer warns when the two commands were given different `--set` values |
| `--shaping-anneal-end 1000000000` | Keeps the shaping weight at 1 |
| `--eval-matches 50` | The goal difference spreads by about 1.6 goals from match to match, so the mean of 20 matches is uncertain by about 0.35 goals, the mean of 50 by about 0.2 |

Only the entropy setting has been compared with its alternative. The other four are the recipe as it was run.

**These results are against the opponent as it was before the receiver's call (`opponent.attack.call`, `opponent.md` Section 6.2).** That opponent passed to teammates marked by a defender the passer could not see, and the trained teams took the ball from 15 to 20 such passes a match. Against the opponent with the call, the two 2M-step teams score W 0.07 and 0.08 (60 matches each), so the numbers below show that the recipe trains a team, not how good a team is against today's opponent.

Against the full-strength opponent (2-2-1 against 2-2-1, memoryless actor, 56 full matches a row unless it says otherwise, W = wins + draws / 2):

| Team | Goals for – against | Wins-draws-losses | W |
|---|---|---|---|
| Easy start, 1M steps, three runs (28 matches each) | 0.5 to 1.1 – 4.2 to 5.5 | 0-1-27 to 1-0-27 | 0.02 to 0.04 |
| Easy start with `context_mask`, 1M steps, one run (28 matches) | 0.75 – 2.64 | 4-2-22 | 0.18 |
| The copy of the rules, not trained | 0.36 – 2.27 | 3-9-44 | 0.13 |
| The copy + 600k steps with `--entropy-coef 0.01` | 0.57 – 2.59 | 5-6-45 | 0.14 |
| The copy + 600k steps with `--entropy-coef 0.001`, seed 0 | 0.75 – 1.36 | 12-11-33 | 0.31 |
| The same, seed 1 | 0.98 – 1.66 | 11-15-30 | 0.33 |
| Seed 0 carried on to 1.5M steps | 1.11 – 1.34 | 15-16-25 | 0.41 |
| The scripted team against itself (64 matches) | 1.34 – 1.41 | | 0.48 |

With the lower entropy bonus the copy improves, on both seeds, and at 1.5M steps it is at the Stage 2 target (W above 0.4 within 2M steps). That last row is one seed. The trainer's own final evaluations, which are other matches, agree: W 0.31 (100 matches) and 0.33 (30 matches) at 600k steps, and 0.42 at 1.5M (30 matches).

The rise is not steady. The training matches are against the full-strength opponent too. In those of seed 0, W by 100k steps was 0.21, 0.19, 0.21, 0.21, 0.14 and 0.24 up to 600k, then 0.28, 0.33, 0.43 and 0.54 up to 1M, and between 0.36 and 0.56 from there to 1.5M. Seed 1 was as flat for its first 500k steps (0.20 to 0.28) and reached 0.36 in the last 100k. The entropy climbs from 0.06 to about 0.6 over the first 500k steps and then settles. So a run of this kind shows little in its first 500k steps, and its evaluations move with it: for seed 0 they were 0.25 at 250k steps and 0.13 at 500k (50 matches each), before the 0.31 at 600k.

What these teams do, and what they do not do yet:

- **They defend.** The opponent gets 2.7 to 4.3 shots a match (7.6 to 11.1 against the easy-start teams), and the four players not nearest the ball stand 6 to 9 m from their spots (15 to 19 m).
- **They do not build up.** They have the ball 21% of the time, complete 16 to 21 passes a match, and score from shots taken 32 m out at a chance of 8 to 9% (8 to 13 shots a match).
- **Not written yet:** a copy for the recurrent actor (`pretrain` writes the memoryless one), and rules for structure level 0, which has no `HOLD_SHAPE`.

### Stage 1: Sanity Check

Config: `T=300`, single formation `"2-2-1"`, full vision (`radius=100`), no switching, no playmaker, `zeta=2`.

Expected: random policy gets near-zero win rate within 100k steps. If not, the reward signal is broken.

### Stage 2: Easy Setting

Config: `T=1200`, single formation, `radius=25` (default), `zeta=2`.

Expected: trained policy clearly beats random (win rate > 80%) and approaches heuristic baseline (win rate > 40% against heuristic) within 2M steps. The easy start has not reached this target (first table above). The rule-based start reached W 0.41 at 1.5M steps, on one seed (second table).

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

`training/train.py` writes `runs/<name>/log.jsonl`. A `"kind": "train"` line after every update holds the losses, the entropy, `shaping_weight`, `opp_eps` and the mean over the training matches that finished in that update of: `win`, `draw`, `goal_diff`, `goals_for`, `shaping_return`, `passes`, `pass_completion`, `shots`, `possession`, `ball_own_third` / `ball_mid_third` / `ball_their_third`, `invalid_action_rate`, `opp_anchor_dist`. A `"kind": "eval"` line every `--eval-interval` steps (and a `"final"` line at the end) holds the same match fields plus `W`, from matches on fixed seeds against the opponent at full strength. `goal_diff_over_time` and the Q2 and Q3 metrics below are not written yet.

Read the behaviour fields, not only the win rate. A team that is not attacking shows `shots` near 0 and `ball_their_third` under 0.1; a team that is hiding the ball shows high `possession` with `ball_own_third` above 0.5.

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

As implemented in `training/train.py`: without `--workers` the games of a run are stepped one after the other in one process. With `--workers` each game has its own process, which holds a copy of the policy, plays its whole rollout alone and sends the buffer back. Sending every single step through a pipe instead was more than four times slower, because the processes wait for each other at every step. Each run uses one thread for the update (`--torch-threads`); more threads make several runs on one machine fight over the cores.

The sketch below is the simple idea behind it:

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
| Hundreds of passes a match, no shots | The pass bonus is being farmed | Keep `reward.w_pass` at 0 (forward passes are paid by `w_progress`) |
| Next to no passes | Passing loses the ball too often to be worth it | Keep `actions.open_pass` on and receiving automatic (`game.md`, Section 8.2). Check `passes` and `pass_completion` in the log |
| No shots, `ball_their_third` under 0.1, many 0-0 draws | The opponent is too strong for a new policy to get near the goal | Use the easy start (`--opp-eps-start 0.5`) and keep `reward.w_progress` on |
| High possession, `ball_own_third` above 0.5, draws | The team is hiding the ball in its own half | The scripted team goes and gets a parked ball (`opponent.hunt`, `opponent.md` 5.9): check `hunt.after` is not 0 |
| Good against the weakened opponent, heavy defeats in the evaluations | The opponent has not been strengthened far enough yet | Train longer; watch `opp_eps` fall in `log.jsonl` |
| Heavy defeats at full strength however long the run, the players off the ball wander | From random play only the carrier learns (Section 7) | Use the rule-based start: `training.pretrain`, then `--init` |
| With `--init`: the entropy climbs and the team gets worse than the copy it started as | The entropy bonus undoes the copied policy | `--entropy-coef 0.001` |
| The evaluations of one run jump about | 20 matches are too few, and with the rule-based start the team itself moves up and down by 0.1 in W between evaluations (Section 7) | `--eval-matches 50`, and judge a run over several evaluations |
| GRU not learning Q2 | Sequences too short | Increase `SEQ_LEN` to 64 or 128 |
| Q2 belief filter beats recurrent | Hidden state not used | Check that hidden state is propagated (not reset each step) |
| NaN loss | Learning rate too high | Reduce `lr_actor`, add gradient clipping |
| Training very slow | `n_envs = 1` | Increase parallel envs to 8–16 |
| Formation probe near chance | Policy not learning to infer | Inspect hidden state activation patterns; check `opp_anchor_dist` (if too high, formation is not readable) |
