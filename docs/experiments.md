# experiments.md: Experiment Protocol and Evaluation

This document specifies exactly how to run each experiment, which conditions to test, what metrics to compute, and how to report results. Read `game.md` (Sections 11–13) and `training.md` first.

---

## 1. Experiment Matrix

### Q1: Structure vs Vision

**Question:** Can formation structure compensate for limited vision? Does the best formation change with vision radius?

| Axis | Values |
|---|---|
| Formation `f` | All 5: 2-2-1, 2-1-2, 1-3-1, 1-2-2, 3-1-1 |
| Structure level `zeta` | 0 (no structure), 1 (information only), 2 (constraint) |
| Vision radius `R_VISION` | Small (12m), Medium (25m), Large (40m), Full (∞) |
| Agent variant | Memoryless MLP |
| Opponent | Scripted, fixed formation `"2-2-1"`, NONE switching |
| Training seeds | 5 |
| Eval matches per condition/seed | 100 |

**Total runs:** 5 formations × 3 zeta levels × 4 radii × 5 seeds = **300 training runs**

**Key outputs:**
- Structure benefit: `B_zeta(f, r) = W(zeta) - W(zeta=0)` at same `f`, `r`
- Best formation: `f*(r) = argmax_f W(f, r, zeta=2)`
- Formation win matrix (also needed for Phase 0)

---

### Q2: Formation Inference

**Question:** How fast can the team infer an opponent formation switch? Does memory help? Is explicit Bayes competitive?

| Axis | Values |
|---|---|
| Formation `f` | `"2-2-1"` (best from Q1 at medium vision) |
| Structure level `zeta` | 2 (fixed) |
| Vision radius | Medium (25m) |
| Agent variant | Random, Heuristic, Memoryless, Recurrent (GRU), Belief-augmented, Oracle Q2 |
| Opponent switching mode | SCHEDULED (switch at T/2 to `"1-2-2"`), RANDOM (`h=0.005`), SCORE_REACTIVE |
| Initial opponent formation `g0` | `"2-2-1"` |
| Training seeds | 5 |
| Eval matches per condition/seed | 100 |

**Total runs:** 6 agent variants × 3 switching modes × 5 seeds = **90 training runs**

**Key outputs:**
- Regret vs oracle: `W(oracle) - W(agent)` per variant
- Detection delay: steps after switch until inferred formation = true formation
- Adaptation lag: steps after switch until rolling win rate recovers
- Belief filter detection curve (probability of correct formation vs time post-switch)

---

### Q3: Playmaker Adaptation

**Question:** How much does a playmaker hurt? Can the team identify and shadow them? How much does adaptation recover?

| Axis | Values |
|---|---|
| Formation `f` | `"2-2-1"` |
| Structure level `zeta` | 2 |
| Vision radius | Medium (25m) |
| Playmaker slot | D (index 0), M (index 2), F (index 4) |
| Agent variant | Memoryless, Recurrent (GRU), Oracle Q3 |
| Training seeds | 5 |
| Eval matches per condition/seed | 100 |

**Total runs:** 3 playmaker slots × 3 agent variants × 5 seeds = **45 training runs**

**Key outputs:**
- Playmaker cost: `E(sigma) = W(no playmaker) - W(with playmaker, unaware)`
- Recovery ratio: `(W(adaptive) - W(unaware)) / E(sigma)`
- Shadow rate: fraction of steps some player shadows `sigma`
- Identification accuracy and delay (from playmaker probe)

---

## 2. Naming and Logging Convention

Every run is identified by a config file + a seed. Save both with results.

**Run ID format:** `{question}_{formation}_{zeta}_{radius}_{agent}_{opp_mode}_{sigma}_{seed}`

Examples:
- `Q1_221_z2_r25_mlp_none_s0_seed3`
- `Q2_221_z2_r25_gru_scheduled_s0_seed1`
- `Q3_221_z2_r25_gru_none_s2_seed0`

**Directory structure:**
```
runs/
  Q1/
    221_z0_r12_mlp_none_s0_seed0/
      config.json
      metrics.json
      checkpoint_final.pt
  Q2/
    ...
  Q3/
    ...
phase0/
  formation_win_matrix.json
  anchor_dist_stats.json
  vision_effect.json
  playmaker_effect.json
```

---

## 3. Metrics Reference

### 3.1 Primary Metrics (All Questions)

| Metric | Formula | Unit |
|---|---|---|
| Win score `W` | (wins + 0.5 × draws) / n_matches | [0, 1] |
| Goal difference | (our goals - their goals) / n_matches | goals/match |
| Goals scored per match | our goals / n_matches | goals/match |
| Goals conceded per match | their goals / n_matches | goals/match |

Report: **mean ± std over 5 training seeds**. Use paired evaluation seeds (same 100 seeds across all runs for the same question and condition).

### 3.2 Q1-Specific Metrics

| Metric | Formula |
|---|---|
| Structure benefit `B_zeta(f, r)` | `W(f, r, zeta) - W(f, r, zeta=0)` |
| Vision benefit | `W(f, r=40, zeta=2) - W(f, r=12, zeta=2)` |
| Best formation at radius `r` | `f*(r) = argmax_f W(f, r, zeta=2)` |
| Formation win matrix | `M[f1][f2]` = W of `f1` vs opponent using `f2` (heuristic vs heuristic) |

### 3.3 Q2-Specific Metrics

| Metric | Formula |
|---|---|
| Regret vs oracle | `W(oracle_Q2) - W(agent)` per variant |
| Detection delay | Median steps from switch time until `argmax(b) == true_g` for 3+ consecutive steps |
| Adaptation lag | Steps after switch until 20-step rolling W recovers to within 5% of pre-switch W |
| Formation accuracy (belief) | `accuracy = 1[argmax(b) == true_g]`, averaged over steps post-switch |
| GRU probe formation accuracy | Probe accuracy on held-out eval trajectories |

### 3.4 Q3-Specific Metrics

| Metric | Formula |
|---|---|
| Playmaker cost `E(sigma)` | `W(no_playmaker) - W(with_playmaker, unaware_mlp)` |
| Recovery ratio | `(W(recurrent) - W(unaware_mlp)) / E(sigma)` |
| Shadow rate | `shadow_steps_on_sigma / T` per match, averaged |
| Identification accuracy | Probe accuracy for `sigma` prediction |
| Identification delay | Steps until probe is correct for 10+ consecutive steps |
| Sigma involvement | passes to/from sigma / total passes |

### 3.5 Behaviour Statistics (All Questions)

Compute from `env.stats` aggregated over 100 eval matches:

| Stat | Source in `env.stats` |
|---|---|
| Possession share | `possession_steps[0] / T` |
| Pass completion rate | `passes_completed[0] / passes[0]` |
| Shots per match | `shots[0]` |
| Tackles won | `tackles_won[0]` |
| Mean opponent anchor distance | `np.mean(env.opp_anchor_dist)` per episode |
| Invalid action rate | `invalid_actions[0] / (T * 5)` |
| Shadow steps | `shadow_steps_on_sigma`, `shadow_steps_total` |
| Formation switches by opponent | `switches` |

---

## 4. Phase 0: Pre-Training Checks

All Phase 0 checks must pass before training any learned agent. These verify the environment is well-configured.

### Check A: Formation Readability

**Method:** run 10 heuristic vs heuristic matches with each opponent formation. Record `env.opp_anchor_dist` per step.

**Pass condition:**
```
mean_anchor_dist < 0.3 × min_inter_formation_dist
```

Where `min_inter_formation_dist` = minimum anchor-to-anchor distance between any two different formations for the same player.

**Compute inter-formation distance:**
```python
from itertools import combinations
import numpy as np

def inter_formation_dist(book, cfg):
    dists = []
    L, W = cfg.pitch.length, cfg.pitch.width
    for f1, f2 in combinations(book.names, 2):
        a1 = book.anchors_team_frame(f1)
        a2 = book.anchors_team_frame(f2)
        d = np.mean([np.linalg.norm(a1[i] - a2[i]) for i in range(5)])
        dists.append(d)
    return min(dists)
```

### Check B: Vision Matters

**Method:** 100 heuristic matches with `radius=25` (team 0) vs same team with `radius=12` (team 1).

**Pass condition:** `W(radius=25) > 0.60`

### Check C: Playmaker Works

**Method:** 100 heuristic matches with `sigma=3` (playmaker, team 1) vs without.

**Pass condition:** team-with-playmaker `W > 0.60`

### Check D: Balance and Determinism

**Method 1 (balance):** 100 heuristic vs heuristic matches (same formation both sides). Record win rate for team 0.

**Pass condition:** `|W - 0.5| < 0.05`

**Method 2 (determinism):** run same seed twice. Compare full `env.event_log`.

**Pass condition:** logs are identical.

### Check E: Formation Win Matrix

**Method:** for each pair of formations (5 × 5 = 25 combinations, including self-play), run 100 heuristic matches. Team 0 uses formation `f`, team 1 uses `g`.

Store as `phase0/formation_win_matrix.json`:
```json
{
  "2-2-1_vs_2-2-1": 0.50,
  "2-2-1_vs_2-1-2": 0.47,
  ...
}
```

This matrix is:
- The Phase 0 benchmark for Q1
- Used to pick `f_att` and `f_def` for `SCORE_REACTIVE` switching
- The expected performance ceiling for trained agents

### Check F: Random Policy Loses

**Method:** 100 matches of random agent vs heuristic opponent.

**Pass condition:** `W(random) < 0.15`

---

## 5. Statistical Reporting

All results report **mean ± standard deviation over 5 seeds**.

For paired comparisons (e.g. zeta=2 vs zeta=0 at same formation and radius):
- Use a **paired t-test** or **Wilcoxon signed-rank test** over the 5 seed means.
- Report p-value alongside the difference.

For learning curves:
- Plot **mean ± 1 std** band over seeds, with x-axis = training steps.
- Mark curriculum stage transitions with vertical dashed lines.
- Always show both win rate and shaping return on the same plot (two y-axes).

For the Q2 detection delay:
- Plot **CDF of detection delay** (fraction of switches detected within N steps) for each agent variant.
- Show separately for each switching mode.

---

## 6. Reproducibility Requirements

Every result must be reproducible from:
1. The config file (`config.json` saved with the run)
2. The training seed (in the run ID)
3. The evaluation seeds (fixed: `range(10_000, 10_100)`)
4. The code commit hash (log it)

```python
import subprocess
commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
log_dict["commit"] = commit
```

---

## 7. Evaluation Script Template

```python
import json
import numpy as np
from football import FootballEnv, load_config
from agents import RandomAgent  # or your trained agent

EVAL_SEEDS = list(range(10_000, 10_100))

def run_eval(agent, cfg_overrides=None, n=100):
    cfg = load_config(overrides=cfg_overrides or {})
    env = FootballEnv(cfg)

    wins = draws = losses = 0
    all_stats = []

    for seed in EVAL_SEEDS[:n]:
        obs, _ = env.reset(seed=seed)
        while not env.done:
            actions = agent.act(obs)
            obs, _, _, info = env.step(actions)

        s = info["final_score"]
        if s[0] > s[1]: wins += 1
        elif s[0] == s[1]: draws += 1
        else: losses += 1
        all_stats.append(dict(env.stats))

    W = (wins + 0.5 * draws) / n
    return {
        "W": W, "wins": wins, "draws": draws, "losses": losses,
        "stats": all_stats,
    }
```

---

## 8. Priorities if Time Runs Short

1. **Phase 0 checks** — always. Without them, training results are uninterpretable.
2. **Q1 (structure vs vision)** — the core experiment. At minimum: one formation × 3 zeta levels × 3 radii × 5 seeds.
3. **Q2 (formation switching)** — GRU vs belief filter is the key comparison.
4. **Q3 (playmaker)** — cut to 1 playmaker slot (midfielder) if needed.

For Q2 and Q3, if training time is limited: reduce `total_steps` to 5M and report that comparisons may not be fully converged.
