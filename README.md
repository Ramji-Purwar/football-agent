# football_adapt: the 5-a-side football game and its opponent

Code for the project described in `game.md` and `opponent.md`.
Run everything from the project root (the folder containing this README), using `python -m ...`.

## File structure

```
football_adapt/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml            all parameters (PLACEHOLDER values, calibrate in Phase 0)
├── football/                   THE GAME
│   ├── __init__.py
│   ├── config.py               load the YAML, dotted overrides, attribute access
│   ├── geometry.py             2D helpers, team-frame conversion (to_frame)
│   ├── formations.py           formations, slot types, anchors
│   ├── actions.py              the action space (all indexing lives here)
│   ├── mechanics.py            control rating, shot probability, pass interception, tackle probability
│   ├── observation.py          shared observation builder, action mask, flat vector, critic state
│   └── env.py                  FootballEnv: reset(), step(), rewards, events, stats
├── opponent/                   THE OPPONENT
│   ├── __init__.py
│   ├── formation_manager.py    NONE / SCHEDULED / RANDOM / SCORE_REACTIVE switching
│   └── controller.py           ScriptedTeam: zonal defence, attack, support, playmaker
├── agents/
│   ├── __init__.py
│   └── random_agent.py         random baseline (uses the action mask)
├── scripts/
│   ├── __init__.py
│   ├── phase0_checks.py        the Phase 0 checks and the formation win matrix
│   └── watch_match.py          render a match to a GIF
└── tests/
    ├── __init__.py
    └── test_smoke.py           a few basic tests
```

Still to be added by the team (not included): `agents/belief_filter.py` (Bayesian filter),
`agents/mappo.py` and `training/` (shared-policy PPO with GRU), `analysis/` (probes, metrics, plots).

## How to use

```python
from football import FootballEnv, load_config
from agents import RandomAgent

cfg = load_config(overrides={"vision.radius": 15.0, "opponent.switching.mode": "SCHEDULED"})
env = FootballEnv(cfg)                                   # our team = learner, opponent = ScriptedTeam
obs, info = env.reset(seed=0, our_formation="2-2-1", opp_formation="1-3-1", sigma=0, zeta=2)
agent = RandomAgent(0)
done = False
while not done:
    obs, reward, done, info = env.step(agent.act(obs))   # obs["vec"] (5, dim), obs["mask"] (5, n_actions), obs["state"]
```

- `FootballEnv(cfg, scripted_ours=True)` drives OUR side with the scripted team too (heuristic vs heuristic, used by Phase 0).
- `info` carries the hidden truth for oracles and probes: `info["g"]` (opponent formation), `info["sigma"]`, `info["switch"]`.
- `env.event_log` and `env.stats` hold passes, tackles, shots, goals, switches, shadow counts. Compute metrics from them.
- Call `env.set_shaping_weight(lambda_t)` from the trainer to anneal the shaping reward.

Commands:

```
python -m scripts.phase0_checks --episodes 20 --T 600 --matrix
python -m scripts.watch_match --mode scripted --out match.gif
python -m pytest -q
```

## Things to know before you trust any result

1. **Every number in `configs/default.yaml` is a placeholder.** Calibrate them in Phase 0 and freeze them.
2. **The playmaker effect has not been shown to work yet.** In a quick trial (not a proper experiment), a team with
   the playmaker scored about the same as without one. Phase 0 check D will likely fail until you tune the opponent
   (for example `opponent.attack.hub_bonus`, or make the playmaker's wider information matter more in
   `opponent/controller.py`). Q3 depends on this, so fix it before anything else.
3. **Balance and sample size.** Results from a handful of episodes are noisy (heuristic vs heuristic swung between
   about 0.43 and 0.58 in small trials). Use many episodes before judging a check.
4. `scripts/watch_match.py` and `tests/test_smoke.py` were written but not run.

## Simplifications built into the code (all stated in game.md)

- Goals come only from shots. The ball is clamped to the pitch (no throw-ins or corners).
- A pass is decided at release (interception geometry), then the ball flies for a few steps.
- Our agent moves in `n_directions` discrete directions. Macros (`HOLD_SHAPE`, `GO_TO_BALL`, `SHADOW_j`) and the
  opponent use the same `MOVE_TO` movement routine and speed limit.
- Teammate positions, possession, score and time are always known. Opponents and the ball are visible only inside the radius.
- No side swap: both teams attack toward +x in their own frame (team 1 is rotated 180 degrees in world coordinates).
