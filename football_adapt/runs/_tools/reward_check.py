"""Offline check of the reward (configs/default.yaml), no training: points per player per match from each small term
(measured alone, without the cap), then the full reward with the cap, for five kinds of team:
  scripted  the scripted team in our seat (good football)        random  a random team
  longshot  shoots whenever it may, otherwise runs at goal        farm    passes to any teammate it may, never shoots
  stall     keeps the ball (STAY on it)
The three last ones go for the ball with the player nearest it and hold shape with the rest.
usage (from football_adapt/): PYTHONPATH=. python runs/_tools/reward_check.py n  (28 processes)"""
import sys
import numpy as np
from multiprocessing import Pool
from football import FootballEnv, load_config
from football.actions import Kind
from football.geometry import dist
from agents import RandomAgent
from training.train import sample_reset

TERMS = ("w_pass_done", "w_pass_accuracy", "w_receive", "w_free", "w_turnover", "w_progress", "w_chance", "w_shot",
         "w_big_chance", "w_poor_shot", "w_missed_shot", "w_chance_against", "w_shot_against", "w_tackle",
         "w_interception", "w_open_lanes", "w_formation")
TEAMS = ("scripted", "random", "longshot", "farm", "stall")
BASE = {"actions.context_mask": True, "actions.open_pass.call": True}
SCEN = dict(our_formation="2-2-1", radius=None, zeta=None, opp_formations=["2-2-1"],
            opp_styles=["normal", "aggressive", "defensive"], sigmas=["0"])


def simple(kind):
    def act(env, obs):
        I = {k: env.space.index_of_kind(k) for k in (Kind.STAY, Kind.HOLD_SHAPE, Kind.GO_TO_BALL, Kind.DRIBBLE_GOAL,
                                                     Kind.SHOOT, Kind.PASS, Kind.TACKLE)}
        near = min(range(5), key=lambda j: dist(env.pos[0][j], env.ball_pos))
        out = {}
        for j in range(5):
            m = obs["mask"][j]
            if m.sum() <= 1:
                out[j] = int(np.flatnonzero(m)[0]); continue
            if env.holder == (0, j):
                passes = [I[Kind.PASS] + q for q in range(4) if m[I[Kind.PASS] + q]]
                if kind == "longshot":
                    a = I[Kind.SHOOT] if m[I[Kind.SHOOT]] else I[Kind.DRIBBLE_GOAL]
                elif kind == "farm":
                    a = int(np.random.default_rng(env.t * 7 + j).choice(passes)) if passes else I[Kind.STAY]
                else:
                    a = I[Kind.STAY]
            elif j == near:
                a = I[Kind.TACKLE] if m[I[Kind.TACKLE]] else I[Kind.GO_TO_BALL]
            else:
                a = I[Kind.HOLD_SHAPE]
            out[j] = a if m[a] else I[Kind.STAY]
        return out
    return act


def one(job):
    team, term, seed = job
    ov = dict(BASE)
    if term != "FULL":                     # this term alone, no cap
        ov.update({f"reward.{t}": 0.0 for t in TERMS if t != term})
        ov["reward.small_cap"] = 0.0
    env = FootballEnv(load_config(overrides=ov), scripted_ours=(team == "scripted"))
    obs, _ = env.reset(seed=seed, **sample_reset(SCEN, np.random.default_rng(seed)))
    if team == "scripted":                 # the shape guide a learning team has, so that its spot points count too
        from opponent.controller import ScriptedTeam
        env.shape_guide = ScriptedTeam(env.team_cfg[0], np.random.default_rng(seed), team=0,
                                       formation=env.formation[0], playmaker=None, mode_override="NONE")
    agent = RandomAgent(seed) if team == "random" else None
    act = None if team in ("scripted", "random") else simple(team)
    result = 0.0
    done = False
    while not done:
        obs, r, done, info = env.step(None if team == "scripted" else (agent.act(obs) if agent else act(env, obs)))
        result += (info["reward_parts"]["team"] + info["reward_parts"]["terminal"]) / env.cfg.reward.scale
    s = env.small_total
    return team, term, float(s.mean()), float(s.min()), float(s.max()), result, env.score[0] - env.score[1]


if __name__ == "__main__":
    n = int(sys.argv[1])
    with Pool(28) as pool:
        res = pool.map(one, [(t, k, s) for t in TEAMS for k in TERMS + ("FULL",) for s in range(n)])
    get = lambda t, k, i: np.mean([r[i] for r in res if r[0] == t and r[1] == k])
    print(f"Points per player per match ({n} matches each, all three opponent styles). Each term alone, no cap:")
    print(f"{'term':18s}" + "".join(f"{t:>10s}" for t in TEAMS))
    for k in TERMS:
        print(f"{k:18s}" + "".join(f"{get(t, k, 2):10.1f}" for t in TEAMS))
    print(f"{'sum of terms':18s}" + "".join(f"{sum(get(t, k, 2) for k in TERMS):10.1f}" for t in TEAMS))
    print("\nThe full reward, with the cap (+-50 per player):")
    print(f"{'small terms (mean)':18s}" + "".join(f"{get(t, 'FULL', 2):10.1f}" for t in TEAMS))
    print(f"{'  lowest player':18s}" + "".join(f"{get(t, 'FULL', 3):10.1f}" for t in TEAMS))
    print(f"{'  highest player':18s}" + "".join(f"{get(t, 'FULL', 4):10.1f}" for t in TEAMS))
    print(f"{'result + goals':18s}" + "".join(f"{get(t, 'FULL', 5):10.0f}" for t in TEAMS))
    print(f"{'goal difference':18s}" + "".join(f"{get(t, 'FULL', 6):10.2f}" for t in TEAMS))
