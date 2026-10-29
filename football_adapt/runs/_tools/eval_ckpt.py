"""Full matches of a trained team against the scripted team at full strength, with the config values stored in its
checkpoint (as the viewers do). Run from football_adapt/. usage: eval_ckpt.py n ckpt [ckpt ...]   (seeds 0 .. n-1)"""
import sys
import numpy as np
from multiprocessing import Pool


def one(a):
    path, seed = a
    from football import FootballEnv, load_config
    from agents.mappo import MAPPOAgent
    ag = MAPPOAgent.load(path, seed=seed)
    env = FootballEnv(load_config(overrides=ag.overrides))
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, done, _ = env.step(ag.act(obs))
    s = env.stats
    return path, (env.score[0], env.score[1], s["shots"][0], s["shots"][1], s["passes_completed"][0],
                  s["possession_steps"][0] / env.T, s["invalid_actions"][0])


if __name__ == "__main__":
    n, paths = int(sys.argv[1]), sys.argv[2:]
    with Pool(14) as pool:
        res = pool.map(one, [(p, seed) for p in paths for seed in range(n)])
    for p in paths:
        r = np.array([v for k, v in res if k == p], float)
        d = r[:, 0] - r[:, 1]
        w, dr = int((d > 0).sum()), int((d == 0).sum())
        print(f"{p}  ({n} full matches, full-strength opponent)")
        print(f"  goals {r[:, 0].mean():.2f} - {r[:, 1].mean():.2f}, wins-draws-losses {w}-{dr}-{n - w - dr}, W {(w + dr / 2) / n:.2f}")
        print(f"  goal difference {d.mean():+.2f}, spread between matches {d.std(ddof=1):.2f}, so the mean of 20 matches is "
              f"uncertain by {d.std(ddof=1) / 20 ** 0.5:.2f}, of 50 by {d.std(ddof=1) / 50 ** 0.5:.2f}, of these {n} by {d.std(ddof=1) / n ** 0.5:.2f}")
        print(f"  shots {r[:, 2].mean():.1f}, their shots {r[:, 3].mean():.1f}, completed passes {r[:, 4].mean():.1f}, "
              f"possession {r[:, 5].mean():.0%}, invalid actions {int(r[:, 6].sum())}")
