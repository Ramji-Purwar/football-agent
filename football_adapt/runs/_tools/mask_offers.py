"""At the same game states (v3 playing), how many passes / through balls the action mask offers our ball carrier
when he sees 120 m versus 25 m. Run from football_adapt/: PYTHONPATH=. python mask_offers.py CKPT N"""
import sys
from multiprocessing import Pool
import numpy as np

FORMS = ["2-2-1", "2-1-2", "1-3-1", "3-1-1"]


def one(job):
    ckpt, form, seed = job
    import torch
    torch.set_num_threads(1)
    from football import FootballEnv, load_config
    from football.actions import Kind
    from football.observation import build_observation
    from agents.mappo import MAPPOAgent
    ag = MAPPOAgent.load(ckpt, seed=seed)
    env = FootballEnv(load_config(overrides=ag.overrides))
    t = ag.train_args
    obs, _ = env.reset(seed=seed, our_formation="2-2-1", opp_formation=form, radius=t["radius"], zeta=t["zeta"])
    kinds = [env.space.decode(i).kind for i in range(env.space.n)]
    out = {r: [0, 0, 0] for r in (120.0, 25.0, 12.0)}        # carrier steps, passes offered, through balls offered
    done = False
    while not done:
        h = env.holder
        if h is not None and h[0] == 0 and env.frozen[0][h[1]] == 0:
            r0 = env.radius[0]
            for r in out:
                env.radius[0] = r
                m = build_observation(env, 0, h[1]).mask
                out[r][0] += 1
                out[r][1] += sum(1 for i in range(len(m)) if m[i] and kinds[i] == Kind.PASS)
                out[r][2] += sum(1 for i in range(len(m)) if m[i] and kinds[i] == Kind.THROUGH)
            env.radius[0] = r0
        obs, _, done, _ = env.step(ag.act(obs))
    return out


if __name__ == "__main__":
    ckpt, n = sys.argv[1], int(sys.argv[2])
    with Pool(48) as p:
        res = p.map(one, [(ckpt, f, 30_000 + s) for f in FORMS for s in range(n)])
    for r in (120.0, 25.0, 12.0):
        c = np.sum([x[r] for x in res], axis=0)
        print(f"carrier sees {r:5.0f} m: per carrier step, passes offered {c[1] / c[0]:.2f} (of 4), "
              f"through balls offered {c[2] / c[0]:.2f}   ({c[0]} carrier steps)")
