"""Has a trained team overfitted to the opponents it trained against? It plays every opponent formation (the ones
it never met are marked NEW), at full strength, in the setting it was trained in (its formation, radius, zeta and
config values, from the checkpoint). The scripted team plays the same matches in our seat, as a yardstick: a
formation that is hard for both is just hard, one that is hard only for the trained team is overfitting.

Run from football_adapt/:
    PYTHONPATH=. ../.venv/bin/python runs/_tools/vs_formations.py runs/v1/ckpts/m039024.pt
    ... --matches 100 --opp-style aggressive     (more matches; an opponent style it never met)"""
import argparse
from multiprocessing import Pool
import numpy as np

FORMS = ["2-2-1", "2-1-2", "1-3-1", "1-2-2", "3-1-1"]
SEED0 = 20_000                    # not the evaluation (10000+) or test (50000+) seeds of training


def one(job):
    who, ckpt, form, style, seed = job
    import torch
    torch.set_num_threads(1)
    from football import FootballEnv, load_config
    from agents.mappo import MAPPOAgent
    ag = MAPPOAgent.load(ckpt, seed=seed)
    t = ag.train_args
    kw = dict(seed=seed, our_formation=t["our_formation"], opp_formation=form, opp_style=style,
              radius=t["radius"], zeta=t["zeta"])
    if who == "scripted":
        env = FootballEnv(load_config(), scripted_ours=True)
        env.reset(**kw)
        while not env.done:
            env.step()
    else:
        env = FootballEnv(load_config(overrides=ag.overrides))
        obs, _ = env.reset(**kw)
        done = False
        while not done:
            obs, _, done, _ = env.step(ag.act(obs))
    return who, form, env.score[0], env.score[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--matches", type=int, default=50, help="matches per opponent formation")
    ap.add_argument("--opp-style", default="normal", help="normal, aggressive or defensive")
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    import torch
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"]
    seen = ck.get("opp_formations") or ["2-2-1"]
    seen_styles = ck.get("opp_styles") or ["normal"]
    print(f"{a.ckpt}: trained as {ck['our_formation']}, radius {ck['radius']}, zeta {ck['zeta']}, against "
          f"{', '.join(seen)} ({', '.join(seen_styles)})")
    print(f"opponent style now: {a.opp_style}{'' if a.opp_style in seen_styles else ' (NEW)'}, "
          f"{a.matches} matches per formation, seeds {SEED0}+\n")
    jobs = [(w, a.ckpt, f, a.opp_style, SEED0 + s) for w in ("trained", "scripted") for f in FORMS
            for s in range(a.matches)]
    with Pool(a.procs) as p:
        res = p.map(one, jobs)
    print(f"{'opponent':12s} {'trained team':>24s}   {'scripted team, same matches':>28s}")
    for f in FORMS:
        row = f"{f + ('' if f in seen else ' NEW'):12s}"
        for w in ("trained", "scripted"):
            r = np.array([x[2:] for x in res if x[0] == w and x[1] == f], float)
            d = r[:, 0] - r[:, 1]
            W = ((d > 0).sum() + 0.5 * (d == 0).sum()) / len(r)
            row += f"   W {W:4.2f}  goals {r[:, 0].mean():4.2f}-{r[:, 1].mean():4.2f}"
        print(row)
    print("\nW = wins + draws/2. With 50 matches a W is good to about +-0.07.")
