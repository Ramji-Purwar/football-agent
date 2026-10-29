"""What the playmaker costs one team (docs/questions.md, Q3: E(sigma) = W(no playmaker) - W(playmaker)), from the
per-match lines of ckpt_vs_form.py, paired by seed: the same seed against the same formation with and without him.
usage: python pm_cost.py runs/q3/impact_matches.jsonl"""
import json
import sys
from collections import defaultdict
import numpy as np


def main(path):
    res = defaultdict(dict)              # (formation, playmaker) -> seed -> (result 1/0.5/0, goal difference, passes he got)
    for line in open(path):
        x = json.loads(line)
        d = x["gf"] - x["ga"]
        res[x["form"], x["sigma"]][x["seed"]] = ((d > 0) + 0.5 * (d == 0), d, x["pm_got"])
    ci = lambda v: 1.96 * np.std(v) / np.sqrt(len(v))
    print(f"{'case':8s} {'n':>4s}  {'W':>10s}  {'GD':>11s}  {'cost E':>11s}  {'GD lost':>11s}  {'he gets':>7s}  verdict")
    for form in sorted({f for f, _ in res}):
        none = res[form, 0]
        for s in sorted(s for f, s in res if f == form):
            r = res[form, s]
            o, g = (np.array([v[i] for v in r.values()], float) for i in (0, 1))
            line = f"{form}:{s:<2d} {len(r):4d}  {o.mean():4.2f}+-{ci(o):.2f}  {g.mean():+5.2f}+-{ci(g):.2f}"
            if s:
                p = [k for k in r if k in none]
                e = np.array([none[k][0] - r[k][0] for k in p])
                dg = np.array([none[k][1] - r[k][1] for k in p], float)
                beat = "they beat v4" if o.mean() + ci(o) < 0.5 else "even" if o.mean() - ci(o) <= 0.5 else "v4 still ahead"
                line += (f"  {e.mean():+5.2f}+-{ci(e):.2f}  {dg.mean():+5.2f}+-{ci(dg):.2f}  {np.mean([v[2] for v in r.values()]):7.1f}"
                         f"  {beat}")
            print(line)


if __name__ == "__main__":
    main(sys.argv[1])
