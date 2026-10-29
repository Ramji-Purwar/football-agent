"""Markdown tables for docs/reportQ1.md, averaged over the seeds of each setting, from runs/q1/summary.json
(q1_summary.py) and runs/q1/style.jsonl (style_stats.py). Run from football_adapt/: python runs/_tools/q1_tables.py"""
import json
import numpy as np

S = json.load(open("runs/q1/summary.json"))
ORDER = ["z1_r12", "z0_r12", "z1_r25", "z0_r25", "z1_r120", "z0_r120"]
NAME = {k: f"zeta {k[1]}, {'full' if k.endswith('120') else k.split('_r')[1] + ' m'}" for k in ORDER}
FORMS = ["2-2-1", "2-1-2", "1-3-1", "3-1-1", "1-2-2"]

style = {}
for line in open("runs/q1/style.jsonl"):
    d = json.loads(line)
    key = d["ckpt"].split("/")[2]
    style.setdefault(key, []).append(d["s"])
ST = {k: {m: float(np.nanmean([x[m] for x in v])) for m in v[0]} for k, v in style.items()}


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


print("## main")
table(["setting", "seeds", "test W", "seed spread", "win", "draw", "goals for", "goals against"],
      [[NAME[k], str(S[k]["seeds"]), f"**{S[k]['W']:.2f}**", f"{S[k]['W_spread']:.2f}", f"{S[k]['win']:.0%}",
        f"{S[k]['draw']:.0%}", f"{S[k]['goals_for']:.1f}", f"{S[k]['goals_against']:.1f}"] for k in ORDER])

print("## B(r)")
for r in ("12", "25", "120"):
    print(r, f"{S['z1_r' + r]['W'] - S['z0_r' + r]['W']:+.2f}")

print("\n## learning")
def reach(k, lv):
    v = [x for x in S[k][f"reach_{lv}"] if x == x]
    n = len(S[k][f"reach_{lv}"])
    return f"{np.mean(v) / 1000:.0f}k ({len(v)}/{n})" if v else f"never (0/{n})"
table(["setting", "W 0.3 reached at", "W 0.5 reached at", "W 0.7 reached at", "mean W, last 10 evaluations",
       "best checkpoint at"],
      [[NAME[k], reach(k, 0.3), reach(k, 0.5), reach(k, 0.7), f"{S[k]['last10_W']:.2f}",
        f"{S[k]['best_at'] / 1000:.0f}k"] for k in ORDER])

print("## formations")
table(["setting"] + [f + (" (never trained)" if f == "1-2-2" else "") for f in FORMS],
      [[NAME[k]] + [f"{S[k]['vs'][f]:.2f}" for f in FORMS] for k in ORDER])
table(["scripted team in our seat"] + FORMS,
      [[f"radius {r}"] + [f"{S['z1_r' + r]['scripted'][f]:.2f}" for f in FORMS] for r in ("12", "25", "120")])
for r in ("12", "25", "120"):
    print("scripted mean over the 4 trained formations, r", r,
          f"{np.mean([S['z1_r' + r]['scripted'][f] for f in FORMS[:4]]):.2f}")

print("\n## test details")
table(["setting", "passes completed", "pass completion", "through balls", "shots", "chance per shot", "turnovers",
       "possession", "ball in own third", "ball in their third"],
      [[NAME[k], f"{S[k]['passes_completed']:.1f}", f"{S[k]['pass_completion']:.0%}", f"{S[k]['through_balls']:.1f}",
        f"{S[k]['shots']:.1f}", f"{S[k]['chance_per_shot']:.2f}", f"{S[k]['turnovers']:.1f}",
        f"{S[k]['possession']:.0%}", f"{S[k]['ball_own_third']:.0%}", f"{S[k]['ball_their_third']:.0%}"]
       for k in ORDER])

if ST:
    print("## style")
    table(["setting", "ball: ours / theirs / in the air", "they hold it in our half", "their passes", "their shots",
           "our tackles (won)", "interceptions"],
          [[NAME[k], f"{ST[k]['ours']:.0%} / {ST[k]['theirs']:.0%} / {ST[k]['free']:.0%}",
            f"{ST[k]['their_in_our_half']:.0%}", f"{ST[k]['their_passes']:.0f}", f"{ST[k]['their_shots']:.1f}",
            f"{ST[k]['tackles']:.1f} ({ST[k]['tackles_won']:.1f})", f"{ST[k]['intercepts']:.1f}"]
           for k in ORDER if k in ST])
    table(["setting", "team depth defending", "team depth attacking", "spread between teammates (m)",
           "distance from formation spot (m)"],
          [[NAME[k], f"{ST[k]['depth_def']:.2f}", f"{ST[k]['depth_att']:.2f}", f"{ST[k]['spread']:.1f}",
            f"{ST[k]['anchor_dist']:.1f}"] for k in ORDER if k in ST])
    table(["setting", "goals started in own third", "goals with 2+ passes", "steps from winning the ball to goal (median)",
           "goals within 60 steps"],
          [[NAME[k], f"{ST[k].get('goal_won_own_third', float('nan')):.0%}",
            f"{ST[k].get('goal_two_passes', float('nan')):.0%}", f"{ST[k].get('goal_steps_median', float('nan')):.0f}",
            f"{ST[k].get('goal_fast', float('nan')):.0%}"] for k in ORDER if k in ST])
