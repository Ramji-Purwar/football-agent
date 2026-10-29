"""W and goals of checkpoints, each against its own opponent formation (and playmaker), at full strength, in the
setting it was trained in (formation, radius, zeta and config values from the checkpoint). One line per match goes to
--out as it finishes.
Run from football_adapt/ (on a server):
    PYTHONPATH=. python ckpt_vs_form.py --matches 100 --seed0 20000 --out res.txt CKPT:2-2-1 CKPT:1-2-2 ...
    ... CKPT:2-2-1:3       (opponent player 3 is the playmaker: he sees the whole pitch; 0 = no playmaker)
    ... CKPT:2-2-1:3:opponent.playmaker.pass_speed=2      (config values for these matches, comma-separated)"""
import argparse
import json
from multiprocessing import Pool
import numpy as np


ORIGINS = ("tackle_pm", "tackle", "cut_pm", "cut", "loose_pm", "loose", "goal_kick", "kickoff")


def goal_origins(ev, pm):
    """How we last won the ball before each of our goals: a tackle, a pass of theirs cut out, a loose ball (each
    "_pm" when we took it off the playmaker or he had played it last), a goal kick (their shot missed) or the
    kick-off after they scored. {kind: our goals}"""
    out, owner, start, last = {}, None, "kickoff", None      # last: the player of theirs who last had or played it
    for e in ev:
        t, gain, how, who = e["type"], None, None, None
        if t == "tackle" and e["success"]:
            gain, how, who = e["by"][0], "tackle", tuple(e["from_"])
        elif t == "pass_result" and e["result"] == "intercepted":
            gain, how, who = e["by"][0], "cut", tuple(e["passer"])
        elif t == "pickup":
            gain, how, who = e["by"][0], "loose", last
        elif t == "goal_kick":
            gain, how = e["by"][0], "goal_kick"
        elif t == "goal":
            if e["team"] == 0:
                out[start] = out.get(start, 0) + 1
            gain, how = 1 - e["team"], "kickoff"
        if gain is not None and gain == 0 and owner != 0:
            start = how + ("_pm" if who == pm else "")
        if gain is not None:
            owner = gain
        for k in ("by", "passer"):
            if k in e and e[k][0] == 1 and (t != "tackle" or e["success"]):
                last = tuple(e[k])
        if t == "pass_result" and e["result"] == "completed" and e["passer"][0] == 1:
            last = tuple(e["by"])
        if t == "goal":
            last = None
    return out


def one(job):
    ckpt, form, sigma, extra, seed = job
    import torch
    torch.set_num_threads(1)
    from football import FootballEnv, load_config
    from agents.mappo import MAPPOAgent
    ag = MAPPOAgent.load(ckpt, seed=seed)
    t = ag.train_args
    env = FootballEnv(load_config(overrides={**ag.overrides, **extra}))
    obs, _ = env.reset(seed=seed, our_formation=t["our_formation"], opp_formation=form, opp_style="normal",
                       radius=t["radius"], zeta=t["zeta"], sigma=sigma)
    done = False
    pm = (1, sigma - 1)
    ctrl = []                                    # the playmaker's control on every step he has the ball
    while not done:
        obs, _, done, _ = env.step(ag.act(obs))
        if sigma and env.holder == pm:
            ctrl.append(env.control(*pm))
    s = env.stats
    ev = env.event_log
    lost = lambda kind: sum(1 for e in ev if e["type"] == "pass_result" and tuple(e["passer"]) == pm and e["result"] == kind)
    return dict(ckpt=ckpt, form=form, sigma=sigma, extra=extra, seed=seed, gf=env.score[0], ga=env.score[1],
                pm_made=s["sigma_passes_made"], pm_got=s["sigma_passes_received"], pm_int=s["sigma_interceptions"],
                pm_ctrl=float(np.mean(ctrl)) if ctrl else None,
                pm_tackled=sum(1 for e in ev if e["type"] == "tackle" and e["success"] and tuple(e["from_"]) == pm),
                pm_cut=lost("intercepted"), pm_loose=lost("loose"),
                pm_goals=sum(1 for e in ev if e["type"] == "shot" and e["scored"] and tuple(e["by"]) == pm),
                origins=goal_origins(ev, pm),
                opp_passes=s["passes_completed"][1], opp_shots=s["shots"][1], opp_poss=s["possession_steps"][1] / env.T)


def summary(rows, spec):
    out = []
    for ckpt, form, sigma, extra in spec:
        r = [x for x in rows if (x["ckpt"], x["form"], x["sigma"], x["extra"]) == (ckpt, form, sigma, extra)]
        if not r:
            continue
        g = np.array([[x["gf"], x["ga"]] for x in r], float)
        d = g[:, 0] - g[:, 1]
        W = ((d > 0).sum() + 0.5 * (d == 0).sum()) / len(r)
        m = lambda k: np.mean([x[k] for x in r])
        tag = "".join(" %s %g" % (k.split(".")[-1], v) for k, v in extra.items())
        line = (f"{ckpt:30s} vs {form}{f' playmaker {sigma}' if sigma else ''}{tag}  n {len(r):3d}  W {W:4.2f}  goals "
                f"{g[:, 0].mean():5.2f}-{g[:, 1].mean():4.2f}  GD {d.mean():+5.2f} (+-{1.96 * d.std() / np.sqrt(len(d)):.2f})"
                f"  | their passes {m('opp_passes'):4.1f}, shots {m('opp_shots'):4.1f}, ball {m('opp_poss'):.0%}")
        if sigma:
            line += f" | playmaker: gets {m('pm_got'):4.1f} passes, gives {m('pm_made'):4.1f}, intercepts {m('pm_int'):3.1f}"
            if "pm_ctrl" in r[0]:
                c = [x["pm_ctrl"] for x in r if x["pm_ctrl"] is not None]
                line += (f", control on the ball {np.mean(c):.2f}, loses it {m('pm_tackled'):3.1f} tackled /"
                         f" {m('pm_cut'):3.1f} cut out / {m('pm_loose'):3.1f} loose passes, scores {m('pm_goals'):3.1f}")
        if "origins" in r[0]:
            line += " | our goals start with: " + ", ".join(
                "%s %.2f" % (k, np.mean([x["origins"].get(k, 0) for x in r])) for k in ORIGINS
                if sigma or not k.endswith("_pm"))
        out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="CKPT:FORMATION[:PLAYMAKER[:KEY=VALUE,...]]")
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=20_000)
    ap.add_argument("--procs", type=int, default=56)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = []
    for s in a.specs:
        ckpt, form, *rest = s.split(":")
        extra = dict((k, float(v)) for k, v in (kv.split("=") for kv in rest[1].split(","))) if len(rest) > 1 else {}
        spec.append((ckpt, form, int(rest[0]) if rest else 0, extra))
    jobs = [(c, f, pm, ex, a.seed0 + s) for s in range(a.matches) for c, f, pm, ex in spec]   # all advance together
    rows = []
    with Pool(a.procs) as p, open(a.out, "w") as fh:
        for x in p.imap_unordered(one, jobs):
            rows.append(x)
            fh.write(json.dumps(x) + "\n")
            fh.flush()
    print(f"{a.matches} matches each, seeds {a.seed0}+\n" + summary(rows, spec), flush=True)
