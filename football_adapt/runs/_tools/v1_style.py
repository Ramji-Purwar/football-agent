"""How does a trained team play? Who has the ball, how attacks end, and how the goals are made. Read-only.
usage (from football_adapt/): PYTHONPATH=. python v1_style.py CKPT N   (N matches per opponent formation)
The scripted team in our seat plays the same seeds, for comparison."""
import sys
from multiprocessing import Pool
import numpy as np

FORMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["2-2-1", "2-1-2", "1-3-1", "3-1-1"]


def one(job):
    who, ckpt, form, seed = job
    import torch
    torch.set_num_threads(1)
    from football import FootballEnv, load_config
    s = 70_000 + seed                         # seeds no evaluation or test used
    import torch as _t
    _a = _t.load(ckpt, map_location="cpu", weights_only=False)["args"]   # the setting it was trained in
    kw = dict(seed=s, our_formation="2-2-1", opp_formation=form, radius=float(_a["radius"]), zeta=int(_a["zeta"]))
    if who == "scripted":
        env = FootballEnv(load_config(), scripted_ours=True)
        env.reset(**kw)
        ag = None
    else:
        from agents.mappo import MAPPOAgent
        ag = MAPPOAgent.load(ckpt, seed=s)
        env = FootballEnv(load_config(overrides=ag.overrides))
        obs, _ = env.reset(**kw)
    hold, state, bx, still, px = [], [], [], [], []
    done = False
    while not done:
        before = None if env.holder is None else (env.holder, env.pos[env.holder[0]][env.holder[1]].copy())
        if ag is None:
            _, _, done, _ = env.step()
        else:
            obs, _, done, _ = env.step(ag.act(obs))
        hold.append(-1 if env.holder is None else env.holder[0])
        state.append(env.ball_state)
        bx.append(float(env.ball_pos[0]))
        px.append(env.pos[:, :, 0].copy())
        still.append(before is not None and env.holder == before[0] and
                     float(np.linalg.norm(env.pos[before[0][0]][before[0][1]] - before[1])) < 0.05)
    hold, state, bx, still = np.array(hold), np.array(state), np.array(bx), np.array(still)
    L, ev = env.L, env.event_log
    goal_t = [e["t"] for e in ev if e["type"] == "goal"]
    # hold[i] is after step i; a goal logged at t has its kickoff in the same step, so hold[t] is the kicking-off side
    goals = []
    for g in [e for e in ev if e["type"] == "goal"]:
        team, t = g["team"], g["t"]
        other = [i for i in range(t - 1, -1, -1) if hold[i] == 1 - team]
        start = other[0] + 1 if other else 0
        kicks = [k for k in goal_t if k < t]
        from_kick = bool(kicks) and kicks[-1] >= start - 1
        if from_kick:
            start = kicks[-1] + 1
        elif not other:
            from_kick = True                   # the opening kickoff
        passes = sum(1 for e in ev if e["type"] == "pass" and e["passer"][0] == team and start <= e["t"] <= t)
        x = bx[min(start, len(bx) - 1)]
        xs = px[min(start, len(px) - 1)][1 - team]      # the other team's players when the ball was won
        back = int((xs > x).sum()) if team == 0 else int((xs < x).sum())   # ... goal-side of the ball
        goals.append(dict(team=team, steps=t - start, passes=passes, kick=from_kick, back=back,
                          won_x=(x if team == 0 else L - x) / L))
    out = dict(who=who, form=form, gf=env.score[0], ga=env.score[1],
               ours=float((hold == 0).mean()), theirs=float((hold == 1).mean()),
               loose=float(((hold == -1) & (state == "loose")).mean()),
               flight=float(((hold == -1) & (state == "flight")).mean()),
               theirs_our_half=float(((hold == 1) & (bx < L / 2)).mean()),
               their_carrier_still=float(still[hold == 1].mean()) if (hold == 1).any() else 0.0,
               goals=goals)
    for team in (0, 1):
        out[f"tackles{team}"] = sum(1 for e in ev if e["type"] == "tackle" and e["by"][0] == team)
        out[f"tackles_won{team}"] = sum(1 for e in ev if e["type"] == "tackle" and e["by"][0] == team and e["success"])
        out[f"intercepts{team}"] = sum(1 for e in ev if e["type"] == "pass_result" and e["result"] == "intercepted"
                                       and e["by"] is not None and e["by"][0] == team)
        out[f"shots{team}"] = sum(1 for e in ev if e["type"] == "shot" and e["by"][0] == team)
        out[f"passes{team}"] = sum(1 for e in ev if e["type"] == "pass" and e["passer"][0] == team)
    return out


if __name__ == "__main__":
    ckpt, n = sys.argv[1], int(sys.argv[2])
    with Pool(50) as p:
        res = p.map(one, [(w, ckpt, f, s) for w in ("trained", "scripted") for f in FORMS for s in range(n)])
    for w in ("trained", "scripted"):
        print(f"=== {w} team in our seat (2-2-1, radius/zeta of the checkpoint), {n} matches per opponent formation")
        for f in FORMS + ["all"]:
            r = [x for x in res if x["who"] == w and (f == "all" or x["form"] == f)]
            d = np.array([x["gf"] - x["ga"] for x in r])
            W = ((d > 0).sum() + 0.5 * (d == 0).sum()) / len(r)
            m = lambda k: np.mean([x[k] for x in r])
            print(f"  vs {f:5s}: W {W:.2f}  goals {m('gf'):.2f}-{m('ga'):.2f}   ball: ours {m('ours'):.2f}  theirs "
                  f"{m('theirs'):.2f} (in our half {m('theirs_our_half'):.2f}, carrier standing still "
                  f"{m('their_carrier_still'):.2f})  loose {m('loose'):.2f}  in flight {m('flight'):.2f}")
        r = [x for x in res if x["who"] == w]
        m = lambda k: np.mean([x[k] for x in r])
        for team, name in ((0, "us"), (1, "them")):
            print(f"  per match, {name:4s}: passes {m(f'passes{team}'):5.1f}  shots {m(f'shots{team}'):4.1f}  "
                  f"tackles {m(f'tackles{team}'):5.1f} (won {m(f'tackles_won{team}'):4.1f})  "
                  f"interceptions {m(f'intercepts{team}'):4.1f}")
        for team, name in ((0, "our"), (1, "their")):
            g = [y for x in r for y in x["goals"] if y["team"] == team]
            if not g:
                continue
            st = np.array([y["steps"] for y in g]); ps = np.array([y["passes"] for y in g])
            wx = np.array([y["won_x"] for y in g]); kk = np.array([y["kick"] for y in g])
            bk = np.array([y["back"] for y in g])
            nk = ~kk
            print(f"  {name} goals ({len(g)}): straight from a kickoff {kk.mean():.2f};  steps from winning the ball: "
                  f"median {np.median(st):.0f}, <=20 {np.mean(st <= 20):.2f}, <=60 {np.mean(st <= 60):.2f};  "
                  f"passes in the move: 0 {np.mean(ps == 0):.2f}, 1 {np.mean(ps == 1):.2f}, 2+ {np.mean(ps >= 2):.2f}"
                  + (f";  won (not kickoffs) in own third {np.mean(wx[nk] < 1 / 3):.2f}, middle "
                     f"{np.mean((wx[nk] >= 1 / 3) & (wx[nk] < 2 / 3)):.2f}, their third {np.mean(wx[nk] >= 2 / 3):.2f}"
                     if nk.any() else ""))
            if nk.any():
                print(f"      defenders goal-side of the ball when it was won (not kickoffs): mean {bk[nk].mean():.2f}, "
                      + ", ".join(f"{k}: {np.mean(bk[nk] == k):.2f}" for k in range(6)))
