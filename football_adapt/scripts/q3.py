"""Q3: can our team tell which opponent player is the playmaker (docs/questions.md, Q3)?

A case is an opponent formation and its playmaker (0 = none), e.g. 2-2-1:4. The playmaker is as configured
(opponent.playmaker: full vision, free role, control). Our general team (v4) plays every case while we record what
it sees (agents/playmaker_filter.py, Evidence); the guesser is then built and judged offline on those recordings.
`play` then plays every case three ways on the same seeds: unaware (the general team all match), oracle (the
specialist for the true playmaker from the kick-off, the general team if he has none) and adaptive (the general team
until the guess commits, then the specialist for the guess; a guess without a specialist keeps the general team).

Run from football_adapt/, on a server:
  python -m scripts.q3 record   --general G.pt --cases 2-2-1:0,2-2-1:3,... --matches 100 --seed0 30000 --out DIR/maps_rec.npz
  python -m scripts.q3 evaluate --maps DIR/maps_rec.npz --tune DIR/tune_rec.npz --test DIR/test_rec.npz --out DIR/guess.txt
  python -m scripts.q3 play     --general G.pt --specialist 2:S2.pt --specialist 4:S4.pt --maps DIR/maps_rec.npz
                                --beta 0.05 --commit 0.95 --cases ... --matches 500 --seed0 70000 --out DIR/play
Seeds: maps 30000+, tuning 40000+, test 60000+, play 70000+."""
from __future__ import annotations
import argparse
import itertools
from multiprocessing import Pool
import numpy as np

CHECK = (100, 300, 600, 1199)              # steps at which the guess is scored


def _record_job(job):
    ckpt, form, sigma, seed = job
    import torch
    torch.set_num_threads(1)
    from football import FootballEnv, load_config
    from agents.mappo import MAPPOAgent
    from agents.playmaker_filter import Evidence
    ag = MAPPOAgent.load(ckpt, seed=seed)
    t = ag.train_args
    env = FootballEnv(load_config(overrides=ag.overrides))
    obs, _ = env.reset(seed=seed, our_formation=t["our_formation"], opp_formation=form, opp_style="normal",
                       radius=t["radius"], zeta=t["zeta"], sigma=sigma)
    ev = Evidence()
    rows = []
    done = False
    while not done:
        bins, poss, e = ev.step(env.last_obs)
        rows.append((*bins, poss, -1 if e is None else e[0], -1 if e is None else e[1]))
        obs, _, done, _ = env.step(ag.act(obs))
    return form, sigma, seed, np.array(rows, np.int8), list(env.score)


def cmd_record(a):
    cases = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cases.split(",")]
    jobs = [(a.general, f, s, a.seed0 + m) for m in range(a.matches) for f, s in cases]
    with Pool(a.procs) as p:
        res = p.map(_record_job, jobs, chunksize=2)
    np.savez(a.out, form=np.array([r[0] for r in res]), sigma=np.array([r[1] for r in res]),
             seed=np.array([r[2] for r in res]), rows=np.stack([r[3] for r in res]), score=np.array([r[4] for r in res]))
    print(f"{a.out}: {len(res)} matches")


def load_rec(path):
    z = np.load(path)
    r = z["rows"].astype(int)                                   # (matches, T, 8): 5 bins, ball state, passer, speed
    return dict(form=z["form"], sigma=z["sigma"], score=z["score"], bins=r[..., :5], poss=r[..., 5], ej=r[..., 6],
                es=r[..., 7])


def maps_from(rec):
    from agents.playmaker_filter import PlaymakerMaps
    maps = PlaymakerMaps()
    M, T = rec["poss"].shape
    for m in range(M):
        pm = (np.arange(5) == rec["sigma"][m] - 1).astype(int)
        j = np.broadcast_to(np.arange(5), (T, 5))
        np.add.at(maps.pos, (j, pm[j], np.broadcast_to(rec["poss"][m][:, None], (T, 5)), rec["bins"][m]), 1)
        e = rec["ej"][m] >= 0
        np.add.at(maps.passes, (rec["ej"][m][e], pm[rec["ej"][m][e]], rec["es"][m][e]), 1)
    return maps


def logliks(flt, rec):
    """Per match, step and hypothesis: the position part and the pass part of the log-likelihood (as flt.loglik)."""
    M, T = rec["poss"].shape
    H = len(flt.hyp)
    lp, lq = np.zeros((M, T, H)), np.zeros((M, T, H))
    e = rec["ej"] >= 0
    ej, es = np.where(e, rec["ej"], 0), np.where(e, rec["es"], 0)
    for h in range(H):
        for j in range(5):
            lp[..., h] += flt.lp_pos[j, flt.is_pm[h, j]][rec["poss"], rec["bins"][..., j]]
        lq[..., h] = np.where(e, flt.lp_pass[ej, flt.is_pm[h][ej], es], 0.0)
    return lp, lq


def posteriors(flt, lp, lq, beta, beta_pass, use_speed):
    """Belief over the hypotheses after every step, (matches, T, H): the filter's update, all steps at once."""
    lb = beta * np.cumsum(lp, 1) + (beta_pass * np.cumsum(lq, 1) if use_speed else 0.0)
    b = np.exp(lb - lb.max(-1, keepdims=True))
    return b / b.sum(-1, keepdims=True)


def judge(flt, post, rec, p_commit):
    """Per match: the guess commits once its top hypothesis reaches p_commit. Returns (truth index, argmax per step,
    commit step or -1, committed hypothesis index or -1, score = share of the match after a right commit minus after a
    wrong one)."""
    T = post.shape[1]
    truth = np.array([flt.hyp.index(s) if s in flt.hyp else -1 for s in rec["sigma"]])
    top, pmax = post.argmax(-1), post.max(-1)
    hit = pmax >= p_commit
    when = np.where(hit.any(1), hit.argmax(1), -1)
    what = np.where(when >= 0, top[np.arange(len(top)), np.maximum(when, 0)], -1)
    share = np.where(when >= 0, (T - when) / T, 0.0)
    score = np.where(what == truth, share, -share)
    return truth, top, when, what, score


def cmd_evaluate(a):
    from agents.playmaker_filter import PlaymakerFilter
    maps = maps_from(load_rec(a.maps))
    tune, test = load_rec(a.tune), load_rec(a.test)
    out = [f"hypotheses: {['none' if h == 0 else f'player {h}' for h in maps.hypotheses()]}",
           f"maps from {a.maps}, tuned on {a.tune}, tested on {a.test}"]
    for use_speed in (True, False):
        flt = PlaymakerFilter(maps, 1.0, 1.0, use_speed)
        lp, lq = logliks(flt, tune)
        best = None
        for beta, beta_pass in itertools.product(a.betas, a.beta_passes if use_speed else [0.0]):
            post = posteriors(flt, lp, lq, beta, beta_pass, use_speed)
            for pc in a.commits:
                s = judge(flt, post, tune, pc)[-1].mean()
                if best is None or s > best[0]:
                    best = (s, beta, beta_pass, pc)
        _, beta, beta_pass, pc = best
        lp, lq = logliks(flt, test)
        truth, top, when, what, score = judge(flt, posteriors(flt, lp, lq, beta, beta_pass, use_speed), test, pc)
        out.append(f"\n== evidence: positions{' + pass speed' if use_speed else ' only'}; tuned beta {beta}"
                   f"{f', beta_pass {beta_pass}' if use_speed else ''}, commit at p >= {pc}")
        out.append(f"{'case':10s} {'n':>4s} {'v4 W':>5s}  " + "  ".join(f"top@{t + 1:<4d}" for t in CHECK)
                   + f"  {'right':>6s} {'wrong':>6s} {'never':>6s}  {'commit step (median, right)':>28s}")
        cases = sorted({(f, s) for f, s in zip(test["form"], test["sigma"])}, key=lambda c: (c[0], c[1]))
        for f, s in cases + [("all", None)]:
            i = np.ones(len(truth), bool) if f == "all" else (test["form"] == f) & (test["sigma"] == s)
            d = test["score"][i, 0] - test["score"][i, 1]
            acc = "  ".join(f"{np.mean(top[i, t] == truth[i]):8.0%}" for t in CHECK)
            right, wrong = (what[i] == truth[i]) & (when[i] >= 0), (what[i] != truth[i]) & (when[i] >= 0)
            med = np.median(when[i][right]) if right.any() else float("nan")
            name = "all" if f == "all" else f"{f}:{s}"
            out.append(f"{name:10s} {i.sum():4d} {np.mean((d > 0) + 0.5 * (d == 0)):5.2f}  {acc}  {right.mean():6.0%} "
                       f"{wrong.mean():6.0%} {(when[i] < 0).mean():6.0%}  {med:28.0f}")
    text = "\n".join(out)
    open(a.out, "w").write(text + "\n")
    print(text)


_MAPS = {}                                 # per worker process: the guess's maps, built once


class SigmaProbe:
    """A linear read-out of a recurrent team's memory: which opponent, if any, is the playmaker (idea.pdf: a probe on
    the hidden state). Fitted by `probe` on practice matches; classes as the guess's hypotheses (0 = none)."""

    def __init__(self, path):
        z = np.load(path)
        self.mu, self.sd, self.W, self.b, self.classes = z["mu"], z["sd"], z["W"], z["b"], [int(c) for c in z["classes"]]

    def says(self, h):
        return self.classes[int(np.argmax(((np.asarray(h).mean(0) - self.mu) / self.sd) @ self.W + self.b))]


def _behaviour(env, pm, holder_steps):
    """How the playmaker fared: steps on the ball, passes he got, passes of his cut out, tackles he lost, his goals."""
    ev = env.event_log
    return dict(pm_ball=holder_steps,
                pm_got=sum(1 for e in ev if e["type"] == "pass_result" and e["result"] == "completed"
                           and tuple(e["by"]) == pm),
                pm_cut=sum(1 for e in ev if e["type"] == "pass_result" and e["result"] == "intercepted"
                           and tuple(e["passer"]) == pm),
                pm_tackled=sum(1 for e in ev if e["type"] == "tackle" and e["success"] and tuple(e["from_"]) == pm),
                pm_goals=sum(1 for e in ev if e["type"] == "shot" and e["scored"] and tuple(e["by"]) == pm))


def _play_job(job):
    mode, general, specsets, maps_path, beta, p_commit, form, sigma, seed, teams, probe = job
    import torch
    torch.set_num_threads(1)
    from football import FootballEnv, load_config
    from agents.mappo import MAPPOAgent
    from agents.playmaker_filter import Evidence, PlaymakerFilter
    kind = "team" if mode in teams else "oracle" if mode.startswith("oracle") else \
        "adaptive" if mode.startswith("adaptive") else "unaware"
    specs = specsets.get(mode[len(kind):], {}) if kind in ("oracle", "adaptive") else {}
    team = MAPPOAgent.load(teams[mode] if kind == "team" else specs[sigma] if kind == "oracle" and sigma in specs
                           else general, seed=seed)
    t = team.train_args
    env = FootballEnv(load_config(overrides=team.overrides))
    obs, _ = env.reset(seed=seed, our_formation=t["our_formation"], opp_formation=form, opp_style="normal",
                       radius=t["radius"], zeta=t["zeta"], sigma=sigma)
    game = {k: v for k, v in team.overrides.items() if k != "actions.shadow"}
    flt = ev = None
    if kind == "adaptive":
        if maps_path not in _MAPS:
            _MAPS[maps_path] = maps_from(load_rec(maps_path))
        flt, ev = PlaymakerFilter(_MAPS[maps_path], beta, 0.0, use_speed=False), Evidence()
    prb = SigmaProbe(probe) if kind == "team" and probe and team.actor.recurrent else None
    said = []                                  # the probe's reading after every step
    pm = (1, sigma - 1)
    when, guess, done, holder = -1, -1, False, 0
    while not done:
        if flt is not None and when < 0:
            h, p = flt.update(*ev.step(env.last_obs))
            if p >= p_commit:
                when, guess = env.t, h
                if h in specs:
                    team = MAPPOAgent.load(specs[h], seed=seed)
                    over = dict(team.overrides)
                    assert {k: v for k, v in over.items() if k != "actions.shadow"} == game, \
                        "the specialist was trained in another game"
                    env.cfg._d["actions"]["shadow"] = over.get("actions.shadow", True)   # a no-shadow specialist
        act = team.act(obs)
        if prb is not None:
            said.append(prb.says(team.h.numpy()))
        holder += int(sigma > 0 and env.holder == pm)
        obs, _, done, _ = env.step(act)
    s = env.stats
    out = dict(mode=mode, form=form, sigma=sigma, seed=seed, gf=env.score[0], ga=env.score[1], commit=when,
               guess=guess, shadow=s["shadow_steps_total"], shadow_pm=s["shadow_steps_on_sigma"],
               **(_behaviour(env, pm, holder) if sigma else {}))
    if prb is not None:
        said = np.array(said)
        wrong = np.flatnonzero(said != sigma)
        out.update(probe_acc=float(np.mean(said == sigma)),
                   probe_id=int(wrong[-1] + 1) if len(wrong) else 0)       # from this step on it is always right
    return out


def _w(r):
    d = np.array([x["gf"] - x["ga"] for x in r])
    return (d > 0) + 0.5 * (d == 0), d


def play_table(rows, cases, modes):
    """Per case: W of each team; the cost E = W(unaware, no playmaker) - W(unaware, playmaker); the recovery
    kappa = (W(team) - W(unaware)) / E of every other team (only when E's interval excludes 0); the regret against the
    oracle; the shadow value (with minus without the shadow action); all paired by seed with 95% intervals (kappa by
    bootstrap over seeds). Then the guess's accuracy and delay, the playmaker's matches under each team, and the probe
    of a recurrent team's memory."""
    rng = np.random.default_rng(0)
    by = {}
    for x in rows:
        by.setdefault((x["mode"], x["form"], x["sigma"]), {})[x["seed"]] = x
    ci = lambda v: 1.96 * np.std(v) / np.sqrt(len(v))
    others = [m for m in modes if m != "unaware"]
    T1 = [f"{'case':9s} {'n':>4s}  " + " ".join(f"{m:>11s}" for m in modes) + f"  {'E (cost)':>12s}  "
          + "  ".join(f"{'kappa ' + m:>21s}" for m in others)]
    T2 = [f"{'case':9s}  " + "  ".join(f"{'regret ' + m:>20s}" for m in others if m != "oracle")
          + "".join(f"  {'shadow value ' + m:>24s}" for m in ("oracle", "adaptive") if m + "_ns" in modes)]
    T3 = [f"{'case':9s} {'guess right/wrong/never':>23s} {'commit':>6s}   shadow on him (" + " / ".join(modes) + ")"]
    T4 = [f"{'case':9s} {'team':>11s}  {'his steps on the ball':>21s} {'passes he got':>13s} {'his passes cut':>14s}"
          f" {'tackled':>7s} {'his goals':>9s}"]
    T5 = []
    for f, s in cases:
        seeds = sorted(set.intersection(*(set(by.get((m, f, s), {})) for m in modes)))
        if not seeds:
            continue
        get = lambda m: [by[(m, f, s)][k] for k in seeds]
        w = {m: _w(get(m))[0] for m in modes}
        line = f"{f}:{s:<3d} {len(seeds):4d}  " + " ".join(f"{w[m].mean():11.2f}" for m in modes) + "  "
        base = by.get(("unaware", f, 0), {})
        both = [k for k in seeds if k in base] if s else []
        if both:
            idx = [seeds.index(k) for k in both]
            e = _w([base[k] for k in both])[0] - w["unaware"][idx]
            line += f"{e.mean():5.2f} +-{ci(e):.2f}  "
            for m in others:
                g = w[m][idx] - w["unaware"][idx]
                if e.mean() - ci(e) > 0:
                    bs = [g[b].mean() / e[b].mean() if e[b].mean() else np.nan
                          for b in (rng.integers(len(e), size=len(e)) for _ in range(2000))]
                    line += f"{g.mean() / e.mean():9.2f} [{np.nanpercentile(bs, 2.5):5.2f},{np.nanpercentile(bs, 97.5):5.2f}]  "
                else:
                    line += f"{'(E ~ 0)':>21s}  "
        else:
            line += f"{'-':>12s}  " + "  ".join(f"{'-':>21s}" for _ in others)
        T1.append(line)
        reg = [w["oracle"] - w[m] for m in others if m != "oracle"]
        line = f"{f}:{s:<3d}  " + "  ".join(f"{r.mean():+13.2f} +-{ci(r):.2f}" for r in reg)
        for m in ("oracle", "adaptive"):
            if m + "_ns" in modes:
                a_sh = w[m] - w[m + "_ns"]
                line += f"  {a_sh.mean():+17.2f} +-{ci(a_sh):.2f}"
        T2.append(line)
        sh = lambda m: np.sum([x["shadow_pm"] for x in get(m)]) / max(1, np.sum([x["shadow"] for x in get(m)]))
        for m in [m for m in modes if m.startswith("adaptive")][:1]:
            ad = get(m)
            com = np.array([x["commit"] for x in ad])
            right = np.array([x["guess"] == s for x in ad]) & (com >= 0)
            wrong = (com >= 0) & ~right
            med = np.median(com[right]) if right.any() else float("nan")
            T3.append(f"{f}:{s:<3d} {right.mean():7.0%} /{wrong.mean():5.0%} /{(com < 0).mean():5.0%} {med:6.0f}   "
                      + (" / ".join(f"{sh(mm):.0%}" for mm in modes) if s else ""))
        if s:
            for m in modes:
                r = get(m)
                mm = lambda k: np.mean([x[k] for x in r])
                T4.append(f"{f}:{s:<3d} {m:>11s}  {mm('pm_ball'):21.0f} {mm('pm_got'):13.1f} {mm('pm_cut'):14.2f}"
                          f" {mm('pm_tackled'):7.2f} {mm('pm_goals'):9.2f}")
        for m in modes:
            r = get(m)
            if "probe_acc" in r[0]:
                pid = np.array([x["probe_id"] for x in r])
                ok = pid < 1200
                T5.append(f"{f}:{s:<3d} {m}: probe right on {np.mean([x['probe_acc'] for x in r]):.0%} of steps; settles on "
                          f"the right answer in {ok.mean():.0%} of matches, from step {np.median(pid[ok]) if ok.any() else '-'}"
                          f" (median)")
    out = (T1 + ["", "Regret against the oracle (W) and the shadow value (W with minus without the shadow action):"] + T2
           + ["", "The guess (adaptive team) and the share of our SHADOW steps on the true playmaker:"] + T3
           + ["", "The playmaker, per match:"] + T4)
    if T5:
        out += ["", "Probe of the recurrent team's memory (who is the playmaker):"] + T5
    out += ["", "W = wins + draws/2. E, kappa, regret, shadow value paired by seed; +- and [ ] are 95% intervals. kappa",
            "= share of the cost won back (1 = all of it). guess: share of matches whose committed guess is right /",
            "wrong / that never commit; commit = median step of a right commit."]
    return "\n".join(out)


def cmd_play(a):
    import json
    specsets = {"": {int(x.split(":")[0]): x.split(":", 1)[1] for x in a.specialist}}
    if a.specialist_ns:
        specsets["_ns"] = {int(x.split(":")[0]): x.split(":", 1)[1] for x in a.specialist_ns}
    teams = dict(x.split("=", 1) for x in a.team)
    modes = ["unaware", "oracle", "adaptive"] + (["oracle_ns", "adaptive_ns"] if a.specialist_ns else []) + list(teams)
    cases = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cases.split(",")]
    jobs = [(m, a.general, specsets, a.maps, a.beta, a.commit, f, s, a.seed0 + k, teams, a.probe)
            for k in range(a.matches) for f, s in cases for m in modes]
    rows = []
    with Pool(a.procs) as p, open(a.out + "_matches.jsonl", "w") as fh:
        for x in p.imap_unordered(_play_job, jobs, chunksize=2):
            rows.append(x)
            fh.write(json.dumps(x) + "\n")
    text = (f"general {a.general}; specialists {specsets}; teams {teams}; guess: positions only, beta {a.beta}, "
            f"commit at p >= {a.commit}, maps {a.maps}; {a.matches} matches per case, seeds {a.seed0}+\n\n"
            + play_table(rows, cases, modes))
    open(a.out + ".txt", "w").write(text + "\n")
    print(text)


def _probe_job(job):
    ckpt, form, sigma, seed, every = job
    import torch
    torch.set_num_threads(1)
    from football import FootballEnv, load_config
    from agents.mappo import MAPPOAgent
    team = MAPPOAgent.load(ckpt, seed=seed)
    t = team.train_args
    env = FootballEnv(load_config(overrides=team.overrides))
    obs, _ = env.reset(seed=seed, our_formation=t["our_formation"], opp_formation=form, opp_style="normal",
                       radius=t["radius"], zeta=t["zeta"], sigma=sigma)
    X, done = [], False
    while not done:
        act = team.act(obs)
        if env.t % every == 0:
            X.append(team.h.numpy().mean(0))
        obs, _, done, _ = env.step(act)
    return np.array(X, np.float32), sigma


def cmd_probe(a):
    """Fit the probe: multinomial logistic regression of who the playmaker is (0 = none) on the players' mean hidden
    state, on practice matches (seeds 40000+, the guess's tuning seeds), every --every steps."""
    cases = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cases.split(",")]
    with Pool(a.procs) as p:
        res = p.map(_probe_job, [(a.team, f, s, a.seed0 + k, a.every) for k in range(a.matches) for f, s in cases])
    classes = sorted({s for _, s in cases})
    X = np.concatenate([r[0] for r in res])
    y = np.concatenate([np.full(len(r[0]), classes.index(r[1])) for r in res])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    Y = np.eye(len(classes))[y]
    W, b = np.zeros((Z.shape[1], len(classes))), np.zeros(len(classes))
    for _ in range(3000):                                               # plain gradient descent, a little L2
        L = Z @ W + b
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        W -= 0.5 * (Z.T @ (P - Y) / len(y) + 1e-3 * W)
        b -= 0.5 * (P - Y).mean(0)
    acc = float(np.mean((Z @ W + b).argmax(1) == y))
    np.savez(a.out, mu=mu, sd=sd, W=W, b=b, classes=np.array(classes))
    print(f"{a.out}: {len(y)} steps from {len(res)} matches, classes {classes}, accuracy on them {acc:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("record")
    s.add_argument("--general", required=True)
    s.add_argument("--cases", required=True, help="FORMATION:PLAYMAKER,... (0 = no playmaker)")
    s.add_argument("--matches", type=int, default=100, help="per case")
    s.add_argument("--seed0", type=int, required=True)
    s.add_argument("--procs", type=int, default=56)
    s.add_argument("--out", required=True)
    s = sub.add_parser("evaluate")
    s.add_argument("--maps", required=True)
    s.add_argument("--tune", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--betas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    s.add_argument("--beta-passes", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    s.add_argument("--commits", type=float, nargs="+", default=[0.9, 0.95, 0.99])
    s = sub.add_parser("probe")
    s.add_argument("--team", required=True, help="a recurrent team's checkpoint")
    s.add_argument("--cases", required=True)
    s.add_argument("--matches", type=int, default=50, help="per case")
    s.add_argument("--seed0", type=int, default=40_000)
    s.add_argument("--every", type=int, default=5, help="keep every n-th step")
    s.add_argument("--procs", type=int, default=56)
    s.add_argument("--out", required=True)
    s = sub.add_parser("play")
    s.add_argument("--general", required=True)
    s.add_argument("--specialist", action="append", required=True, help="PLAYMAKER:CKPT, repeated")
    s.add_argument("--specialist-ns", action="append", default=[],
                   help="PLAYMAKER:CKPT of specialists trained without the shadow action (oracle_ns, adaptive_ns)")
    s.add_argument("--team", action="append", default=[], metavar="NAME=CKPT",
                   help="a team that plays the whole match alone (the recurrent mixed team), repeated")
    s.add_argument("--probe", default=None, help="a probe (from `probe`) for the recurrent teams' memory")
    s.add_argument("--maps", required=True, help="a record file (DIR/maps_rec.npz)")
    s.add_argument("--beta", type=float, required=True)
    s.add_argument("--commit", type=float, required=True)
    s.add_argument("--cases", required=True)
    s.add_argument("--matches", type=int, default=500, help="per case")
    s.add_argument("--seed0", type=int, required=True)
    s.add_argument("--procs", type=int, default=56)
    s.add_argument("--out", required=True, help="writes OUT.txt and OUT_matches.jsonl")
    a = ap.parse_args()
    dict(record=cmd_record, evaluate=cmd_evaluate, play=cmd_play, probe=cmd_probe)[a.cmd](a)
