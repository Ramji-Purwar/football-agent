"""Q2: the formation guesser and the three teams it is compared with (docs/questions.md, Q2).

Before the switch our general team plays (trained against 2-2-1, 2-1-2, 1-3-1, 3-1-1). The opponent starts in one of
the --starts formations and switches to 1-2-2 at a random step between 100 and 1100. From the switch on:
  none    the general team plays on: what a switch that is never noticed costs
  oracle  the 1-2-2 specialist comes on at the switch (it is told): the best one can do
  guess   the specialist comes on while the guesser (agents/belief_filter.py) says 1-2-2
All three play the same matches (same seeds): until one of them changes team, the matches are identical.

Run from football_adapt/, on a server:
  python -m scripts.q2 maps   --general G.pt --specialist S.pt --out DIR/maps.npz
  python -m scripts.q2 traces --general G.pt --starts 2-2-1,3-1-1 --out DIR/traces.npz
  python -m scripts.q2 tune   --maps DIR/maps.npz --traces DIR/traces.npz --c-late 0.44 --c-false 2-2-1=0.34,3-1-1=0.27
                              --out DIR/filter.json          (costs in goals per 100 steps, runs/q2/pairs/README.md)
                              [--anonymous]                  (the guesser is not told the opponents' player numbers)
  python -m scripts.q2 test   --general G.pt --specialist S.pt --maps DIR/maps.npz --filter DIR/filter.json
                              --starts 2-2-1,3-1-1 --out DIR/test.jsonl
Seeds: maps 30000+, traces (tuning) 40000+, test 60000+, so the test plays matches the guesser was not tuned on."""
from __future__ import annotations
import argparse
import itertools
import json
from multiprocessing import Pool
import numpy as np

TARGET = "1-2-2"
T_SWITCH = (100, 1100)


def load(ckpt, seed):
    import torch
    torch.set_num_threads(1)
    from agents.mappo import MAPPOAgent
    return MAPPOAgent.load(ckpt, seed=seed)


def start_match(ag, seed, start, mode="window"):
    """The env of the agent's setting (config values, formation, radius, zeta), reset for match `seed`. mode window:
    the opponent switches only when told (env.set_formation); score: it switches by itself on the score
    (opponent.switching.score_reactive: trailing -> 1-2-2, leading -> 3-1-1)."""
    from football import FootballEnv, load_config
    over = {**ag.overrides, "opponent.switching.mode": "SCORE_REACTIVE" if mode == "score" else "NONE"}
    env = FootballEnv(load_config(overrides=over))
    t = ag.train_args
    obs, _ = env.reset(seed=seed, our_formation=t["our_formation"], opp_formation=start, opp_style="normal",
                       radius=t["radius"], zeta=t["zeta"])
    return env, obs


def schedule(seed, starts):
    """The starting formation and the switch step of match `seed`, the same for every team that plays it."""
    rng = np.random.default_rng([seed, 2])
    return starts[int(rng.integers(len(starts)))], int(rng.integers(T_SWITCH[0], T_SWITCH[1] + 1))


# ---------------------------------------------------------------------------------------------- jobs (one match each)
def _maps_job(job):
    ckpt, form, seed = job
    from agents.belief_filter import PositionMaps
    ag = load(ckpt, seed)
    env, obs = start_match(ag, seed, form)
    maps = PositionMaps(env.book.names, env.L, env.W)
    done = False
    while not done:
        maps.add(form, *maps.view(env.last_obs))
        obs, _, done, _ = env.step(ag.act(obs))
    return maps.counts


def _trace_job(job):
    ckpt, starts, seed = job
    from agents.belief_filter import PositionMaps
    ag = load(ckpt, seed)
    start, ts = schedule(seed, starts)
    env, obs = start_match(ag, seed, start)
    maps = PositionMaps(env.book.names, env.L, env.W)                  # only for view()
    boxes, poss = [], []
    done = False
    while not done:
        if env.t == ts:
            env.set_formation(1, TARGET)
        b, p = maps.view(env.last_obs)
        boxes.append(b)
        poss.append(p)
        obs, _, done, _ = env.step(ag.act(obs))
    return start, ts, np.array(boxes), np.array(poss)


class Probe:
    """A linear read-out of a recurrent team's memory: is the opponent in 1-2-2 now? (idea.pdf: a probe on the hidden
    state). Fitted by `probe` on practice matches, applied to the players' mean hidden state."""

    def __init__(self, path):
        z = np.load(path)
        self.mu, self.sd, self.w, self.b = z["mu"], z["sd"], z["w"], float(z["b"])

    def says_target(self, h):
        return float(((np.asarray(h).mean(0) - self.mu) / self.sd) @ self.w + self.b) > 0.0


def _test_job(job):
    general, specialist, maps_path, params, starts, seed, variant, teams, mode, probe = job
    from agents.belief_filter import PositionMaps, FormationFilter
    if variant in teams:                       # one team plays the whole match (trained on switching opponents)
        gen = spec = load(teams[variant], seed)
    else:
        gen, spec = load(general, seed), load(specialist, seed)
    start, ts = schedule(seed, starts)
    env, obs = start_match(gen, seed, start, mode)
    if variant == "guess":
        maps = PositionMaps.load(maps_path)
        flt = FormationFilter(maps, **params)
    prb = Probe(probe) if variant in teams and probe and gen.actor.recurrent else None
    on, at_switch, swaps, false_steps, late_steps = False, None, 0, 0, 0
    was, entered, delays = False, None, []     # one delay (or None: never noticed) per time the opponent went into 1-2-2
    done = False
    while not done:
        if mode == "window" and env.t == ts:
            env.set_formation(1, TARGET)
            at_switch = list(env.score)
        truth = env.formation[1] == TARGET
        if truth and not was:
            entered = env.t
            delays.append(None)
        was = truth
        if variant == "oracle":
            want = truth
        elif variant == "guess":
            want = flt.update(*maps.view(env.last_obs)) == TARGET
        else:
            want = False
        swaps += int(want != on)
        on = want
        act = (spec if on else gen).act(obs)
        if prb is not None:
            on = prb.says_target(gen.h.numpy())                # what the team's memory says, after seeing this step
        if variant != "none" and (variant not in teams or prb is not None):
            false_steps += int(on and not truth)
            late_steps += int(truth and not on)
            if truth and on and delays[-1] is None:
                delays[-1] = env.t - entered
        obs, _, done, _ = env.step(act)
    goals = [(e["t"], e["team"]) for e in env.event_log if e["type"] == "goal"]
    switches = [(e["t"], e["to"]) for e in env.event_log if e["type"] == "switch" and e["team"] == 1]
    first = delays[0] if delays else None
    return dict(seed=seed, variant=variant, start=start, ts=ts if mode == "window" else None, score=list(env.score),
                at_switch=at_switch, delay=first, delays=delays, false_steps=false_steps, late_steps=late_steps,
                swaps=swaps, goals=goals, switches=switches, probed=prb is not None)


def _probe_job(job):
    ckpt, starts, seed, every = job
    ag = load(ckpt, seed)
    start, ts = schedule(seed, starts)
    env, obs = start_match(ag, seed, start)
    X, y = [], []
    done = False
    while not done:
        if env.t == ts:
            env.set_formation(1, TARGET)
        act = ag.act(obs)
        if env.t % every == 0:
            X.append(ag.h.numpy().mean(0))
            y.append(env.formation[1] == TARGET)
        obs, _, done, _ = env.step(act)
    return np.array(X, np.float32), np.array(y, bool)


# ---------------------------------------------------------------------------------------------- commands
def cmd_maps(a):
    from agents.belief_filter import PositionMaps
    from football import load_config
    cfg = load_config()
    names = list(cfg.formations.anchors.to_dict().keys())              # the order of env.book.names
    jobs = [(c, f, a.seed0 + s) for c in (a.general, a.specialist) for f in names for s in range(a.matches)]
    with Pool(a.procs) as p:
        counts = sum(p.imap_unordered(_maps_job, jobs, chunksize=2))
    PositionMaps(names, cfg.pitch.length, cfg.pitch.width, counts=counts).save(a.out)
    print(f"{a.out}: {len(jobs)} matches, {int(counts.sum() / 5):,} steps")


def cmd_traces(a):
    starts = a.starts.split(",")
    with Pool(a.procs) as p:
        res = p.map(_trace_job, [(a.general, starts, a.seed0 + s) for s in range(a.matches)], chunksize=2)
    np.savez(a.out, start=np.array([r[0] for r in res]), ts=np.array([r[1] for r in res]),
             boxes=np.stack([r[2] for r in res]), poss=np.stack([r[3] for r in res]))
    print(f"{a.out}: {len(res)} matches")


def costs(on, ts, c_late, c_false):
    """Goals a match is expected to lose to the guesser: per step on the wrong team, before the switch (a false
    alarm, c_false per trace) and after it (late, c_late). on: (N, T) bool, the specialist is playing."""
    t = np.arange(on.shape[1])[None, :]
    before = t < ts[:, None]
    false_steps = (on & before).sum(1)
    late_steps = (~on & ~before).sum(1)
    return c_false * false_steps + c_late * late_steps, false_steps, late_steps


def cmd_tune(a):
    from agents.belief_filter import PositionMaps, filter_step
    maps = PositionMaps.load(a.maps)
    logp, logn = maps.anonymous_log_probs() if a.anonymous else (maps.log_probs(), None)   # as FormationFilter
    z = np.load(a.traces)
    boxes, poss, ts, start = z["boxes"], z["poss"], z["ts"], z["start"]
    N, T = poss.shape
    ll = sum(logp[:, j][:, poss, boxes[..., j]] for j in range(5))      # (K, N, T)
    if logn is not None:
        ll = ll + logn[:, poss, (boxes != maps.unseen).sum(-1)]
    ll = ll.transpose(1, 2, 0)                                          # (N, T, K)
    k = maps.names.index(TARGET)
    c_late = a.c_late / 100.0
    cf = dict((s, float(v) / 100.0) for s, v in (x.split("=") for x in a.c_false.split(",")))
    c_false = np.array([cf[s] for s in start])
    rows = []
    for beta, h in itertools.product(a.betas, a.hs):
        b = np.full((N, len(maps.names)), 1.0 / len(maps.names))
        tops = np.zeros((N, T), int)
        for t in range(T):
            b = filter_step(b, ll[:, t], beta, h)
            tops[:, t] = b.argmax(1)
        for m in a.ms:
            prev, run, guess = np.full(N, -1), np.zeros(N, int), np.full(N, -1)
            on = np.zeros((N, T), bool)
            for t in range(T):
                run = np.where(tops[:, t] == prev, run + 1, 1)
                prev = tops[:, t]
                guess = np.where(run >= m, prev, guess)
                on[:, t] = guess == k
            cost, fs, ls = costs(on, ts, c_late, c_false)
            after = np.arange(T)[None, :] >= ts[:, None]
            first = np.where((on & after).any(1), (on & after).argmax(1) - ts, -1)
            rows.append(dict(beta=beta, h=h, m=m, cost=float(cost.mean()), false_steps=float(fs.mean()),
                             false_matches=float((fs > 0).mean()), late_steps=float(ls.mean()),
                             detected=float((first >= 0).mean()),
                             median_delay=float(np.median(first[first >= 0])) if (first >= 0).any() else None))
    rows.sort(key=lambda r: r["cost"])
    print(f"{N} matches{', without player numbers' if a.anonymous else ''}. Goals lost per match: {a.c_late} per "
          f"100 steps late, {a.c_false} per 100 steps of false alarm")
    print(f"{'beta':>6s} {'h':>7s} {'m':>4s} {'cost':>6s} {'late':>6s} {'false':>6s} {'false m':>7s} "
          f"{'found':>6s} {'delay':>6s}")
    for r in rows[:15]:
        print(f"{r['beta']:6.3f} {r['h']:7.4f} {r['m']:4d} {r['cost']:6.3f} {r['late_steps']:6.1f} "
              f"{r['false_steps']:6.1f} {r['false_matches']:7.0%} {r['detected']:6.0%} {r['median_delay']!s:>6s}")
    best = dict(beta=rows[0]["beta"], h=rows[0]["h"], m=rows[0]["m"], anonymous=a.anonymous)
    json.dump(best, open(a.out, "w"))
    json.dump(rows, open(a.out.replace(".json", "_grid.json"), "w"), indent=1)
    print(f"{a.out}: {best}")


LAG_W, LAG_EPS, LAG_STEP = 100, 0.05, 10     # adaptation lag: window (steps), tolerance (goals per 100 steps), grid


def steps_in_target(switches, T=1200):
    """Steps of a match the opponent played 1-2-2, from its switches [(t, to), ...] (it never starts in 1-2-2)."""
    n, since = 0, None
    for t, to in switches + [(T, None)]:
        if since is not None:
            n += t - since
        since = t if to == TARGET else None
    return n


def gd_curve(r, T=1200):
    """Goal difference (ours minus theirs) after every step of a match, from its goals."""
    g = np.zeros(T + 1)
    for t, team in r["goals"]:
        g[min(t, T - 1) + 1:] += 1 if team == 0 else -1
    return g


def adaptation(rows, T=1200):
    """idea.pdf eq. (10), matches aligned at their own switch: the goal-difference rate (per 100 steps) in the LAG_W
    steps before the switch, the rate in the LAG_W steps from every step after it, and the lag = first step after the
    switch from which the rate is back within LAG_EPS of the rate before (None: not within the match)."""
    curves = [(gd_curve(r, T), r["ts"]) for r in rows]
    before = np.mean([(g[ts] - g[ts - LAG_W]) for g, ts in curves if ts >= LAG_W]) * 100 / LAG_W
    after, lag = {}, None
    for k in range(0, T - LAG_W, LAG_STEP):
        c = [(g[ts + k + LAG_W] - g[ts + k]) for g, ts in curves if ts + k + LAG_W <= T]
        if len(c) < 30:
            break
        after[k] = float(np.mean(c)) * 100 / LAG_W
        if lag is None and after[k] >= before - LAG_EPS:
            lag = k
    return before, after, lag


def summary(res, variants, mode="window"):
    """W and goals of every team, the regret of each against the oracle (paired by seed), how the guesser (and the
    probe of a recurrent team) did, and, for a single switch, the adaptation lag."""
    def points(d):                                                      # per match: 1 win, 0.5 draw, 0 loss
        return (d > 0) + 0.5 * (d == 0)

    by = {v: {r["seed"]: r for r in res if r["variant"] == v} for v in variants}
    seeds = sorted(set.intersection(*(set(by[v]) for v in variants)))
    out = []
    for group, ss in [("all", seeds)] + [(s, [x for x in seeds if by["oracle"][x]["start"] == s])
                                         for s in sorted({r["start"] for r in res})]:
        if not ss:
            continue
        out.append(f"\n{group}{f' -> {TARGET}' if mode == 'window' else ' (switches on the score)'}: {len(ss)} matches")
        gd = {}
        for v in variants:
            sc = np.array([by[v][x]["score"] for x in ss], float)
            d = sc[:, 0] - sc[:, 1]
            gd[v] = d
            line = f"  {v:9s} W {np.mean(points(d)):4.2f}  goals {sc[:, 0].mean():4.2f}-{sc[:, 1].mean():4.2f}  GD {d.mean():+5.2f}"
            if mode == "window":
                aft = sc - np.array([by[v][x]["at_switch"] for x in ss], float)
                line += f"  after the switch {aft[:, 0].mean():4.2f}-{aft[:, 1].mean():4.2f} ({(aft[:, 0] - aft[:, 1]).mean():+5.2f})"
            else:
                sw = [by[v][x]["switches"] for x in ss]
                line += (f"  | the opponent switches {np.mean([len(w) for w in sw]):.1f} times a match, is in 1-2-2 "
                         f"{np.mean([steps_in_target(w) for w in sw]):.0f} steps")
            out.append(line)
        for v in variants:
            if v == "oracle":
                continue
            dd, dw = gd["oracle"] - gd[v], points(gd["oracle"]) - points(gd[v])
            out.append(f"  regret of {v:9s} vs oracle: W {dw.mean():+5.2f} (+-{1.96 * dw.std() / np.sqrt(len(dw)):.2f}),"
                       f" GD {dd.mean():+5.2f} (+-{1.96 * dd.std() / np.sqrt(len(dd)):.2f})")
        for v in variants:
            g = [by[v][x] for x in ss]
            if v in ("none", "oracle") or (v != "guess" and not g[0]["probed"]):
                continue
            dl = np.array([d for r in g for d in r["delays"] if d is not None])
            n = sum(len(r["delays"]) for r in g)
            who = "guesser" if v == "guess" else f"probe of {v}"
            out.append(f"  {who}: found {len(dl) / max(1, n):.0%} of {n} switches into 1-2-2, delay median "
                       f"{np.median(dl) if len(dl) else '-'} steps (mean {dl.mean() if len(dl) else float('nan'):.0f}); "
                       f"says 1-2-2 wrongly in {np.mean([r['false_steps'] > 0 for r in g]):.0%} of matches, "
                       f"{np.mean([r['false_steps'] for r in g]):.1f} steps a match; misses {np.mean([r['late_steps'] for r in g]):.0f}"
                       f" steps a match while it is 1-2-2")
        if mode == "window":
            out.append(f"  adaptation lag (goal-difference rate per 100 steps over {LAG_W}-step windows, aligned at the "
                       f"switch; lag = first step it is back within {LAG_EPS} of the rate before):")
            for v in variants:
                before, after, lag = adaptation([by[v][x] for x in ss])
                pts = "  ".join(f"{k}:{after[k]:+.2f}" for k in (0, 100, 200, 300, 500) if k in after)
                out.append(f"    {v:9s} before {before:+.2f} | after, from step  {pts} | lag "
                           f"{lag if lag is not None else 'not within the match'}")
    return "\n".join(out)


def cmd_probe(a):
    """Fit the probe: logistic regression of 'the opponent is in 1-2-2 now' on the players' mean hidden state, on
    practice switch matches (seeds 40000+, the tuning seeds), every --every steps."""
    with Pool(a.procs) as p:
        res = p.map(_probe_job, [(a.team, a.starts.split(","), a.seed0 + s, a.every) for s in range(a.matches)])
    X, y = np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]).astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(3000):                                               # plain gradient descent, a little L2
        q = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        w -= 0.5 * (Z.T @ (q - y) / len(y) + 1e-3 * w)
        b -= 0.5 * float(np.mean(q - y))
    acc = float(np.mean(((Z @ w + b) > 0) == (y > 0.5)))
    np.savez(a.out, mu=mu, sd=sd, w=w, b=b)
    print(f"{a.out}: {len(y)} steps from {a.matches} matches, share in 1-2-2 {y.mean():.2f}, accuracy on them {acc:.3f}")


def cmd_test(a):
    params = json.load(open(a.filter))
    starts = a.starts.split(",")
    teams = dict(x.split("=", 1) for x in a.team)
    jobs = [(a.general, a.specialist, a.maps, params, starts, a.seed0 + s, v, teams, a.mode, a.probe)
            for s in range(a.matches) for v in ("none", "oracle", "guess", *teams)]
    res = []
    with Pool(a.procs) as p, open(a.out, "w") as fh:
        for r in p.imap_unordered(_test_job, jobs):
            res.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    text = (f"general {a.general}\nspecialist {a.specialist}\nteams {teams}\nfilter {params}, starts {a.starts}, "
            f"switch mode {a.mode}, seeds {a.seed0}+" + summary(res, ["none", "oracle", "guess", *teams], a.mode))
    open(a.out.replace(".jsonl", ".txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("maps", "traces", "tune", "test", "probe"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--procs", type=int, default=56)
        if name in ("maps", "traces", "test"):
            s.add_argument("--general", required=True)
        if name in ("maps", "test"):
            s.add_argument("--specialist", required=True)
        if name in ("traces", "test", "probe"):
            s.add_argument("--starts", required=True, help="starting formations, e.g. 2-2-1,3-1-1")
        if name == "probe":
            s.add_argument("--team", required=True, help="a recurrent team's checkpoint")
            s.add_argument("--matches", type=int, default=200)
            s.add_argument("--seed0", type=int, default=40_000)
            s.add_argument("--every", type=int, default=5, help="keep every n-th step")
        if name == "maps":
            s.add_argument("--matches", type=int, default=40, help="per team and opponent formation")
            s.add_argument("--seed0", type=int, default=30_000)
        if name == "traces":
            s.add_argument("--matches", type=int, default=200)
            s.add_argument("--seed0", type=int, default=40_000)
        if name in ("tune", "test"):
            s.add_argument("--maps", required=True)
        if name == "tune":
            s.add_argument("--traces", required=True)
            s.add_argument("--c-late", type=float, required=True, help="goals lost per 100 steps late")
            s.add_argument("--c-false", required=True, help="per starting formation, e.g. 2-2-1=0.34,3-1-1=0.27")
            s.add_argument("--anonymous", action="store_true", help="the guesser is not told the player numbers")
            s.add_argument("--betas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
            s.add_argument("--hs", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 5e-2])
            s.add_argument("--ms", type=int, nargs="+", default=[1, 3, 5, 10, 20, 40])
        if name == "test":
            s.add_argument("--filter", required=True)
            s.add_argument("--team", action="append", default=[], metavar="NAME=CKPT",
                           help="also a team that plays the whole match alone (recurrent or memoryless), repeated")
            s.add_argument("--probe", default=None, help="a probe (from `probe`) for the recurrent teams' memory")
            s.add_argument("--mode", choices=["window", "score"], default="window",
                           help="window: one switch to 1-2-2 at a random step; score: the opponent switches on the score")
            s.add_argument("--matches", type=int, default=300)
            s.add_argument("--seed0", type=int, default=60_000)
    a = ap.parse_args()
    dict(maps=cmd_maps, traces=cmd_traces, tune=cmd_tune, test=cmd_test, probe=cmd_probe)[a.cmd](a)
