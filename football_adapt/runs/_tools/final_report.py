"""Final results for the Q1/Q2/Q3 reports, built only from the files the runs and evaluations wrote.

Run from football_adapt/ (PC or server; needs numpy + matplotlib only, no torch):
    python runs/_tools/final_report.py
Writes ../docs/results_final.md (every table, figure and the automatic verdicts), ../docs/img/final/*.png, and fills
the blocks between  <!-- AUTO:name -->  and  <!-- /AUTO:name -->  in ../docs/reportQ1.md, reportQ2.md, reportQ3.md.
Text outside those blocks is never touched. Missing results (a run or evaluation not finished yet) are listed as
missing; run the script again later and everything is recomputed from scratch.

Statistics: a run's test W = wins + draws/2 over its 300 test matches (result.json). A setting's W is the mean over
its training seeds, with a 95% t-interval over seeds (the seed-to-seed spread is much larger than the 300-match
noise of one run, so the interval over seeds is the honest one). Differences between settings use Welch's interval.
"With n seeds" is always stated: one seed gives no interval at all."""
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs")
import os
DOCS = Path(os.environ.get("FINAL_DOCS", "../docs"))      # FINAL_DOCS=dir writes there instead
IMG = DOCS / "img" / "final"
T = 1200                                                   # steps per match
# two-sided 95% t quantiles by degrees of freedom (no scipy on the servers)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 1000: 1.96}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"                # categorical slots 1-3 (validated default palette)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})


def t95(df):
    df = max(1.0, df)
    ks = sorted(T95)
    for a, b in zip(ks, ks[1:]):
        if df <= b:
            return T95[a] + (T95[b] - T95[a]) * (df - a) / (b - a)
    return 1.96


# ---------------------------------------------------------------- reading runs
def load_run(d: Path):
    r = json.loads((d / "result.json").read_text())
    ev = [json.loads(l) for l in open(d / "log.jsonl")] if (d / "log.jsonl").exists() else []
    ev = [e for e in ev if e.get("kind") == "eval"]
    seen, curve = set(), []
    for e in ev:                                           # one row per checkpoint (15 Nov: a few doubled rows)
        if e["step"] in seen:
            continue
        seen.add(e["step"])
        curve.append((e["step"] / T, e["W"]))
    vs = {}
    if (d / "vs_formations.txt").exists():
        for f, w in re.findall(r"^(\S+)(?: NEW)?\s+W ([\d.]+)", (d / "vs_formations.txt").read_text(), re.M):
            vs[f] = float(w)
    return dict(seed=d.name, W=r["win"] + 0.5 * r["draw"], win=r["win"], draw=r["draw"], gd=r.get("goal_diff"),
                gf=r.get("goals_for"), best=r.get("best_ckpt"), val=r.get("validation", {}).get("W"),
                curve=sorted(curve), vs=vs, args=r.get("args", {}))


def setting(path):
    p = RUNS / path
    runs = [load_run(d) for d in sorted(p.glob("s*")) if (d / "result.json").exists()] if p.exists() else []
    running = [d.name for d in sorted(p.glob("s*")) if not (d / "result.json").exists()] if p.exists() else []
    return dict(path=path, runs=runs, running=running)


def stats(ws):
    ws = np.asarray(ws, float)
    n = len(ws)
    if n == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), ci=float("nan"))
    sd = float(ws.std(ddof=1)) if n > 1 else float("nan")
    return dict(n=n, mean=float(ws.mean()), sd=sd, ci=t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan"))


def diff(a, b):
    """Welch: mean(a) - mean(b) with a 95% interval; nan interval if either side has fewer than 2 seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean() if len(a) and len(b) else float("nan")
    if len(a) < 2 or len(b) < 2:
        return d, float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / ((va ** 2) / (len(a) - 1) + (vb ** 2) / (len(b) - 1)) if se > 0 else 1000
    return d, t95(df) * se, se


def fmt_ci(m, ci, n=None):
    if m != m:
        return "–"
    s = f"{m:.2f}" if ci != ci else f"{m:.2f} ± {ci:.2f}"
    return s + (f" (n={n})" if n is not None else "")


def verdict(d, ci, what):
    if d != d:
        return f"{what}: not available yet."
    if ci != ci:
        return f"{what}: {d:+.2f}, from single seeds, so no interval: **not a result on its own**."
    lo, hi = d - ci, d + ci
    if lo > 0:
        return f"{what}: **{d:+.2f}** [{lo:+.2f}, {hi:+.2f}]: **positive, the interval excludes 0**."
    if hi < 0:
        return f"{what}: **{d:+.2f}** [{lo:+.2f}, {hi:+.2f}]: **negative, the interval excludes 0**."
    return f"{what}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]: **not distinguishable from 0** with these seeds."


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def band(ax, runs, color, label):
    """Mean evaluation curve over seeds (3-eval moving average) with the min-max range of the seeds shaded."""
    cs = [np.array(r["curve"]) for r in runs if len(r["curve"]) > 3]
    if not cs:
        return
    n = min(len(c) for c in cs)
    m = cs[0][:n, 0] / 1000
    W = np.array([np.convolve(c[:n, 1], np.ones(3) / 3, mode="same") for c in cs])
    ax.plot(m[1:-1], W.mean(0)[1:-1], color=color, lw=2, label=f"{label} ({len(cs)} seed{'s' * (len(cs) > 1)})")
    if len(cs) > 1:
        ax.fill_between(m[1:-1], W.min(0)[1:-1], W.max(0)[1:-1], color=color, alpha=0.15, lw=0)


# ---------------------------------------------------------------- Q1
R = [12, 25, 120]
RN = {12: "12 m", 25: "25 m", 120: "full (120 m)"}


def q1(md, auto, verdicts):
    S = {}
    for z in (0, 1, 2):
        for r in R:
            S[(f"z{z}", r)] = setting(f"q1/z{z}_r{r}")
    for f in ("f131", "f122"):
        for r in R:
            S[(f, r)] = setting(f"q1/{f}_z1_r{r}")
    name = {"z0": "ζ 0 (no formation)", "z1": "ζ 1 (spot + HOLD_SHAPE)", "z2": "ζ 2 (ζ 1 + control penalty)",
            "f131": "ζ 1, our formation 1-3-1", "f122": "ζ 1, our formation 1-2-2"}
    rows = []
    for k in ("z0", "z1", "z2", "f131", "f122"):
        for r in R:
            s = S[(k, r)]
            st = stats([x["W"] for x in s["runs"]])
            per = ", ".join(f"{x['seed']} {x['W']:.2f}" for x in s["runs"]) or "–"
            pend = f" (still training: {', '.join(s['running'])})" if s["running"] else ""
            rows.append([name[k], RN[r], st["n"], fmt_ci(st["mean"], st["ci"]),
                         "–" if st["sd"] != st["sd"] else f"{st['sd']:.2f}", per + pend])
    t_main = table(["setting", "radius", "seeds", "test W, mean ± 95% interval over seeds", "seed sd", "each seed"], rows)

    rows, vlines = [], []
    W = {k: [x["W"] for x in S[k]["runs"]] for k in S}
    for a, b, lab in (("z1", "z0", "B₁(r) = W(ζ 1) − W(ζ 0)"), ("z2", "z0", "B₂(r) = W(ζ 2) − W(ζ 0)"),
                      ("z2", "z1", "W(ζ 2) − W(ζ 1): the control penalty")):
        cells = []
        for r in R:
            d, ci, _ = diff(W[(a, r)], W[(b, r)])
            cells.append("–" if d != d else (f"{d:+.2f}" if ci != ci else f"{d:+.2f} ± {ci:.2f}"))
            vlines.append(verdict(d, ci, f"{lab} at {RN[r]}"))
        rows.append([lab] + cells)
    t_B = table(["difference", *[RN[r] for r in R]], rows)
    # does the benefit grow as r shrinks: B(12) - B(25), B(12) - B(120)
    grow = []
    for z in ("z1", "z2"):
        for r2 in (25, 120):
            ds = [diff(W[(z, r)], W[("z0", r)]) for r in (12, r2)]
            if any(x[0] != x[0] for x in ds):
                continue
            g = ds[0][0] - ds[1][0]
            ci = t95(4) * math.sqrt(ds[0][2] ** 2 + ds[1][2] ** 2) if all(x[2] == x[2] for x in ds) else float("nan")
            grow.append(verdict(g, ci, f"B for ζ {z[1]}: B(12 m) − B({RN[r2]}) (does the benefit grow as r shrinks?)"))
    # best formation of ours per radius (zeta 1): 2-2-1 (z1), 1-3-1, 1-2-2
    rows, best = [], []
    for r in R:
        cand = [("2-2-1", W[("z1", r)]), ("1-3-1", W[("f131", r)]), ("1-2-2", W[("f122", r)])]
        cells = [f"{np.mean(w):.2f} (n={len(w)})" if w else "–" for _, w in cand]
        have = [(f, np.mean(w)) for f, w in cand if w]
        rows.append([RN[r]] + cells + [max(have, key=lambda x: x[1])[0] if len(have) == 3 else "–"])
        if len(have) == 3:
            best.append(f"{RN[r]}: {max(have, key=lambda x: x[1])[0]}")
    t_form = table(["radius", "2-2-1", "1-3-1", "1-2-2", "best"], rows)
    # against each opponent formation, mean over seeds
    FO = ["2-2-1", "2-1-2", "1-3-1", "3-1-1", "1-2-2"]
    rows = []
    for k in ("z0", "z1", "z2", "f131", "f122"):
        for r in R:
            rs = [x for x in S[(k, r)]["runs"] if x["vs"]]
            if rs:
                rows.append([name[k], RN[r], len(rs)] + [f"{np.mean([x['vs'].get(f, np.nan) for x in rs]):.2f}" for f in FO])
    t_vs = table(["setting", "radius", "seeds", *FO[:4], "1-2-2 (never trained)"], rows)

    # figures
    IMG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for j, (z, c) in enumerate((("z0", C2), ("z1", C1), ("z2", C3))):
        xs = np.arange(3) + (j - 1) * 0.18
        means = []
        for x, r in zip(xs, R):
            w = W[(z, r)]
            ax.scatter([x] * len(w), w, s=22, color=c, alpha=0.55, lw=0)
            means.append(np.mean(w) if w else np.nan)
        ax.plot(xs, means, color=c, lw=2, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2,
                label=name[z])
    ax.set_xticks(range(3), [RN[r] for r in R])
    ax.set_ylim(0, 1)
    ax.set_xlabel("vision radius of our players")
    ax.set_ylabel("test W (300 matches)")
    ax.set_title("Q1: test W by radius and structure (big dot = mean, small dots = seeds)", loc="left", color=INK,
                 fontsize=10)
    ax.legend(frameon=False, loc="lower center", fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(IMG / "q1_w_by_radius.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for j, (k, lab, c) in enumerate((("z1", "2-2-1", C1), ("f131", "1-3-1", C2), ("f122", "1-2-2", C3))):
        xs = np.arange(3) + (j - 1) * 0.18
        means = []
        for x, r in zip(xs, R):
            w = W[(k, r)]
            ax.scatter([x] * len(w), w, s=22, color=c, alpha=0.55, lw=0)
            means.append(np.mean(w) if w else np.nan)
        n = "/".join(str(len(W[(k, r)])) for r in R)
        ax.plot(xs, means, color=c, lw=2, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2,
                label=f"our formation {lab} (seeds per radius: {n})")
    ax.set_xticks(range(3), [RN[r] for r in R])
    ax.set_ylim(0, 1)
    ax.set_xlabel("vision radius of our players")
    ax.set_ylabel("test W (300 matches)")
    ax.set_title("Which of our formations works best at each radius (ζ 1; big dot = mean, small = seeds)", loc="left",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(IMG / "q1_formations.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, r in zip(axs, R):
        for z, c in (("z0", C2), ("z1", C1), ("z2", C3)):
            band(ax, S[(z, r)]["runs"], c, f"ζ {z[1]}")
        ax.set_title(f"radius {RN[r]}", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("training matches (thousands)")
        ax.set_ylim(0, 1)
        style(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    axs[0].set_ylabel("evaluation W (50 matches)")
    fig.suptitle("Q1: learning curves, mean over seeds; shaded = range of the seeds", x=0.01, ha="left", color=INK,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG / "q1_learning_curves.png", dpi=150)
    plt.close(fig)

    md += ["## Q1. Can formation structure compensate for limited vision?", "", "### Test W of every setting", "",
           t_main, "", "![Q1 test W by radius](img/final/q1_w_by_radius.png)", "",
           "### Benefit of structure (Welch 95% intervals over seeds)", "", t_B, "", *[f"- {v}" for v in vlines],
           "", "### Does the benefit grow as the radius shrinks?", "", *([f"- {g}" for g in grow] or ["- not available yet"]),
           "", "### Our formation, by radius (ζ 1)", "", t_form, "",
           "The 1-3-1 and 1-2-2 teams have one seed each, so these are single runs, not settled differences.", "",
           "### Against each opponent formation (W over 50 matches each, mean over seeds)", "", t_vs, "",
           "![Q1 learning curves](img/final/q1_learning_curves.png)", ""]
    ans = []
    for a, lab in (("z1", "ζ 1 (formation spot) against ζ 0"), ("z2", "ζ 2 (enforced formation) against ζ 0")):
        parts, sig = [], []
        for r in R:
            d, ci, _ = diff(W[(a, r)], W[("z0", r)])
            if d == d:
                parts.append(f"{d:+.2f}{'' if ci != ci else f' ± {ci:.2f}'} at {RN[r]} ({len(W[(a, r)])} vs {len(W[('z0', r)])} seeds)")
                if ci == ci and (d - ci > 0 or d + ci < 0):
                    sig.append(RN[r])
        if parts:
            ans.append(f"- **{lab}:** " + ", ".join(parts) + ". " + (f"The 95% interval excludes 0 at {', '.join(sig)}."
                       if sig else "No interval excludes 0: no difference can be claimed with these seeds."))
        else:
            ans.append(f"- **{lab}:** not available yet.")
    # pooled over the radii: each radius centred on its own mean (so the radius effect is removed), then Welch
    for a, lab in (("z1", "ζ 1 against ζ 0"), ("z2", "ζ 2 against ζ 0")):
        xa, xb = [], []
        for r in R:
            if len(W[(a, r)]) and len(W[("z0", r)]):
                c = float(np.mean(list(W[(a, r)]) + list(W[("z0", r)])))
                xa += [w - c for w in W[(a, r)]]
                xb += [w - c for w in W[("z0", r)]]
        if len(xa) >= 2 and len(xb) >= 2:
            d, ci, _ = diff(xa, xb)
            ans.append(f"- **Pooled over the radii, {lab}:** {d:+.2f} ± {ci:.2f} ({len(xa)} vs {len(xb)} runs, each radius "
                       f"centred on its own mean). " + ("The interval excludes 0." if d - ci > 0 or d + ci < 0 else
                                                        "The interval includes 0: no effect of structure can be claimed."))
    worst = []
    for r in R:
        if W[("z1", r)] and W[("z0", r)]:
            worst.append(f"{RN[r]} {min(W[('z1', r)]):.2f} vs {min(W[('z0', r)]):.2f}")
    if worst:
        ans.append("- **Worst seed, ζ 1 vs ζ 0:** " + ", ".join(worst) + ".")
    auto["q1_answer"] = "\n".join(ans)
    # conclusion sentence on the enforced formation (zeta 2), written from the numbers (reportQ1 Section 5)
    z2p, z2sig, z2n = [], [], []
    for r in R:
        d, ci, _ = diff(W[("z2", r)], W[("z0", r)])
        if d == d:
            z2p.append(f"{d:+.2f}{'' if ci != ci else f' ± {ci:.2f}'} at {RN[r]}")
            z2n.append(len(W[("z2", r)]))
            if ci == ci and (d - ci > 0 or d + ci < 0):
                z2sig.append(RN[r])
    if not z2p:
        auto["q1_z2_concl"] = "Enforcing the formation (ζ 2): not available yet."
    else:
        few = min(z2n) < 2
        auto["q1_z2_concl"] = (
            f"**Enforcing the formation (ζ 2) on top of the spot** changes test W against ζ 0 by {', '.join(z2p)} "
            f"({min(z2n) if min(z2n) == max(z2n) else f'{min(z2n)}–{max(z2n)}'} ζ 2 seed{'s' if max(z2n) > 1 else ''} against 4 ζ 0 seeds). "
            + ("With single ζ 2 seeds at some radii these are not results on their own." if few else
               (f"The 95% interval excludes 0 at {', '.join(z2sig)}: there the enforced formation makes a difference."
                if z2sig else "No interval excludes 0: the control penalty does not change the result by more than "
                              "the seeds do.")))
    auto["q1_main"] = t_main + "\n\n![Q1 test W by radius](img/final/q1_w_by_radius.png)"
    auto["q1_B"] = t_B + "\n\n" + "\n".join(f"- {v}" for v in vlines + grow)
    auto["q1_form"] = t_form
    # formation order per radius, tested (19 Nov: 1-3-1 / 1-2-2 got more seeds at 12 m)
    fl, fv, sig_any = [], [], []
    for r in R:
        cand = [(f, W[(k, r)]) for f, k in (("2-2-1", "z1"), ("1-3-1", "f131"), ("1-2-2", "f122")) if W[(k, r)]]
        if len(cand) < 3:
            continue
        cand.sort(key=lambda x: -np.mean(x[1]))
        (fb, wb), (fs, ws), (fw, ww) = cand
        parts = []
        for fo, wo in ((fs, ws), (fw, ww)):
            d, ci, _ = diff(wb, wo)
            v = verdict(d, ci, f"Q1 formation at {RN[r]}: W({fb}) − W({fo})")
            fv.append(v)
            n = f"{len(wb)} vs {len(wo)} seed{'s' if max(len(wb), len(wo)) > 1 else ''}"
            if ci != ci:
                parts.append(f"{fo} by {d:+.2f} ({n}, no interval)")
            elif d - ci > 0:
                parts.append(f"**{fo} by {d:+.2f} ± {ci:.2f}** ({n}, interval excludes 0)")
                sig_any.append((r, fb, fo))
            else:
                parts.append(f"{fo} by {d:+.2f} ± {ci:.2f} ({n}, not distinguishable from 0)")
        fl.append(f"- **{RN[r]}:** {fb} is best (W {np.mean(wb):.2f}), ahead of " + " and of ".join(parts) + ".")
    if fl:
        flip = [(r1, a, b) for r1, a, b in sig_any for r2, a2, b2 in sig_any if r2 != r1 and a2 == b and b2 == a]
        if flip:
            r1, a, b = flip[0]
            tail = (f"The order of {a} and {b} reverses between radii with both differences tested: "
                    "**the best formation depends on r.**")
        elif sig_any:
            tail = ("Only these tested differences can be claimed: " +
                    "; ".join(f"at {RN[r]} {a} beats {b}" for r, a, b in sig_any) +
                    ". Where the best formation changes between radii without a tested difference, it is a lead, not a result.")
        else:
            tail = ("No difference between our formations is distinguishable from 0 with these seeds, so a dependence "
                    "of the best formation on r cannot be claimed.")
        auto["q1_form_concl"] = "\n".join(fl) + "\n\n" + tail
    else:
        auto["q1_form_concl"] = "Our formations by radius: not available yet."
    vlines_form = fv
    auto["q1_vs"] = t_vs
    auto["q1_curves"] = "![Q1 learning curves](img/final/q1_learning_curves.png)"
    verdicts += [v for v in vlines if "not available" not in v] + grow + vlines_form
    return S


# ---------------------------------------------------------------- Q2
def points(d):
    return (d > 0) + 0.5 * (d == 0)


def q2_test(path):
    """Per variant: W, goal difference, after-switch goal difference, paired regret vs the oracle (95%)."""
    if not Path(path).exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    BASE = ["none", "oracle", "guess"]
    vs = sorted({r["variant"] for r in rows}, key=lambda v: (BASE.index(v) if v in BASE else 3, v))
    by = {v: {r["seed"]: r for r in rows if r["variant"] == v} for v in vs}
    seeds = sorted(set.intersection(*(set(by[v]) for v in vs)))
    out = {}
    for v in vs:
        sc = np.array([by[v][s]["score"] for s in seeds], float)
        d = sc[:, 0] - sc[:, 1]
        o = dict(n=len(seeds), W=float(points(d).mean()), gd=float(d.mean()))
        if all(by[v][s].get("at_switch") for s in seeds):
            a = sc - np.array([by[v][s]["at_switch"] for s in seeds], float)
            o["gd_after"] = float((a[:, 0] - a[:, 1]).mean())
            o["after_share"] = float(np.mean([(T - by[v][s]["ts"]) / T for s in seeds]))
            b = np.array([by[v][s]["at_switch"] for s in seeds], float)
            o["rate_before"] = float(((b[:, 0] - b[:, 1]) / np.maximum(1, [by[v][s]["ts"] for s in seeds])).mean() * 100)
            o["rate_after"] = float(((a[:, 0] - a[:, 1]) / np.maximum(1, [T - by[v][s]["ts"] for s in seeds])).mean() * 100)
        if "oracle" in by:
            od = np.array([by["oracle"][s]["score"][0] - by["oracle"][s]["score"][1] for s in seeds], float)
            dw = points(od) - points(d)
            o["regret"] = float(dw.mean())
            o["regret_ci"] = float(1.96 * dw.std() / math.sqrt(len(dw)))
        dl = [x for s in seeds for x in by[v][s]["delays"] if x is not None] if v == "guess" else []
        if dl:
            o["delay"] = float(np.median(dl))
            o["false_matches"] = float(np.mean([by[v][s]["false_steps"] > 0 for s in seeds]))
        out[v] = o
    return out


def q2(md, auto, verdicts):
    tr = {}
    for k in ("gru_r25_z0", "mlp_r25_z0", "gru_r12_z1", "mlp_r12_z1", "f122_r25_z0", "f122_r12_z1"):
        tr[k] = setting(f"q2/{k}")
    rows = []
    for k, s in tr.items():
        st = stats([x["W"] for x in s["runs"]])
        rows.append([k, st["n"], fmt_ci(st["mean"], st["ci"]), ", ".join(f"{x['seed']} {x['W']:.2f}" for x in s["runs"])
                     + (f" (still training: {', '.join(s['running'])})" if s["running"] else "")])
    t_train = table(["run", "seeds", "test W (training opponent), mean ± 95%", "each seed"], rows)
    # how many seeds learned at all
    learned = []
    for k in ("gru_r25_z0", "mlp_r25_z0"):
        rs = tr[k]["runs"]
        if rs:
            ok = sum(x["W"] >= 0.5 for x in rs)
            learned.append(f"{k}: {ok} of {len(rs)} seeds reach test W ≥ 0.5 (each: "
                           + ", ".join(f"{x['seed']} {x['W']:.2f}" for x in rs) + ")")

    blocks, rows_sw = [], []
    for R_, label in ((25, "radius 25"), (12, "radius 12")):
        for mode in ("window", "score"):
            for g in ("guess", "guess_anon"):
                res = q2_test(RUNS / f"q2/test/r{R_}_{mode}_{g}.jsonl")
                if res is None:
                    continue
                for v, o in res.items():
                    if g == "guess_anon" and v != "guess":
                        continue
                    rows_sw.append([label, mode, v if g == "guess" else "guess (anonymous)", o["n"], f"{o['W']:.2f}",
                                    f"{o['gd']:+.2f}", f"{o['gd_after']:+.2f}" if "gd_after" in o else "–",
                                    "–" if v == "oracle" or "regret" not in o else f"{o['regret']:+.2f} ± {o['regret_ci']:.2f}",
                                    f"{o['delay']:.0f}" if "delay" in o else "–",
                                    f"{o['false_matches']:.0%}" if "false_matches" in o else "–"])
    t_sw = table(["radius", "switch", "team", "matches", "W", "goal diff", "goal diff after the switch",
                  "regret vs oracle (W)", "guess delay (median steps)", "matches with a false alarm"], rows_sw)
    # seeds of the end-to-end teams on the same switch test (eval 00_e10)
    rows_seed = []
    seed_res = {}
    for mode in ("window", "score"):
        res = q2_test(RUNS / f"q2/test/r25_{mode}_seeds.jsonl")
        if res is None:
            continue
        seed_res[mode] = res
        for v, o in res.items():
            if v in ("none", "guess"):
                continue
            rows_seed.append([mode, v, f"{o['W']:.2f}", f"{o['gd']:+.2f}",
                              f"{o['rate_before']:+.2f} → {o['rate_after']:+.2f}" if "rate_before" in o else "–"])
    t_seed = table(["switch", "team", "W", "goal diff", "goal-diff rate per 100 steps before → after the switch"],
                   rows_seed) if rows_seed else "Not available yet (evaluation `00_e10_q2_test_r25_seeds`)."
    # verdicts
    r25 = q2_test(RUNS / "q2/test/r25_window_guess.jsonl")
    if r25:
        g = r25["guess"]
        verdicts.append(f"Q2 (r 25, single switch): the Bayesian guess detects the switch after a median "
                        f"{g.get('delay', float('nan')):.0f} steps and loses {g['regret']:+.2f} ± {g['regret_ci']:.2f} W "
                        f"against the oracle; keeping the wrong team loses {r25['none']['regret']:+.2f}.")
    if "window" in seed_res:
        res = seed_res["window"]
        gw = [o["W"] for v, o in res.items() if v.startswith("gru")]
        mw = [o["W"] for v, o in res.items() if v.startswith("mlp")]
        d, ci, _ = diff(gw, mw)
        verdicts.append(verdict(d, ci, "Q2 switch test, W(GRU seeds) − W(MLP seeds): does memory help?"))
    # figure: training curves of the end-to-end teams, one line per seed
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, (k, c) in zip(axs, (("gru_r25_z0", C1), ("mlp_r25_z0", C2))):
        for i, x in enumerate(tr[k]["runs"]):
            cv = np.array(x["curve"])
            if len(cv) > 3:
                ax.plot(cv[1:-1, 0] / 1000, np.convolve(cv[:, 1], np.ones(3) / 3, mode="same")[1:-1], color=c,
                        lw=2, alpha=1.0 - 0.3 * i, label=f"{x['seed']} (test W {x['W']:.2f})")
        ax.set_title(k.replace("_r25_z0", ", radius 25, switching opponent"), loc="left", color=INK, fontsize=10)
        ax.set_xlabel("training matches (thousands)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
        style(ax)
    axs[0].set_ylabel("evaluation W (50 matches)")
    fig.tight_layout()
    fig.savefig(IMG / "q2_seeds.png", dpi=150)
    plt.close(fig)

    md += ["## Q2. How quickly can the team infer an opponent formation change?", "",
           "### Training results (W against the training opponent)", "", t_train, "", *[f"- {x}" for x in learned], "",
           "![Q2 seeds](img/final/q2_seeds.png)", "",
           "### Switch test (300 matches, seeds 60000+; the specialist pair is deliberately moderate, see runs/q2/pairs)",
           "", t_sw, "", "### Every seed of the end-to-end teams on the radius-25 switch test", "", t_seed, ""]
    gs = [x["W"] for x in tr["gru_r25_z0"]["runs"]]
    ms = [x["W"] for x in tr["mlp_r25_z0"]["runs"]]
    if gs and ms:
        gk, mk = sum(w >= 0.5 for w in gs), sum(w >= 0.5 for w in ms)
        each = lambda v: ", ".join(f"{w:.2f}" for w in sorted(v, reverse=True))
        if len(ms) < 2:
            how = "Only one MLP seed exists, so it cannot be said yet whether the failures belong to memory."
        elif mk > gk:
            how = ("Without memory more seeds learned, so in these runs **training with memory was less reliable**.")
        elif mk == gk:
            how = ("Without memory just as many seeds failed, so **the failures are not caused by memory**: training "
                   "on the switching opponent is unreliable for both networks.")
        else:
            how = "With memory more seeds learned than without."
        auto["q2_concl"] = (f"**Seeds.** {gk} of {len(gs)} GRU seeds reached test W ≥ 0.5 ({each(gs)}), and {mk} of "
                            f"{len(ms)} MLP seeds ({each(ms)}). {how}")
    else:
        auto["q2_concl"] = "Seeds: not available yet."
    auto["q2_train"] = t_train + "\n\n" + "\n".join(f"- {x}" for x in learned) + "\n\n![Q2 seeds](img/final/q2_seeds.png)"
    auto["q2_switch"] = t_sw
    auto["q2_seeds"] = t_seed


# ---------------------------------------------------------------- Q3
def q3(md, auto, verdicts):
    # mixw*: the mix teams warm-started from the Q2 GRU seed 0 (17 Nov), the ones the play evaluation uses;
    # mix/oracle/mix_ns: the same teams from scratch, stopped early (never reached the full-strength opponent)
    tr = {k: setting(f"q3/{k}") for k in ("mixw_r25_z0", "mixw_told_r25_z0", "mixw_ns_r25_z0", "mix_r25_z0",
                                         "oracle_r25_z0", "mix_ns_r25_z0", "pm2_r25_z0", "pm4_r25_z0", "pm3_r25_z0")}
    rows = [[k, ", ".join(f"{x['seed']} {x['W']:.2f}" for x in s["runs"]) or "–",
             ", ".join(s["running"]) or "–"] for k, s in tr.items()]
    t_train = table(["run", "test W on its training cases (each seed)", "still training"], rows)
    p = RUNS / "q3/play_all_matches.jsonl"
    t_play = "Not available yet (evaluation `00_e7_q3_play`)."
    pair = {}
    if p.exists():
        rows = [json.loads(l) for l in open(p)]
        BASE = ["unaware", "oracle", "adaptive"]
        modes = sorted({r["mode"] for r in rows}, key=lambda m: (BASE.index(m) if m in BASE else 3, m))
        cases = sorted({(r["form"], r["sigma"]) for r in rows}, key=lambda c: (c[1] == 0, c))
        by = {}
        for r in rows:
            by.setdefault((r["form"], r["sigma"], r["mode"]), {})[r["seed"]] = r["gf"] - r["ga"]
        head = ["case (formation:playmaker)"] + modes
        out, reg = [], {m: [] for m in modes}
        for f, s in cases:
            cells = []
            for m in modes:
                d = np.array(list(by.get((f, s, m), {}).values()), float)
                cells.append(f"{points(d).mean():.2f}" if len(d) else "–")
                if len(d) and (f, s, "oracle") in by:
                    common = sorted(set(by[(f, s, m)]) & set(by[(f, s, "oracle")]))
                    reg[m] += [points(np.array([by[(f, s, "oracle")][x]])) - points(np.array([by[(f, s, m)][x]]))
                               for x in common]
            out.append([f"{f}:{s}"] + cells)
        out.append(["**mean regret vs the specialist oracle**"] +
                   [f"{np.mean(reg[m]):+.2f}" if reg[m] and m != "oracle" else "–" for m in modes])
        t_play = table(head, out)
        for a, b in (("mix", "mix_ns"), ("mix_told", "mix"), ("mix", "adaptive"), ("mix", "oracle")):
            if a in modes and b in modes:
                da = []
                for f, s in cases:
                    common = sorted(set(by.get((f, s, a), {})) & set(by.get((f, s, b), {})))
                    da += [points(np.array([by[(f, s, a)][x]]))[0] - points(np.array([by[(f, s, b)][x]]))[0] for x in common]
                if da:
                    m, ci = float(np.mean(da)), float(1.96 * np.std(da) / math.sqrt(len(da)))
                    verdicts.append(verdict(m, ci, f"Q3, W({a}) − W({b}) over the played cases (paired by match)"))
                    pair[(a, b)] = (m, ci)
    md += ["## Q3. Can the team identify and neutralize an opponent playmaker?", "", "### Training", "", t_train, "",
           "### All teams on the same matches (W per case; regret paired by seed)", "", t_play, ""]
    def said(key, bigger, smaller, same):
        if key not in pair:
            return None
        m, ci = pair[key]
        txt = bigger if m - ci > 0 else smaller if m + ci < 0 else same
        return f"{txt} ({m:+.2f} W, 95% interval ±{ci:.2f}, paired over the same matches)"
    sc = {k: [x["W"] for x in tr[k]["runs"]] for k in ("mixw_r25_z0", "mixw_told_r25_z0", "mixw_ns_r25_z0",
                                                        "mix_r25_z0", "oracle_r25_z0", "mix_ns_r25_z0")}
    lines = []
    if all(sc[k] for k in ("mixw_r25_z0", "mixw_told_r25_z0", "mixw_ns_r25_z0")):
        lines.append("**One team for every case needs a start.** From scratch the three mix teams did not learn (test W "
                     + ", ".join(f"{sc[k][0]:.2f}" for k in ("mix_r25_z0", "oracle_r25_z0", "mix_ns_r25_z0") if sc[k])
                     + "); warm-started from a team that could already play, they reached test W "
                     + ", ".join(f"{sc[k][0]:.2f}" for k in ("mixw_r25_z0", "mixw_told_r25_z0", "mixw_ns_r25_z0"))
                     + " (mix, mix_told, mix_ns) on the 24 training cases.")
    s = said(("mix", "adaptive"), "**End-to-end beats identify-then-act:** mix scores higher than the adaptive team",
             "**Identify-then-act beats end-to-end:** mix scores lower than the adaptive team",
             "**End-to-end and identify-then-act are not distinguishable:** mix against the adaptive team")
    if s:
        lines.append(s + ". Note that mix started from a trained team (§3.7).")
    s = said(("mix", "oracle"), "mix even beats the oracle of specialists", "mix stays below the oracle of specialists",
             "mix is level with the oracle of specialists")
    if s:
        lines.append(s + ".")
    s = said(("mix_told", "mix"), "**Being told who the playmaker is helps** the mix team",
             "Being told who the playmaker is makes the mix team worse",
             "**Being told who the playmaker is does not measurably help** the mix team")
    if s:
        lines.append(s + ".")
    s = said(("mix", "mix_ns"), "**The SHADOW action helps:** mix scores higher than the same team without it",
             "The team without SHADOW scores higher than mix",
             "**The SHADOW action makes no measurable difference** (mix against mix_ns)")
    if s:
        lines.append(s + (", in line with §3.5 (the specialists do not use SHADOW on him either)."
                          if "no measurable" in s else "."))
    if len(lines) <= 1:
        lines.append("The comparisons with the adaptive team and the oracle come from `00_e7_q3_play`: not available yet.")
    auto["q3_concl"] = "\n".join(f"- {x}" for x in lines)
    auto["q3_train"] = t_train
    auto["q3_play"] = t_play


# ---------------------------------------------------------------- summary figures for FINAL_REPORT.md
BASE = "#a3a29e"                                           # neutral grey for the do-nothing baseline


def summary_figs():
    # Q2: switch test (window switch, 300 matches each; docs/reportQ2.md §3.2)
    q2 = {"radius 25 m": [0.46, 0.68, 0.74, 0.75], "radius 12 m": [0.25, 0.39, 0.46, 0.53]}
    labs = ["keep the general team", "guess (no player numbers)", "guess", "oracle (told the switch)"]
    cols = [BASE, "#f2a07d", C2, C3]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (t, v) in zip(axs, q2.items()):
        b = ax.bar(range(4), v, color=cols, width=0.7, edgecolor=SURF, linewidth=2)
        for i, x in enumerate(v):
            ax.text(i, x + 0.02, f"{x:.2f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_title(t, loc="left", fontsize=10, color=INK)
        ax.set_ylim(0, 1)
        style(ax)
    axs[0].set_ylabel("W after a mid-match formation switch")
    fig.legend(b, labs, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(IMG / "q2_switch.png", dpi=150)
    plt.close(fig)
    # Q3: per playmaker case (500 matches each; docs/reportQ3.md §3.7 table)
    cases = ["1-3-1, player 2", "3-1-1, player 4", "2-2-1, player 4", "1-3-1, player 4",
             "2-1-2, player 3*", "2-2-1, player 3*", "1-3-1, player 3*"]
    d = {"unaware team": [0.22, 0.36, 0.21, 0.02, 0.37, 0.41, 0.22],
         "identify-then-act": [0.50, 0.56, 0.32, 0.04, 0.37, 0.41, 0.23],
         "oracle (right specialist)": [0.71, 0.65, 0.46, 0.12, 0.37, 0.41, 0.22],
         "one recurrent team (mix)": [0.68, 0.65, 0.74, 0.54, 0.80, 0.81, 0.69]}
    cols = [BASE, C2, C3, C1]
    mk = ["o", "s", "D", "o"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    y = np.arange(len(cases))[::-1]
    for (lab, v), c, m in zip(d.items(), cols, mk):
        ax.scatter(v, y, s=60, color=c, marker=m, label=lab, edgecolor=SURF, linewidth=1.5, zorder=3)
    for yi in y:
        ax.axhline(yi, color=GRID, lw=0.8, zorder=1)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_yticks(y, cases)
    ax.set_xlim(0, 1)
    ax.set_xlabel("W against the team with a playmaker (500 matches per case)")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=4, handletextpad=0.2,
              columnspacing=1.0)
    style(ax)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(IMG / "q3_cases.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- write
def fill(path: Path, auto):
    if not path.exists():
        return []
    s = path.read_text()
    done = []
    for k, v in auto.items():
        pat = re.compile(rf"(<!-- AUTO:{k} -->).*?(<!-- /AUTO:{k} -->)", re.S)
        if pat.search(s):
            s = pat.sub(lambda m: m.group(1) + "\n" + v + "\n" + m.group(2), s)
            done.append(k)
    path.write_text(s)
    return done


def main():
    md, auto, verdicts = [], {}, []
    IMG.mkdir(parents=True, exist_ok=True)
    q1(md, auto, verdicts)
    q2(md, auto, verdicts)
    q3(md, auto, verdicts)
    summary_figs()
    import datetime
    head = ["# Final results (generated)", "",
            f"Generated by `football_adapt/runs/_tools/final_report.py` on "
            f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC from the result files. Do not edit by hand: "
            "run the script again.", "",
            "Test W = wins + draws/2 over 300 new matches per run. Intervals are 95% over training seeds "
            "(Welch for differences). A result from one seed has no interval and is not a finding on its own.", "",
            "## Automatic verdicts", "", *[f"- {v}" for v in verdicts], ""]
    auto["verdicts"] = "\n".join(f"- {v}" for v in verdicts)
    (DOCS / "results_final.md").write_text("\n".join(head + md) + "\n")
    filled = {p.name: fill(p, auto) for p in (DOCS / "reportQ1.md", DOCS / "reportQ2.md", DOCS / "reportQ3.md", DOCS / "FINAL_REPORT.md")}
    print("wrote", DOCS / "results_final.md", "and", sorted(x.name for x in IMG.iterdir()))
    print("filled AUTO blocks:", filled)


if __name__ == "__main__":
    main()
