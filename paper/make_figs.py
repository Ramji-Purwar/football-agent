"""Figures and the number sheet for the paper (paper/main.tex), computed only from the run and evaluation files.

Run from football_adapt/:   python ../paper/make_figs.py
Writes ../paper/media/*.pdf (vector figures) and ../paper/numbers.txt (every number quoted in the paper, with where it
comes from). With PAPER_OUT=dir set, the figures go to dir/media and the number sheet to dir/numbers.txt instead
(reproduce.sh uses this to recompute everything without touching the committed files). Statistics are the ones of runs/_tools/final_report.py, whose loaders are reused here unchanged.
"""
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("fr", "runs/_tools/final_report.py")
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)
plt = fr.plt

PAPER = Path(os.environ.get("PAPER_OUT", "../paper"))
OUT = PAPER / "media"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "legend.fontsize": 8, "axes.titlesize": 9})
C1, C2, C3, BASE, INK, MUTED, GRID = fr.C1, fr.C2, fr.C3, fr.BASE, fr.INK, fr.MUTED, fr.GRID
R, RN = [12, 25, 120], {12: "12 m", 25: "25 m", 120: "full (120 m)"}
NUM = []


def say(key, val, src):
    NUM.append(f"{key:58s} {val}    [{src}]")


def style(ax):
    fr.style(ax)


def W(rs):
    return [x["W"] for x in rs]


# ------------------------------------------------------------------ Q1
Z = {z: {r: fr.setting(f"q1/z{z}_r{r}")["runs"] for r in R} for z in (0, 1, 2)}
F = {f: {r: fr.setting(f"q1/{f}_z1_r{r}")["runs"] for r in R} for f in ("f131", "f122")}
ZN = {0: r"$\zeta=0$ (no formation)", 1: r"$\zeta=1$ (formation as information)", 2: r"$\zeta=2$ (formation enforced)"}
ZC = {0: C2, 1: C1, 2: C3}

for z in (0, 1, 2):
    for r in R:
        st = fr.stats(W(Z[z][r]))
        say(f"Q1 W zeta{z} r{r} (n, mean, ci, sd)", f"{st['n']} {st['mean']:.3f} ±{st['ci']:.3f} sd {st['sd']:.3f} "
            + " ".join(f"{x['seed']}:{x['W']:.2f}" for x in Z[z][r]), f"runs/q1/z{z}_r{r}/s*/result.json")
for z in (1, 2):
    for r in R:
        d, ci, _ = fr.diff(W(Z[z][r]), W(Z[0][r]))
        say(f"Q1 B{z}(r{r}) = W(z{z})-W(z0)", f"{d:+.3f} ±{ci:.3f} [{d-ci:+.3f},{d+ci:+.3f}]", "Welch")
    for r in R:
        d, ci, _ = fr.diff(W(Z[2][r]), W(Z[1][r]))
    # pooled, each radius centred on its own mean (as final_report.py)
for z in (1, 2):
    a, b = [], []
    for r in R:
        mu = np.mean(W(Z[0][r]) + W(Z[z][r]))
        a += [w - mu for w in W(Z[z][r])]
        b += [w - mu for w in W(Z[0][r])]
    d, ci, _ = fr.diff(a, b)
    say(f"Q1 pooled B{z}", f"{d:+.3f} ±{ci:.3f} ({len(a)} vs {len(b)} runs)", "Welch, radius-centred")
for r in R:
    d, ci, _ = fr.diff(W(Z[2][r]), W(Z[1][r]))
    say(f"Q1 W(z2)-W(z1) r{r}", f"{d:+.3f} ±{ci:.3f}", "Welch")
for z in (1, 2):
    for r2 in (25, 120):
        # B(12) - B(r2): difference of two Welch differences, se added in quadrature (as final_report.py)
        d1, _, s1 = fr.diff(W(Z[z][12]), W(Z[0][12]))
        d2, _, s2 = fr.diff(W(Z[z][r2]), W(Z[0][r2]))
        se = math.sqrt(s1 ** 2 + s2 ** 2)
        say(f"Q1 B{z}(12)-B{z}({r2})", f"{d1-d2:+.3f} ±{1.96*se:.3f} (normal approx.)", "derived")
allz0 = [w for r in R for w in W(Z[0][r])]
allz1 = [w for r in R for w in W(Z[1][r])]
say("Q1 worst seed z1 vs z0 per radius", " ".join(f"r{r}:{min(W(Z[1][r])):.2f}/{min(W(Z[0][r])):.2f}" for r in R), "")
for f in ("f131", "f122"):
    for r in R:
        st = fr.stats(W(F[f][r]))
        say(f"Q1 W {f} r{r}", f"n {st['n']} mean {st['mean']:.3f} ci {st['ci']:.3f} " +
            " ".join(f"{x['seed']}:{x['W']:.2f}" for x in F[f][r]), f"runs/q1/{f}_z1_r{r}")
for r in (12, 25):
    for f in ("f131", "f122"):
        d, ci, _ = fr.diff(W(Z[1][r]), W(F[f][r]))
        say(f"Q1 W(2-2-1)-W({f}) r{r}", f"{d:+.3f} ±{ci:.3f}", "Welch")
vsF = ["2-2-1", "2-1-2", "1-3-1", "3-1-1", "1-2-2"]
for z in (0, 1, 2):
    for r in R:
        v = {f: np.mean([x["vs"].get(f, np.nan) for x in Z[z][r]]) for f in vsF}
        say(f"Q1 vs formations z{z} r{r}", " ".join(f"{f}:{v[f]:.2f}" for f in vsF), "vs_formations.txt")
nruns_q1 = sum(len(Z[z][r]) for z in Z for r in R) + sum(len(F[f][r]) for f in F for r in R)
say("Q1 number of runs", nruns_q1, "")

# Figure: (a) W by radius and structure level, (b) structure benefit with 95% CI
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
xs = np.arange(3)
for k, z in enumerate((0, 1, 2)):
    off = (k - 1) * 0.12
    m = [np.mean(W(Z[z][r])) for r in R]
    for i, r in enumerate(R):
        ax.scatter([xs[i] + off] * len(Z[z][r]), W(Z[z][r]), s=10, color=ZC[z], alpha=0.45, lw=0, zorder=2)
    ax.plot(xs + off, m, "-o", color=ZC[z], ms=5, lw=1.6, label=ZN[z], zorder=3)
ax.set_xticks(xs, [RN[r] for r in R])
ax.set_ylim(0, 1.02)
ax.set_xlabel("vision radius $r$ of our players")
ax.set_ylabel("test $W$ (300 matches per run)")
ax.legend(frameon=False, loc="lower center", fontsize=7)
ax.set_title("(a) every run (small dots) and setting means", loc="left")
style(ax)
ax = axs[1]
for k, z in enumerate((1, 2)):
    off = (k - 0.5) * 0.18
    for i, r in enumerate(R):
        d, ci, _ = fr.diff(W(Z[z][r]), W(Z[0][r]))
        ax.errorbar(xs[i] + off, d, yerr=ci, fmt="o", color=ZC[z], ms=5, capsize=3, lw=1.4,
                    label=(r"$B_1=W_1-W_0$" if z == 1 else r"$B_2=W_2-W_0$") if i == 0 else None)
ax.axhline(0, color=MUTED, lw=0.9)
ax.set_xticks(xs, [RN[r] for r in R])
ax.set_xlabel("vision radius $r$")
ax.set_ylabel("structure benefit (95% CI over seeds)")
ax.legend(frameon=False, loc="lower left", fontsize=7)
ax.set_title("(b) benefit of a formation", loc="left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "q1_main.pdf")
plt.close(fig)

# Figure: learning curves
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
for ax, r in zip(axs, R):
    for z in (0, 1, 2):
        fr.band(ax, Z[z][r], ZC[z], rf"$\zeta={z}$")
    ax.set_title(f"$r$ = {RN[r]}", loc="left")
    ax.set_xlabel("training matches (thousands)")
    ax.set_ylim(0, 1)
    style(ax)
axs[0].set_ylabel("evaluation $W$ (50 matches)")
axs[0].legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "q1_curves.pdf")
plt.close(fig)

# Figure: our formations by radius
FS = {"2-2-1": {r: Z[1][r] for r in R}, "1-3-1": F["f131"], "1-2-2": F["f122"]}
FC = {"2-2-1": C1, "1-3-1": C2, "1-2-2": C3}
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for k, (f, d) in enumerate(FS.items()):
    off = (k - 1) * 0.12
    m = []
    for i, r in enumerate(R):
        ax.scatter([xs[i] + off] * len(d[r]), W(d[r]), s=10, color=FC[f], alpha=0.45, lw=0, zorder=2)
        m.append(np.mean(W(d[r])))
    ns = "/".join(str(len(d[r])) for r in R)
    ax.plot(xs + off, m, "-o", color=FC[f], ms=5, lw=1.6, label=f"{f} (seeds {ns})", zorder=3)
ax.set_xticks(xs, [RN[r] for r in R])
ax.set_ylim(0, 1.02)
ax.set_xlabel("vision radius $r$")
ax.set_ylabel("test $W$")
ax.legend(frameon=False, loc="lower right", fontsize=7)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "q1_formations.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Q2
SW = {}
for r in (25, 12):
    for mode in ("window", "score"):
        g = fr.q2_test(fr.RUNS / f"q2/test/r{r}_{mode}_guess.jsonl")
        a = fr.q2_test(fr.RUNS / f"q2/test/r{r}_{mode}_guess_anon.jsonl")
        SW[(r, mode)] = (g, a)
        for v, o in g.items():
            say(f"Q2 r{r} {mode} {v}", " ".join(f"{k}={o[k]:.3f}" if isinstance(o[k], float) else f"{k}={o[k]}"
                                                for k in o), f"runs/q2/test/r{r}_{mode}_guess.jsonl")
        o = a["guess"]
        say(f"Q2 r{r} {mode} guess_anon", " ".join(f"{k}={o[k]:.3f}" if isinstance(o[k], float) else f"{k}={o[k]}"
                                                   for k in o), f"runs/q2/test/r{r}_{mode}_guess_anon.jsonl")
for mode in ("window", "score"):
    res = fr.q2_test(fr.RUNS / f"q2/test/r25_{mode}_seeds.jsonl")
    for v, o in res.items():
        say(f"Q2 seeds r25 {mode} {v}", f"W={o['W']:.3f} gd={o['gd']:+.2f}" +
            (f" rate {o['rate_before']:+.2f}->{o['rate_after']:+.2f}" if "rate_before" in o else ""),
            f"runs/q2/test/r25_{mode}_seeds.jsonl")
Q2T = {k: fr.setting(f"q2/{k}")["runs"] for k in ("gru_r25_z0", "mlp_r25_z0", "gru_r12_z1", "mlp_r12_z1",
                                                     "f122_r25_z0", "f122_r12_z1")}
for k, rs in Q2T.items():
    say(f"Q2 train {k}", " ".join(f"{x['seed']}:{x['W']:.2f}" for x in rs), f"runs/q2/{k}")
d, ci, _ = fr.diff(W(Q2T["gru_r25_z0"]), W(Q2T["mlp_r25_z0"]))
say("Q2 GRU-MLP test W r25 (3 v 3 seeds)", f"{d:+.3f} ±{ci:.3f}", "Welch")

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})
labs = ["keep general team", "guess, no player IDs", "guess (Bayes filter)", "oracle (told)"]
cols = [BASE, "#f2a07d", C2, C3]
for ax, r in zip(axs[:2], (25, 12)):
    g, a = SW[(r, "window")]
    v = [g["none"]["W"], a["guess"]["W"], g["guess"]["W"], g["oracle"]["W"]]
    b = ax.bar(range(4), v, color=cols, width=0.72, edgecolor="white", lw=1.5)
    for i, x in enumerate(v):
        ax.text(i, x + 0.02, f"{x:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_title(f"({'a' if r == 25 else 'b'}) switch test, $r$ = {r} m", loc="left")
    style(ax)
axs[0].set_ylabel("$W$ (300 matches, same seeds)")
axs[1].legend(b, labs, frameon=False, fontsize=6.5, loc="upper right")
ax = axs[2]
for k, c in (("gru_r25_z0", C1), ("mlp_r25_z0", C2)):
    for j, x in enumerate(sorted(Q2T[k], key=lambda x: -x["W"])):
        cv = np.array(x["curve"])
        sm = np.convolve(cv[:, 1], np.ones(3) / 3, mode="same")
        ax.plot(cv[1:-1, 0] / 1000, sm[1:-1], color=c, lw=1.3, alpha=[1, 0.6, 0.35][j],
                label=f"{k[:3].upper()} {x['seed']} (test {x['W']:.2f})")
ax.set_ylim(0, 1)
ax.set_xlabel("training matches (thousands)")
ax.set_ylabel("evaluation $W$")
ax.set_title("(c) end-to-end training", loc="left")
ax.legend(frameon=False, fontsize=6, loc="upper left")
style(ax)
fig.tight_layout()
fig.savefig(OUT / "q2.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Q3
rows = [json.loads(l) for l in open(fr.RUNS / "q3/play_all_matches.jsonl")]
by = {}
for r in rows:
    by.setdefault((r["form"], r["sigma"], r["mode"]), {})[r["seed"]] = r["gf"] - r["ga"]
modes = sorted({r["mode"] for r in rows})
cases = sorted({(r["form"], r["sigma"]) for r in rows})
Wc = {(f, s, m): float(fr.points(np.array(list(by[(f, s, m)].values()), float)).mean())
      for (f, s, m) in by}
for (f, s) in cases:
    say(f"Q3 W case {f}:{s}", " ".join(f"{m}={Wc[(f, s, m)]:.3f}(n{len(by[(f, s, m)])})" for m in modes
                                      if (f, s, m) in Wc), "runs/q3/play_all_matches.jsonl")
pm = [c for c in cases if c[1] != 0]
nopm = [c for c in cases if c[1] == 0]
spec_cases = [("1-3-1", 2), ("3-1-1", 4), ("2-2-1", 4), ("1-3-1", 4)]
for grp, cs in (("7 playmaker cases", pm), ("4 cases with a specialist", spec_cases), ("4 no-playmaker cases", nopm)):
    say(f"Q3 mean W over {grp}", " ".join(f"{m}={np.mean([Wc[(f, s, m)] for f, s in cs]):.3f}" for m in modes), "")


def paired(a, b, cs):
    da = []
    for f, s in cs:
        common = sorted(set(by[(f, s, a)]) & set(by[(f, s, b)]))
        da += [fr.points(np.array([by[(f, s, a)][x]]))[0] - fr.points(np.array([by[(f, s, b)][x]]))[0] for x in common]
    return float(np.mean(da)), float(1.96 * np.std(da) / math.sqrt(len(da))), len(da)


for a, b in (("mix", "adaptive"), ("mix", "oracle"), ("mix_told", "mix"), ("mix_ns", "mix"), ("adaptive", "unaware"),
             ("oracle", "adaptive")):
    for grp, cs in (("all 11", cases), ("7 pm", pm), ("4 spec", spec_cases), ("4 no-pm", nopm)):
        m, ci, n = paired(a, b, cs)
        say(f"Q3 paired W({a})-W({b}) {grp}", f"{m:+.3f} ±{ci:.3f} (n={n})", "paired by seed")
for f, s in spec_cases:
    E = Wc[(f, 0, "unaware")] - Wc[(f, s, "unaware")]
    ka = (Wc[(f, s, "adaptive")] - Wc[(f, s, "unaware")]) / E
    ko = (Wc[(f, s, "oracle")] - Wc[(f, s, "unaware")]) / E
    km = (Wc[(f, s, "mix")] - Wc[(f, s, "unaware")]) / E
    say(f"Q3 E, kappa adaptive/oracle/mix {f}:{s}", f"E={E:.3f} ka={ka:.3f} ko={ko:.3f} kmix={km:.3f}", "")
Q3T = {k: fr.setting(f"q3/{k}")["runs"] for k in ("pm2_r25_z0", "pm4_r25_z0", "pm3_r25_z0", "mix_r25_z0",
                                                     "oracle_r25_z0", "mix_ns_r25_z0", "mixw_r25_z0",
                                                     "mixw_told_r25_z0", "mixw_ns_r25_z0")}
for k, rs in Q3T.items():
    say(f"Q3 train {k}", " ".join(f"{x['seed']}:{x['W']:.2f}" for x in rs), f"runs/q3/{k}")

CN = {2: "player 2", 3: "player 3", 4: "player 4"}
order = spec_cases + [("2-1-2", 3), ("2-2-1", 3), ("1-3-1", 3)]
fig, ax = plt.subplots(figsize=(7.0, 2.9))
y = np.arange(len(order))[::-1]
for m, c, mk, lab in (("unaware", BASE, "o", "unaware (best Q1 team)"), ("adaptive", C2, "s", "identify-then-act"),
                      ("oracle", C3, "D", "oracle (right specialist from kick-off)"),
                      ("mix", C1, "o", "one recurrent team (mix)")):
    ax.scatter([Wc[(f, s, m)] for f, s in order], y, s=34, color=c, marker=mk, label=lab, edgecolor="white",
               lw=1.0, zorder=3)
for (f, s), yi in zip(order, y):
    ax.axhline(yi, color=GRID, lw=0.7, zorder=1)
    ax.scatter([Wc[(f, 0, "unaware")]], [yi], s=34, marker="|", color=INK, zorder=3,
               label="unaware, same formation without playmaker" if (f, s) == order[0] else None)
ax.axvline(0.5, color=MUTED, lw=0.7, ls=":")
ax.set_yticks(y, [f"{f}, {CN[s]}" + ("$^*$" if s == 3 else "") for f, s in order])
ax.set_xlim(0, 1)
ax.set_xlabel("$W$ against an opponent with a playmaker (500 matches per case, same seeds for every team)")
ax.legend(frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, handletextpad=0.2,
          columnspacing=0.9)
fr.style(ax)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT / "q3.pdf")
plt.close(fig)

shutil.copy("../docs/img/viewer.png", OUT / "viewer.png")
(PAPER / "numbers.txt").write_text("Numbers used in paper/main.tex, computed by paper/make_figs.py from the run files\n\n"
                                        + "\n".join(NUM) + "\n")
print("\n".join(NUM))
