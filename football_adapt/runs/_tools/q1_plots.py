"""Figures for docs/reportQ1.md from runs/q1/summary.json (q1_summary.py) and runs/q1/style.jsonl (style_stats.py).
Run from football_adapt/:  python runs/_tools/q1_plots.py  -> ../docs/img/q1/*.png"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = json.load(open("runs/q1/summary.json"))
OUT = Path("../docs/img/q1")
OUT.mkdir(parents=True, exist_ok=True)
COL = {1: "#2a78d6", 0: "#eb6834"}                  # zeta 1 blue, zeta 0 orange (validated pair)
LAB = {1: "zeta 1 (formation spot + HOLD_SHAPE)", 0: "zeta 0 (no formation)"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
R = [12, 25, 120]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 1. test W against radius, mean over seeds, bar = range over seeds
fig, ax = plt.subplots(figsize=(6.4, 4))
for z in (1, 0):
    rows = [S[f"z{z}_r{r}"] for r in R]
    m = np.array([d["W"] for d in rows])
    spread = np.array([d["W_spread"] for d in rows])
    x = np.arange(3) + (-0.06 if z == 1 else 0.06)
    ax.errorbar(x, m, yerr=spread / 2, color=COL[z], lw=2, marker="o", ms=8, capsize=4,
                markeredgecolor=SURF, markeredgewidth=2)
    for xi, mi, d in zip(x, m, rows):
        ax.annotate(f"{mi:.2f}", (xi, mi), textcoords="offset points", xytext=(10, -4) if z == 0 else (-32, 6),
                    color=INK, fontsize=9)
    ax.plot([], [], color=COL[z], lw=2, marker="o", label=LAB[z])
ax.legend(frameon=False, loc="lower center")
ax.set_xticks(range(3), ["12 m", "25 m", "full (120 m)"])
ax.set_xlabel("vision radius of our players")
ax.set_xlim(-0.45, 2.4)
ax.set_ylabel("test W (300 matches), mean over seeds")
ax.set_ylim(0, 1)
ax.set_title("Q1: win score by vision radius and structure", color=INK, loc="left")
style(ax)
fig.text(0.01, 0.01, "Bars: seed spread (half the range above and below the mean). zeta 0 at 12 m and 120 m: 4 seeds, "
         "others 2.", color=MUTED, fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "w_by_radius.png", dpi=150)

# 2. learning curves, one panel per radius, mean W over seeds (3-evaluation moving average)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, r in zip(axs, R):
    for z in (1, 0):
        d = S[f"z{z}_r{r}"]
        m = np.array(d["curve_matches"]) / 1000
        w = np.convolve(d["curve_W"], np.ones(3) / 3, mode="same")
        ax.plot(m[1:-1], w[1:-1], color=COL[z], lw=2, label=f"zeta {z}")
    ax.set_title(f"radius {r if r != 120 else 'full (120)'} m", color=INK, loc="left", fontsize=10)
    ax.set_xlabel("training matches (thousands)")
    ax.set_ylim(0, 1)
    style(ax)
axs[0].set_ylabel("evaluation W, mean over seeds")
axs[0].legend(frameon=False, loc="upper left")
fig.suptitle("Q1: learning curves (evaluation every 1,000 matches, 50 matches each, smoothed over 3)", color=INK,
             x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "learning_curves.png", dpi=150)

print("written", sorted(p.name for p in OUT.iterdir()))
