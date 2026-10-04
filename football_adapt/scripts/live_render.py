"""Live rendering of a match using matplotlib.

Run from project root:
    python -m scripts.live_render --mode scripted
    python -m scripts.live_render --mode random --radius 15 --sigma 3 --opp-mode SCHEDULED
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import time

from football import FootballEnv, load_config
from agents import RandomAgent

def draw(ax, env, show_vision=True):
    ax.clear()
    L, W = env.L, env.W
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(-3, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), L, W, fc="#e8f5e9", ec="#333", lw=1.5))
    ax.plot([L / 2, L / 2], [0, W], color="#333", lw=0.8)
    gw = env.cfg.pitch.goal_width
    for x0 in (-1.5, L):
        ax.add_patch(Rectangle((x0, W / 2 - gw / 2), 1.5, gw, fc="#333"))
    cols = {0: "#2b6cb0", 1: "#c53030"}
    for team in (0, 1):
        for i in range(5):
            p = env.pos[team][i]
            a = env.anchors[team][i]
            ax.plot([a[0]], [a[1]], "x", color=cols[team], alpha=0.3, ms=5)
            if show_vision and np.isfinite(env.vision_radius(team, i)) and i == 0 and team == 0:
                ax.add_patch(Circle(p, env.vision_radius(team, i), fill=False, ec=cols[team], alpha=0.25, ls="--"))
            lw = 2.5 if env.holder == (team, i) else 0.8
            ec = "gold" if (team == 1 and env.sigma_idx == i) else "white"
            ax.add_patch(Circle(p, 1.4, fc=cols[team], ec=ec, lw=lw if ec == "white" else 3, zorder=3))
            ax.text(p[0], p[1], str(i + 1), color="white", ha="center", va="center", fontsize=7, zorder=4)
    ax.add_patch(Circle(env.ball_pos, 0.8, fc="black", zorder=5))
    ax.set_title(f"t={env.t}  score {env.score[0]}-{env.score[1]}   ours={env.formation[0]}  opp={env.formation[1]}"
                 f"{'  (gold ring = playmaker)' if env.sigma else ''}", fontsize=9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "scripted"], default="scripted")
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--sigma", type=int, default=0)
    ap.add_argument("--opp-mode", default="NONE")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(overrides={"time.T": a.T, "opponent.switching.mode": a.opp_mode,
                                 **({"vision.radius": a.radius} if a.radius else {})})
    env = FootballEnv(cfg, scripted_ours=(a.mode == "scripted"))
    obs, _ = env.reset(seed=a.seed, sigma=a.sigma)
    ag = RandomAgent(a.seed)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.canvas.manager.set_window_title("Football Game Live")

    done = False
    while not done:
        draw(ax, env)
        plt.draw()
        plt.pause(1.0 / a.fps)
        
        obs, _, done, _ = env.step(ag.act(obs) if a.mode == "random" else None)
        
        if not plt.fignum_exists(fig.number):
            print("Window closed by user.")
            break

    plt.ioff()
    print("Match finished!")
    if plt.fignum_exists(fig.number):
        plt.show()

if __name__ == "__main__":
    main()
