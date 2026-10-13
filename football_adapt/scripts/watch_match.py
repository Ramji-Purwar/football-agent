"""Render a match to a GIF so the team can SEE what happens.

   python -m scripts.watch_match --mode scripted --out match.gif
   python -m scripts.watch_match --mode random --radius 15 --sigma 3 --opp-mode SCHEDULED
   python -m scripts.watch_match --ckpt runs/stage2_easy_start/ckpt.pt --T 1200 --out trained.gif
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from football import FootballEnv, load_config
from agents import RandomAgent


def draw(ax, env, show_vision=True):
    ax.clear()
    L, W = env.L, env.W
    ax.set_xlim(-3, L + 3); ax.set_ylim(-3, W + 3); ax.set_aspect("equal"); ax.axis("off")
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
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--sigma", type=int, default=0)
    ap.add_argument("--opp-mode", default="NONE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="match.gif")
    ap.add_argument("--ckpt", default=None,
                    help="a trained team (runs/<name>/ckpt.pt) plays blue, with the vision and formation it was trained with")
    ap.add_argument("--opp-eps", type=float, default=None,
                    help="red players act at random this often, 0..1 (the weakened opponent of the easy start)")
    a = ap.parse_args()
    ag, trained, trained_cfg = (RandomAgent(a.seed) if a.mode == "random" else None), {}, {}
    if a.ckpt:
        from agents.mappo import MAPPOAgent        # needs torch, so it is imported only when asked for
        ag = MAPPOAgent.load(a.ckpt, seed=a.seed)
        t = ag.train_args
        trained = dict(zeta=t.get("zeta"), radius=None if a.radius else t.get("radius"), our_formation=t.get("our_formation"))
        trained_cfg = ag.overrides                 # the config values it was trained with (the trainer's --set)
    cfg = load_config(overrides={**trained_cfg, "time.T": a.T, "opponent.switching.mode": a.opp_mode,
                                 **({"vision.radius": a.radius} if a.radius else {}),
                                 **({"opponent.randomness.eps_opp": a.opp_eps} if a.opp_eps is not None else {})})
    env = FootballEnv(cfg, scripted_ours=(ag is None))
    obs, _ = env.reset(seed=a.seed, sigma=a.sigma, **trained)
    frames = []
    done = False
    while not done:
        if env.t % a.stride == 0:
            frames.append((env.pos.copy(), env.ball_pos.copy(), env.holder, env.anchors.copy(),
                           env.t, tuple(env.score), tuple(env.formation)))
        obs, _, done, _ = env.step(ag.act(obs) if ag else None)
    fig, ax = plt.subplots(figsize=(8, 5))
    def update(k):
        pos, ball, holder, anchors, t, score, form = frames[k]
        env.pos, env.ball_pos, env.holder, env.anchors, env.t, env.score, env.formation = \
            pos, ball, holder, anchors, t, list(score), list(form)
        draw(ax, env)
    FuncAnimation(fig, update, frames=len(frames)).save(a.out, writer=PillowWriter(fps=10))
    print("saved", a.out, "frames:", len(frames))


if __name__ == "__main__":
    main()
