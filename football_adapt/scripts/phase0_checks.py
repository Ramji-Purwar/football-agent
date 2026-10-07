"""Phase 0 checks (opponent.md Section 14, game.md Section 14).

Run from the project root:   python -m scripts.phase0_checks --episodes 20
Use --T 400 for a faster run. Every check prints PASS/FAIL against a threshold you can change below.
The thresholds are placeholders: decide them with your team.
"""
from __future__ import annotations
import argparse
import itertools
import numpy as np

from football import FootballEnv, load_config
from agents import RandomAgent


def score_of(final):
    return 1.0 if final[0] > final[1] else (0.5 if final[0] == final[1] else 0.0)


def play(env, agent=None, seed=0, **reset_kw):
    obs, _ = env.reset(seed=seed, **reset_kw)
    done = False
    info = {}
    while not done:
        acts = agent.act(obs) if agent is not None else None
        obs, _, done, info = env.step(acts)
    return info["final_score"]


def run_scripted(cfg, episodes, **reset_kw):
    """Scripted team 0 vs scripted team 1. Returns mean score of team 0."""
    env = FootballEnv(cfg, scripted_ours=True)
    return float(np.mean([score_of(play(env, None, seed=s, **reset_kw)) for s in range(episodes)]))


def check_random_loses(cfg, n):
    env = FootballEnv(cfg)
    ag = RandomAgent(0)
    w = float(np.mean([score_of(play(env, ag, seed=s)) for s in range(n)]))
    return w, w <= 0.2


def check_balance(cfg, n):
    w = run_scripted(cfg, n)
    return w, abs(w - 0.5) <= 0.15


def check_vision_matters(cfg, n):
    big = cfg.to_dict()["vision"]["radius"]
    w = run_scripted(cfg, n, radius=10.0, radius_opp=big)      # small radius (ours) vs large (theirs)
    return w, w < 0.4


def check_playmaker(cfg, n):
    w_with = 1.0 - run_scripted(cfg, n, sigma=3)               # score of team 1 (with playmaker) vs team 0 (without)
    return w_with, w_with > 0.55


def check_readability(cfg, n):
    env = FootballEnv(cfg)
    ag = RandomAgent(0)
    excursions = []
    for s in range(n):
        play(env, ag, seed=s)
        excursions.append(np.mean(env.opp_anchor_dist))
    exc = float(np.mean(excursions))
    names = env.book.names
    seps = []
    for a, b in itertools.combinations(names, 2):
        A, B = env.book.anchors_team_frame(a), env.book.anchors_team_frame(b)
        seps.append(np.mean(np.linalg.norm(A - B, axis=1)))
    sep = float(np.mean(seps))
    return (exc, sep), exc < 0.5 * sep


def check_determinism(cfg):
    env = FootballEnv(cfg, scripted_ours=True)
    a = play(env, None, seed=123)
    log_a = list(env.event_log)
    b = play(env, None, seed=123)
    return (a, b), a == b and log_a == env.event_log


def formation_matrix(cfg, n):
    env_names = FootballEnv(cfg).book.names
    M = np.zeros((len(env_names), len(env_names)))
    for i, f in enumerate(env_names):
        for j, g in enumerate(env_names):
            env = FootballEnv(cfg, scripted_ours=True)
            M[i, j] = np.mean([score_of(play(env, None, seed=s, our_formation=f, opp_formation=g)) for s in range(n)])
    return env_names, M


def style_matrix(cfg, n):
    names = list(cfg.styles.presets.to_dict())
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            env = FootballEnv(cfg, scripted_ours=True)
            M[i, j] = np.mean([score_of(play(env, None, seed=s, our_style=a, opp_style=b)) for s in range(n)])
    return names, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--matrix", action="store_true", help="also compute the formation win matrix")
    ap.add_argument("--styles", action="store_true", help="also compute the play-style win matrix")
    args = ap.parse_args()
    over = {"time.T": args.T} if args.T else {}
    cfg = load_config(overrides=over)
    n = args.episodes

    rows = []
    v, ok = check_random_loses(cfg, n);       rows.append(("A random loses (score of random, want <= 0.2)", f"{v:.2f}", ok))
    v, ok = check_balance(cfg, n);            rows.append(("B heuristic vs heuristic balanced (want 0.35-0.65)", f"{v:.2f}", ok))
    v, ok = check_vision_matters(cfg, n);     rows.append(("C small radius loses to large (score, want < 0.4)", f"{v:.2f}", ok))
    v, ok = check_playmaker(cfg, n);          rows.append(("D playmaker team beats same team (score, want > 0.55)", f"{v:.2f}", ok))
    v, ok = check_readability(cfg, max(3, n // 4)); rows.append(("E opponent excursion vs formation separation (want exc < 0.5 sep)", f"exc={v[0]:.1f} sep={v[1]:.1f}", ok))
    v, ok = check_determinism(cfg);           rows.append(("F same seed, same match", str(v[0]), ok))
    print("\nPhase 0 checks (placeholder constants; episodes per check = %d, T = %d)" % (n, cfg.time.T))
    for name, val, ok in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")
    if args.matrix:
        names, M = formation_matrix(cfg, max(3, n // 2))
        print("\nFormation win matrix (row = team 0 formation, col = team 1 formation, score of team 0):")
        print("          " + " ".join(f"{x:>6}" for x in names))
        for nm, row in zip(names, M):
            print(f"  {nm:>6}  " + " ".join(f"{x:6.2f}" for x in row))
    if args.styles:
        names, M = style_matrix(cfg, max(3, n // 2))
        print("\nPlay-style win matrix (row = team 0 style, col = team 1 style, score of team 0):")
        print("              " + " ".join(f"{x:>10}" for x in names))
        for nm, row in zip(names, M):
            print(f"  {nm:>10}  " + " ".join(f"{x:10.2f}" for x in row))


if __name__ == "__main__":
    main()
