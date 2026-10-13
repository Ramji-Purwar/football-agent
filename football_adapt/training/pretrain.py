"""The rule-based start: an actor and a critic that begin as a copy of the hand-written team in training/rules.py.

Run from football_adapt/:

   python -m training.pretrain --out runs/warm.pt --set actions.context_mask=true
   python -m training.train --name warm --workers --init runs/warm.pt --critic-warmup 10 --entropy-coef 0.001 \\
          --shaping-anneal-end 1000000000 --eval-matches 50 --set actions.context_mask=true

A policy that starts from random play learns what to do with the ball and little else: after 1M steps its four
players off the ball still run about as a random team's do (training.md, Section 7). So here the rules play
--matches matches against the scripted team, the actor is trained to make the same choices from the same
observations, and the critic to predict the return of that play. train.py --init carries on from there with PPO.

Give the same --set values here and to train.py: the copy fits the game it was made in. The file can also be
watched like a trained team (scripts.pygame_render --ckpt runs/warm.pt). Memoryless actor only (--actor mlp).
"""
from __future__ import annotations
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from football import FootballEnv, load_config
from agents.mappo import Actor, Critic, masked_dist
from training.rules import rule_actions
from training.train import set_overrides

SEED0 = 50_000                     # the matches use seeds 50000, 50001, ... (not the evaluation seeds of train.py)
OPP_EPS = (0.0, 0.0, 0.1, 0.3)     # opponent randomness, match by match: half at full strength, the rest a little
                                   # weaker, so that the rules are also seen with the ball in the opponent's half
GAMMA = 0.99                       # discount of the critic's targets (--gamma of train.py)


def play(job):
    """One match by the rules. Returns every step's observations, masks, the rules' choices, global states and
    discounted returns, and the final score."""
    overrides, seed, eps, noise = job
    env = FootballEnv(load_config(overrides=overrides))
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    env.controllers[1].eps_opp = eps
    O, M, A, S, R = [], [], [], [], []
    done = False
    while not done:
        want = rule_actions(env)
        O.append(obs["vec"].copy()); M.append(obs["mask"].copy()); S.append(obs["state"].copy())
        A.append([want[j] for j in range(5)])
        do = dict(want)
        for j in range(5):                         # now and then a player does something else, so that the rules are
            if rng.random() < noise:               # also seen (and copied) in the situations a learner's mistakes lead to
                do[j] = int(rng.choice(np.flatnonzero(obs["mask"][j])))
        obs, r, done, _ = env.step(do)
        R.append(r)
    G, acc = np.zeros(len(R), np.float32), 0.0
    for t in reversed(range(len(R))):
        acc = R[t] + GAMMA * acc
        G[t] = acc
    return np.array(O, np.float32), np.array(M), np.array(A), np.array(S, np.float32), G, env.score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="file to write, e.g. runs/warm.pt (give it to train.py --init)")
    ap.add_argument("--matches", type=int, default=96, help="matches played by the rules")
    ap.add_argument("--noise", type=float, default=0.15, help="share of the choices swapped for a random valid one")
    ap.add_argument("--epochs", type=int, default=12, help="passes over the recorded steps")
    ap.add_argument("--hidden", type=int, default=256, help="as --hidden of train.py")
    ap.add_argument("--procs", type=int, default=6, help="processes that play the matches")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="config value, as in train.py: give both the same ones (can be repeated)")
    args = ap.parse_args()
    overrides = set_overrides(args.set)
    if load_config(overrides=overrides).structure.zeta == 0:
        raise SystemExit("the rules hold the team's shape with HOLD_SHAPE, which structure level 0 does not have")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    jobs = [(overrides, SEED0 + i, OPP_EPS[i % len(OPP_EPS)], args.noise) for i in range(args.matches)]
    with Pool(args.procs) as pool:
        res = pool.map(play, jobs)
    O = torch.as_tensor(np.concatenate([r[0] for r in res]).reshape(-1, res[0][0].shape[-1]))     # one row per player
    M = torch.as_tensor(np.concatenate([r[1] for r in res]).reshape(-1, res[0][1].shape[-1]))
    A = torch.as_tensor(np.concatenate([r[2] for r in res]).reshape(-1))
    S = torch.as_tensor(np.concatenate([r[3] for r in res]))                                     # one row per step
    G = torch.as_tensor(np.concatenate([r[4] for r in res]))
    free = M.sum(1) > 1                            # the steps on which the player had a choice
    print(f"{args.matches} matches by the rules ({args.noise:.0%} of the choices swapped for a random one): goals "
          f"{np.mean([r[5][0] for r in res]):.2f} - {np.mean([r[5][1] for r in res]):.2f}, {int(free.sum())} choices to copy",
          flush=True)

    torch.manual_seed(0)
    torch.set_num_threads(4)
    actor = Actor(O.shape[1], M.shape[1], args.hidden, False)
    critic = Critic(S.shape[1], args.hidden)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    idx = torch.nonzero(free)[:, 0]
    perm = idx[torch.randperm(len(idx))]
    test, fit = perm[:len(idx) // 10], perm[len(idx) // 10:]      # a tenth is kept aside to measure the copy on
    h0 = torch.zeros(1, 1)                                         # the MLP's dummy hidden state
    for ep in range(args.epochs):
        p = fit[torch.randperm(len(fit))]
        for i in range(0, len(p), 1024):
            b = p[i:i + 1024]
            logits, _ = actor(O[b], h0, None)
            loss = -masked_dist(logits, M[b]).log_prob(A[b]).mean()
            opt_a.zero_grad(); loss.backward(); opt_a.step()
        p = torch.randperm(len(S))
        for i in range(0, len(p), 1024):
            b = p[i:i + 1024]
            v_loss = F.mse_loss(critic(S[b]), G[b])
            opt_c.zero_grad(); v_loss.backward(); opt_c.step()
        with torch.no_grad():
            logits, _ = actor(O[test], h0, None)
            d = masked_dist(logits, M[test])
            same = (d.probs.argmax(-1) == A[test]).float().mean().item()
            prob = d.log_prob(A[test]).exp().mean().item()
        print(f"  epoch {ep + 1:2d}: same choice as the rules on steps kept aside {same:.1%} (mean probability given "
              f"to it {prob:.2f}), critic loss {v_loss.item():.4f}", flush=True)

    # the same keys as a checkpoint of train.py, so MAPPOAgent.load can play it
    torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), obs_dim=O.shape[1], n_actions=M.shape[1],
                    args=dict(hidden=args.hidden, actor="mlp"), overrides=overrides, step=0), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
