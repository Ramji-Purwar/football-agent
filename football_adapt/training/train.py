"""Train one parameter-shared MAPPO policy against the scripted team (training.md). CPU only.

Run from football_adapt/:

   python -m training.train --name sanity --T 300 --radius 100 --total-steps 600000 --workers --opp-eps-start 0.5
   python -m training.train --name easy --total-steps 2000000 --workers --opp-eps-start 0.5          # Stage 2
   python -m training.train --actor gru --opp-formations 2-2-1,1-2-2,1-3-1 --switching RANDOM

--opp-eps-start 0.5 is the easy start: the opponent's players act at random half of the time, and the opponent is
made stronger step by step as our team gets ahead of it (see EPS_STEP below). The evaluations are always against
the opponent at full strength. Against the full-strength opponent from step 0 a new policy never gets near the goal.

--matches N trains by whole matches: every update is one full match of each game, from kick-off to the final
whistle. --eval-every sets the matches between evaluations. --recipe loads a file of config values: the study of
docs/questions.md uses configs/study.yaml, e.g. its first Q1 setting:

   python -m training.train --name v1 --recipe configs/study.yaml --zeta 1 --radius 25 \
          --opp-formations 2-2-1,2-1-2,1-3-1,3-1-1 --opp-eps-start 0.5 --workers --n-envs 48 --batch-size 1024 \
          --matches 20000 --eval-every 1000 --eval-matches 50

Everything goes to runs/<name>/: config.json, log.jsonl (one line per update, per evaluation and for the test),
ckpt.pt (to resume), ckpts/ and, when the run is finished, result.json. Every evaluation (seeds 10000, 10001, ...)
keeps its checkpoint in ckpts/. At the end the best of them plays --test-matches matches on seeds no evaluation used
(50000, ...), and that is the run's result. Running the SAME command again resumes from ckpt.pt, so a run survives
a node that expires; with a larger --total-steps it carries a finished run on. Our formation, radius and structure
level are fixed for a run; the opponent's formation, style and playmaker are drawn fresh at every kick-off from the
lists given.

Two ways to use the cores:
  * one fast run:        --workers gives each of the --n-envs games its own process
  * many runs at once:   leave --workers off (one process per run) and start them with training/sweep.py
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from football import FootballEnv, load_config
from agents.mappo import Actor, Critic, MAPPOAgent, masked_dist

EVAL_SEED0 = 10_000        # evaluation matches use seeds 10000, 10001, ... for every run (paired comparisons)
TEST_SEED0 = 50_000        # the test of a run's best checkpoint: seeds no evaluation used, so the pick does not flatter it

# The easy start (--opp-eps-start): each opponent player acts at random with probability eps (opponent.md,
# opponent.randomness.eps_opp). After every EPS_WINDOW training matches our win score (wins + draws / 2) is looked
# at: EPS_HARDER or more and eps drops by EPS_STEP (down to the config value), EPS_EASIER or less and it goes back
# up by EPS_STEP (up to the start value). In between it stays. So the opponent is always one our team can just beat.
EPS_STEP, EPS_WINDOW, EPS_HARDER, EPS_EASIER = 0.05, 50, 0.6, 0.4


# ---------------------------------------------------------------------------------------------- the games
def sample_reset(scen: dict, rng) -> dict:
    """reset() arguments for one match. Fixed things are passed through, lists are drawn from."""
    ours = scen.get("our_formations") or [scen["our_formation"]]
    out = dict(our_formation=ours[rng.integers(len(ours))], radius=scen["radius"], zeta=scen["zeta"],
               opp_formation=scen["opp_formations"][rng.integers(len(scen["opp_formations"]))],
               opp_style=scen["opp_styles"][rng.integers(len(scen["opp_styles"]))],
               sigma=int(scen["sigmas"][rng.integers(len(scen["sigmas"]))]))
    if scen.get("cases"):                      # formation and playmaker drawn together
        out["opp_formation"], out["sigma"] = scen["cases"][rng.integers(len(scen["cases"]))]
    return out


def summary(env, shaping: float) -> dict:
    """What we keep from a finished match."""
    s, T = env.stats, env.t
    gf, ga = env.score
    return {"win": float(gf > ga), "draw": float(gf == ga), "goal_diff": gf - ga, "goals_for": gf,
            "shaping_return": shaping,
            "passes": s["passes"][0], "pass_completion": s["passes_completed"][0] / max(1, s["passes"][0]),
            "passes_completed": s["passes_completed"][0],
            "shots": s["shots"][0], "chance_per_shot": s["shot_chance"][0] / max(1, s["shots"][0]),
            "poor_shots": s["poor_shots"][0], "missed_shots": s["missed_shots"][0],
            "through_balls": s["through_balls"][0], "turnovers": s["turnovers"][0],
            "possession": s["possession_steps"][0] / T,
            "ball_own_third": s["ball_thirds"][0] / T, "ball_mid_third": s["ball_thirds"][1] / T,
            "ball_their_third": s["ball_thirds"][2] / T,
            "invalid_action_rate": s["invalid_actions"][0] / (5 * T),
            "opp_anchor_dist": float(np.mean(env.opp_anchor_dist))}


class Slot:
    """One game that restarts itself: when a match ends, the next one is kicked off straight away."""

    def __init__(self, overrides: dict, scen: dict, seed: int, idx: int):
        self.env = FootballEnv(load_config(overrides=overrides))
        self.scen = scen
        self.rng = np.random.default_rng([seed, idx])
        self.w, self.opp_eps = 1.0, None

    def set(self, w: float, opp_eps):
        """The two things the trainer changes during a run: the shaping weight and the opponent's randomness
        (None = as configured)."""
        self.w, self.opp_eps = w, opp_eps
        if not self.env.done:
            self._apply()

    def _apply(self):                              # reset() puts the config values back, so this runs every match
        self.env.set_shaping_weight(self.w)
        if self.opp_eps is not None:
            self.env.controllers[1].eps_opp = self.opp_eps

    def reset(self):
        obs, _ = self.env.reset(seed=int(self.rng.integers(2 ** 31)), **sample_reset(self.scen, self.rng))
        self._apply()
        self.shaping = 0.0
        return obs

    def step(self, actions):
        obs, _, done, info = self.env.step({i: int(a) for i, a in enumerate(actions)})
        r = info["reward_players"]                # (5,): the team reward, plus each player's own (reward.per_player)
        self.shaping += info["reward_parts"]["shaping"]
        ep = None
        if done:
            ep = summary(self.env, self.shaping)
            obs = self.reset()
        return obs, r, done, ep


class Collector:
    """A few games played with the current policy. Keeps the games and the GRU hidden states between rollouts."""

    def __init__(self, slots, dims, hidden: int, recurrent: bool, per_player: bool = False):
        self.slots = slots
        self.obs_dim, self.n_act, self.state_dim = dims
        self.per_player = per_player
        self.actor = Actor(self.obs_dim, self.n_act, hidden, recurrent)
        self.critic = Critic(self.state_dim + (5 if per_player else 0), hidden)
        self.obs = None

    @staticmethod
    def _stack(obs_list):
        return {k: np.stack([o[k] for o in obs_list]) for k in ("vec", "mask", "state")}

    @torch.no_grad()
    def rollout(self, weights, R: int, knobs):
        """R steps of every game. Returns the buffer (R, E, ...), the values (R + 1, E, 5) and the finished matches.
        knobs = (shaping weight, opponent randomness), see Slot.set."""
        self.actor.load_state_dict(weights[0])
        self.critic.load_state_dict(weights[1])
        E, H, D, A = len(self.slots), self.actor.h_dim, self.obs_dim, self.n_act
        for s in self.slots:
            s.set(*knobs)
        if self.obs is None:
            self.obs = self._stack([s.reset() for s in self.slots])
            self.h = torch.zeros(E * 5, H)
            self.reset = np.ones(E, np.float32)                  # 1 = this step starts a new match
        buf = dict(obs=np.zeros((R, E, 5, D), np.float32), mask=np.zeros((R, E, 5, A), bool),
                   act=np.zeros((R, E, 5), np.int64), logp=np.zeros((R, E, 5), np.float32),
                   h=np.zeros((R, E, 5, H), np.float32), reset=np.zeros((R, E), np.float32),
                   state=np.zeros((R, E, self.state_dim), np.float32), rew=np.zeros((R, E, 5), np.float32),
                   done=np.zeros((R, E), np.float32))
        val = np.zeros((R + 1, E, 5), np.float32)
        episodes = []
        for t in range(R):
            obs = self.obs
            buf["obs"][t], buf["mask"][t], buf["state"][t] = obs["vec"], obs["mask"], obs["state"]
            buf["h"][t], buf["reset"][t] = self.h.numpy().reshape(E, 5, H), self.reset
            logits, self.h = self.actor(torch.as_tensor(obs["vec"]).reshape(E * 5, 1, D), self.h,
                                        torch.as_tensor(np.repeat(self.reset, 5))[:, None])
            dist = masked_dist(logits[:, 0], torch.as_tensor(obs["mask"]).reshape(E * 5, A))
            a = dist.sample()
            val[t] = values(self.critic, obs["state"], self.per_player)
            buf["act"][t] = a.numpy().reshape(E, 5)
            buf["logp"][t] = dist.log_prob(a).numpy().reshape(E, 5)
            out = [s.step(x) for s, x in zip(self.slots, buf["act"][t])]
            self.obs = self._stack([o[0] for o in out])
            buf["rew"][t] = [o[1] for o in out]
            buf["done"][t] = [o[2] for o in out]
            self.reset = buf["done"][t].copy()
            episodes += [o[3] for o in out if o[3] is not None]
        val[R] = values(self.critic, self.obs["state"], self.per_player)
        return buf, val, episodes


def _serve(conn, slot_args, coll_args):
    torch.set_num_threads(1)
    torch.manual_seed(slot_args[2] * 1000 + slot_args[3])
    coll = Collector([Slot(*slot_args)], *coll_args)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            conn.send(coll.rollout(*msg))
        except Exception as e:                     # tell the trainer, or it would wait for this game for ever
            conn.send(e)
            raise


class VecEnv:
    """n games: all in this process, or one process each (workers=True). A worker plays its whole rollout alone
    with its own copy of the policy, so the processes never wait for each other."""

    def __init__(self, n: int, workers: bool, overrides: dict, scen: dict, seed: int, coll_args):
        self.local, self.conns, self.procs = None, [], []
        if not workers:
            self.local = Collector([Slot(overrides, scen, seed, i) for i in range(n)], *coll_args)
        for i in range(n if workers else 0):
            here, there = mp.Pipe()
            p = mp.Process(target=_serve, args=(there, (overrides, scen, seed, i), coll_args), daemon=True)
            p.start()
            self.conns.append(here)
            self.procs.append(p)

    def rollout(self, actor, critic, R: int, knobs):
        weights = (actor.state_dict(), critic.state_dict())
        if self.local:
            return self.local.rollout(weights, R, knobs)
        for c in self.conns:
            c.send((weights, R, knobs))
        out = [c.recv() for c in self.conns]
        for o in out:
            if isinstance(o, Exception):
                raise o
        buf = {k: np.concatenate([o[0][k] for o in out], axis=1) for k in out[0][0]}
        return buf, np.concatenate([o[1] for o in out], axis=1), [e for o in out for e in o[2]]

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join()


def _eval_matches(job):
    """Some evaluation matches, each with its own fixed seed (one process of evaluate)."""
    weights, dims, overrides, scen, seeds = job
    torch.set_num_threads(1)
    actor = Actor(*dims)
    actor.load_state_dict(weights)
    env = FootballEnv(load_config(overrides=overrides))
    eps = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed, **sample_reset(scen, np.random.default_rng(seed)))
        env.set_shaping_weight(0.0)
        agent = MAPPOAgent(actor, seed=seed)
        done = False
        while not done:
            obs, _, done, _ = env.step(agent.act(obs))
        eps.append(summary(env, 0.0))
    return eps


def evaluate(actor: Actor, overrides: dict, scen: dict, n: int, procs: int = 1, seed0: int = EVAL_SEED0) -> dict:
    """n full matches with fixed seeds (seed0, seed0 + 1, ...), played by `procs` processes side by side (the same
    matches either way). Returns the mean of every summary field, plus W = wins + draws / 2."""
    first = actor.encoder[0] if actor.recurrent else actor.body[0]
    dims = (first.in_features, actor.head.out_features, actor.head.in_features, actor.recurrent)
    seeds = list(range(seed0, seed0 + n))
    jobs = [({k: v.detach().clone() for k, v in actor.state_dict().items()}, dims, overrides, scen, seeds[i::procs])
            for i in range(max(1, min(procs, n)))]
    if len(jobs) == 1:
        eps = _eval_matches(jobs[0])
    else:
        with mp.Pool(len(jobs)) as pool:
            eps = [e for part in pool.map(_eval_matches, jobs) for e in part]
    actor.train()
    out = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
    out["W"] = out["win"] + 0.5 * out["draw"]
    out["matches"] = n
    return out


# ------------------------------------------------------------------------------------------------ PPO
class ValueNorm:
    """Running mean and spread of the critic's targets (MAPPO's value normalisation). The critic learns targets of
    about unit size; its outputs are scaled back with denormalize before they are used."""

    def __init__(self, beta: float = 0.99999):
        self.beta, self.mean, self.mean_sq, self.debias = beta, 0.0, 0.0, 0.0

    def update(self, x):
        x = np.asarray(x, np.float64)
        self.mean = self.beta * self.mean + (1.0 - self.beta) * float(x.mean())
        self.mean_sq = self.beta * self.mean_sq + (1.0 - self.beta) * float((x ** 2).mean())
        self.debias = self.beta * self.debias + (1.0 - self.beta)

    def stats(self):
        """(mean, spread) so far; (0, 1) before the first update, so that nothing is scaled."""
        if self.debias == 0.0:
            return 0.0, 1.0
        m = self.mean / self.debias
        return m, float(np.sqrt(max(self.mean_sq / self.debias - m * m, 1e-8)))

    def normalize(self, x):
        m, s = self.stats()
        return ((np.asarray(x) - m) / s).astype(np.float32)

    def denormalize(self, x):
        m, s = self.stats()
        return (np.asarray(x) * s + m).astype(np.float32)

    def state(self) -> dict:
        return dict(beta=self.beta, mean=self.mean, mean_sq=self.mean_sq, debias=self.debias)

    def load(self, d: dict):
        self.beta, self.mean, self.mean_sq, self.debias = d["beta"], d["mean"], d["mean_sq"], d["debias"]


def explained_variance(y, v) -> float:
    """1 - Var(y - v) / Var(y): how much of y the critic's values v explain (1 all, 0 nothing, below 0 worse than a
    constant)."""
    vy = float(np.var(y))
    return float(1.0 - np.var(np.asarray(y) - np.asarray(v)) / vy) if vy > 0 else float("nan")


def gae(rew, val, done, gamma, lam):
    """rew: (R, E, 5). done: (R, E). val: (R + 1, E, 5). A match ends on time, so nothing is bootstrapped past
    `done`."""
    adv = np.zeros_like(rew)
    last = 0.0
    for t in reversed(range(len(rew))):
        nd = 1.0 - done[t][:, None]
        delta = rew[t] + gamma * val[t + 1] * nd - val[t]
        last = delta + gamma * lam * nd * last
        adv[t] = last
    return adv, adv + val[:-1]


def with_ids(state):
    """(..., 5, S) -> (..., 5, S + 5): each player's row of the global state with his one-hot slot, the critic's input
    when every player has his own value (reward.per_player)."""
    eye = np.broadcast_to(np.eye(5, dtype=np.float32), state.shape[:-1] + (5,))
    return np.concatenate([state, eye], -1)


@torch.no_grad()
def values(critic, state, per_player: bool):
    """(E, S) global states -> (E, 5) values: one per player, or the team value five times."""
    if per_player:
        return critic(torch.as_tensor(with_ids(np.repeat(state[:, None], 5, axis=1)))).numpy()
    return np.repeat(critic(torch.as_tensor(state)).numpy()[:, None], 5, axis=1)


def to_sequences(x, S: int):
    """(R, E, 5, ...) -> (R/S * E * 5, S, ...): chunks of S consecutive steps of one player."""
    R, E, P = x.shape[:3]
    x = x.reshape(R // S, S, E, P, *x.shape[3:])
    x = np.moveaxis(x, 1, 3)                                    # (C, E, 5, S, ...)
    return torch.as_tensor(np.ascontiguousarray(x).reshape(-1, S, *x.shape[4:]))


def per_player(x):
    """(R, E, ...) -> (R, E, 5, ...): the same team value for each of the five players."""
    return np.repeat(x[:, :, None], 5, axis=2)


def ppo_update(actor, critic, opt_a, opt_c, buf, args, rng, vnorm):
    S = args.seq_len if actor.recurrent else 1
    obs, mask, act = (to_sequences(buf[k], S) for k in ("obs", "mask", "act"))
    logp_old = to_sequences(buf["logp"], S)
    reset = to_sequences(per_player(buf["reset"]), S)
    state = per_player(buf["state"])
    state = to_sequences(with_ids(state) if args.per_player else state, S)
    adv = to_sequences(buf["adv"], S)
    ret = to_sequences(vnorm.normalize(buf["ret"]), S)                       # the critic learns normalised targets
    h0 = torch.as_tensor(buf["h"][::S].reshape(-1, buf["h"].shape[-1]))      # hidden state at each chunk start
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N, B = obs.shape[0], max(1, args.batch_size // S)
    stats, last = [], []
    for epoch in range(args.epochs):
        perm = rng.permutation(N)
        for i in range(0, N, B):
            b = torch.as_tensor(perm[i:i + B])
            logits, _ = actor(obs[b], h0[b], reset[b])
            dist = masked_dist(logits, mask[b])
            log_ratio = dist.log_prob(act[b]) - logp_old[b]
            ratio = log_ratio.exp()
            pi_loss = -torch.min(ratio * adv[b], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[b]).mean()
            ent = dist.entropy().mean()
            v_loss = F.mse_loss(critic(state[b]), ret[b])
            opt_a.zero_grad()
            opt_c.zero_grad()
            (pi_loss - args.entropy_coef * ent + args.value_coef * v_loss).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            opt_a.step()
            opt_c.step()
            with torch.no_grad():                  # how far the policy has moved from the one that played the matches
                kl = ((ratio - 1) - log_ratio).mean().item()
                clipped = ((ratio - 1).abs() > args.clip).float().mean().item()
            stats.append((pi_loss.item(), v_loss.item(), ent.item()))
            if epoch == args.epochs - 1:
                last.append((kl, clipped))
    out = dict(zip(("pi_loss", "v_loss", "entropy"), np.mean(stats, axis=0).tolist()))
    out["approx_kl"], out["clip_frac"] = np.mean(last, axis=0).tolist()          # in the last epoch: the whole move
    return out


# ----------------------------------------------------------------------------------------------- main
def csv(s):
    return [x for x in str(s).split(",") if x]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="run folder under --runs-dir (default: built from the settings)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actor", choices=["mlp", "gru"], default="mlp")
    # the setting (None = the value in configs/default.yaml)
    ap.add_argument("--our-formation", default=None)
    ap.add_argument("--our-formations", type=csv, default=None, help="comma list, one is drawn per match for our team")
    ap.add_argument("--opp-formations", type=csv, default=None, help="comma list, one is drawn per match")
    ap.add_argument("--opp-styles", type=csv, default=None, help="comma list, one is drawn per match")
    ap.add_argument("--sigmas", type=csv, default=["0"], help="playmaker, comma list of 0 (none) or 1..5")
    ap.add_argument("--cases", type=csv, default=None,
                    help="opponent formation and playmaker drawn together, e.g. 2-2-1:0,2-2-1:4 (replaces the two lists)")
    ap.add_argument("--radius", type=float, default=None, help="our vision radius")
    ap.add_argument("--zeta", type=int, default=None, help="structure level 0, 1 or 2")
    ap.add_argument("--switching", default=None, help="NONE | SCHEDULED | RANDOM | SCORE_REACTIVE | WINDOW")
    ap.add_argument("--T", type=int, default=None, help="match length in steps")
    ap.add_argument("--recipe", default=None,
                    help="a YAML file of config values for this run, e.g. configs/study.yaml (--set wins over it)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="any other config value, e.g. --set reward.w_progress=0.005 (can be repeated)")
    # how long, how parallel
    ap.add_argument("--total-steps", type=int, default=2_000_000, help="game steps, summed over the games")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--workers", action="store_true", help="one process per game (for a single fast run)")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="threads for the update. More than 1 only helps a single run that has the machine to itself")
    # PPO (training.md Section 5)
    ap.add_argument("--rollout-len", type=int, default=256, help="steps per game between updates")
    ap.add_argument("--seq-len", type=int, default=32, help="GRU chunk length")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr-actor", type=float, default=3e-4)
    ap.add_argument("--lr-critic", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="discount: 0.99 looks about 100 steps ahead. At 1.0 a critic learns nothing that carries over "
                         "to new matches (measured 2026-10-18)")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--value-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="normalise the critic's targets by their running mean and spread (MAPPO)")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--shaping-anneal-end", type=int, default=5_000_000,
                    help="steps over which shaping falls to --shaping-floor (with the easy start: counted from full strength)")
    ap.add_argument("--shaping-floor", type=float, default=0.0,
                    help="the shaping weight never falls below this (0 = the small terms fade out completely)")
    ap.add_argument("--opp-eps-start", type=float, default=None,
                    help="easy start: opponent randomness to begin with, see EPS_STEP (default: the opponent as configured)")
    ap.add_argument("--matches", type=int, default=None,
                    help="train by whole matches: this many matches in all (summed over the games), each update one "
                         "full match of every game (sets --rollout-len to the match length and --total-steps)")
    # evaluation and checkpoints
    ap.add_argument("--eval-every", type=int, default=400, help="with --matches: matches between evaluations")
    ap.add_argument("--eval-interval", type=int, default=250_000)
    ap.add_argument("--eval-matches", type=int, default=20)
    ap.add_argument("--eval-procs", type=int, default=1, help="processes that play the evaluation matches side by side")
    ap.add_argument("--test-matches", type=int, default=300,
                    help="matches the best checkpoint plays at the end, on seeds no evaluation used (the run's result)")
    ap.add_argument("--ckpt-interval", type=int, default=50_000)
    args = ap.parse_args(argv)
    if args.rollout_len % args.seq_len:
        ap.error("--rollout-len must be a multiple of --seq-len")
    return args


def _known(config: dict, key: str, where: str):
    """A misspelt key must not run the wrong experiment silently."""
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f"{where} {key}: there is no such value in the config")
        node = node[part]


def set_overrides(pairs) -> dict:
    """The --set KEY=VALUE pairs as config overrides."""
    known, out = load_config().to_dict(), {}
    for kv in pairs:
        key, value = kv.split("=", 1)
        _known(known, key, "--set")
        out[key] = yaml.safe_load(value)
    return out


def recipe_overrides(path) -> dict:
    """A recipe file (a partial copy of configs/default.yaml, e.g. configs/study.yaml) as config overrides, one for
    every value it sets."""
    if not path:
        return {}
    known, out = load_config().to_dict(), {}

    def walk(d, prefix):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            else:
                _known(known, f"{prefix}{k}", f"--recipe {path}:")
                out[f"{prefix}{k}"] = v
    with open(path) as f:
        walk(yaml.safe_load(f) or {}, "")
    return out


def save_ckpt(path: Path, **ck):
    tmp = path.with_suffix(".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)                      # never leaves a half-written checkpoint behind


def train(args):
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    overrides = recipe_overrides(args.recipe)
    if args.T:
        overrides["time.T"] = args.T
    if args.switching:
        overrides["opponent.switching.mode"] = args.switching
    overrides.update(set_overrides(args.set))
    overrides["reward.potential.gamma"] = args.gamma       # the potential-based shaping uses the trainer's own discount
    cfg = load_config(overrides=overrides)
    if args.matches:                           # by whole matches: one update = one full match of every game
        T = int(cfg.time.T)
        if args.actor == "gru" and T % args.seq_len:
            raise SystemExit(f"--matches with --actor gru needs a match length that is a multiple of --seq-len")
        args.rollout_len, args.total_steps, args.eval_interval = T, args.matches * T, args.eval_every * T
    scen = dict(our_formation=args.our_formation or cfg.formations.ours, our_formations=args.our_formations,
                radius=args.radius, zeta=args.zeta,
                opp_formations=args.opp_formations or [cfg.formations.opponent],
                opp_styles=args.opp_styles or [cfg.styles.opponent], sigmas=args.sigmas,
                cases=[(c.split(":")[0], int(c.split(":")[1])) for c in args.cases] if args.cases else None)
    name = args.name or (f"{args.actor}_f{scen['our_formation']}_r{args.radius or cfg.vision.radius:g}"
                         f"_z{cfg.structure.zeta if args.zeta is None else args.zeta}_s{args.seed}")
    run = Path(args.runs_dir) / name
    run.mkdir(parents=True, exist_ok=True)
    done = run / "result.json"                 # a finished run is skipped, unless more steps are asked for now
    if done.exists() and json.load(open(done))["step"] >= args.total_steps:
        print(f"[{name}] already finished (result.json exists)")
        return

    probe = FootballEnv(cfg)
    o, _ = probe.reset(seed=0)
    obs_dim, n_act, state_dim = o["vec"].shape[1], o["mask"].shape[1], o["state"].shape[0]
    actor = Actor(obs_dim, n_act, args.hidden, args.actor == "gru")
    args.per_player = bool(cfg.reward.per_player)        # every player his own value and advantage
    critic = Critic(state_dim + (5 if args.per_player else 0), args.hidden)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
    vnorm = ValueNorm()                         # never updated with --no-value-norm: then it scales nothing

    step = 0
    eps, recent = args.opp_eps_start, []       # opponent randomness now (None = as configured), our latest win scores
    eps_end = float(cfg.opponent.randomness.eps_opp)
    # the shaping anneal starts at this step. With the easy start it waits until the opponent is at full strength:
    # when the shaping faded before that, the attack faded with it
    anneal_from = 0 if eps is None or eps <= eps_end else None
    ckpt = run / "ckpt.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic"])
        opt_a.load_state_dict(ck["opt_a"]); opt_c.load_state_dict(ck["opt_c"])
        step = ck["step"]
        if eps is not None:
            eps = ck.get("opp_eps", eps)
        anneal_from = ck.get("anneal_from", anneal_from)
        if "vnorm" in ck:
            vnorm.load(ck["vnorm"])
        rng = np.random.default_rng([args.seed, step])         # new games after a restart, not the same ones again
        print(f"[{name}] resuming from step {step}")
    with open(run / "config.json", "w") as f:
        json.dump({"args": vars(args), "scenario": scen, "cfg": cfg.to_dict()}, f, indent=2)
    log = open(run / "log.jsonl", "a")

    def write(kind, **row):
        log.write(json.dumps({"kind": kind, "step": step, **row}) + "\n")
        log.flush()

    def checkpoint():
        save_ckpt(ckpt, actor=actor.state_dict(), critic=critic.state_dict(), opt_a=opt_a.state_dict(),
                  opt_c=opt_c.state_dict(), step=step, args=vars(args), obs_dim=obs_dim, n_actions=n_act, opp_eps=eps,
                  anneal_from=anneal_from, overrides=overrides, vnorm=vnorm.state())

    E, R = args.n_envs, args.rollout_len

    def shaping_w():
        return 1.0 if anneal_from is None else max(args.shaping_floor, 1.0 - (step - anneal_from) / args.shaping_anneal_end)

    vec = VecEnv(E, args.workers, overrides, scen, int(rng.integers(2 ** 31)),
                 ((obs_dim, n_act, state_dim), args.hidden, actor.recurrent, args.per_player))
    next_eval = (step // args.eval_interval + 1) * args.eval_interval
    next_ckpt = (step // args.ckpt_interval + 1) * args.ckpt_interval

    t_start, step_start, last_eval = time.time(), step, None

    def progress():
        """runs/<name>/progress.txt: how far the run is, for a look at any time (cat it)."""
        T_ = R if args.matches else cfg.time.T
        done_m, total_m = step // T_, args.total_steps // T_
        el = time.time() - t_start
        rate = (step - step_start) / T_ / el if el > 0 else 0.0
        left = max(0, total_m - done_m) / rate if rate > 0 else float("nan")
        hms = lambda x: "?" if x != x else time.strftime("%H:%M:%S", time.gmtime(x))
        txt = (f"{name}: {done_m:,} / {total_m:,} matches finished ({100 * done_m / max(total_m, 1):.1f}%), "
               f"{rate * 60:.0f} matches a minute, elapsed {hms(el)}, about {hms(left)} left (evaluations not counted)\n")
        if last_eval is not None:
            txt += (f"last evaluation (after {last_eval[0]:,} matches, {last_eval[1]['matches']} matches at full strength): "
                    f"W {last_eval[1]['W']:.2f}, goals {last_eval[1]['goals_for']:.2f} - "
                    f"{last_eval[1]['goals_for'] - last_eval[1]['goal_diff']:.2f}\n")
        with open(run / "progress.txt", "w") as f:
            f.write(txt)

    while step < args.total_steps:
        t0, w = time.time(), shaping_w()
        buf, val, episodes = vec.rollout(actor, critic, R, (w, eps))
        val = vnorm.denormalize(val)                           # the critic's outputs are normalised values
        buf["adv"], buf["ret"] = gae(buf["rew"], val, buf["done"], args.gamma, args.lam)
        _, disc = gae(buf["rew"], val, buf["done"], args.gamma, 1.0)      # the discounted return itself
        # how much the critic explains: of its own targets, and of the return that really followed (exact for whole
        # matches). The second is the honest one: near 0 means the critic carries no credit to earlier choices
        fit = dict(ev_targets=explained_variance(buf["ret"], val[:-1]), ev_return=explained_variance(disc, val[:-1]))
        if args.value_norm:
            vnorm.update(buf["ret"])
        t_roll = time.time() - t0
        losses = ppo_update(actor, critic, opt_a, opt_c, buf, args, rng, vnorm)
        step += R * E

        dt = time.time() - t0
        row = dict(losses, **fit, shaping_weight=w, opp_eps=eps_end if eps is None else eps, sps=R * E / dt,
                   rollout_frac=t_roll / dt, reward_per_step=float(buf["rew"].mean()), matches=len(episodes))
        if episodes:
            row.update({k: float(np.mean([e[k] for e in episodes])) for k in episodes[0]})
        write("train", **row)
        progress()
        print(f"[{name}] " + (f"match {step // R:>6,}  " if args.matches else "") + f"step {step:>9,}  {row['sps']:6.0f} steps/s  ent {row['entropy']:.2f}  v_loss {row['v_loss']:.4f}"
              + f"  ev {row['ev_return']:+.2f}  kl {row['approx_kl']:.4f}"
              + (f"  win {row['win']:.2f}  goal_diff {row['goal_diff']:+.2f}  shaping {row['shaping_return']:+.2f}"
                 if episodes else "") + ("" if eps is None else f"  opp_eps {eps:.2f}"), flush=True)
        if eps is not None:                    # the easy start: see EPS_STEP at the top of this file
            recent += [e["win"] + 0.5 * e["draw"] for e in episodes]
            if len(recent) >= EPS_WINDOW:
                score, recent = float(np.mean(recent)), []
                if score >= EPS_HARDER:
                    eps = max(eps_end, round(eps - EPS_STEP, 6))
                elif score <= EPS_EASIER:
                    eps = min(max(eps, args.opp_eps_start), round(eps + EPS_STEP, 6))
            if anneal_from is None and eps <= eps_end:
                anneal_from = step

        if step >= next_ckpt or step >= args.total_steps:
            checkpoint()
            next_ckpt = (step // args.ckpt_interval + 1) * args.ckpt_interval
        if step >= next_eval or step >= args.total_steps:      # the last weights are a candidate too
            ev = evaluate(actor, overrides, scen, args.eval_matches, args.eval_procs)
            m = step // (R if args.matches else cfg.time.T)
            cand = run / "ckpts" / f"m{m:06d}.pt"                  # every evaluation keeps its checkpoint
            cand.parent.mkdir(exist_ok=True)
            save_ckpt(cand, actor=actor.state_dict(), step=step, args=vars(args), obs_dim=obs_dim, n_actions=n_act,
                      overrides=overrides)
            write("eval", ckpt=cand.name, **ev)
            last_eval = (m, ev)
            print(f"[{name}] EVAL at {step:,}: W {ev['W']:.2f}  win {ev['win']:.2f}  goal_diff {ev['goal_diff']:+.2f}", flush=True)
            next_eval = (step // args.eval_interval + 1) * args.eval_interval

    vec.close()
    # the run's result: the evaluated checkpoint with the best W (then goal difference), played on the test seeds,
    # which no evaluation used, so that picking the best does not flatter it
    rows = [json.loads(line) for line in open(run / "log.jsonl")]
    cands = [r for r in rows if r["kind"] == "eval" and "ckpt" in r and (run / "ckpts" / r["ckpt"]).exists()]
    best = max(cands, key=lambda r: (r["W"], r["goal_diff"], r["step"]))
    actor.load_state_dict(torch.load(run / "ckpts" / best["ckpt"], map_location="cpu", weights_only=False)["actor"])
    ev = evaluate(actor, overrides, scen, args.test_matches, args.eval_procs, seed0=TEST_SEED0)
    write("test", ckpt=best["ckpt"], validation_W=best["W"], **ev)
    with open(done, "w") as f:
        json.dump({"name": name, "step": step, "args": vars(args), "scenario": scen, "best_ckpt": best["ckpt"],
                   "validation": {k: best[k] for k in ("W", "goal_diff", "goals_for", "matches")}, **ev}, f, indent=2)
    print(f"[{name}] TEST of {best['ckpt']} (best evaluation, W {best['W']:.2f}) over {ev['matches']} new matches: "
          f"W {ev['W']:.2f}  win {ev['win']:.2f}  goal_diff {ev['goal_diff']:+.2f}")


if __name__ == "__main__":
    train(parse_args())
