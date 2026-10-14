"""Train one parameter-shared MAPPO policy against the scripted team (training.md). CPU only.

Run from football_adapt/:

   python -m training.train --name sanity --T 300 --radius 100 --total-steps 600000 --workers --opp-eps-start 0.5
   python -m training.train --name easy --total-steps 2000000 --workers --opp-eps-start 0.5          # Stage 2
   python -m training.train --actor gru --opp-formations 2-2-1,1-2-2,1-3-1 --switching RANDOM

--opp-eps-start 0.5 is the easy start: the opponent's players act at random half of the time, and the opponent is
made stronger step by step as our team gets ahead of it (see EPS_STEP below). The evaluations are always against
the opponent at full strength. Against the full-strength opponent from step 0 a new policy never gets near the goal.

Everything goes to runs/<name>/: config.json, log.jsonl (one line per update and per evaluation), ckpt.pt and,
when the run is finished, result.json. Running the SAME command again resumes from ckpt.pt, so a run survives a
node that expires; with a larger --total-steps it carries a finished run on. Our formation, radius and structure
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

# The easy start (--opp-eps-start): each opponent player acts at random with probability eps (opponent.md,
# opponent.randomness.eps_opp). After every EPS_WINDOW training matches our win score (wins + draws / 2) is looked
# at: EPS_HARDER or more and eps drops by EPS_STEP (down to the config value), EPS_EASIER or less and it goes back
# up by EPS_STEP (up to the start value). In between it stays. So the opponent is always one our team can just beat.
EPS_STEP, EPS_WINDOW, EPS_HARDER, EPS_EASIER = 0.05, 50, 0.6, 0.4


# ---------------------------------------------------------------------------------------------- the games
def sample_reset(scen: dict, rng) -> dict:
    """reset() arguments for one match. Fixed things are passed through, lists are drawn from."""
    return dict(our_formation=scen["our_formation"], radius=scen["radius"], zeta=scen["zeta"],
                opp_formation=scen["opp_formations"][rng.integers(len(scen["opp_formations"]))],
                opp_style=scen["opp_styles"][rng.integers(len(scen["opp_styles"]))],
                sigma=int(scen["sigmas"][rng.integers(len(scen["sigmas"]))]))


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


def evaluate(actor: Actor, overrides: dict, scen: dict, n: int) -> dict:
    """n full matches with fixed seeds. Returns the mean of every summary field, plus W = wins + draws / 2."""
    env = FootballEnv(load_config(overrides=overrides))
    eps = []
    for seed in range(EVAL_SEED0, EVAL_SEED0 + n):
        obs, _ = env.reset(seed=seed, **sample_reset(scen, np.random.default_rng(seed)))
        env.set_shaping_weight(0.0)
        agent = MAPPOAgent(actor, seed=seed)
        done = False
        while not done:
            obs, _, done, _ = env.step(agent.act(obs))
        eps.append(summary(env, 0.0))
    actor.train()
    out = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
    out["W"] = out["win"] + 0.5 * out["draw"]
    out["matches"] = n
    return out


# ------------------------------------------------------------------------------------------------ PPO
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


def ppo_update(actor, critic, opt_a, opt_c, buf, args, rng):
    S = args.seq_len if actor.recurrent else 1
    obs, mask, act = (to_sequences(buf[k], S) for k in ("obs", "mask", "act"))
    logp_old = to_sequences(buf["logp"], S)
    reset = to_sequences(per_player(buf["reset"]), S)
    state = per_player(buf["state"])
    state = to_sequences(with_ids(state) if args.per_player else state, S)
    adv = to_sequences(buf["adv"], S)
    ret = to_sequences(buf["ret"], S)
    h0 = torch.as_tensor(buf["h"][::S].reshape(-1, buf["h"].shape[-1]))      # hidden state at each chunk start
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N, B = obs.shape[0], max(1, args.batch_size // S)
    stats = []
    for _ in range(args.epochs):
        perm = rng.permutation(N)
        for i in range(0, N, B):
            b = torch.as_tensor(perm[i:i + B])
            logits, _ = actor(obs[b], h0[b], reset[b])
            dist = masked_dist(logits, mask[b])
            ratio = (dist.log_prob(act[b]) - logp_old[b]).exp()
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
            stats.append((pi_loss.item(), v_loss.item(), ent.item()))
    return dict(zip(("pi_loss", "v_loss", "entropy"), np.mean(stats, axis=0).tolist()))


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
    ap.add_argument("--opp-formations", type=csv, default=None, help="comma list, one is drawn per match")
    ap.add_argument("--opp-styles", type=csv, default=None, help="comma list, one is drawn per match")
    ap.add_argument("--sigmas", type=csv, default=["0"], help="playmaker, comma list of 0 (none) or 1..5")
    ap.add_argument("--radius", type=float, default=None, help="our vision radius")
    ap.add_argument("--zeta", type=int, default=None, help="structure level 0, 1 or 2")
    ap.add_argument("--switching", default=None, help="NONE | SCHEDULED | RANDOM | SCORE_REACTIVE")
    ap.add_argument("--T", type=int, default=None, help="match length in steps")
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
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="discount: 1 = every choice is judged by the whole match (configs: reward)")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--shaping-anneal-end", type=int, default=5_000_000,
                    help="steps over which shaping falls to 0 (with the easy start: counted from full strength)")
    ap.add_argument("--opp-eps-start", type=float, default=None,
                    help="easy start: opponent randomness to begin with, see EPS_STEP (default: the opponent as configured)")
    # evaluation and checkpoints
    ap.add_argument("--eval-interval", type=int, default=250_000)
    ap.add_argument("--eval-matches", type=int, default=20)
    ap.add_argument("--final-eval-matches", type=int, default=100)
    ap.add_argument("--ckpt-interval", type=int, default=50_000)
    args = ap.parse_args(argv)
    if args.rollout_len % args.seq_len:
        ap.error("--rollout-len must be a multiple of --seq-len")
    return args


def set_overrides(pairs) -> dict:
    """The --set KEY=VALUE pairs as config overrides."""
    known, out = load_config().to_dict(), {}
    for kv in pairs:
        key, value = kv.split("=", 1)
        node = known
        for part in key.split("."):                # a misspelt key must not run the wrong experiment silently
            if not isinstance(node, dict) or part not in node:
                raise SystemExit(f"--set {key}: there is no such value in the config")
            node = node[part]
        out[key] = yaml.safe_load(value)
    return out


def save_ckpt(path: Path, **ck):
    tmp = path.with_suffix(".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)                      # never leaves a half-written checkpoint behind


def train(args):
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    overrides = {}
    if args.T:
        overrides["time.T"] = args.T
    if args.switching:
        overrides["opponent.switching.mode"] = args.switching
    overrides.update(set_overrides(args.set))
    cfg = load_config(overrides=overrides)
    scen = dict(our_formation=args.our_formation or cfg.formations.ours, radius=args.radius, zeta=args.zeta,
                opp_formations=args.opp_formations or [cfg.formations.opponent],
                opp_styles=args.opp_styles or [cfg.styles.opponent], sigmas=args.sigmas)
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
                  anneal_from=anneal_from, overrides=overrides)

    E, R = args.n_envs, args.rollout_len

    def shaping_w():
        return 1.0 if anneal_from is None else max(0.0, 1.0 - (step - anneal_from) / args.shaping_anneal_end)

    vec = VecEnv(E, args.workers, overrides, scen, int(rng.integers(2 ** 31)),
                 ((obs_dim, n_act, state_dim), args.hidden, actor.recurrent, args.per_player))
    next_eval = (step // args.eval_interval + 1) * args.eval_interval
    next_ckpt = (step // args.ckpt_interval + 1) * args.ckpt_interval

    while step < args.total_steps:
        t0, w = time.time(), shaping_w()
        buf, val, episodes = vec.rollout(actor, critic, R, (w, eps))
        buf["adv"], buf["ret"] = gae(buf["rew"], val, buf["done"], args.gamma, args.lam)
        t_roll = time.time() - t0
        losses = ppo_update(actor, critic, opt_a, opt_c, buf, args, rng)
        step += R * E

        dt = time.time() - t0
        row = dict(losses, shaping_weight=w, opp_eps=eps_end if eps is None else eps, sps=R * E / dt,
                   rollout_frac=t_roll / dt, reward_per_step=float(buf["rew"].mean()), matches=len(episodes))
        if episodes:
            row.update({k: float(np.mean([e[k] for e in episodes])) for k in episodes[0]})
        write("train", **row)
        print(f"[{name}] step {step:>9,}  {row['sps']:6.0f} steps/s  ent {row['entropy']:.2f}  v_loss {row['v_loss']:.4f}"
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
        if step >= next_eval and step < args.total_steps:
            ev = evaluate(actor, overrides, scen, args.eval_matches)
            write("eval", **ev)
            print(f"[{name}] EVAL at {step:,}: W {ev['W']:.2f}  win {ev['win']:.2f}  goal_diff {ev['goal_diff']:+.2f}", flush=True)
            next_eval = (step // args.eval_interval + 1) * args.eval_interval

    vec.close()
    ev = evaluate(actor, overrides, scen, args.final_eval_matches)
    write("final", **ev)
    with open(done, "w") as f:
        json.dump({"name": name, "step": step, "args": vars(args), "scenario": scen, **ev}, f, indent=2)
    print(f"[{name}] FINAL over {ev['matches']} matches: W {ev['W']:.2f}  win {ev['win']:.2f}  goal_diff {ev['goal_diff']:+.2f}")


if __name__ == "__main__":
    train(parse_args())
