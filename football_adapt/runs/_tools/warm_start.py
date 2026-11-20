"""Warm start for a stuck run (17 Nov): write runs/NEW/ckpt.pt with the weights of a trained run, step 0, fresh Adam,
and no opp_eps (so the easy start begins again at --opp-eps-start). training.train then "resumes" from it: the new run
trains from those weights with the normal curriculum, logs and evaluations.

The Q3 mixture runs trained from scratch never reached the full-strength opponent (eval W 0.00-0.03 after
11k-27k of 40k matches). Starting them from the Q2 GRU seed 0 (test W 0.93 against the full-strength 2-2-1, same
observation and action space) gives them a team that can already play.

--like: a checkpoint with the shapes the new run needs (its own from-scratch run). Every tensor of the source that has
the same shape is copied. The only other case allowed is the actor's first layer with extra inputs at the END of the
observation (actions.show_playmaker adds 6 at the end, football/observation.py): those columns start at 0, so the
warm-started team first plays exactly like the source and learns to use them.

usage (server, from football_adapt/):
  ~/fa-venv/bin/python runs/_tools/warm_start.py --src runs/q2/gru_r25_z0/s0/ckpt.pt \
      --like runs/q3/oracle_r25_z0/s0/ckpt.pt --out runs/q3/mixw_told_r25_z0/s0
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from agents.mappo import Actor, Critic  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--like", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

src = torch.load(a.src, map_location="cpu", weights_only=False)
like = torch.load(a.like, map_location="cpu", weights_only=False)
out = Path(a.out)
if (out / "ckpt.pt").exists() or (out / "log.jsonl").exists():
    sys.exit(f"{out} already has a run: not touching it")


def merge(name, s, t):
    m = {}
    for k, tv in t.items():
        sv = s[k]
        if sv.shape == tv.shape:
            m[k] = sv.clone()
        elif name == "actor" and k == "encoder.0.weight" and sv.shape[0] == tv.shape[0] and sv.shape[1] < tv.shape[1]:
            w = torch.zeros_like(tv)
            w[:, :sv.shape[1]] = sv                # the extra inputs are appended at the end: their weights start at 0
            m[k] = w
            print(f"  actor {k}: {tuple(sv.shape)} -> {tuple(tv.shape)}, {tv.shape[1] - sv.shape[1]} new inputs at 0")
        else:
            sys.exit(f"{name} {k}: shapes {tuple(sv.shape)} and {tuple(tv.shape)} do not fit")
    return m


actor_sd, critic_sd = merge("actor", src["actor"], like["actor"]), merge("critic", src["critic"], like["critic"])
hidden = like["actor"]["head.weight"].shape[1]
actor = Actor(like["obs_dim"], like["n_actions"], hidden, recurrent="gru.weight_ih" in like["actor"])
critic = Critic(like["critic"]["net.0.weight"].shape[1], hidden)
actor.load_state_dict(actor_sd)
critic.load_state_dict(critic_sd)
largs = like["args"]
ck = dict(actor=actor.state_dict(), critic=critic.state_dict(),
          opt_a=torch.optim.Adam(actor.parameters(), lr=largs["lr_actor"]).state_dict(),
          opt_c=torch.optim.Adam(critic.parameters(), lr=largs["lr_critic"]).state_dict(),
          step=0, args=largs, obs_dim=like["obs_dim"], n_actions=like["n_actions"], overrides=like["overrides"],
          vnorm=src["vnorm"], warm_start_from=a.src)
out.mkdir(parents=True, exist_ok=True)
tmp = out / "ckpt.tmp"
torch.save(ck, tmp)
os.replace(tmp, out / "ckpt.pt")
(out / "WARM_START.txt").write_text(f"weights from {a.src} (step {src['step']}), shapes from {a.like}, step 0, fresh Adam,\n"
                                    "easy start from --opp-eps-start again (runs/_tools/warm_start.py)\n")
print(f"wrote {out}/ckpt.pt  (obs {like['obs_dim']}, actions {like['n_actions']}, from {a.src} step {src['step']})")
