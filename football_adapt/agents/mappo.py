"""Parameter-shared MAPPO networks (training.md Sections 2 and 4).

All five of our players run the SAME actor, each on its own observation. The critic sees the global state
(training only). The trainer is training/train.py.

Not imported from agents/__init__.py on purpose: the game, the scripted team and the tests must keep working
without torch installed.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    """recurrent=False: memoryless MLP (Q1). recurrent=True: encoder + GRU (Q2, Q3). Same call for both.

    forward(obs, h, reset):
        obs   (B, S, obs_dim)   S consecutive steps of one player
        h     (B, h_dim)        hidden state BEFORE the first step (ignored by the MLP)
        reset (B, S)            1.0 where that step is the first of a new episode: the hidden state is zeroed there
    returns logits (B, S, n_actions) and the hidden state after the last step.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, recurrent: bool = False):
        super().__init__()
        self.recurrent = recurrent
        self.h_dim = hidden if recurrent else 1          # the MLP carries a dummy hidden state
        if recurrent:
            self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
            self.gru = nn.GRUCell(hidden, hidden)
        else:
            self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs, h, reset):
        if not self.recurrent:
            return self.head(self.body(obs)), h
        x = self.encoder(obs)
        outs = []
        for t in range(x.shape[1]):
            h = self.gru(x[:, t], h * (1.0 - reset[:, t, None]))
            outs.append(h)
        return self.head(torch.stack(outs, 1)), h


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, state):                            # (..., state_dim) -> (...)
        return self.net(state).squeeze(-1)


def masked_dist(logits, mask):
    """Categorical over the VALID actions only. -1e9 (not -inf) keeps the entropy finite."""
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


class MAPPOAgent:
    """A trained actor behind the same act(obs) interface as RandomAgent. Call reset() at every kick-off."""

    def __init__(self, actor: Actor, seed=None, greedy: bool = False):
        self.actor = actor.eval()
        self.greedy = greedy
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(int(seed))
        self.reset()

    @classmethod
    def load(cls, path, seed=None, greedy: bool = False):
        torch.set_num_threads(1)                 # the networks are tiny: more threads only keep the cores busy
        ck = torch.load(path, map_location="cpu", weights_only=False)
        actor = Actor(ck["obs_dim"], ck["n_actions"], ck["args"]["hidden"], ck["args"]["actor"] == "gru")
        actor.load_state_dict(ck["actor"])
        agent = cls(actor, seed, greedy)
        agent.train_args = ck["args"]            # the setting it was trained in: T, radius, zeta, our_formation, ...
        agent.overrides = ck.get("overrides") or {}   # ... and the config values it was trained with (--set)
        return agent

    def reset(self, seed=None):
        self.h = torch.zeros(5, self.actor.h_dim)
        if seed is not None:                     # same seed, same match again
            self.gen.manual_seed(int(seed))

    @torch.no_grad()
    def act(self, obs: dict) -> dict:
        vec = torch.as_tensor(obs["vec"]).unsqueeze(1)                       # (5, 1, obs_dim)
        logits, self.h = self.actor(vec, self.h, torch.zeros(5, 1))
        probs = masked_dist(logits[:, 0], torch.as_tensor(obs["mask"])).probs
        a = probs.argmax(-1) if self.greedy else torch.multinomial(probs, 1, generator=self.gen)[:, 0]
        return {i: int(a[i]) for i in range(5)}


__all__ = ["Actor", "Critic", "masked_dist", "MAPPOAgent"]
