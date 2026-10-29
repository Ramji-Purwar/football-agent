"""The formation guesser of Q2: which formation does the opponent play NOW, from what our players see?

Position maps, counted once on practice matches: for every formation, opponent player and ball state (we have it, they
have it, loose), how often that player stands in each 10 m box of the pitch, or is seen by none of us (one more box).
Our team pools its view: one of us seeing a player is enough.

Without player numbers (anonymous=True) the maps say where AN opponent stands (all five pooled) and how many of them
we see at once, so the guess has to come from the shape alone, not from who stands where.

The filter keeps a belief over the formations. Every step it first allows for a switch (chance h that the formation
has just changed to another one), then weights every formation by how well its maps explain what we see now, raised
to the power beta < 1: successive steps show nearly the same picture, so each step counts for little. A formation
becomes the guess once it has been on top for m steps in a row, and stays the guess until another one has."""
from __future__ import annotations
import numpy as np

BOX = 10.0                     # box size of the maps (m)


class PositionMaps:
    def __init__(self, names, L: float, W: float, box: float = BOX, counts=None, seen=None):
        self.names, self.L, self.W, self.box = list(names), float(L), float(W), float(box)
        self.nx, self.ny = int(np.ceil(self.L / self.box)), int(np.ceil(self.W / self.box))
        self.unseen = self.nx * self.ny                                  # the last box: seen by none of us
        self.counts = np.zeros((len(self.names), 5, 3, self.unseen + 1)) if counts is None else counts
        self.seen = np.zeros((len(self.names), 3, 6)) if seen is None else seen   # how many we see at once

    def view(self, obs_by_player: dict):
        """What our team sees this step: the box of every opponent player (or `unseen`) and the ball state."""
        boxes = np.full(5, self.unseen)
        for o in obs_by_player.values():
            for j in np.flatnonzero(o.opp_visible):
                ix = int(np.clip(o.opp_pos[j][0] // self.box, 0, self.nx - 1))
                iy = int(np.clip(o.opp_pos[j][1] // self.box, 0, self.ny - 1))
                boxes[j] = ix * self.ny + iy
        return boxes, int(next(iter(obs_by_player.values())).possession)

    def add(self, formation: str, boxes, poss: int):
        f = self.names.index(formation)
        self.counts[f, np.arange(5), poss, boxes] += 1
        self.seen[f, poss, int((boxes != self.unseen).sum())] += 1

    def log_probs(self, prior: float = 1.0):
        """log P(box | formation, player, ball state). `prior` is added to every count, so no box is impossible."""
        c = self.counts + prior
        return np.log(c / c.sum(-1, keepdims=True))

    def anonymous_log_probs(self, prior: float = 1.0):
        """The same without player numbers: log P(an opponent we see stands in box | formation, ball state), all
        five pooled, shaped like log_probs() (the same for every player, 0 for `unseen`) so the filter adds it up the
        same way; and log P(we see n of them | formation, ball state) for n = 0..5."""
        c = self.counts[..., :self.unseen].sum(1) + prior                   # (formations, ball states, boxes)
        logp = np.zeros_like(self.counts)
        logp[..., :self.unseen] = np.log(c / c.sum(-1, keepdims=True))[:, None]
        n = self.seen + prior
        return logp, np.log(n / n.sum(-1, keepdims=True))

    def save(self, path):
        np.savez(path, counts=self.counts, seen=self.seen, names=np.array(self.names),
                 size=np.array([self.L, self.W, self.box]))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        L, W, box = z["size"]
        return cls([str(n) for n in z["names"]], L, W, box, z["counts"], z["seen"] if "seen" in z.files else None)


def filter_step(belief, ll, beta: float, h: float):
    """One step of the filter, for one belief (K,) or for many at once (N, K). ll = log-likelihood of what we see
    now under each formation."""
    K = belief.shape[-1]
    b = (1.0 - h) * belief + h * (1.0 - belief) / (K - 1)               # the formation may just have changed
    lb = np.log(b) + beta * ll
    b = np.exp(lb - lb.max(-1, keepdims=True))
    return b / b.sum(-1, keepdims=True)


class FormationFilter:
    def __init__(self, maps: PositionMaps, beta: float, h: float, m: int, anonymous: bool = False):
        self.names, self.unseen = maps.names, maps.unseen
        self.logp, self.logn = maps.anonymous_log_probs() if anonymous else (maps.log_probs(), None)
        self.beta, self.h, self.m = float(beta), float(h), int(m)
        self.reset()

    def reset(self):
        """A new match: every formation equally likely (the guesser is not told how the opponent starts)."""
        self.belief = np.full(len(self.names), 1.0 / len(self.names))
        self.top, self.run, self.guess = None, 0, None

    def loglik(self, boxes, poss: int):
        ll = self.logp[:, np.arange(5), poss, boxes].sum(1)
        if self.logn is not None:                                       # without player numbers: also how many
            ll = ll + self.logn[:, poss, int((boxes != self.unseen).sum())]
        return ll

    def update(self, boxes, poss: int):
        """One step: what our team sees now (PositionMaps.view). Returns the guess (None until there is one)."""
        self.belief = filter_step(self.belief, self.loglik(boxes, poss), self.beta, self.h)
        top = int(np.argmax(self.belief))
        self.run = self.run + 1 if top == self.top else 1
        self.top = top
        if self.run >= self.m:
            self.guess = self.names[top]
        return self.guess
