"""The playmaker guesser of Q3: which opponent player, if any, is the playmaker, from what our players see?

Evidence, every step, for every opponent player (by number): how far from the ball he stands (or that none of us sees
him), in each ball state (their ball / ours / loose); and when one of them passes, how fast the ball then flies. Maps,
counted on practice matches, say how likely each is for a playmaker and for an ordinary player of that number.

The filter keeps a belief over "no playmaker" and "player k is the playmaker" (k = the numbers that were the playmaker
in practice) and multiplies in each step's evidence, the positions raised to beta < 1 (successive steps show nearly
the same picture) and a pass to beta_pass. The playmaker does not change during a match, so there is no switching."""
from __future__ import annotations
import numpy as np
from football.observation import OTHER, LOOSE

DIST = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 50.0])    # distance from the ball (m): 7 bins, and one more = not seen
UNSEEN = len(DIST) + 1
SPEED = np.array([2.0, 2.75, 3.25, 3.75])                # ball speed (m per step) on the 2nd step of a pass: 5 bins


class Evidence:
    """What our team sees, step by step, in the form the maps count. Pooled: one of us seeing a player is enough.
    A pass is noticed when the ball leaves an opponent we see on the ball, and timed on its second step in the air."""

    def __init__(self):
        self.prev_ball, self.prev_poss, self.holder, self.flight = None, None, None, 0

    def step(self, obs_by_player: dict):
        """Returns (distance bin of every opponent player, ball state, pass) where pass = (player, speed bin) or None."""
        o0 = next(iter(obs_by_player.values()))
        ball, poss = o0.ball_pos, int(o0.possession)
        seen = {}
        for o in obs_by_player.values():
            for j in np.flatnonzero(o.opp_visible):
                seen[int(j)] = o.opp_pos[j]
        bins = np.full(5, UNSEEN)
        for j, p in seen.items():
            bins[j] = int(np.searchsorted(DIST, np.linalg.norm(p - ball), side="right"))
        event = None
        if poss == OTHER:                                       # who has it: one of them we see, on the ball
            near = min(((np.linalg.norm(p - ball), j) for j, p in seen.items()), default=None)
            self.holder = near[1] if near is not None and near[0] < 2.0 else None
            self.flight = 0
        elif poss == LOOSE and self.holder is not None and self.prev_poss in (OTHER, LOOSE):
            self.flight += 1
            if self.flight == 2:
                event = (self.holder, int(np.searchsorted(SPEED, np.linalg.norm(ball - self.prev_ball), side="right")))
                self.holder = None
        else:
            self.holder, self.flight = None, 0
        self.prev_ball, self.prev_poss = ball.copy(), poss
        return bins, poss, event


class PlaymakerMaps:
    def __init__(self, pos=None, passes=None):
        self.pos = np.zeros((5, 2, 3, UNSEEN + 1)) if pos is None else pos       # player, playmaker?, ball state, bin
        self.passes = np.zeros((5, 2, len(SPEED) + 1)) if passes is None else passes   # player, playmaker?, speed

    def add(self, sigma: int, bins, poss: int, event):
        """One step of a practice match whose playmaker is `sigma` (0 = none)."""
        pm = (np.arange(5) == sigma - 1).astype(int)
        self.pos[np.arange(5), pm, poss, bins] += 1
        if event is not None:
            self.passes[event[0], pm[event[0]], event[1]] += 1

    def log_probs(self, prior: float = 1.0):
        p, q = self.pos + prior, self.passes + prior
        return np.log(p / p.sum(-1, keepdims=True)), np.log(q / q.sum(-1, keepdims=True))

    def hypotheses(self):
        """0 (no playmaker) and every player number that was the playmaker in practice."""
        return [0] + [j + 1 for j in range(5) if self.pos[j, 1].sum() > 0]

    def save(self, path):
        np.savez(path, pos=self.pos, passes=self.passes)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["pos"], z["passes"])


class PlaymakerFilter:
    def __init__(self, maps: PlaymakerMaps, beta: float, beta_pass: float, use_speed: bool = True):
        self.hyp = maps.hypotheses()
        self.lp_pos, self.lp_pass = maps.log_probs()
        self.beta, self.beta_pass, self.use_speed = float(beta), float(beta_pass), bool(use_speed)
        self.is_pm = np.array([[int(h == j + 1) for j in range(5)] for h in self.hyp])   # (hypotheses, players)
        self.reset()

    def reset(self):
        self.belief = np.full(len(self.hyp), 1.0 / len(self.hyp))

    def loglik(self, bins, poss: int, event):
        """(position part, pass part) of the log-likelihood of one step under every hypothesis."""
        lp = self.lp_pos[np.arange(5)[None, :], self.is_pm, poss, np.asarray(bins)[None, :]].sum(1)
        lq = np.zeros(len(self.hyp))
        if self.use_speed and event is not None:
            lq = self.lp_pass[event[0], self.is_pm[:, event[0]], event[1]]
        return lp, lq

    def update(self, bins, poss: int, event):
        """One step. Returns (the most likely hypothesis: 0 = no playmaker, else his number; its probability)."""
        lp, lq = self.loglik(bins, poss, event)
        lb = np.log(self.belief) + self.beta * lp + self.beta_pass * lq
        b = np.exp(lb - lb.max())
        self.belief = b / b.sum()
        k = int(np.argmax(self.belief))
        return self.hyp[k], float(self.belief[k])
