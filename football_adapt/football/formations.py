"""Formations: slot types (D/M/F) and anchors. Anchors are fractions of the pitch in the team frame."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Formation:
    name: str
    slot_types: tuple          # e.g. ("D", "D", "M", "M", "F"), player 1..5 in order
    anchors_frac: np.ndarray   # shape (5, 2), team frame, fractions of the pitch


def slot_types_from_name(name: str) -> tuple:
    d, m, f = (int(x) for x in name.split("-"))
    assert d + m + f == 5, f"formation {name} must have 5 outfield players"
    return tuple(["D"] * d + ["M"] * m + ["F"] * f)


class FormationBook:
    """All formations available in this experiment."""

    def __init__(self, cfg):
        self.L, self.W = cfg.pitch.length, cfg.pitch.width
        self.formations = {}
        for name, anchors in cfg.formations.anchors.to_dict().items():
            a = np.asarray(anchors, float)
            assert a.shape == (5, 2), f"{name}: need 5 anchors"
            self.formations[name] = Formation(name, slot_types_from_name(name), a)
        self.names = list(self.formations.keys())
        self.phases = cfg.formations.phases

    def __getitem__(self, name) -> Formation:
        return self.formations[name]

    def index(self, name) -> int:
        return self.names.index(name)

    def anchors_team_frame(self, name, phase: float = 0.0, phases=None):
        """Anchors in the team frame, world units. Shape (5, 2).

        phase in [-1, 1] blends the base shape toward the defending shape (-1) or the attacking shape (+1).
        Each shape keeps the same slots but puts every line (D/M/F) at its own depth and rescales the width.
        phases = the shapes to use (a team's play style can have its own); default = the config's."""
        f = self.formations[name]
        a = f.anchors_frac.copy()
        ph = self.phases if phases is None else phases
        if phase != 0.0 and ph.enabled:
            sh = ph.attack if phase > 0 else ph.defend
            tgt = np.array([[sh.x[t], 0.5 + (y - 0.5) * sh.y_scale] for t, (_, y) in zip(f.slot_types, a)])
            a += min(abs(phase), 1.0) * (tgt - a)
        return a * np.array([self.L, self.W])

    def kickoff_slot(self, name) -> int:
        """Player index (0-based) who takes the kickoff: the first forward."""
        return self.formations[name].slot_types.index("F")
