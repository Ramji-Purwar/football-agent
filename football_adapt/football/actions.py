"""Action space. One place for all action indexing (the mask and the policy both depend on it)."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


class Kind(Enum):
    MOVE_DIR = auto()      # arg = direction index 0..D-1
    STAY = auto()
    HOLD_SHAPE = auto()
    GO_TO_BALL = auto()
    PASS = auto()          # arg = teammate index 0..3 (teammates in ascending player number, self excluded)
    DRIBBLE_GOAL = auto()
    SHOOT = auto()
    TACKLE = auto()
    SHADOW = auto()        # arg = opponent index 0..4
    MOVE_TO = auto()       # macro for scripted controllers: arg = point in TEAM frame (not in the learner's set)
    THROUGH = auto()       # macro for scripted controllers: through ball. arg = [teammate index 0..3, x, y] with the
                           # point in TEAM frame; the ball is played to that point for the teammate to run onto
    CLEAR = auto()         # macro for scripted controllers: clearance. arg = point in TEAM frame to aim at (None =
                           # straight upfield); the ball flies up to mechanics.clear.dist m and lands loose


@dataclass(frozen=True)
class Action:
    kind: Kind
    arg: object = None

    def __hash__(self):
        a = self.arg
        if hasattr(a, "tolist"):
            a = tuple(a.tolist())
        return hash((self.kind, a))

    def __eq__(self, other):
        return isinstance(other, Action) and self.kind == other.kind and _argeq(self.arg, other.arg)


def _argeq(a, b):
    if a is None or b is None:
        return a is b
    if hasattr(a, "tolist") or hasattr(b, "tolist"):
        return tuple(a.tolist() if hasattr(a, "tolist") else [a]) == tuple(b.tolist() if hasattr(b, "tolist") else [b])
    return a == b


STAY = Action(Kind.STAY)


class ActionSpace:
    """Index layout: MOVE_DIR x D, STAY, HOLD_SHAPE, GO_TO_BALL, PASS x4, DRIBBLE_GOAL, SHOOT, TACKLE, SHADOW x5."""

    def __init__(self, n_directions: int):
        self.D = n_directions
        t = [Action(Kind.MOVE_DIR, d) for d in range(self.D)]
        t += [Action(Kind.STAY), Action(Kind.HOLD_SHAPE), Action(Kind.GO_TO_BALL)]
        t += [Action(Kind.PASS, k) for k in range(4)]
        t += [Action(Kind.DRIBBLE_GOAL), Action(Kind.SHOOT), Action(Kind.TACKLE)]
        t += [Action(Kind.SHADOW, j) for j in range(5)]
        self._table = t
        self.n = len(t)
        self._index = {}
        for i, a in enumerate(t):
            self._index[(a.kind, a.arg)] = i

    def decode(self, idx: int) -> Action:
        return self._table[int(idx)]

    def index_of(self, action: Action):
        """Index in the learner's action set, or None for macros not in it (MOVE_TO)."""
        if action.kind in (Kind.MOVE_TO, Kind.THROUGH, Kind.CLEAR):
            return None
        return self._index.get((action.kind, action.arg))

    def index_of_kind(self, kind):
        """Index of the first action of this kind (used for STAY)."""
        for i, a in enumerate(self._table):
            if a.kind == kind:
                return i
        raise KeyError(kind)

    def names(self):
        return [a.kind.name if a.arg is None else f"{a.kind.name}_{a.arg}" for a in self._table]
