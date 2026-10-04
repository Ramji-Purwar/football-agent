"""Small 2D geometry helpers (numpy arrays of shape (2,))."""
from __future__ import annotations
import numpy as np

EPS = 1e-9


def dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def norm(v) -> float:
    return float(np.hypot(v[0], v[1]))


def unit(v):
    n = norm(v)
    return np.asarray(v, float) / n if n > EPS else np.zeros(2)


def clip_norm(v, r):
    """Limit the length of vector v to r."""
    v = np.asarray(v, float)
    n = norm(v)
    return v if n <= r or n < EPS else v * (r / n)


def move_toward(pos, target, step):
    """Move from pos toward target by at most step."""
    pos = np.asarray(pos, float)
    d = np.asarray(target, float) - pos
    n = norm(d)
    if n <= step:
        return np.asarray(target, float).copy()
    return pos + d * (step / n)


def closest_point_on_segment(p, a, b):
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a
    l2 = float(ab @ ab)
    if l2 < EPS:
        return a.copy()
    t = float(np.clip(((p - a) @ ab) / l2, 0.0, 1.0))
    return a + t * ab


def dist_point_segment(p, a, b) -> float:
    return dist(p, closest_point_on_segment(p, a, b))


def rotate(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def to_frame(team: int, p, L: float, W: float):
    """World coordinates <-> team frame. Team 0 is identity. Team 1 is rotated 180 degrees.
    The map is its own inverse, so the same function converts both ways."""
    p = np.asarray(p, float)
    if team == 0:
        return p.copy()
    return np.array([L - p[0], W - p[1]])


def dir_to_frame(team: int, v):
    """Convert a DIRECTION vector between world and team frame (self-inverse)."""
    v = np.asarray(v, float)
    return v.copy() if team == 0 else -v
