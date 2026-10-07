"""Pure physics functions. Shared by the environment and the opponent, so both use identical rules."""
from __future__ import annotations
import numpy as np
from .geometry import dist, dist_point_segment, closest_point_on_segment


def control_rating(pos, anchor, zeta: int, c_min: float, l_ctrl: float) -> float:
    """c = C_MIN + (1 - C_MIN) * exp(-||x - anchor||^2 / L^2) at structure level 2, else 1."""
    if zeta < 2 or anchor is None:
        return 1.0
    d2 = float(np.sum((np.asarray(pos, float) - np.asarray(anchor, float)) ** 2))
    return float(c_min + (1.0 - c_min) * np.exp(-d2 / l_ctrl ** 2))


def shot_openness(shooter, goal_center, defenders, scale: float) -> float:
    """1 when no defender is near the shot line, tending to 0 as a defender stands on it."""
    o = 1.0
    for d in defenders:
        dd = dist_point_segment(d, shooter, goal_center)
        o *= 1.0 - float(np.exp(-(dd / scale) ** 2))
    return o


def shot_probability(shooter, goal_center, defenders, c: float, p_max: float, d0: float, scale: float,
                     sure=None) -> float:
    """p = P_MAX * exp(-d / D0) * openness * c_shooter.

    sure = (depth, half_width) of the goal area: a shot from inside it always scores (there is no goalkeeper)."""
    if sure is not None and abs(shooter[0] - goal_center[0]) <= sure[0] and abs(shooter[1] - goal_center[1]) <= sure[1]:
        return 1.0
    d = dist(shooter, goal_center)
    return float(p_max * np.exp(-d / d0) * shot_openness(shooter, goal_center, defenders, scale) * c)


def pass_interception(passer, target, defenders, v_pass: float, v_player: float, margin: float):
    """Interception geometry for one pass.

    defenders: iterable of (key, position) that are ALLOWED to react.
    For each defender: q = closest point on the pass segment, t_ball = |q - passer| / V_PASS,
    t_def = |q - defender| / V_PLAYER. The defender intercepts if t_def <= t_ball + margin.
    Returns (key, q, slack) of the best interceptor (smallest t_def - t_ball), or None.
    """
    best = None
    passer = np.asarray(passer, float)
    for key, pos in defenders:
        q = closest_point_on_segment(pos, passer, target)
        slack = dist(q, pos) / v_player - dist(q, passer) / v_pass
        if slack <= margin and (best is None or slack < best[2]):
            best = (key, q, slack)
    return best


def tackle_probability(c_def: float, c_carrier: float) -> float:
    """p_win = c_defender / (c_defender + c_carrier)."""
    return c_def / (c_def + c_carrier)


def sure_goal(m):
    """The goal area as (depth, half_width) for shot_probability, or None if the rule is off."""
    sg = m.sure_goal
    return (sg.depth, sg.half_width) if sg.enabled else None
