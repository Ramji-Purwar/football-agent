"""A team played by a few hand-written rules. It uses the learner's own observation, mask and 23 actions and nothing
else, so whatever it does a trained team can do too. It is the teacher of the rule-based start (training/pretrain.py).

   on the ball                SHOOT when the chance is `shoot_min` or better. If not, the open pass that gains the
                              most ground (`gain_min` m or more). If there is none and a defender is within
                              `release_dist` m, the open pass that gains the most ground (or loses the least) to a
                              teammate with `room_min` m or more from the defenders he can see: keep the ball instead
                              of being tackled. Otherwise DRIBBLE_GOAL. release_dist=0 gives the rules before
                              2026-10-14 (runs/warm2.pt)
   we have the ball           HOLD_SHAPE (also while our pass is in the air)
   they have it, or it is     the `n_press` players nearest the ball go for it: TACKLE when in range, GO_TO_BALL
   loose                      when not. The others HOLD_SHAPE
"""
from __future__ import annotations

import numpy as np

from football.actions import Kind
from football.geometry import dist
from football.mechanics import shot_probability, sure_goal


def rule_actions(env, n_press: int = 2, shoot_min: float = 0.3, gain_min: float = 5.0,
                 release_dist: float = 3.0, room_min: float = 5.0) -> dict:
    """One action for each of our five players, from the observations of the step that is about to be played."""
    sp, m = env.space, env.cfg.mechanics
    I = {k: sp.index_of_kind(k) for k in (Kind.STAY, Kind.HOLD_SHAPE, Kind.GO_TO_BALL, Kind.DRIBBLE_GOAL, Kind.SHOOT,
                                          Kind.TACKLE, Kind.PASS)}
    goal = np.array([env.L, env.W / 2])
    obs = env.last_obs
    h = env.holder
    ours = (h is not None and h[0] == 0) or (h is None and env.ball_state == "flight" and env.flight["passer"][0] == 0)
    order = sorted(range(5), key=lambda j: dist(obs[j].pos, obs[j].ball_pos))
    acts = {}
    for j in range(5):
        o = obs[j]
        mask = o.mask
        if mask.sum() <= 1:                        # frozen, or meeting a pass: nothing to choose
            acts[j] = int(np.flatnonzero(mask)[0])
            continue
        if o.has_ball:
            defs = [o.opp_pos[k] for k in range(5) if o.opp_visible[k]]
            p = (shot_probability(o.pos, goal, defs, o.control, m.p_max, m.d0, m.shot_block_scale, sure_goal(m))
                 if mask[I[Kind.SHOOT]] else 0.0)
            best, safe = None, None
            for k in range(4):                     # the mask has already left out the passes that are not open
                if mask[I[Kind.PASS] + k]:
                    gain = o.teammate_pos[k][0] - o.pos[0]
                    if gain >= gain_min and (best is None or gain > best[0]):
                        best = (gain, k)
                    room = min([dist(d, o.teammate_pos[k]) for d in defs] + [np.inf])
                    if room >= room_min and (safe is None or (gain, room) > safe[0]):
                        safe = ((gain, room), k)
            pressed = any(dist(d, o.pos) <= release_dist for d in defs)
            if p >= shoot_min:
                acts[j] = I[Kind.SHOOT]
            elif best:
                acts[j] = I[Kind.PASS] + best[1]
            elif safe and pressed:
                acts[j] = I[Kind.PASS] + safe[1]
            else:
                acts[j] = I[Kind.DRIBBLE_GOAL]
        elif ours or j not in order[:n_press]:
            acts[j] = I[Kind.HOLD_SHAPE]
        else:
            acts[j] = I[Kind.TACKLE] if mask[I[Kind.TACKLE]] else I[Kind.GO_TO_BALL]
        if not mask[acts[j]]:                      # e.g. GO_TO_BALL for a ball he cannot see (vision.ball_always: false)
            acts[j] = I[Kind.STAY]
    return acts
