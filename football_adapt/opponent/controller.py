"""ScriptedTeam: the hand-written team (opponent.md). Also usable to drive OUR side as a baseline.

Rules of the road:
  * It receives ONLY observations (built by the shared observation function), never the full env state.
  * Everything is in the team frame (the team attacks toward +x).
  * Priority list per player: transition walk, ball carrier, support, loose ball, zonal defence.
  * No man-marking. No learning. Deterministic given the seed (separate random stream).
"""
from __future__ import annotations
import numpy as np

from football.actions import Action, ActionSpace, Kind, STAY
from football.geometry import dist, clip_norm, unit
from football.mechanics import shot_probability, pass_interception
from football.observation import OWN, OTHER, LOOSE
from .formation_manager import FormationManager


def MOVE_TO(p):
    return Action(Kind.MOVE_TO, np.asarray(p, float))


class ScriptedTeam:
    def __init__(self, cfg, rng, team: int, formation: str, playmaker=None, mode_override=None):
        self.cfg = cfg
        self.rng = rng
        self.team = team
        self.L, self.W = cfg.pitch.length, cfg.pitch.width
        self.z = cfg.opponent.zone
        self.at = cfg.opponent.attack
        self.su = cfg.opponent.support
        self.eps_opp = cfg.opponent.randomness.eps_opp
        self.m = cfg.mechanics
        self.v_pl, self.v_pa = cfg.speeds.player, cfg.speeds["pass"]
        self.margin = cfg.mechanics.intercept_margin
        self.playmaker = playmaker          # player index 0..4, or None
        self.goal = np.array([self.L, self.W / 2])
        names = list(cfg.formations.anchors.to_dict().keys())
        self.fm = FormationManager(cfg, rng, formation, names, mode_override)
        self.in_transition = False
        self.transition_steps = 0
        self._tackle_idx = ActionSpace(cfg.actions.n_directions).index_of_kind(Kind.TACKLE)

    # ------------------------------------------------------------------ interface used by the env
    def update_formation(self, t, own_goals, other_goals):
        new = self.fm.update(t, own_goals, other_goals)
        if new is not None:
            self.in_transition = True
            self.transition_steps = 0
        return new

    def act(self, t, obs_by_player: dict) -> dict:
        ctx = self._team_context(obs_by_player)
        if self.in_transition:
            self.transition_steps += 1
            settled = all(dist(o.pos, o.anchor) <= self.z.eps_arrive for o in obs_by_player.values())
            if settled or self.transition_steps >= self.z.transition_max:
                self.in_transition = False
        return {j: self._decide(j, o, ctx) for j, o in obs_by_player.items()}

    # ------------------------------------------------------------------ team-level information
    def _team_context(self, obs):
        pos = np.zeros((5, 2))
        anchors = np.zeros((5, 2))
        for j, o in obs.items():
            pos[j] = o.pos
            anchors[j] = o.anchor
        ball = None
        for o in obs.values():
            if o.ball_visible:
                ball = o.ball_pos
                break
        poss = next(iter(obs.values())).possession
        presser = nearest = None
        if ball is not None:
            seers = [j for j, o in obs.items() if o.ball_visible]
            cands = [j for j in seers if dist(ball, anchors[j]) <= self.z.r_zone]
            if cands:
                presser = min(cands, key=lambda j: (dist(pos[j], ball), j))
            nearest = min(range(5), key=lambda j: (dist(pos[j], ball), j))
        return dict(pos=pos, anchors=anchors, ball=ball, poss=poss, presser=presser, nearest=nearest)

    # ------------------------------------------------------------------ per-player priority list
    def _decide(self, j, o, ctx):
        if self.eps_opp > 0 and self.rng.random() < self.eps_opp:
            valid = np.flatnonzero(o.mask)
            return STAY if len(valid) == 0 else int(self.rng.choice(valid))   # an index into the action set
        if self.in_transition and not o.has_ball and dist(o.pos, o.anchor) > self.z.eps_arrive:
            return MOVE_TO(o.anchor)
        if o.has_ball:
            return self._attack_with_ball(o)
        if o.possession == OWN:
            return self._support(j, o, ctx)
        if o.possession == LOOSE:
            return self._loose_ball(j, o, ctx)
        return self._zonal(j, o, ctx)

    # ---- defence --------------------------------------------------------------------------------
    def _zonal(self, j, o, ctx):
        anchor = o.anchor
        if not o.ball_visible:
            return MOVE_TO(anchor)                                   # hold the home position
        ball = o.ball_pos
        if ctx["presser"] == j:                                      # the single presser
            if o.mask[self._tackle_idx] and o.possession == OTHER:
                return Action(Kind.TACKLE)
            return MOVE_TO(ball)
        return MOVE_TO(anchor + self.z.omega * clip_norm(ball - anchor, self.z.r_zone))  # small ball-side drift

    def _loose_ball(self, j, o, ctx):
        if o.ball_visible and ctx["nearest"] == j:
            return MOVE_TO(o.ball_pos)
        return self._zonal(j, o, ctx)

    # ---- attack ---------------------------------------------------------------------------------
    def _visible_defenders(self, o):
        return [o.opp_pos[k] for k in range(5) if o.opp_visible[k]]

    def _attack_with_ball(self, o):
        defs = self._visible_defenders(o)
        pos = o.pos
        m = self.m
        # 1. shoot
        if dist(pos, self.goal) <= m.shoot_max:
            p = shot_probability(pos, self.goal, defs, o.control, m.p_max, m.d0, m.shot_block_scale)
            if p >= self.at.shoot_min:
                return Action(Kind.SHOOT)
        # 2. pass: score every teammate, keep only passes with no visible interceptor
        cands = []
        for k, tid in enumerate(o.teammate_ids):
            tp = o.teammate_pos[k]
            ic = pass_interception(pos, tp, list(enumerate(defs)), self.v_pa, self.v_pl, self.margin)
            if ic is not None:
                continue
            open_ = (min(dist(d, tp) for d in defs) if defs else self.at.open_max)
            open_ = min(open_, self.at.open_max) / self.at.open_max
            fwd = (tp[0] - pos[0]) / self.L
            score = (self.at.w_fwd * fwd + self.at.w_open * open_ - self.at.w_dist * dist(pos, tp) / self.L
                     + (self.at.hub_bonus if tid == self.playmaker else 0.0))
            cands.append((score, open_, k))
        if cands:
            best = max(cands)
            if best[0] >= self.at.pass_min:
                return Action(Kind.PASS, best[2])
        # 3. dribble if no visible defender is close ahead
        ahead = [d for d in defs if d[0] > pos[0] and dist(d, pos) <= self.at.d_danger]
        if not ahead:
            return Action(Kind.DRIBBLE_GOAL)
        # 4. recycle: safest allowed pass, even backward
        if cands:
            return Action(Kind.PASS, max(cands, key=lambda c: (c[1], c[2]))[2])
        return STAY

    # ---- off-ball attacking support ------------------------------------------------------------------
    def _support(self, j, o, ctx):
        carrier = None
        for k, tid in enumerate(o.teammate_ids):
            if o.teammate_has_ball[k]:
                carrier = o.teammate_pos[k]
        shift = np.zeros(2)
        if o.slot_type == "F":
            shift = np.array([self.su.adv, 0.0])
        elif o.slot_type == "M" and carrier is not None:
            shift = clip_norm(carrier - o.anchor, self.su.mid)
        elif o.slot_type == "D":
            shift = np.array([self.su["def"], 0.0])
        return MOVE_TO(o.anchor + clip_norm(shift, self.su.r_support))
