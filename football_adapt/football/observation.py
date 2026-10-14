"""Observation building. ONE function is used for both teams, so nobody sees more than their radius allows.

All positions in `Obs` are in the observing player's TEAM FRAME (each team attacks toward +x)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .actions import Kind
from .geometry import dist, to_frame, closest_point_on_segment
from .mechanics import control_rating, pass_interception

OWN, OTHER, LOOSE = 0, 1, 2   # possession from the observer's point of view


@dataclass
class Obs:
    team: int
    idx: int                       # 0-based player index (player number = idx + 1)
    pos: np.ndarray                # own position, team frame
    anchor: np.ndarray | None      # own anchor (None if structure level 0)
    control: float
    has_ball: bool
    frozen: bool
    teammate_ids: list             # 4 ids (0-based), ascending, self excluded
    teammate_pos: np.ndarray       # (4, 2)
    teammate_has_ball: np.ndarray  # (4,)
    opp_pos: np.ndarray            # (5, 2), zeros where not visible
    opp_visible: np.ndarray        # (5,) bool
    ball_pos: np.ndarray           # (2,), zeros if not visible
    ball_visible: bool
    possession: int                # OWN / OTHER / LOOSE
    score_diff: int                # own goals minus the other team's goals
    t: int
    t_frac: float
    formation: str                 # own formation name
    slot_type: str                 # "D", "M" or "F"
    radius: float
    mask: np.ndarray = field(default=None)       # (n_actions,) bool
    incoming: np.ndarray | None = None           # where a pass in flight will land, if this player is the one
                                                 # who gets it (receiver or interceptor). Not in the flat vector.
    prev_action: int = -1


def open_pass(env, team: int, a, b, opp_visible) -> bool:
    """Can the pass a -> b (world frame) be played without a defender the passer SEES getting to the ball?

    The same two tests the scripted team makes before it passes: no visible defender intercepts even with
    `slack` extra steps in hand, and none is within `lane` m of the cut-out radius of the ball's path (the first
    metres next to the passer do not count). A defender the passer cannot see can still take the ball."""
    op, m, sp = env.cfg.actions.open_pass, env.cfg.mechanics, env.cfg.speeds
    defs = [(j, env.pos[1 - team][j]) for j in range(5) if opp_visible[j]]
    if not defs:
        return True
    if pass_interception(a, b, defs, sp["pass"], sp.player, m.intercept_margin + op.slack) is not None:
        return False
    fc, n = m.flight_cut, dist(a, b)
    if n <= fc.free or fc.radius <= 0:
        return True
    start = a + (b - a) * (fc.free / n)
    return all(dist(d, closest_point_on_segment(d, start, b)) > fc.radius + op.lane for _, d in defs)


def compute_mask(env, team: int, idx: int, obs_partial: dict) -> np.ndarray:
    sp = env.space
    m = np.zeros(sp.n, bool)
    if obs_partial["frozen"] or obs_partial["receiving"]:      # nothing to choose: frozen, or meeting a pass (env.step)
        m[sp.index_of_kind(Kind.STAY)] = True
        return m
    cfg = env.cfg.mechanics
    has_ball = obs_partial["has_ball"]
    for i in range(sp.n):
        a = sp.decode(i)
        k = a.kind
        if k in (Kind.MOVE_DIR, Kind.STAY):
            ok = True
        elif k == Kind.HOLD_SHAPE:
            ok = obs_partial["anchor"] is not None
        elif k == Kind.GO_TO_BALL:
            ok = obs_partial["ball_visible"]
        elif k == Kind.DRIBBLE_GOAL:
            ok = has_ball
        elif k == Kind.PASS:
            ok = has_ball
            # a learning team is only offered the passes it can see are open (the scripted team checks for itself)
            if ok and env.cfg.actions.open_pass.enabled and env.controllers.get(team) is None:
                mate = [j for j in range(5) if j != idx][a.arg]
                seen = obs_partial["opp_visible"]
                if env.cfg.actions.open_pass.call:         # the receiver calls: the defenders HE sees count too
                    rho = env.vision_radius(team, mate)
                    seen = seen | np.array([dist(env.pos[1 - team][j], env.pos[team][mate]) <= rho for j in range(5)])
                ok = open_pass(env, team, env.pos[team][idx], env.pos[team][mate], seen)
        elif k == Kind.SHOOT:
            ok = has_ball and obs_partial["in_shoot_zone"]
            # a learning player is offered only the shot that is the best he can do: close in, or a real chance he
            # could not carry closer (actions.early_shot; judged with the defenders he sees)
            if ok and env.cfg.actions.early_shot.enabled and env.controllers.get(team) is None:
                L, W = env.L, env.W
                defs = [to_frame(team, env.pos[1 - team][j], L, W) for j in range(5) if obs_partial["opp_visible"][j]]
                ok = not env.early_shot(to_frame(team, env.pos[team][idx], L, W), defs, env.control(team, idx))
        elif k == Kind.TACKLE:
            ok = obs_partial["opp_carrier_close"]
        elif k == Kind.SHADOW:
            ok = bool(obs_partial["opp_visible"][a.arg])
        else:
            ok = False
        # a learning player is offered only the choices that belong to his situation: no HOLD_SHAPE, GO_TO_BALL or
        # SHADOW on the ball, no steps in a bare direction off it
        if ok and env.cfg.actions.context_mask and env.controllers.get(team) is None:
            if has_ball:
                ok = k not in (Kind.HOLD_SHAPE, Kind.GO_TO_BALL, Kind.SHADOW)
            else:
                ok = k != Kind.MOVE_DIR
        m[i] = ok
    # take the shot (actions.early_shot.take): a defender is closing in and the shot is on, so a learning carrier
    # shoots or passes now. Carrying on only lets the chance fall and the defender take the ball
    es = env.cfg.actions.early_shot
    if (es.enabled and es.take and has_ball and env.controllers.get(team) is None
            and m[sp.index_of_kind(Kind.SHOOT)]
            and any(obs_partial["opp_visible"][j] and dist(env.pos[1 - team][j], env.pos[team][idx]) <= es.pressed
                    for j in range(5))):
        for i in range(sp.n):
            if sp.decode(i).kind in (Kind.MOVE_DIR, Kind.STAY, Kind.DRIBBLE_GOAL):
                m[i] = False
    return m


def build_observation(env, team: int, idx: int) -> Obs:
    L, W = env.L, env.W
    other = 1 - team
    me_w = env.pos[team][idx]
    me = to_frame(team, me_w, L, W)
    rho = env.vision_radius(team, idx)

    tids = [k for k in range(5) if k != idx]
    tm_pos = np.array([to_frame(team, env.pos[team][k], L, W) for k in tids])
    tm_has = np.array([env.holder == (team, k) for k in tids])

    opp_pos = np.zeros((5, 2))
    opp_vis = np.zeros(5, bool)
    for j in range(5):
        if dist(env.pos[other][j], me_w) <= rho:
            opp_pos[j] = to_frame(team, env.pos[other][j], L, W)
            opp_vis[j] = True

    # everyone knows where the ball is (vision.ball_always); the radius hides opponents only
    ball_vis = env.cfg.vision.ball_always or bool(dist(env.ball_pos, me_w) <= rho) or env.holder == (team, idx)
    ball_pos = to_frame(team, env.ball_pos, L, W) if ball_vis else np.zeros(2)

    if env.holder is None:
        poss = LOOSE
    else:
        poss = OWN if env.holder[0] == team else OTHER

    anchor = to_frame(team, env.anchors[team][idx], L, W) if env.zeta[team] >= 1 else None
    ctrl = env.control(team, idx)
    has_ball = env.holder == (team, idx)
    frozen = bool(env.frozen[team][idx] > 0)

    goal_c = np.array([L, W / 2])
    in_zone = dist(me, goal_c) <= env.cfg.mechanics.shoot_max
    opp_close = (env.holder is not None and env.holder[0] == other and
                 env.steal_chain < env.cfg.mechanics.max_steal_chain and
                 dist(env.pos[other][env.holder[1]], me_w) <= env.cfg.mechanics.tackle_dist)

    incoming = None
    if env.ball_state == "flight" and env.flight is not None and env.flight["designated"] == (team, idx):
        incoming = to_frame(team, env.flight["end"], L, W)
    partial = dict(frozen=frozen, has_ball=has_ball, anchor=anchor, ball_visible=ball_vis,
                   in_shoot_zone=in_zone, opp_carrier_close=opp_close, opp_visible=opp_vis,
                   receiving=incoming is not None and env.controllers.get(team) is None)
    o = Obs(team=team, idx=idx, pos=me, anchor=anchor, control=ctrl, has_ball=has_ball, frozen=frozen,
            teammate_ids=tids, teammate_pos=tm_pos, teammate_has_ball=tm_has,
            opp_pos=opp_pos, opp_visible=opp_vis, ball_pos=ball_pos, ball_visible=ball_vis,
            possession=poss, score_diff=env.score[team] - env.score[other], t=env.t,
            t_frac=env.t / env.T, formation=env.formation[team],
            slot_type=env.book[env.formation[team]].slot_types[idx], radius=rho,
            prev_action=int(env.prev_action[team][idx]))
    o.mask = compute_mask(env, team, idx, partial)
    o.incoming = incoming
    return o


def flat_dim(env) -> int:
    return len(flatten(env, build_observation(env, 0, 0)))


def flatten(env, o: Obs) -> np.ndarray:
    """Flat vector for the learner. Relative positions, normalized by the pitch size."""
    L, W = env.L, env.W
    scale = np.array([L, W])
    v = []
    v += list(o.pos / scale)
    pa = np.zeros(env.space.n + 1)
    pa[o.prev_action if o.prev_action >= 0 else env.space.n] = 1.0     # last slot = "none / macro"
    v += list(pa)
    v += [float(o.has_ball), float(o.frozen)]
    oh = np.zeros(5); oh[o.idx] = 1.0; v += list(oh)                    # slot one-hot (player number)
    st = np.zeros(3); st["DMF".index(o.slot_type)] = 1.0; v += list(st)  # slot type one-hot
    if o.anchor is not None:
        v += list((o.anchor - o.pos) / scale) + [o.control]
    else:
        v += [0.0, 0.0, 0.0]                                            # zeros if structure level 0
    fo = np.zeros(len(env.book.names)); fo[env.book.index(o.formation)] = 1.0; v += list(fo)
    for k in range(4):
        v += list((o.teammate_pos[k] - o.pos) / scale) + [float(o.teammate_has_ball[k])]
    for j in range(5):
        if o.opp_visible[j]:
            v += list((o.opp_pos[j] - o.pos) / scale) + [1.0]
        else:
            v += [0.0, 0.0, 0.0]
    if o.ball_visible:
        v += list((o.ball_pos - o.pos) / scale) + [1.0]
    else:
        v += [0.0, 0.0, 0.0]
    p = np.zeros(3); p[o.possession] = 1.0; v += list(p)
    v += [float(np.clip(o.score_diff, -3, 3)) / 3.0, o.t_frac]
    return np.asarray(v, np.float32)


def global_state(env) -> np.ndarray:
    """Full state for the centralized critic (training only). Team-0 frame."""
    L, W = env.L, env.W
    scale = np.array([L, W])
    v = []
    for team in (0, 1):
        for i in range(5):
            v += list(to_frame(0, env.pos[team][i], L, W) / scale)
    v += list(env.ball_pos / scale)
    h = np.zeros(11)
    if env.holder is None:
        h[10] = 1.0
    else:
        h[env.holder[0] * 5 + env.holder[1]] = 1.0
    v += list(h)
    for team in (0, 1):
        fo = np.zeros(len(env.book.names)); fo[env.book.index(env.formation[team])] = 1.0
        v += list(fo)
    sg = np.zeros(6); sg[env.sigma] = 1.0; v += list(sg)
    v += [float(np.clip(env.score[0] - env.score[1], -3, 3)) / 3.0, env.t / env.T]
    return np.asarray(v, np.float32)
