"""ScriptedTeam: the hand-written team (opponent.md). Also usable to drive OUR side as a baseline.

Rules of the road:
  * It receives ONLY observations (built by the shared observation function), never the full env state.
  * Everything is in the team frame (the team attacks toward +x).
  * Priority list per player: transition walk, ball carrier, support, loose ball, zonal defence.
  * No man-marking (one exception: cutting passing lanes, cfg.opponent.lanes, used by the aggressive style).
    No learning. Deterministic given the seed (separate random stream).
"""
from __future__ import annotations
import numpy as np

from football.formations import slot_types_from_name
from football.actions import Action, ActionSpace, Kind, STAY
from football.geometry import dist, clip_norm, unit, closest_point_on_segment
from football.mechanics import shot_probability, pass_interception, sure_goal
from football.observation import OWN, OTHER, LOOSE
from .formation_manager import FormationManager


def MOVE_TO(p):
    return Action(Kind.MOVE_TO, np.asarray(p, float))


class ScriptedTeam:
    def __init__(self, cfg, rng, team: int, formation: str, playmaker=None, mode_override=None):
        self.rng = rng
        self.team = team
        self.set_cfg(cfg)
        self.playmaker = playmaker          # player index 0..4, or None
        self.goal = np.array([self.L, self.W / 2])
        names = list(cfg.formations.anchors.to_dict().keys())
        self.fm = FormationManager(cfg, rng, formation, names, mode_override)
        self.in_transition = False
        self.transition_steps = 0
        self.block = 0.0                    # team-wide x shift of the home positions (push up / drop back)
        self.spread = 0.0                   # team-wide y stretch (+ wider in attack, - tighter in defence)
        self.lat = 0.0                      # team-wide y slide toward the ball side when defending
        self.last_poss = None               # last non-loose possession, to notice when the ball is won or lost
        self.cp_timer = 0                   # counter-press: steps left of chasing the ball after losing it
        self.ca_timer = 0                   # counter-attack: steps left of breaking forward after winning it
        self.mover = None                   # the player who just passed and is now running on (pass and move)
        self.mover_timer = 0
        self.poss_steps = 0                 # how long the team has kept the ball without a break (see patience)
        self._lanes = {}                    # passing lanes cut on the last step: {player: point} (see lanes)
        self._tackle_idx = ActionSpace(cfg.actions.n_directions).index_of_kind(Kind.TACKLE)

    # ------------------------------------------------------------------ interface used by the env
    def set_cfg(self, cfg):
        """Load every tunable from cfg. cfg is this team's own view of the config, i.e. with its play style
        applied (see config.apply_style). Called again when the style changes mid-match; state is kept."""
        self.cfg = cfg
        self.L, self.W = cfg.pitch.length, cfg.pitch.width
        self.z = cfg.opponent.zone
        self.at = cfg.opponent.attack
        self.su = cfg.opponent.support
        self.bl = cfg.opponent.block
        self.run = cfg.opponent.run
        self.wd = cfg.opponent.width
        self.cv = cfg.opponent.cover
        self.ro = cfg.opponent.roles
        self.cr = cfg.opponent.carrier
        self.sp = cfg.opponent.space
        self.mv = cfg.opponent.moves
        self.pt = cfg.opponent.patience
        self.ln = cfg.opponent.lanes
        self.cl = cfg.opponent.clear
        self.eps_opp = cfg.opponent.randomness.eps_opp
        self.m = cfg.mechanics
        self.v_pl, self.v_pa = cfg.speeds.player, cfg.speeds["pass"]
        self.margin = cfg.mechanics.intercept_margin

    def update_formation(self, t, own_goals, other_goals):
        new = self.fm.update(t, own_goals, other_goals)
        if new is not None:
            self.in_transition = True
            self.transition_steps = 0
        return new

    def set_formation(self, name):
        """The formation was changed from outside (env.set_formation): walk to the new anchors."""
        self.fm.formation, self.fm.dwell = name, 0
        self.in_transition = True
        self.transition_steps = 0

    def act(self, t, obs_by_player: dict) -> dict:
        self._update_timers(next(iter(obs_by_player.values())).possession)
        ctx = self._team_context(obs_by_player)
        self._update_shape(ctx)
        if self.in_transition:
            self.transition_steps += 1
            settled = all(dist(o.pos, o.anchor) <= self.z.eps_arrive for o in obs_by_player.values())
            if settled or self.transition_steps >= self.z.transition_max:
                self.in_transition = False
        out = {j: self._decide(j, o, ctx) for j, o in obs_by_player.items()}
        for j, a in out.items():                                     # whoever passes now keeps running (pass and move)
            if isinstance(a, Action) and a.kind in (Kind.PASS, Kind.THROUGH) and self.mv.pass_move.steps > 0:
                self.mover, self.mover_timer = j, self.mv.pass_move.steps
        return out

    def _update_timers(self, poss):
        self.cp_timer = max(0, self.cp_timer - 1)
        self.ca_timer = max(0, self.ca_timer - 1)
        self.mover_timer = max(0, self.mover_timer - 1)
        if poss == OTHER:
            self.poss_steps = 0
        elif poss == OWN or self.last_poss == OWN:                   # we hold it, or our own pass is in the air
            self.poss_steps += 1
        if poss != LOOSE:
            if self.last_poss == OWN and poss == OTHER:              # just lost it: counter-press
                self.cp_timer, self.ca_timer, self.mover_timer = self.mv.counter_press.steps, 0, 0
            elif self.last_poss == OTHER and poss == OWN:            # just won it: counter-attack
                self.ca_timer, self.cp_timer = self.mv.counter.steps, 0
            self.last_poss = poss

    def _update_shape(self, ctx):
        """Slide the block (x), stretch it (y) and shift it toward the ball side (y). All driven by possession
        and the team's shared ball sighting, so the vision limit is respected."""
        poss = ctx["poss"]
        step = lambda cur, tgt, rate: cur + float(np.clip(tgt - cur, -rate, rate))
        tgt = self.bl.push if poss == OWN else -self.bl.drop if poss == OTHER else 0.0
        self.block = step(self.block, tgt, self.bl.rate)
        tgt = self.wd.attack if poss == OWN else -self.wd.defend if poss == OTHER else 0.0
        self.spread = step(self.spread, tgt, self.wd.rate)
        tgt = 0.0
        if poss == OTHER and ctx["ball"] is not None:
            tgt = self.wd.slide * float(np.clip((ctx["ball"][1] - self.W / 2) / (self.W / 2), -1.0, 1.0))
        self.lat = step(self.lat, tgt, self.wd.slide_rate)

    def _home(self, anchor, slot):
        """The anchor moved by the current block shift (scaled per role), width stretch and ball-side slide."""
        scale = (self.ro.push_scale if self.block >= 0 else self.ro.drop_scale)[slot]
        x = np.clip(anchor[0] + self.block * scale, 1.0, self.L - 1.0)
        y = np.clip(self.W / 2 + (anchor[1] - self.W / 2) * (1.0 + self.spread) + self.lat, 1.0, self.W - 1.0)
        return np.array([x, y])

    def _urgency(self):
        """0 while the team is happy to keep the ball, rising to 1 once it has kept it for too long without
        getting anywhere. At 1 the carrier forces the play: no more passes backward, he runs at the defence."""
        pt = self.pt
        if not pt.enabled or pt.full <= pt.start:
            return 0.0
        return float(np.clip((self.poss_steps - pt.start) / (pt.full - pt.start), 0.0, 1.0))

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
        presser = presser2 = nearest = cover = None
        chasers = []
        lanes = {}
        if ball is not None:
            seers = [j for j, o in obs.items() if o.ball_visible]
            # may press: the ball is in my zone, OR it is simply close to me (go straight for it, do not walk home first)
            cands = [j for j in seers
                     if dist(ball, anchors[j]) <= self.z.r_zone or dist(ball, pos[j]) <= self.z.engage]
            if cands:
                presser = min(cands, key=lambda j: (dist(pos[j], ball), j))
            nearest = min(range(5), key=lambda j: (dist(pos[j], ball), j))
            # cutting passing lanes comes before more pressers. It goes on while their pass is in the air: the
            # lanes are then drawn from the moving ball, so players are already in place when it arrives.
            if self.ln.enabled and (poss == OTHER or (poss == LOOSE and self.last_poss == OTHER)):
                lanes = self._assign_lanes(obs, ball, presser)
            rest = sorted((j for j in seers if j != presser and j not in lanes), key=lambda j: (dist(pos[j], ball), j))
            # double team: a second defender who is already close to the ball goes for it too (no jogging alongside)
            if presser is not None and rest and dist(pos[rest[0]], ball) <= self.z.double_dist:
                presser2 = rest.pop(0)
            if presser is not None and rest and self.cv.enabled:
                cover = rest[0]
            if self.cp_timer > 0:                                    # counter-press: the nearest few all go for the ball
                chasers = sorted((j for j in seers if j not in lanes),
                                 key=lambda j: (dist(pos[j], ball), j))[:self.mv.counter_press.n]
        self._lanes = lanes                                          # kept for inspection (tests, viewer)
        return dict(pos=pos, anchors=anchors, ball=ball, poss=poss, presser=presser, nearest=nearest, cover=cover,
                    chasers=chasers, presser2=presser2, lanes=lanes)

    def _assign_lanes(self, obs, ball, presser):
        """Who cuts which passing lane while the other team has the ball: {player: point to stand on}.

        A lane runs from the ball to a receiver the carrier could pass to: an opponent level with or ahead of the
        ball (they attack toward x = 0 in our frame). The point is `frac` of the way along it, i.e. in the middle
        between the two for 0.5. The ball position is the team's shared sighting (teammates call it), but a player
        only takes a lane to a receiver he sees himself, and only if the point is within `max_dev` (per role) of his
        home position. The presser is left out. The most dangerous lane (the receiver nearest our goal) is filled
        first, by the nearest player who can take it. One player per lane, one lane per player."""
        ln = self.ln
        options = {}                                                 # receiver -> [(distance, player, point)]
        for j, o in obs.items():
            if j == presser:
                continue
            home = self._home(o.anchor, o.slot_type)
            for k in range(5):
                if not o.opp_visible[k]:
                    continue
                r = o.opp_pos[k]
                if dist(r, ball) < 1.0 or r[0] > ball[0] + ln.side:  # the carrier himself / a receiver behind the ball
                    continue
                pt = ball + ln.frac * (r - ball)
                if dist(pt, home) <= ln.max_dev[o.slot_type]:
                    options.setdefault(k, (r[0], []))[1].append((dist(o.pos, pt), j, pt))
        lanes = {}
        for k in sorted(options, key=lambda k: (options[k][0], k)):
            free = [c for c in options[k][1] if c[1] not in lanes]
            if free:
                _, j, pt = min(free, key=lambda c: c[:2])
                lanes[j] = pt
        return lanes

    # ------------------------------------------------------------------ per-player priority list
    def _decide(self, j, o, ctx):
        if self.eps_opp > 0 and self.rng.random() < self.eps_opp:
            valid = np.flatnonzero(o.mask)
            return STAY if len(valid) == 0 else int(self.rng.choice(valid))   # an index into the action set
        if o.incoming is not None:                                   # a pass is coming to me: go and meet it
            return MOVE_TO(o.incoming)
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
        anchor = self._home(o.anchor, o.slot_type)
        ball = o.ball_pos
        # the presser, the second man of a double team, or a counter-presser: go for the ball
        if o.ball_visible and (ctx["presser"] == j
                               or (o.possession == OTHER and (ctx["presser2"] == j or j in ctx["chasers"]))):
            if o.mask[self._tackle_idx] and o.possession == OTHER:
                return Action(Kind.TACKLE)
            return MOVE_TO(ball)
        if j in ctx["lanes"]:                                        # stand on the passing lane I was given
            if o.mask[self._tackle_idx] and o.possession == OTHER:   # (the carrier ran into me: take it off him)
                return Action(Kind.TACKLE)
            return MOVE_TO(ctx["lanes"][j])
        if not o.ball_visible:
            return MOVE_TO(anchor)                                   # hold the home position
        if ctx["cover"] == j and o.possession == OTHER:              # cover: sit goal-side of the presser
            behind = ball + self.cv.dist * unit(np.array([0.0, self.W / 2]) - ball)
            return MOVE_TO(anchor + clip_norm(behind - anchor, self.cv.max_dev))
        return MOVE_TO(anchor + self.z.omega * clip_norm(ball - anchor, self.z.r_zone))  # small ball-side drift

    def _loose_ball(self, j, o, ctx):
        if o.ball_visible and ctx["nearest"] == j:
            return MOVE_TO(o.ball_pos)
        if self.last_poss == OWN and o.ball_visible:                 # our own pass is on its way: keep supporting,
            return self._support(j, o, ctx, carrier=o.ball_pos)      # following the ball, instead of standing and watching
        return self._zonal(j, o, ctx)

    # ---- attack ---------------------------------------------------------------------------------
    def _visible_defenders(self, o):
        return [o.opp_pos[k] for k in range(5) if o.opp_visible[k]]

    def _through_point(self, o, tp, defs):
        """Where to play a through ball for the teammate at tp: a point ahead of him that he reaches before the
        ball does, with space around it and no visible interceptor. None if there is no such point."""
        th = self.mv.through
        if not th.enabled or tp[0] < o.pos[0] - self.cr.back_tol:
            return None
        d = unit(self.goal - tp) if tp[0] > 0.6 * self.L else np.array([1.0, 0.0])
        lead = th.lead
        for _ in range(2):                                           # he must get there first: lead <= his run in flight
            pt = np.clip(tp + lead * d, [2.0, 2.0], [self.L - 3.0, self.W - 2.0])
            lead = min(lead, 0.8 * self.v_pl * np.ceil(dist(o.pos, pt) / self.v_pa))
        pt = np.clip(tp + lead * d, [2.0, 2.0], [self.L - 3.0, self.W - 2.0])
        if dist(pt, tp) < 3.0:
            return None
        # only when I KNOW he is free: he and the spot are inside my vision, and no defender is near either
        if max(dist(o.pos, tp), dist(o.pos, pt)) > o.radius:
            return None
        if defs and min(min(dist(x, pt), dist(x, tp)) for x in defs) < th.space:
            return None
        if not self._lane_safe(o.pos, pt, defs) or not self._path_uncuttable(o.pos, pt, defs):
            return None
        return pt

    def _path_uncuttable(self, a, b, defs):
        """Walk along the ball's path a -> b: at every point, could a visible defender have run there (to within the
        cut-out radius) by the time the ball arrives? If any can, the through ball would be cut out."""
        fc = self.m.flight_cut
        n = dist(a, b)
        if fc.radius <= 0 or n <= fc.free or not defs:
            return True
        for s in np.arange(fc.free, n + 1e-9, 1.0):
            p = a + (b - a) * (s / n)
            reach = fc.radius + self.v_pl * (s / self.v_pa) + self.mv.through.cut_margin
            if any(dist(d, p) <= reach for d in defs):
                return False
        return True

    def _lane_safe(self, a, b, defs):
        """No visible defender can get to the pass a -> b: neither by the interception rule, nor by standing or
        stepping within `lane_margin` of the ball's path (the first metres next to the passer do not count)."""
        # the game lets a defender intercept if he is at most `margin` steps late. I only pass if he would be
        # `safe_slack` steps later still: a pass that is safe by a hair gets intercepted in practice.
        if pass_interception(a, b, list(enumerate(defs)), self.v_pa, self.v_pl,
                             self.margin + self.at.safe_slack) is not None:
            return False
        fc = self.m.flight_cut
        ab = b - a
        n = dist(a, b)
        if n <= fc.free or fc.radius <= 0:
            return True
        start = a + ab * (fc.free / n)
        return all(dist(d, closest_point_on_segment(d, start, b)) > fc.radius + self.at.lane_margin for d in defs)

    def _pass_candidates(self, o, defs):
        """Safe passes (no visible interceptor) as
        (score, openness, teammate_k, shot_prob_there, known_free, through_point or None)."""
        pos, m = o.pos, self.m
        types = slot_types_from_name(o.formation)
        w_fwd = self.at.w_fwd
        if self.ca_timer > 0:
            w_fwd *= self.mv.counter.fwd_mult                        # counter-attack: go forward early
        cands = []
        for k, tid in enumerate(o.teammate_ids):
            best = None
            for through, pt in ((False, o.teammate_pos[k]), (True, self._through_point(o, o.teammate_pos[k], defs))):
                if pt is None:
                    continue
                if dist(pos, pt) > (self.ro.d_max_pass if o.slot_type == "D" else self.at.max_pass):
                    continue
                if not through and not self._lane_safe(pos, pt, defs):
                    continue
                open_ = (min(dist(d, pt) for d in defs) if defs else self.at.open_max)
                open_ = min(open_, self.at.open_max) / self.at.open_max
                fwd = min((pt[0] - pos[0]) / self.L, self.at.fwd_cap)     # a longer jump upfield earns no extra credit
                score = (w_fwd * fwd + self.at.w_open * open_ - self.at.w_dist * dist(pos, pt) / self.L
                         + (self.at.hub_bonus if tid == self.playmaker else 0.0) + self.ro.pass_bias[types[tid]]
                         + (self.mv.through.bonus if through else 0.0)
                         + (self.mv.pass_move.one_two_bonus if tid == self.mover and self.mover_timer > 0 else 0.0))
                sp = (shot_probability(pt, self.goal, defs, o.control, m.p_max, m.d0, m.shot_block_scale, sure_goal(m))
                      if dist(pt, self.goal) <= m.shoot_max else 0.0)
                # known free: I can see the area around him (he is inside my vision) and no defender is near him
                free = (dist(pos, pt) <= o.radius and
                        (not defs or min(dist(d, pt) for d in defs) >= self.cr.free_dist))
                c = (score, open_, k, sp, free, pt if through else None)
                if best is None or c[0] > best[0]:
                    best = c
            if best is not None:
                cands.append(best)
        return cands

    @staticmethod
    def _pass_act(c):
        """The action for a pass candidate: a normal pass to feet, or a through ball to its point."""
        if c[5] is None:
            return Action(Kind.PASS, c[2])
        return Action(Kind.THROUGH, np.array([float(c[2]), c[5][0], c[5][1]]))

    def _attack_with_ball(self, o):
        defs = self._visible_defenders(o)
        pos = o.pos
        m = self.m
        is_def = o.slot_type == "D"
        cands = self._pass_candidates(o, defs)
        pressed = bool(defs) and min(dist(d, pos) for d in defs) <= self.cr.press_dist
        # a defender is closing in: get rid of the ball (shot or free teammate) BEFORE he arrives
        threat = bool(defs) and min(dist(d, pos) for d in defs) <= self.cr.release_dist
        urgency = self._urgency()
        force = urgency >= 1.0                                       # kept the ball too long: force the play
        shoot_min = self.at.shoot_min - (self.cr.release_shoot if threat else 0.0) - self.pt.shoot * urgency
        # 1. shoot, or set up a teammate who is clearly better placed (cross, cut-back, square ball)
        p = (shot_probability(pos, self.goal, defs, o.control, m.p_max, m.d0, m.shot_block_scale, sure_goal(m))
             if dist(pos, self.goal) <= m.shoot_max else 0.0)
        can_shoot = p > 0 and p >= shoot_min + (self.ro.d_shoot_extra if is_def else 0.0)
        need = p + self.cr.better_margin if (can_shoot or not self.mv.assist.enabled) else max(shoot_min, p + self.cr.better_margin)
        better = [c for c in cands if c[3] >= need] if (can_shoot or self.mv.assist.enabled) else []
        if better:
            return self._pass_act(max(better, key=lambda c: (c[3], c[2])))
        if can_shoot:
            # nobody near me and still far out: do not shoot from anywhere, carry it closer for a better chance
            # (nobody blocking the way to goal within shoot_free, and nobody close enough to tackle me)
            alone = not threat and not any(d[0] > pos[0] and dist(d, pos) <= self.at.shoot_free for d in defs)
            if not (alone and dist(pos, self.goal) > self.at.shoot_close):
                return Action(Kind.SHOOT)
            return Action(Kind.DRIBBLE_GOAL)
        # 2. pass forward or square: best safe teammate; under pressure the bar is lower. Never backward here.
        fwd_cands = [c for c in cands if o.teammate_pos[c[2]][0] >= pos[0] - self.cr.back_tol]
        if fwd_cands:
            best = max(fwd_cands, key=lambda c: (c[0], c[1], c[2]))
            # with open grass ahead a midfielder would rather carry the ball than pass it straight away
            lane_open = not any(d[0] > pos[0] and dist(d, pos) <= self.cr.carry_free for d in defs)
            bar = self.at.pass_min + (self.ro.carry_bias[o.slot_type] if lane_open and not threat else 0.0)
            if best[0] >= bar - (self.cr.press_relief if threat else 0.0) - self.pt.pass_relief * urgency:
                return self._pass_act(best)
        if threat:                                                   # release to a teammate I know is free
            free = [c for c in (fwd_cands if force else cands) if c[4]]
            if free:
                return self._pass_act(max(free, key=lambda c: (c[0], c[1], c[2])))
        # a pressed defender with no teammate he knows to be free does not risk it near his own goal: he clears
        if (self.cl.enabled and is_def and pressed and pos[0] <= self.cl.max_x * self.L
                and not any(c[4] for c in cands)):
            return Action(Kind.CLEAR, self._clear_target(o, defs))
        # 3. dribble if no visible defender is close ahead (defenders do not run into the final third)
        ahead = [d for d in defs if d[0] > pos[0] and dist(d, pos) <= self.at.d_danger]
        too_far = is_def and pos[0] > self.ro.d_dribble_max_x * self.L and bool(cands)
        if force or (not ahead and not too_far):
            return Action(Kind.DRIBBLE_GOAL)
        # 4. recycle, only when blocked ahead AND under pressure: back to a teammate known to be free, else any safe pass
        if cands and pressed:
            free = [c for c in cands if c[4]]                        # nearest teammate known to be free, if any
            if free:
                return self._pass_act(min(free, key=lambda c: (dist(pos, o.teammate_pos[c[2]]), c[2])))
            return self._pass_act(max(cands, key=lambda c: (c[1], c[2])))   # about to lose it: any safe pass
        return self._carry_aside(o, defs)

    def _clear_target(self, o, defs):
        """Where to aim a clearance: upfield, into the most space from visible defenders, and rather toward a
        teammate who is ahead of the ball than away from all of them."""
        d, cap = self.m.clear.dist, self.at.open_max
        ahead = [t for t in o.teammate_pos if t[0] > o.pos[0]]
        best = None
        for ang in (0.0, 0.35, -0.35, 0.7, -0.7):
            pt = np.clip(o.pos + d * np.array([np.cos(ang), np.sin(ang)]), [2.0, 2.0], [self.L - 2.0, self.W - 2.0])
            space = min([dist(x, pt) for x in defs] + [cap])
            mate = min([dist(t, pt) for t in ahead] + [cap])
            key = (round(space - self.cl.w_mate * mate, 3), -abs(ang))
            if best is None or key > best[0]:
                best = (key, pt)
        return best[1]

    def _carry_aside(self, o, defs):
        """Blocked ahead with no pass on: carry the ball diagonally or sideways into the most space, do not stand still."""
        best = None
        for ang in (np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2):
            pt = o.pos + self.cr.carry_step * np.array([np.cos(ang), np.sin(ang)])
            if not (2.0 <= pt[0] <= self.L - 2.0 and 2.0 <= pt[1] <= self.W - 2.0):
                continue
            space = min([dist(d, pt) for d in defs] + [self.at.open_max])
            key = (round(space, 3), -abs(ang))
            if best is None or key > best[0]:
                best = (key, pt)
        if best is None or best[0][0] <= self.cr.press_dist:          # nowhere to go: shield it and wait for support
            return STAY
        return MOVE_TO(best[1])

    # ---- off-ball attacking support ------------------------------------------------------------------
    def _support(self, j, o, ctx, carrier=None):
        for k, tid in enumerate(o.teammate_ids):
            if o.teammate_has_ball[k]:
                carrier = o.teammate_pos[k]
        home = self._home(o.anchor, o.slot_type)
        mv, L, W = self.mv, self.L, self.W
        spot = lambda pt: MOVE_TO(self._free_spot(o, np.clip(pt, [2.0, 2.0], [L - 2.0, W - 2.0])))
        # pass and move: I just passed, so I keep running forward for a few steps (the return pass makes a one-two)
        if j == self.mover and self.mover_timer > 0:
            return spot(np.array([min(o.pos[0] + mv.pass_move.run, L - self.run.goal_gap), o.pos[1]]))
        if carrier is not None:
            c_side = np.sign(carrier[1] - W / 2)
            wide = abs(carrier[1] - W / 2) > mv.box.wide
            # ball in the final third: attack the box. Far post for forwards, cut-back spot for the far-side midfielder
            if carrier[0] > mv.box.x * L and wide:
                if o.slot_type == "F":
                    return spot(np.array([L - mv.box.post_gap, W / 2 - c_side * 4.0]))
                if o.slot_type == "M" and np.sign(home[1] - W / 2) != c_side:
                    return spot(np.array([L - mv.box.cutback_gap, W / 2]))
            # overlap: a wide midfielder just behind the carrier on his own wing runs round the outside
            side = np.sign(home[1] - W / 2)
            if (o.slot_type == "M" and side != 0 and mv.overlap.trigger > 0 and carrier[0] > 0.45 * L
                    and o.pos[0] < carrier[0] and dist(o.pos, carrier) <= mv.overlap.trigger
                    and (c_side == side or not wide)):
                pt = carrier + np.array([mv.overlap.ahead, side * mv.overlap.wide])
                return spot(home + clip_norm(pt - home, mv.overlap.max_dev))
            # dropping deep: the ball is too far back to reach me, so the forward comes short to show for it
            if o.slot_type == "F" and mv.drop_deep.dist > 0 and carrier[0] < home[0] - mv.drop_deep.trigger:
                return spot(home + clip_norm(carrier - home, mv.drop_deep.dist))
        counter = self.ca_timer > 0
        if carrier is not None and self.run.enabled and (o.slot_type == "F" or (counter and o.slot_type == "M")):
            return MOVE_TO(self._free_spot(o, self._run_target(o, home, carrier)))
        shift = np.zeros(2)
        if o.slot_type == "F":
            shift = np.array([self.su.adv, 0.0])
        elif o.slot_type == "M" and carrier is not None:
            if carrier[0] > o.pos[0] and self.su.offer_dist > 0:     # ball is ahead of me: offer a free back pass
                return MOVE_TO(self._free_spot(o, self._offer_target(o, home, carrier)))
            shift = clip_norm(carrier - home, self.su.mid)
        elif o.slot_type == "D":
            shift = np.array([self.su["def"], 0.0])
        return MOVE_TO(self._free_spot(o, home + clip_norm(shift, self.su.r_support)))

    def _free_spot(self, o, target):
        """Drift toward free space: among the target and 8 points around it, take the one farthest from visible
        defenders and from teammates (no bunching). A small cost on distance from where I am keeps it steady."""
        R = self.sp.radius
        if R <= 0:
            return target
        defs = self._visible_defenders(o)
        best = None
        for i in range(9):
            ang = i * np.pi / 4
            pt = target if i == 8 else target + R * np.array([np.cos(ang), np.sin(ang)])
            pt = np.clip(pt, [2.0, 2.0], [self.L - 2.0, self.W - 2.0])
            cap = self.at.open_max
            score = (min([dist(d, pt) for d in defs] + [cap])
                     + self.sp.w_mates * min([dist(t, pt) for t in o.teammate_pos] + [cap])
                     - self.sp.w_move * dist(pt, o.pos))
            if best is None or score > best[0] + 1e-9:
                best = (score, pt)
        return best[1]

    def _offer_target(self, o, home, carrier):
        """Midfielder support: a spot behind or beside the carrier with the most space from visible defenders,
        so the carrier has a free man to give the ball back to."""
        d = self.su.offer_dist
        defs = self._visible_defenders(o)
        mates = [t for k, t in enumerate(o.teammate_pos) if not o.teammate_has_ball[k]]
        best = None
        # spots around the carrier: level with him and ahead of him as well as behind, so he gets close support
        for off in ((-0.7 * d, 0.0), (0.0, d), (0.0, -d), (0.6 * d, 0.8 * d), (0.6 * d, -0.8 * d)):
            pt = np.clip(carrier + np.array(off), [2.0, 2.0], [self.L - 2.0, self.W - 2.0])
            # a spot a teammate is nearer to than I am is his, not mine: two players never offer in the same place
            if any(dist(t, pt) < self.su.offer_sep and dist(t, pt) < dist(o.pos, pt) for t in mates):
                continue
            space = min([dist(x, pt) for x in defs] + [self.at.open_max])
            # enough room is enough: among spots with room, take the most advanced one
            key = (min(round(space, 3), self.su.offer_space), pt[0], -dist(pt, home))
            if best is None or key > best[0]:
                best = (key, pt)
        if best is None:                                             # every spot is taken: stay on my own side
            return home
        return home + clip_norm(best[1] - home, self.su.offer_max)

    def _run_target(self, o, home, carrier):
        """Forward run: get ahead of the ball and pick the laneway with the most space from visible defenders."""
        r = self.run
        lead = r.lead * (self.mv.counter.lead_mult if self.ca_timer > 0 else 1.0)   # counter-attack: run further
        x = min(self.L - r.goal_gap, max(home[0] + self.su.adv, carrier[0] + lead))
        defs = self._visible_defenders(o)
        best = None
        for dy in (-r.dy, 0.0, r.dy):
            pt = np.array([x, float(np.clip(home[1] + dy, 2.0, self.W - 2.0))])
            space = min([dist(d, pt) for d in defs] + [self.at.open_max])
            key = (round(space, 3), -abs(dy))
            if best is None or key > best[0]:
                best = (key, pt)
        return home + clip_norm(best[1] - home, r.r_run)
