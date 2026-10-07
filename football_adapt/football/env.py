"""FootballEnv: the 5-a-side game (no goalkeepers, limited vision). See game.md.

World frame: team 0 ("ours") attacks toward +x, team 1 (the opponent) attacks toward -x.
Both teams' controllers work in a TEAM FRAME where they attack toward +x (see geometry.to_frame).
Teams are indexed 0 (ours) and 1 (opponent). Players are indexed 0..4 (player number = index + 1).
"""
from __future__ import annotations
import math
import numpy as np

from .actions import ActionSpace, Action, Kind, STAY
from .config import load_config, apply_style
from .formations import FormationBook
from .geometry import (dist, unit, clip_norm, move_toward, rotate, to_frame, dir_to_frame,
                       dist_point_segment, closest_point_on_segment)
from .mechanics import (control_rating, shot_probability, sure_goal, pass_interception, tackle_probability)
from .observation import build_observation, flatten, global_state, flat_dim


class FootballEnv:
    def __init__(self, cfg=None, scripted_ours: bool = False):
        self.cfg = cfg or load_config()
        c = self.cfg
        self.L, self.W = c.pitch.length, c.pitch.width
        self.T = c.time.T
        self.space = ActionSpace(c.actions.n_directions)
        self.book = FormationBook(c)
        self.scripted_ours = scripted_ours
        self.done = True
        self.t = 0
        self.obs_dim = None

    # ------------------------------------------------------------------ setup
    def reset(self, seed=None, our_formation=None, opp_formation=None, sigma=None,
              zeta=None, radius=None, radius_opp=None, start_team=None, our_style=None, opp_style=None):
        from opponent.controller import ScriptedTeam   # lazy import (separate package)

        c = self.cfg
        ss = np.random.SeedSequence(seed)
        s_env, s_opp, s_ours = ss.spawn(3)
        self.rng = np.random.default_rng(s_env)

        self.formation = [our_formation or c.formations.ours, opp_formation or c.formations.opponent]
        # play style per team: each team gets its own view of the config (behaviour and shape only, same rules)
        self.style = [our_style or c.styles.ours, opp_style or c.styles.opponent]
        self.team_cfg = [apply_style(c, s) for s in self.style]
        self.sigma = int(c.opponent.playmaker.sigma if sigma is None else sigma)   # 0 = none, else 1..5
        self.sigma_idx = self.sigma - 1 if self.sigma > 0 else None
        z0 = c.structure.zeta if zeta is None else zeta
        self.zeta = [2 if self.scripted_ours else int(z0), 2]
        r0 = c.vision.radius if radius is None else radius
        r1 = c.vision.radius if radius_opp is None else radius_opp
        self.radius = [float(r0), float(r1)]

        self.anchors = np.zeros((2, 5, 2))
        self.phase = np.zeros(2)           # per team: -1 = defending shape, 0 = base shape, +1 = attacking shape
        for team in (0, 1):
            self._apply_formation(team, self.formation[team])

        self.pos = self.anchors.copy()
        self.ball_pos = np.array([self.L / 2, self.W / 2])
        self.holder = None
        self.steal_chain = 0               # consecutive tackle steals since the ball was last passed/shot/picked up
        self.steal_lock = 0                # steps left before a full chain is forgotten and tackles are allowed again
        self.ball_state = "loose"          # "held" | "loose" | "flight"
        self.flight = None
        self.frozen = np.zeros((2, 5), int)
        self.prev_action = -np.ones((2, 5), int)
        self.score = [0, 0]
        self.t = 0
        self.done = False
        self.shaping_used = 0.0
        self.shaping_weight = float(c.reward.shaping_weight)
        self.event_log = []
        self.stats = self._new_stats()
        self.opp_anchor_dist = []

        self.controllers = {1: ScriptedTeam(self.team_cfg[1], np.random.default_rng(s_opp), team=1,
                                            formation=self.formation[1], playmaker=self.sigma_idx)}
        self.controllers[0] = (ScriptedTeam(self.team_cfg[0], np.random.default_rng(s_ours), team=0,
                                            formation=self.formation[0], playmaker=None, mode_override="NONE")
                               if self.scripted_ours else None)

        first = int(self.rng.integers(0, 2)) if start_team is None else int(start_team)
        self._kickoff(first)
        return self._obs_out(), self._info([], None, {"team": 0.0, "shaping": 0.0, "terminal": 0.0})

    def _new_stats(self):
        z = lambda: [0, 0]
        return dict(passes=z(), passes_completed=z(), interceptions=z(), tackles_won=z(), tackles_failed=z(),
                    shots=z(), goals=z(), possession_steps=z(), invalid_actions=z(),
                    sigma_passes_made=0, sigma_passes_received=0, sigma_interceptions=0,
                    shadow_steps_on_sigma=0, shadow_steps_total=0, switches=0, clearances=z())

    def _apply_formation(self, team, name):
        self.formation[team] = name
        self._refresh_anchors(team)

    def set_formation(self, team, name):
        """Change a team's formation in the middle of a match. Nobody teleports: the anchors move at once and the
        players walk to them, exactly as when the scripted team switches by itself."""
        old = self.formation[team]
        if name == old:
            return
        self.book[name]                    # KeyError for an unknown formation
        self._apply_formation(team, name)
        if self.controllers.get(team) is not None:
            self.controllers[team].set_formation(name)
        self.event_log.append({"type": "switch", "team": team, "from_": old, "to": name, "t": self.t})

    def set_style(self, team, name):
        """Change a team's play style in the middle of a match. Positions, timers and the score are kept."""
        self.team_cfg[team] = apply_style(self.cfg, name)
        self.style[team] = name
        if self.controllers.get(team) is not None:
            self.controllers[team].set_cfg(self.team_cfg[team])
        self._refresh_anchors(team)

    def _refresh_anchors(self, team):
        a = self.book.anchors_team_frame(self.formation[team], float(self.phase[team]),
                                         self.team_cfg[team].formations.phases)
        for i in range(5):
            self.anchors[team][i] = to_frame(team, a[i], self.L, self.W)

    def vision_radius(self, team, idx) -> float:
        if team == 1 and self.sigma_idx is not None and idx == self.sigma_idx:
            return float("inf")
        return self.radius[team]

    def control(self, team, idx) -> float:
        cc = self.cfg.control
        anchor = self.anchors[team][idx] if self.zeta[team] >= 2 else None
        return control_rating(self.pos[team][idx], anchor, self.zeta[team], cc.c_min, cc.l_ctrl)

    def _kickoff(self, team_with_ball):
        self.phase[:] = 0.0                # both teams line up in their base shape
        for team in (0, 1):
            self._refresh_anchors(team)
        self.pos = self.anchors.copy()
        self.frozen[:] = 0
        self.flight = None
        self.steal_chain = 0
        self.goal_kick = None              # (team, player) about to take a goal kick: the other team stays out of his box
        k = self.book.kickoff_slot(self.formation[team_with_ball])
        self.holder = (team_with_ball, k)

        # Kickoff player stands at the centre spot.
        centre = np.array([self.L / 2, self.W / 2])
        self.pos[team_with_ball][k] = centre.copy()
        self.ball_pos = centre.copy()
        self.ball_state = "held"

        # Every other player must be inside their own half.
        # Team 0 owns x < L/2; team 1 owns x > L/2.
        # If a player's anchor is across the line, pull them back 3 m inside their half
        # but keep their anchor y so they spread naturally across the pitch width.
        HALF_MARGIN = 3.0
        mid = self.L / 2
        for team in (0, 1):
            for i in range(5):
                if team == team_with_ball and i == k:
                    continue  # kickoff player is already at centre
                x = self.pos[team][i][0]
                if team == 0 and x >= mid:
                    self.pos[team][i][0] = mid - HALF_MARGIN
                elif team == 1 and x <= mid:
                    self.pos[team][i][0] = mid + HALF_MARGIN

    # ------------------------------------------------------------------ outputs
    def _obs_out(self):
        self.last_obs = {i: build_observation(self, 0, i) for i in range(5)}
        vec = np.stack([flatten(self, self.last_obs[i]) for i in range(5)])
        mask = np.stack([self.last_obs[i].mask for i in range(5)])
        if self.obs_dim is None:
            self.obs_dim = vec.shape[1]
        return {"vec": vec, "mask": mask, "state": global_state(self)}

    def _info(self, events, switch, parts):
        return {"t": self.t, "score": tuple(self.score), "g": self.formation[1], "f": self.formation[0],
                "styles": tuple(self.style), "sigma": self.sigma, "events": events, "switch": switch, "reward_parts": parts,
                "opp_anchor_dist": self.opp_anchor_dist[-1] if self.opp_anchor_dist else 0.0}

    def _log(self, step_events, **kw):
        kw["t"] = self.t
        step_events.append(kw)
        self.event_log.append(kw)

    def set_shaping_weight(self, w: float):
        """The trainer anneals lambda_t toward 0 by calling this."""
        self.shaping_weight = float(w)

    # ------------------------------------------------------------------ step
    def step(self, our_actions=None):
        assert not self.done, "call reset()"
        c, m = self.cfg, self.cfg.mechanics
        v_pl, v_dr = c.speeds.player, c.speeds.dribble
        L, W = self.L, self.W
        ev = []
        shaping = 0.0
        goal_by = None
        switch_event = None
        self.shadow_targets = []

        # 0. team shape: slide toward the attacking shape with the ball, the defending shape without it.
        #    While the ball is loose or in flight the current target is kept.
        if self.holder is not None:
            for team in (0, 1):
                ph = self.team_cfg[team].formations.phases
                if not ph.enabled:
                    continue
                tgt = 1.0 if self.holder[0] == team else -1.0
                self.phase[team] += float(np.clip(tgt - self.phase[team], -ph.rate, ph.rate))
                self._refresh_anchors(team)

        # 1. observations and actions -------------------------------------------------
        obs = {team: {i: build_observation(self, team, i) for i in range(5)} for team in (0, 1)}
        raw = {1: self.controllers[1].act(self.t, obs[1])}
        raw[0] = self.controllers[0].act(self.t, obs[0]) if self.controllers[0] is not None else (our_actions or {})
        acts = {0: {}, 1: {}}
        for team in (0, 1):
            for i in range(5):
                a = raw[team].get(i, STAY)
                if isinstance(a, (int, np.integer)):
                    idx = int(a)
                    act = self.space.decode(idx)
                    valid = bool(obs[team][i].mask[idx])
                    self.prev_action[team][i] = idx
                else:
                    act = a
                    idx = self.space.index_of(act)
                    valid = True if idx is None else bool(obs[team][i].mask[idx])
                    if obs[team][i].frozen:
                        valid = False
                    self.prev_action[team][i] = -1 if idx is None else idx
                if not valid:
                    self.stats["invalid_actions"][team] += 1
                    act = STAY
                acts[team][i] = act

        # 2. tackles ---------------------------------------------------------------------
        if self.steal_chain >= m.max_steal_chain:          # chain is full: tackles are off until the lock runs out
            self.steal_lock -= 1
            if self.steal_lock <= 0:
                self.steal_chain = 0
        if self.holder is not None and self.steal_chain < m.max_steal_chain:
            ht, hi = self.holder
            attempts = [(tt, ti) for tt in (0, 1) for ti in range(5)
                        if tt != ht and acts[tt][ti].kind == Kind.TACKLE]
            for pick in self.rng.permutation(len(attempts)) if attempts else []:
                tt, ti = attempts[int(pick)]
                if dist(self.pos[tt][ti], self.pos[ht][hi]) > m.tackle_dist:
                    continue
                p = tackle_probability(self.control(tt, ti), self.control(ht, hi))
                if self.rng.random() < p:
                    self.holder = (tt, ti)
                    self.steal_chain += 1
                    self.steal_lock = m.steal_lock
                    self.stats["tackles_won"][tt] += 1
                    self._log(ev, type="tackle", success=True, by=(tt, ti), from_=(ht, hi))
                    if tt == 0:
                        shaping += c.reward.w_tackle
                    if ht == 0:
                        shaping += c.reward.w_turnover
                    break
                else:
                    self.frozen[tt][ti] = m.tackle_recover
                    self.stats["tackles_failed"][tt] += 1
                    self._log(ev, type="tackle", success=False, by=(tt, ti), from_=(ht, hi))

        # 3. the ball holder passes or shoots -------------------------------------------
        acted = set()
        if self.holder is not None:
            ht, hi = self.holder
            a = acts[ht][hi]
            if a.kind == Kind.SHOOT:
                shaping += self._shoot(ht, hi, ev)
                goal_by = ht if self._goal_flag else None
                acted.add((ht, hi))
            elif a.kind == Kind.PASS:
                self._pass(ht, hi, a.arg, ev)
                acted.add((ht, hi))
            elif a.kind == Kind.THROUGH:
                self._pass(ht, hi, int(a.arg[0]), ev, lead=to_frame(ht, np.asarray(a.arg[1:], float), L, W))
                acted.add((ht, hi))
            elif a.kind == Kind.CLEAR:
                self._clear(ht, hi, a.arg, ev)
                acted.add((ht, hi))

        # 4. move players ----------------------------------------------------------------
        for team in (0, 1):
            for i in range(5):
                if (team, i) in acted or self.frozen[team][i] > 0:
                    continue
                tgt = self._target(team, i, acts[team][i])
                if acts[team][i].kind == Kind.SHADOW:
                    self.shadow_targets.append(acts[team][i].arg)
                    self.stats["shadow_steps_total"] += 1
                    if self.sigma_idx is not None and acts[team][i].arg == self.sigma_idx:
                        self.stats["shadow_steps_on_sigma"] += 1
                if tgt is None:
                    continue
                speed = v_dr if self.holder == (team, i) else v_pl
                new = move_toward(self.pos[team][i], tgt, speed)
                self.pos[team][i] = np.clip(new, [0, 0], [L, W])
                if self.goal_kick is not None and team != self.goal_kick[0]:   # goal kick not taken yet: stay out
                    self.pos[team][i] = self._out_of_box(self.goal_kick[0], self.pos[team][i])
        self.frozen = np.maximum(self.frozen - 1, 0)

        # 5. ball ------------------------------------------------------------------------
        if self.holder is not None:
            self.ball_pos = self.pos[self.holder[0]][self.holder[1]].copy()
            self.ball_state = "held"
        elif self.ball_state == "flight":
            shaping += self._advance_flight(ev)
        if self.holder is None and self.ball_state == "loose":
            self._pickup(ev)
        # the goal kick is taken once the kicker plays the ball or carries it out of his box
        if self.goal_kick is not None and (self.holder != self.goal_kick
                                           or not self._in_box(self.goal_kick[0], self.ball_pos)):
            self.goal_kick = None

        # 6. goal -------------------------------------------------------------------------
        if goal_by is not None:
            self.score[goal_by] += 1
            self.stats["goals"][goal_by] += 1
            self._log(ev, type="goal", team=goal_by, score=tuple(self.score))
            self._kickoff(1 - goal_by)

        # 7. opponent formation switch (visible from the next step) ----------------------
        for team in (1, 0):
            ctrl = self.controllers.get(team)
            if ctrl is None:
                continue
            new = ctrl.update_formation(self.t, self.score[team], self.score[1 - team])
            if new is not None and new != self.formation[team]:
                old = self.formation[team]
                self._apply_formation(team, new)
                self.stats["switches"] += int(team == 1)
                switch_event = {"t": self.t, "team": team, "from": old, "to": new}
                self._log(ev, type="switch", team=team, from_=old, to=new)

        # 8. rewards, logs, advance --------------------------------------------------------
        team_r = (1.0 if goal_by == 0 else 0.0) - (1.0 if goal_by == 1 else 0.0)
        shaped = self.shaping_weight * shaping
        room = self.cfg.reward.shape_cap - self.shaping_used
        shaped = float(np.clip(shaped, -room, room)) if room > 0 else 0.0
        self.shaping_used += abs(shaped)
        if self.holder is not None:
            self.stats["possession_steps"][self.holder[0]] += 1
        ad = np.mean([dist(self.pos[1][j], self.anchors[1][j]) for j in range(5)])
        self.opp_anchor_dist.append(float(ad))

        self.t += 1
        terminal = 0.0
        if self.t >= self.T:
            self.done = True
            terminal = c.reward.eta * float(np.sign(self.score[0] - self.score[1]))
        parts = {"team": team_r, "shaping": shaped, "terminal": terminal}
        reward = team_r + shaped + terminal
        info = self._info(ev, switch_event, parts)
        info["final_score"] = tuple(self.score) if self.done else None
        return self._obs_out(), reward, self.done, info

    # ------------------------------------------------------------------ movement targets
    def _target(self, team, i, a: Action):
        L, W, m = self.L, self.W, self.cfg.mechanics
        k = a.kind
        pos = self.pos[team][i]
        if k == Kind.MOVE_DIR:
            ang = 2 * math.pi * a.arg / self.space.D
            d = dir_to_frame(team, np.array([math.cos(ang), math.sin(ang)]))
            return pos + d * self.cfg.speeds.player
        if k == Kind.HOLD_SHAPE:
            return self.anchors[team][i]
        if k == Kind.GO_TO_BALL:
            return self.ball_pos
        if k == Kind.DRIBBLE_GOAL:
            return to_frame(team, np.array([L, W / 2]), L, W)
        if k == Kind.SHADOW:
            j = a.arg
            opp = self.pos[1 - team][j]
            own_goal = to_frame(team, np.array([0.0, W / 2]), L, W)
            return opp + m.shadow_dist * unit(own_goal - opp)
        if k == Kind.MOVE_TO:
            return to_frame(team, a.arg, L, W)
        return None   # STAY, PASS, SHOOT, TACKLE

    # ------------------------------------------------------------------ pass / shot / flight
    def _pass(self, team, idx, k, ev, lead=None):
        m, c = self.cfg.mechanics, self.cfg
        other = 1 - team
        ids = [j for j in range(5) if j != idx]
        recv = ids[k]
        passer = self.pos[team][idx].copy()
        target = self.pos[team][recv].copy() if lead is None else np.asarray(lead, float)   # lead = through ball
        ctrl = self.control(team, idx)
        std = m.s_pass * math.sqrt(max(0.0, 1.0 - ctrl))
        ang = float(self.rng.normal(0.0, std)) if std > 0 else 0.0
        target = np.clip(passer + rotate(target - passer, ang), [0, 0], [self.L, self.W])

        defenders = [(j, self.pos[other][j]) for j in range(5)
                     if self.frozen[other][j] == 0 and dist(self.pos[other][j], passer) <= self.vision_radius(other, j)]
        res = pass_interception(passer, target, defenders, c.speeds["pass"],
                                c.speeds.player, m.intercept_margin)
        if res is not None:
            key, q, _ = res
            end, designated, predicted = q, (other, key), "intercepted"
        else:
            end, designated, predicted = target, (team, recv), "completed"
        n = max(1, math.ceil(dist(passer, end) / c.speeds["pass"]))
        self.flight = dict(start=passer, end=np.asarray(end, float), n=n, k=0, passer=(team, idx),
                           designated=designated, intended=(team, recv), predicted=predicted)
        self.holder = None
        self.steal_chain = 0
        self.ball_state = "flight"
        self.stats["passes"][team] += 1
        if team == 1 and self.sigma_idx is not None and idx == self.sigma_idx:
            self.stats["sigma_passes_made"] += 1
        self._log(ev, type="pass", passer=(team, idx), receiver=(team, recv), predicted=predicted,
                  through=lead is not None)

    def _clear(self, team, idx, point, ev):
        """Clearance: a long ball kicked upfield to nobody. It goes over everyone, so it cannot be intercepted or
        cut out on the way. It lands with a wide aiming error and is loose: whoever gets there first has it."""
        cl, L, W = self.cfg.mechanics.clear, self.L, self.W
        start = self.pos[team][idx].copy()
        aim = np.array([L, start[1]]) if point is None else np.asarray(point, float)   # team frame
        v = to_frame(team, aim, L, W) - start
        d = float(np.hypot(*v))
        if d < 1e-6:                                   # no direction given: straight upfield
            v, d = dir_to_frame(team, np.array([1.0, 0.0])), cl.dist
        v = unit(v) * min(d, cl.dist)
        ang = float(self.rng.normal(0.0, cl.spread)) if cl.spread > 0 else 0.0
        end = np.clip(start + rotate(v, ang), [0, 0], [L, W])
        n = max(1, math.ceil(dist(start, end) / self.cfg.speeds["pass"]))
        self.flight = dict(start=start, end=end, n=n, k=0, passer=(team, idx),
                           designated=None, intended=None, predicted="clear")
        self.holder = None
        self.steal_chain = 0
        self.ball_state = "flight"
        self.stats["clearances"][team] += 1
        self._log(ev, type="clear", by=(team, idx))

    def _advance_flight(self, ev) -> float:
        f = self.flight
        f["k"] += 1
        prev = self.ball_pos.copy()
        self.ball_pos = f["start"] + (f["end"] - f["start"]) * min(1.0, f["k"] / f["n"])
        if f["designated"] is None:                    # a clearance: nobody is picked, it just lands
            if f["k"] >= f["n"]:
                self.ball_state = "loose"
                self.ball_pos = f["end"].copy()
                self.flight = None
            return 0.0
        m, c = self.cfg.mechanics, self.cfg
        team, idx = f["passer"]
        # cut-out: an opponent standing on the ball's path this step takes it, whoever was picked at the kick.
        # The first `free` metres are exempt, so a defender right on the passer does not block every pass.
        fc = m.flight_cut
        if fc.radius > 0 and f["designated"][0] == team and dist(self.ball_pos, f["start"]) >= fc.free:
            near = [(dist_point_segment(self.pos[1 - team][j], prev, self.ball_pos), j) for j in range(5)
                    if self.frozen[1 - team][j] == 0]
            near = [x for x in near if x[0] <= fc.radius]
            if near:
                j = min(near)[1]
                f["designated"], f["n"] = (1 - team, j), f["k"]
                f["end"] = closest_point_on_segment(self.pos[1 - team][j], prev, self.ball_pos)
                self.ball_pos = f["end"].copy()
        if f["k"] < f["n"]:
            return 0.0
        dt, di = f["designated"]
        reach = m.pickup + c.speeds.player * m.intercept_margin
        shaping = 0.0
        if dist(self.pos[dt][di], f["end"]) <= reach and self.frozen[dt][di] == 0:
            self.holder = (dt, di)
            self.steal_chain = 0
            self.ball_pos = self.pos[dt][di].copy()
            self.ball_state = "held"
            if dt == team:
                self.stats["passes_completed"][team] += 1
                result = "completed"
                if team == 0:
                    adv = (f["end"][0] - f["start"][0]) / self.L
                    shaping += c.reward.w_pass * (1.0 + float(np.clip(adv, 0.0, 1.0)))
                if team == 1 and self.sigma_idx is not None and di == self.sigma_idx:
                    self.stats["sigma_passes_received"] += 1
            else:
                self.stats["interceptions"][dt] += 1
                result = "intercepted"
                if dt == 1 and self.sigma_idx is not None and di == self.sigma_idx:
                    self.stats["sigma_interceptions"] += 1
                if team == 0:
                    shaping += c.reward.w_turnover
        else:
            self.holder = None
            self.ball_state = "loose"
            self.ball_pos = f["end"].copy()
            result = "loose"
        self._log(ev, type="pass_result", passer=(team, idx), result=result, by=(dt, di))
        self.flight = None
        return shaping

    def _shoot(self, team, idx, ev) -> float:
        m, c = self.cfg.mechanics, self.cfg
        L, W = self.L, self.W
        shooter = self.pos[team][idx]
        goal = to_frame(team, np.array([L, W / 2]), L, W)
        self._goal_flag = False
        if dist(to_frame(team, shooter, L, W), np.array([L, W / 2])) > m.shoot_max:
            return 0.0
        defenders = [self.pos[1 - team][j] for j in range(5)]
        p = shot_probability(shooter, goal, defenders, self.control(team, idx),
                             m.p_max, m.d0, m.shot_block_scale, sure_goal(m))
        self.stats["shots"][team] += 1
        scored = bool(self.rng.random() < p)
        self._log(ev, type="shot", by=(team, idx), p=float(p), scored=scored)
        self.holder = None
        self.steal_chain = 0
        if scored:
            self._goal_flag = True
            self.ball_state = "loose"
            self.ball_pos = goal.copy()
        else:
            gk = m.goal_kick
            if gk.enabled:
                # goal kick: the defending player nearest to his own goal restarts from the goal-kick spot.
                # Nobody may tackle him for gk.protect steps (same lock as the steal chain).
                other = 1 - team
                own_goal = to_frame(other, np.array([0.0, W / 2]), L, W)
                j = min(range(5), key=lambda k: (dist(self.pos[other][k], own_goal), k))
                self.pos[other][j] = to_frame(other, np.array([gk.x, W / 2]), L, W)
                self.frozen[other][j] = 0
                self.holder = (other, j)
                self.ball_pos = self.pos[other][j].copy()
                self.ball_state = "held"
                self.steal_chain = m.max_steal_chain
                self.steal_lock = gk.protect
                if gk.clear_box:                       # the shooting team leaves the box before the kick is taken
                    self.goal_kick = (other, j)
                    for k in range(5):
                        self.pos[team][k] = self._out_of_box(other, self.pos[team][k])
                self._log(ev, type="goal_kick", by=(other, j))
            else:
                y = float(np.clip(W / 2 + self.rng.normal(0, m.shot_miss_spread), 1.0, W - 1.0))
                self.ball_pos = to_frame(team, np.array([L - 2.0, y]), L, W)
                self.ball_state = "loose"
        return c.reward.w_shot if team == 0 else 0.0

    def _in_box(self, team, p) -> bool:
        """Is the world point p inside `team`'s own penalty box (mechanics.goal_kick.box)?"""
        b = self.cfg.mechanics.goal_kick.box
        q = to_frame(team, p, self.L, self.W)
        return bool(q[0] < b.depth and abs(q[1] - self.W / 2) < b.half_width)

    def _out_of_box(self, team, p):
        """p itself if it is outside `team`'s penalty box, else the nearest point on the edge of the box."""
        if not self._in_box(team, p):
            return p
        b = self.cfg.mechanics.goal_kick.box
        q = to_frame(team, p, self.L, self.W).astype(float)
        dy = q[1] - self.W / 2
        if b.depth - q[0] <= b.half_width - abs(dy):
            q[0] = b.depth
        else:
            q[1] = self.W / 2 + (b.half_width if dy >= 0 else -b.half_width)
        return to_frame(team, q, self.L, self.W)

    def _pickup(self, ev):
        m = self.cfg.mechanics
        cands = []
        for team in (0, 1):
            for i in range(5):
                d = dist(self.pos[team][i], self.ball_pos)
                if d <= m.pickup and self.frozen[team][i] == 0:
                    cands.append((d, float(self.rng.random()), team, i))
        if cands:
            _, _, team, i = min(cands)
            self.holder = (team, i)
            self.steal_chain = 0
            self.ball_pos = self.pos[team][i].copy()
            self.ball_state = "held"
            self._log(ev, type="pickup", by=(team, i))
