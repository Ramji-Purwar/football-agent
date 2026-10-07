"""Screenshots for the docs: real moments from scripted matches, with the players' decisions drawn on top.

Each scene plays matches with fixed seeds, looks for a step where one behaviour happens, and draws that step: the
positions the players saw, plus lines, arrows and labels for what they decided. RED (team 1, circles) is the team
being explained, because red is the scripted opponent. BLUE (team 0, triangles) is the other side, scripted too.

Run from football_adapt/ (needs pygame; no window is opened). Pictures go to ../docs/img/opponent, and the picture
of the viewer window to ../docs/img/viewer.png:
    python -m scripts.doc_screenshots                        # every picture (a few minutes)
    python -m scripts.doc_screenshots --only lanes,pressing  # only some of them
"""
from __future__ import annotations
import argparse
import math
import os
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")     # draw off-screen
import numpy as np
import pygame

from football import FootballEnv, load_config
from football.actions import Action, Kind
from football.geometry import dist, to_frame
from scripts import pygame_render as pr

RED, BLUE = 1, 0
YELLOW, WHITE, CYAN, GREY, ORANGE = (255, 235, 90), (255, 255, 255), (130, 220, 255), (170, 176, 186), (255, 170, 70)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "img")


# ─────────────────────────────────────────────  recording a match

def hook(env):
    """Record on every step what each scripted team saw and decided: ct.dbg = {obs, ctx, out} (team frame)."""
    for ct in env.controllers.values():
        if ct is None:
            continue
        ct.dbg = {}

        def ctx_wrap(obs, _f=ct._team_context, _ct=ct):
            _ct.dbg["ctx"] = _f(obs)
            return _ct.dbg["ctx"]

        def act_wrap(t, obs, _f=ct.act, _ct=ct):
            _ct.dbg["obs"], _ct.dbg["sup"] = obs, {}
            _ct.dbg["out"] = _f(t, obs)
            return _ct.dbg["out"]

        def support_wrap(j, o, ctx, carrier=None, _f=ct._support, _ct=ct):
            _ct.dbg["sup"][j] = (_ct.mover == j and _ct.mover_timer > 0, _ct.ca_timer > 0)   # as the rule sees them
            return _f(j, o, ctx, carrier)

        ct._team_context, ct.act, ct._support = ctx_wrap, act_wrap, support_wrap


def snapshot(env):
    """Everything the viewer needs to draw the current state (a copy, so it can be drawn later)."""
    f = env.flight
    return SimpleNamespace(
        L=env.L, W=env.W, cfg=env.cfg, book=env.book, sigma_idx=env.sigma_idx, T=env.T,
        pos=env.pos.copy(), ball_pos=env.ball_pos.copy(), holder=env.holder, ball_state=env.ball_state,
        flight=None if f is None else dict(f, start=f["start"].copy(), end=f["end"].copy()),
        frozen=env.frozen.copy(), anchors=env.anchors.copy(), phase=env.phase.copy(),
        formation=list(env.formation), style=list(env.style), score=list(env.score), t=env.t)


class Moment:
    """One step of a match: the state the players saw (s), what each team decided on it, and what then happened."""

    def __init__(self, s, env, events, seed, memo):
        self.s, self.events, self.seed, self.memo = s, events, seed, memo
        self.L, self.W = env.L, env.W
        self.dirty = any(e["type"] in ("goal", "switch") for e in events)   # positions or anchors were reset
        self.team = {}
        for t in (RED, BLUE):
            ct = env.controllers[t]
            self.team[t] = SimpleNamespace(ct=ct, obs=ct.dbg["obs"], ctx=ct.dbg["ctx"], out=ct.dbg["out"],
                                           sup=ct.dbg["sup"], cp=ct.cp_timer, ca=ct.ca_timer, in_transition=ct.in_transition)

    def w(self, team, p):
        """A point in a team's own frame -> world coordinates (the same map goes both ways)."""
        return to_frame(team, np.asarray(p, float), self.L, self.W)

    def target(self, team, j):
        """Where player j of the team decided to go (world), or None if the action is not a move to a point."""
        a = self.team[team].out.get(j)
        if isinstance(a, Action) and a.kind == Kind.MOVE_TO:
            return self.w(team, a.arg)
        return None

    def act(self, team, j):
        a = self.team[team].out.get(j)
        return a.kind if isinstance(a, Action) else None

    def slot(self, team, j):
        return self.s.book[self.s.formation[team]].slot_types[j]


def find(pick, styles=("normal", "normal"), seeds=range(6), overrides=None, **reset):
    """Play matches (blue style, red style) and return the moment for which pick() gives the highest score.
    pick(moment) returns a number, or None when the moment does not show the behaviour. It may also return
    (number, earlier_moment), e.g. once it knows that the pass played at that earlier moment was completed."""
    best = (None, None)
    for seed in seeds:
        env = FootballEnv(load_config(overrides=overrides or {}), scripted_ours=True)
        env.reset(seed=seed, our_style=styles[0], opp_style=styles[1], **reset)
        hook(env)
        memo = {}                                    # the scene's own memory across the steps of one match
        while not env.done:
            s = snapshot(env)
            _, _, _, info = env.step()
            s.anchors, s.phase = env.anchors.copy(), env.phase.copy()   # as used for this step's decisions
            m = Moment(s, env, info["events"], seed, memo)
            score = pick(m)
            if isinstance(score, tuple):
                score, m = score
            if score is not None and (best[0] is None or score > best[0]):
                best = (score, m)
    if best[1] is None:
        raise RuntimeError("no moment found for this scene")
    return best[1]


# ─────────────────────────────────────────────  drawing

class Pen:
    """Draws notes on one picture. All positions are world coordinates in metres."""

    def __init__(self, surf, m):
        self.surf, self.m = surf, m
        self.labels = []
        self.players_drawn = False
        self.u = lambda n: max(1, int(round(n * pr.UI)))

    def px(self, p):
        return pr.w2s(float(p[0]), float(p[1]), self.m.L, self.m.W)

    def line(self, a, b, col=YELLOW, width=2, dash=False):
        a, b = self.px(a), self.px(b)
        if dash:
            pr.draw_dashed_line(self.surf, col, a, b, dash=self.u(8), gap=self.u(6), width=self.u(width))
        else:
            pygame.draw.line(self.surf, col, a, b, self.u(width))

    def arrow(self, a, b, col=WHITE, width=2, skip=1.8, stop=0.0):
        """Arrow from a to b. It starts `skip` metres along, so it does not cover the player standing at a,
        and ends `stop` metres early (use it when b is a player too)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = dist(a, b)
        if n < skip + stop + 0.8:
            return
        a, b = a + (b - a) * (skip / n), b - (b - a) * (stop / n)
        pa, pb = self.px(a), self.px(b)
        pygame.draw.line(self.surf, col, pa, pb, self.u(width))
        ang = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
        for d in (2.6, -2.6):
            tip = (pb[0] + self.u(11) * math.cos(ang + d), pb[1] + self.u(11) * math.sin(ang + d))
            pygame.draw.line(self.surf, col, pb, tip, self.u(width))

    def ring(self, p, r=10, col=WHITE, width=2):
        pygame.draw.circle(self.surf, col, self.px(p), self.u(r), self.u(width))

    def cross(self, p, col=WHITE, size=7, width=2):
        x, y = self.px(p)
        s = self.u(size)
        pygame.draw.line(self.surf, col, (x - s, y), (x + s, y), self.u(width))
        pygame.draw.line(self.surf, col, (x, y - s), (x, y + s), self.u(width))

    def circle(self, p, metres, col=CYAN, width=1):
        """Dashed circle of a radius given in metres."""
        cx, cy = self.px(p)
        r = pr.w2r(metres, self.m.L)
        n = max(24, int(r / 3))
        for i in range(0, n, 2):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            pygame.draw.line(self.surf, col, (cx + r * math.cos(a0), cy + r * math.sin(a0)),
                             (cx + r * math.cos(a1), cy + r * math.sin(a1)), self.u(width))

    def vline(self, x, text, col=CYAN, top=True):
        """Dashed line across the pitch at depth x, with a label at one end."""
        self.line((x, 0.0), (x, self.m.W), col, 1, dash=True)
        self.label((x, self.m.W - 1.5 if top else 1.5), text, col, prefer="below" if top else "above", near=True)

    def label(self, p, text, col=WHITE, prefer="above", near=False):
        """A short note next to point p. It is placed at the end, where it does not cover a player or another note."""
        self.labels.append((np.asarray(p, float), text, col, prefer, near))

    def players(self):
        pr.draw_flight(self.surf, self.m.s)
        pr.draw_players(self.surf, self.m.s)
        pr.draw_ball(self.surf, self.m.s)
        self.players_drawn = True

    def finish(self):
        """Players (if the scene did not draw them itself), then the notes, placed so that they do not overlap."""
        if not self.players_drawn:
            self.players()
        font = pr._FONTS["team"]
        pad, r_pl = self.u(5), pr.w2r(1.5, self.m.L) + self.u(4)
        taken = [pygame.Rect(x - r_pl, y - r_pl, 2 * r_pl, 2 * r_pl + self.u(10))
                 for x, y in (self.px(q) for t in (0, 1) for q in self.m.s.pos[t])]
        field = pygame.Rect(pr.PX0, pr.PY0, pr.PITCH_PX_W, pr.PITCH_PX_H).inflate(-self.u(6), -self.u(6))
        ring = {"above": [(0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (1, 1), (-1, 1)],
                "below": [(0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0), (0, -1), (1, -1), (-1, -1)],
                "right": [(1, 0), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)],
                "left": [(-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1), (1, 0), (1, -1), (1, 1)]}
        for p, text, col, prefer, near in self.labels:
            txt = font.render(text, True, col)
            w, h = txt.get_width() + 2 * pad, txt.get_height() + 2 * pad
            x, y = self.px(p)
            gap = self.u(5) if near else r_pl + self.u(5)
            spot = None
            for reach in (1.0, 2.2, 3.6, 5.0):
                for dx, dy in ring[prefer]:
                    rect = pygame.Rect(0, 0, w, h)
                    rect.center = (int(x + dx * (gap * reach + w / 2)), int(y + dy * (gap * reach + h / 2)))
                    if dx == 0:
                        rect.centerx = x
                    if dy == 0:
                        rect.centery = y
                    if field.contains(rect) and rect.collidelist(taken) < 0:
                        spot = rect
                        break
                if spot:
                    break
            if spot is None:                           # crowded: accept an overlap rather than drop the note
                spot = pygame.Rect(0, 0, w, h)
                spot.center = (x, int(y - gap - h / 2))
                spot.clamp_ip(field)
            near_pt = (min(max(x, spot.left), spot.right), min(max(y, spot.top), spot.bottom))
            if math.hypot(near_pt[0] - x, near_pt[1] - y) > gap * 1.6:       # moved away: tie it to its point
                pygame.draw.line(self.surf, col, (x, y), near_pt, 1)
            box = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(box, (12, 14, 18, 225), (0, 0, w, h), border_radius=self.u(6))
            pygame.draw.rect(box, (*col, 255), (0, 0, w, h), width=1, border_radius=self.u(6))
            box.blit(txt, (pad, pad))
            self.surf.blit(box, spot)
            taken.append(spot)


def picture(m, title, notes, width=1180):
    """One annotated picture of moment m: a title strip, then the pitch (the viewer's bars are cut away)."""
    height = int(width * 0.64)
    pygame.display.set_mode((width, height))
    pr.set_layout(width, height, m.L, m.W)
    surf = pr.build_background(m.s)
    pen = Pen(surf, m)
    notes(pen)
    pen.finish()
    gx, gy = pr.w2r(4.5, m.L), int(12 * pr.UI)             # keep the goals behind the goal lines
    crop = pygame.Rect(pr.PX0 - gx, pr.PY0 - gy, pr.PITCH_PX_W + 2 * gx, pr.PITCH_PX_H + 2 * gy)
    strip = int(40 * pr.UI)
    out = pygame.Surface((crop.w, crop.h + strip))
    out.fill(pr.C_HUD_BG)
    txt = pr._FONTS["med"].render(title, True, WHITE)
    out.blit(txt, (int(14 * pr.UI), (strip - txt.get_height()) // 2))
    out.blit(surf, (0, strip), crop)
    return out


def stack(pics, gap=8):
    """Several pictures one above the other (same depth scale, so the lines of the teams can be compared)."""
    out = pygame.Surface((max(p.get_width() for p in pics), sum(p.get_height() for p in pics) + gap * (len(pics) - 1)))
    out.fill(pr.C_HUD_BG)
    y = 0
    for p in pics:
        out.blit(p, (0, y))
        y += p.get_height() + gap
    return out


# ─────────────────────────────────────────────  helpers for the scenes

def room(m, points):
    """The smallest distance from any of the points to another player: big = the picture will not be crowded."""
    everyone = [q for t in (0, 1) for q in m.s.pos[t]]
    return min(min(d for d in (dist(p, q) for q in everyone) if d > 0.3) for p in points)


def when_completed(m, score):
    """For moments at which red plays (or has just played) a pass: keep the best one until the pass is over, and
    hand it back as (score, moment) only if the pass was completed. score = None means "not such a moment"."""
    if score is not None and ("pending" not in m.memo or score > m.memo["pending"][0]):
        m.memo["pending"] = (score, m)
    for e in m.events:
        if e["type"] == "pass_result" and e["passer"][0] == RED:
            held = m.memo.pop("pending", None)
            if held is not None and e["result"] == "completed":
                return held
    return None


def carrier_of(o):
    """Where the teammate with the ball is, as player o sees it (team frame), or None."""
    for k in range(len(o.teammate_ids)):
        if o.teammate_has_ball[k]:
            return o.teammate_pos[k]
    return None


def support_rule(m, j):
    """Which rule of ScriptedTeam._support moved red player j on this step. Mirrors its order of checks."""
    r = m.team[RED]
    ct, o = r.ct, r.obs[j]
    mv, L, W = ct.mv, m.L, m.W
    is_mover, counter = r.sup[j]
    if is_mover:
        return "pass_move"
    car = carrier_of(o)
    home = ct._home(o.anchor, o.slot_type)
    if car is not None:
        c_side, wide = np.sign(car[1] - W / 2), abs(car[1] - W / 2) > mv.box.wide
        if car[0] > mv.box.x * L and wide:
            if o.slot_type == "F":
                return "far_post"
            if o.slot_type == "M" and np.sign(home[1] - W / 2) != c_side:
                return "cut_back"
        side = np.sign(home[1] - W / 2)
        if (o.slot_type == "M" and side != 0 and mv.overlap.trigger > 0 and car[0] > 0.45 * L and o.pos[0] < car[0]
                and dist(o.pos, car) <= mv.overlap.trigger and (c_side == side or not wide)):
            return "overlap"
        if o.slot_type == "F" and mv.drop_deep.dist > 0 and car[0] < home[0] - mv.drop_deep.trigger:
            return "drop_deep"
    if car is not None and ct.run.enabled and (o.slot_type == "F" or (counter and o.slot_type == "M")):
        return "run"
    if o.slot_type == "M" and car is not None:
        return "offer" if car[0] > o.pos[0] and ct.su.offer_dist > 0 else "toward_ball"
    return {"F": "advance", "D": "step_up"}.get(o.slot_type, "hold")


def red_supporting(m):
    """{player: rule} for red's off-ball players while a red player holds the ball, or None if that is not the case."""
    h = m.s.holder
    if m.dirty or h is None or h[0] != RED:
        return None
    sup = m.team[RED].sup
    return {j: support_rule(m, j) for j in sup} if len(sup) == 4 else None


SUPPORT_NOTE = {"run": "runs ahead of the ball", "offer": "offers a pass", "overlap": "overlaps", "step_up": "steps up",
                "far_post": "far post", "cut_back": "cut-back spot", "drop_deep": "comes short", "pass_move": "passed, runs on",
                "toward_ball": "moves toward the ball", "advance": "pushes up", "hold": "holds"}


def draw_support(pen, m, rules, strong=()):
    """Arrows from red's off-ball players to where they are going, each with the name of its rule."""
    for j, rule in rules.items():
        tgt = m.target(RED, j)
        if tgt is None:
            continue
        col = WHITE if (not strong or rule in strong) else GREY
        pen.arrow(m.s.pos[RED][j], tgt, col)
        pen.label(m.s.pos[RED][j], SUPPORT_NOTE[rule], col)


def depth_lines(pen, m):
    """Red's home positions (+) and a dashed line at the depth of each of its lines, in metres from its own goal."""
    seen = {}
    for j in range(5):
        a = m.s.anchors[RED][j]
        pen.cross(a, CYAN)
        seen.setdefault(m.slot(RED, j), []).append(a[0])
    names = {"D": "defenders", "M": "midfielders", "F": "forward"}
    for slot, xs in seen.items():
        x = float(np.mean(xs))
        pen.vline(x, f"{names[slot]} {m.L - x:.0f} m", CYAN)


# ─────────────────────────────────────────────  scenes

def scene_lanes():
    """Aggressive style: the presser on the ball, the others in the middle of each passing line."""
    def pick(m):
        r, h = m.team[RED], m.s.holder
        if m.dirty or h is None or h[0] != BLUE or r.ctx["presser"] is None or len(r.ctx["lanes"]) < 3:
            return None
        pts = [m.w(RED, p) for p in r.ctx["lanes"].values()]
        arrived = sum(dist(m.s.pos[RED][j], m.w(RED, p)) <= 1.5 for j, p in r.ctx["lanes"].items())
        d_press = dist(m.s.pos[RED][r.ctx["presser"]], m.s.ball_pos)
        if arrived < 3 or d_press > 5.0:
            return None
        return min(dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1:]) - d_press   # lines well apart

    m = find(pick, styles=("normal", "aggressive"))

    def notes(pen):
        r, ball, carrier = m.team[RED], m.s.ball_pos, m.s.holder[1]
        ball_tf = m.w(RED, ball)
        for k in range(5):
            if k == carrier:
                continue
            back = m.w(RED, m.s.pos[BLUE][k])[0] > ball_tf[0] + r.ct.ln.side      # behind the ball as red sees it
            pen.line(ball, m.s.pos[BLUE][k], GREY if back else YELLOW, 1 if back else 2, dash=back)
            if back:
                pen.label((ball + m.s.pos[BLUE][k]) / 2, "back pass: left open", GREY, near=True)
        pen.players()
        for j, p in r.ctx["lanes"].items():
            k = min((k for k in range(5) if k != carrier), key=lambda k: dist((ball + m.s.pos[BLUE][k]) / 2, m.w(RED, p)))
            pen.label(m.s.pos[RED][j], f"blocks the pass to {k + 1}", WHITE)
        pen.label(m.s.pos[RED][r.ctx["presser"]], "presser", ORANGE, prefer="right")

    return picture(m, "Aggressive style (red): one presser, the others in the middle of each passing line", notes)


def scene_pressing():
    """Zonal defence: the presser, the second presser of a double team, and the cover."""
    def pick(m):
        r, h = m.team[RED], m.s.holder
        c = r.ctx
        if m.dirty or h is None or h[0] != BLUE or None in (c["presser"], c["presser2"], c["cover"]):
            return None
        ball_tf = m.w(RED, m.s.ball_pos)
        three = [m.s.pos[RED][c[k]] for k in ("presser", "presser2", "cover")]
        d1, d2 = dist(three[0], m.s.ball_pos), dist(three[1], m.s.ball_pos)
        tgt = m.target(RED, c["cover"])
        if not (14 <= ball_tf[0] <= 50) or not (3.0 <= d1 <= 7.0) or d2 < d1 + 2.0 or tgt is None:
            return None
        if dist(m.s.ball_pos, m.s.anchors[RED][c["presser"]]) > r.ct.z.r_zone:        # show the plain zone case
            return None
        return min(room(m, three), 6.0) + min(dist(three[2], tgt), 4.0)               # spread out, cover on the move

    m = find(pick)

    def notes(pen):
        r = m.team[RED]
        c, z = r.ctx, r.ct.z
        ball, home = m.s.ball_pos, m.s.anchors[RED][c["presser"]]
        pen.circle(home, z.r_zone, CYAN)
        pen.cross(home, CYAN)
        pen.label(home, f"presser's home (+) and zone ({z.r_zone:.0f} m)", CYAN, prefer="below", near=True)
        pen.players()
        for j in range(5):
            press = j in (c["presser"], c["presser2"])
            tgt = ball if press else m.target(RED, j)
            if tgt is not None:
                pen.arrow(m.s.pos[RED][j], tgt, ORANGE if press else (WHITE if j == c["cover"] else GREY))
        pen.label(m.s.pos[RED][c["presser"]], "presser", ORANGE)
        pen.label(m.s.pos[RED][c["presser2"]], "second presser", ORANGE)
        pen.label(m.s.pos[RED][c["cover"]], "cover", WHITE)
        for j in range(5):
            if j not in (c["presser"], c["presser2"], c["cover"]):
                pen.label(m.s.pos[RED][j], "drifts to the ball side", GREY, prefer="below")

    return picture(m, "Defending (red): presser, second presser, cover, and the ball-side drift of the rest", notes)


def scene_counter_press():
    """Just after losing the ball: the two nearest players chase it."""
    def pick(m):
        r, h = m.team[RED], m.s.holder
        prev = m.memo.get("prev")
        cur = h[0] if h is not None else (m.s.flight["passer"][0] if m.s.flight else prev)
        if prev == RED and cur == BLUE:
            m.memo["lost"] = (m.s.ball_pos.copy(), m.s.t)
        m.memo["prev"] = cur
        if m.dirty or h is None or h[0] != BLUE or "lost" not in m.memo or not (2 <= m.s.t - m.memo["lost"][1] <= 4):
            return None
        ch = r.ctx["chasers"]
        if r.cp <= 0 or len(ch) != 2:
            return None
        ds = [dist(m.s.pos[RED][j], m.s.ball_pos) for j in ch]
        if not all(3.0 <= d <= 14.0 for d in ds) or dist(m.s.pos[RED][ch[0]], m.s.pos[RED][ch[1]]) < 5.0:
            return None
        m.lost = m.memo["lost"][0]
        return min(room(m, [m.s.pos[RED][j] for j in ch]), 5.0) - abs(ds[0] - ds[1]) / 4

    m = find(pick)

    def notes(pen):
        r = m.team[RED]
        pen.cross(m.lost, YELLOW, 6)
        pen.label(m.lost, "red lost the ball here", YELLOW, prefer="below", near=True)
        pen.players()
        for j in r.ctx["chasers"]:
            pen.arrow(m.s.pos[RED][j], m.s.ball_pos, ORANGE)
            pen.label(m.s.pos[RED][j], "counter-press", ORANGE)

    return picture(m, f"Counter-press (red): for {m.team[RED].ct.mv.counter_press.steps} steps after losing the ball, the two nearest players chase it", notes)


def scene_release():
    """The carrier gets rid of the ball when a defender closes in."""
    def score(m):
        r, h = m.team[RED], m.s.holder
        if m.dirty or h is None or h[0] != RED or m.act(RED, h[1]) != Kind.PASS:
            return None
        o, cr = r.obs[h[1]], r.ct.cr
        k = int(r.out[h[1]].arg)
        me, mate = m.s.pos[RED][h[1]], m.s.pos[RED][o.teammate_ids[k]]
        d = min(dist(me, q) for q in m.s.pos[BLUE])
        free = min(dist(mate, q) for q in m.s.pos[BLUE])
        back = o.teammate_pos[k][0] < o.pos[0] - 3.0             # a back pass can only come from this rule here
        if not (cr.press_dist + 0.4 < d <= cr.release_dist) or not back or free < cr.free_dist + 1 or o.pos[0] > 62:
            return None
        m.mate = o.teammate_ids[k]
        return min(free, 16) / 4 - abs(d - 4.6) + min(dist(me, mate), 18) / 9 + min(room(m, [me, mate]), 4)

    m = find(lambda m: when_completed(m, score(m)))

    def notes(pen):
        cr = m.team[RED].ct.cr
        me, mate = m.s.pos[RED][m.s.holder[1]], m.s.pos[RED][m.mate]
        near = min(m.s.pos[BLUE], key=lambda q: dist(me, q))
        pen.circle(me, cr.release_dist, CYAN)
        pen.circle(mate, cr.free_dist, GREY)
        pen.players()
        pen.arrow(me, mate, YELLOW, stop=1.8)
        pen.label(near, f"defender closing in: {dist(me, near):.1f} m", ORANGE, prefer="below")
        pen.label(me - np.array([cr.release_dist, 0.0]), f"{cr.release_dist:.0f} m round the carrier", CYAN, prefer="left", near=True)
        pen.label(mate + np.array([0.0, cr.free_dist]), f"free: no defender within {cr.free_dist:.0f} m", GREY, near=True)

    return picture(m, "On the ball (red): a defender is closing in, so the carrier gives it to a free teammate", notes)


def scene_through():
    """A through ball: played to a point ahead of a runner."""
    def pick(m):
        r, h = m.team[RED], m.s.holder
        if m.dirty or h is None or h[0] != RED or m.act(RED, h[1]) != Kind.THROUGH:
            return None
        arg = r.out[h[1]].arg
        m.mate = r.obs[h[1]].teammate_ids[int(arg[0])]
        m.pt = m.w(RED, arg[1:])
        run = dist(m.s.pos[RED][m.mate], m.pt)
        if run < 6.5:
            return None
        return run + min(dist(m.pt, q) for q in m.s.pos[BLUE]) / 3 + min(room(m, [m.pt, m.s.pos[RED][m.mate]]), 5)

    m = find(lambda m: when_completed(m, pick(m)))

    def notes(pen):
        me, mate = m.s.pos[RED][m.s.holder[1]], m.s.pos[RED][m.mate]
        pen.line(me, m.pt, YELLOW, 2, dash=True)
        pen.players()
        pen.ring(m.pt, 9, YELLOW)
        pen.arrow(mate, m.pt, WHITE)
        pen.label(m.pt, f"ball lands {dist(mate, m.pt):.0f} m ahead of the runner", YELLOW, near=True)
        pen.label(mate, "runs onto it", WHITE, prefer="below")
        pen.label(me, "through ball", YELLOW, prefer="below")

    return picture(m, "Through ball (red): played into space ahead of a runner who is known to be free", notes)


def scene_support():
    """Off-ball support while a midfielder or defender builds up."""
    def pick(m):
        rules = red_supporting(m)
        if rules is None or m.slot(RED, m.s.holder[1]) == "F" or m.s.phase[RED] < 0.9 or m.team[RED].ca > 0:
            return None
        car = m.w(RED, m.s.pos[RED][m.s.holder[1]])
        want = set(rules.values())
        if not (52 <= car[0] <= 68) or not {"run", "offer"} <= want or want - {"run", "offer", "step_up", "overlap"}:
            return None
        moves = [dist(m.s.pos[RED][j], m.target(RED, j)) for j in rules if m.target(RED, j) is not None]
        if len(moves) < 4:
            return None
        m.rules = rules
        return sum(min(x, 9.0) for x in moves) / 4 + min(room(m, [m.s.pos[RED][j] for j in rules]), 5.0)

    m = find(pick)

    def notes(pen):
        pen.players()
        draw_support(pen, m, m.rules)
        pen.label(m.s.pos[RED][m.s.holder[1]], "has the ball", YELLOW, prefer="below")

    return picture(m, "Off the ball (red): the forward runs ahead, midfielders offer a pass, defenders step up", notes)


def scene_pass_move():
    """Pass and move: the passer keeps running forward."""
    def pick(m):
        r, f = m.team[RED], m.s.flight
        if m.dirty or m.s.ball_state != "flight" or f is None or f["passer"][0] != RED or f["designated"] is None:
            return None
        j = f["passer"][1]
        tgt = m.target(RED, j)
        if j not in r.sup or not r.sup[j][0] or tgt is None or f["k"] < 2:
            return None
        me = m.s.pos[RED][j]
        run = m.w(RED, tgt)[0] - m.w(RED, me)[0]                 # how far forward the passer is still going
        across = abs(f["end"][1] - f["start"][1])                # a pass across the pitch does not hide that run
        if run < 6.0 or across < 9.0 or f["designated"][0] != RED:
            return None
        return run + min(room(m, [me, tgt]), 5.0) + min(across, 20) / 4

    m = find(lambda m: when_completed(m, pick(m)))

    def notes(pen):
        f = m.s.flight
        j = f["passer"][1]
        pen.players()
        pen.arrow(m.s.pos[RED][j], m.target(RED, j), WHITE)
        pen.label(m.s.pos[RED][j], "has just passed, runs on", WHITE)
        pen.label(m.s.pos[RED][f["intended"][1]], "receives the pass", YELLOW, prefer="below")

    return picture(m, f"Pass and move (red): the passer runs forward for {m.team[RED].ct.mv.pass_move.steps} steps after passing", notes)


def support_scene(rule_names, title, strong, seeds=range(6), styles=("normal", "normal"), extra=None, watch=None,
                  carrier_note="has the ball"):
    """A picture of the off-ball rules in rule_names all being used on the same step. watch(m) is called on every
    step (to remember things); extra(m, rules) can reject a moment."""
    def pick(m):
        if watch:
            watch(m)
        rules = red_supporting(m)
        if rules is None or not set(rule_names) <= set(rules.values()) or (extra and not extra(m, rules)):
            return None
        who = [j for j, x in rules.items() if x in rule_names]
        runs = [dist(m.s.pos[RED][j], m.target(RED, j)) for j in who if m.target(RED, j) is not None]
        if len(runs) < len(who) or min(runs) < 5.0:
            return None
        m.rules = rules
        return min(runs) + min(room(m, [m.s.pos[RED][j] for j in who]), 5.0)

    m = find(pick, styles=styles, seeds=seeds)

    def notes(pen):
        pen.players()
        draw_support(pen, m, m.rules, strong=strong)
        pen.label(m.s.pos[RED][m.s.holder[1]], carrier_note, YELLOW, prefer="below")

    return picture(m, title, notes)


def scene_overlap():
    return support_scene(["overlap"], "Overlap (red): a wide midfielder just behind the carrier runs round the outside", ["overlap"])


def scene_box():
    return support_scene(["far_post", "cut_back"], "Ball wide in the final third (red): far-post run and cut-back spot",
                         ["far_post", "cut_back"], carrier_note="has the ball, out wide")


def scene_drop_deep():
    return support_scene(["drop_deep"], "Dropping deep (red): the ball is far behind the forward, who comes short for it",
                         ["drop_deep"])


def scene_counter():
    """Counter-attack: just after winning the ball in open play, midfielders run forward too."""
    def watch(m):                                                # when did red last win the ball from blue?
        for e in m.events:
            won = ((e["type"] == "tackle" and e["success"] and e["by"][0] == RED)
                   or (e["type"] == "pass_result" and e["result"] == "intercepted" and e["by"][0] == RED))
            if won:
                m.memo["won"] = m.s.t

    def extra(m, rules):
        depth = m.w(RED, m.s.pos[RED][m.s.holder[1]])[0]         # where the ball is, from red's own goal
        mids = [j for j, x in rules.items() if x == "run" and m.slot(RED, j) == "M"]
        return (m.team[RED].ca > 0 and 2 <= m.s.t - m.memo.get("won", -99) <= 9 and 20 <= depth <= 55 and len(mids) >= 1
                and all(m.target(RED, j) is not None
                        and m.w(RED, m.target(RED, j))[0] - m.w(RED, m.s.pos[RED][j])[0] >= 6 for j in mids))

    steps = load_config().opponent.moves.counter.steps
    return support_scene(["run"], f"Counter-attack (red): for {steps} steps after winning the ball, midfielders run forward too",
                         ["run"], extra=extra, watch=watch, carrier_note="has just won the ball")


def shape_moment(style, attacking):
    """A moment when red is in its settled shape and the ball is in a standard place, so that pictures of
    different styles can be compared. Players who are pressing or blocking a pass are allowed to be away from home."""
    def pick(m):
        h = m.s.holder
        if m.dirty or h is None or h[0] != (RED if attacking else BLUE) or m.s.frozen[RED].any():
            return None
        if (m.s.phase[RED] < 0.97) if attacking else (m.s.phase[RED] > -0.97):
            return None
        c = m.team[RED].ctx
        busy = {h[1]} if attacking else {c["presser"], c["presser2"], *c["lanes"], *c["chasers"]}
        off = [dist(m.s.pos[RED][j], m.s.anchors[RED][j]) for j in range(5) if j not in busy]
        if len(off) < 2 or (np.mean(off) > 6.5 if attacking else max(off) > 6.0):
            return None                                          # the players have not reached the shape yet
        ball = m.w(RED, m.s.ball_pos)
        return -abs(ball[0] - (68 if attacking else 55)) - abs(ball[1] - m.W / 2) / 3
    return find(pick, styles=("normal", style))


def shape_picture(style, attacking, width=1000):
    m = shape_moment(style, attacking)
    what = "has the ball: attacking shape" if attacking else "defends: defending shape"

    def notes(pen):
        depth_lines(pen, m)

    return picture(m, f"{style.capitalize()} style (red) {what}", notes, width=width)


def scene_shapes():
    return stack([shape_picture("normal", True), shape_picture("normal", False)])


def scene_styles_defending():
    return stack([shape_picture(s, False) for s in ("normal", "aggressive", "defensive")])


def scene_styles_attacking():
    return stack([shape_picture(s, True) for s in ("normal", "aggressive", "defensive")])


def scene_long_ball():
    """Defensive style: deep block, long ball to the forward."""
    def pick(m):
        r, h = m.team[RED], m.s.holder
        if m.dirty or h is None or h[0] != RED or m.act(RED, h[1]) != Kind.PASS or m.slot(RED, h[1]) != "D":
            return None
        m.mate = r.obs[h[1]].teammate_ids[int(r.out[h[1]].arg)]
        n = dist(m.s.pos[RED][h[1]], m.s.pos[RED][m.mate])
        if m.slot(RED, m.mate) != "F" or n <= r.ct.cfg.opponent.roles.d_max_pass - 12:
            return None
        return n + min(room(m, [m.s.pos[RED][m.mate]]), 6.0)

    m = find(lambda m: when_completed(m, pick(m)), styles=("normal", "defensive"))

    def notes(pen):
        me, mate = m.s.pos[RED][m.s.holder[1]], m.s.pos[RED][m.mate]
        pen.line(me, mate, YELLOW, 2, dash=True)
        pen.players()
        pen.label((me + mate) / 2, f"long ball: {dist(me, mate):.0f} m", YELLOW, near=True)
        pen.label(mate, "forward: the outlet", WHITE)
        pen.label(me, "defender", WHITE, prefer="below")

    return picture(m, "Defensive style (red): a defender breaks with a long ball to the forward", notes)


def scene_switch():
    """A formation switch: players walk to their new home positions."""
    def pick(m):
        for e in m.events:
            if e["type"] == "switch" and e["team"] == RED:
                m.memo["sw"] = (m.s.t, e["from_"], e["to"])
        if "sw" not in m.memo or not m.team[RED].in_transition:
            return None
        m.sw = m.memo["sw"]
        walk = [dist(m.s.pos[RED][j], m.s.anchors[RED][j]) for j in range(5)]
        return -abs(m.s.t - m.sw[0] - 6) + min(max(walk), 12) / 4 + min(room(m, list(m.s.pos[RED])), 5.0)

    m = find(pick, overrides={"opponent.switching.mode": "SCHEDULED", "opponent.switching.scheduled.t_switch": 240})

    def notes(pen):
        for j in range(5):
            pen.cross(m.s.anchors[RED][j], CYAN)
        pen.players()
        for j in range(5):
            if m.s.holder != (RED, j):
                pen.arrow(m.s.pos[RED][j], m.s.anchors[RED][j], WHITE)
        far = max(range(5), key=lambda j: dist(m.s.pos[RED][j], m.s.anchors[RED][j]))
        pen.label(m.s.anchors[RED][far], f"new home position in {m.sw[2]}", CYAN, near=True)

    return picture(m, f"Formation switch (red): {m.sw[1]} to {m.sw[2]}, {m.s.t - m.sw[0]} steps ago. Players walk to the new positions", notes)


def scene_viewer():
    """The viewer window as it looks during a match (for rendering.md)."""
    env = FootballEnv(load_config(), scripted_ours=True)
    env.reset(seed=4, our_style="normal", opp_style="aggressive")
    while env.t < 420 or env.ball_state != "flight":
        env.step()
    w, h = 1280, 800
    screen = pygame.display.set_mode((w, h))
    pr.set_layout(w, h, env.L, env.W)
    screen.blit(pr.build_background(env), (0, 0))
    pr.draw_anchors(screen, env)
    pr.draw_flight(screen, env)
    pr.draw_players(screen, env)
    pr.draw_ball(screen, env)
    pr.draw_footer(screen, env)
    pr.draw_hud(screen, env, dict(fps=20, paused=False, seed=4, show_vision=False, show_anchors=True, show_ghosts=False))
    return screen.copy()


SCENES = {"shapes": scene_shapes, "pressing": scene_pressing, "counter_press": scene_counter_press, "lanes": scene_lanes,
          "release": scene_release, "through": scene_through, "support": scene_support, "pass_move": scene_pass_move,
          "overlap": scene_overlap, "box": scene_box, "drop_deep": scene_drop_deep, "counter": scene_counter,
          "styles_defending": scene_styles_defending, "styles_attacking": scene_styles_attacking,
          "long_ball": scene_long_ball, "switch": scene_switch, "viewer": scene_viewer}


def make(job):
    """Build one picture and save it. Runs in its own process."""
    name, out = job
    pygame.init()
    pr._ft.init()
    pygame.display.set_mode((1180, 755))
    try:
        pic = SCENES[name]()
    except RuntimeError as e:
        return f"{name}: FAILED ({e})"
    path = os.path.join(out, "viewer.png" if name == "viewer" else os.path.join("opponent", name + ".png"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pygame.image.save(pic, path)
    return f"{name}: {pic.get_width()}x{pic.get_height()} -> {os.path.normpath(path)}"


def main():
    ap = argparse.ArgumentParser(description="Annotated screenshots for the docs")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--only", default="", help="comma-separated scene names (default: all)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(SCENES)
    from multiprocessing import Pool
    with Pool(max(1, min(args.jobs, len(names)))) as pool:
        for line in pool.imap_unordered(make, [(n, args.out) for n in names]):
            print(line)


if __name__ == "__main__":
    main()
