"""Top-down 2D Pygame renderer for the football game.

The game is strictly 2D — ball and players have only (x, y) coordinates, no height.
This renderer faithfully represents that with a top-down pitch view.

Run from football_adapt/:
    .venv/bin/python -m scripts.pygame_render --mode scripted --fps 20
    .venv/bin/python -m scripts.pygame_render --mode random --sigma 3 --opp-mode SCHEDULED
    .venv/bin/python -m scripts.pygame_render --style aggressive --opp-style defensive
    .venv/bin/python -m scripts.pygame_render --formation 1-3-1 --opp-formation 3-1-1

Keys: Space pause, Right step, Up/Down speed, R new match (random seed), Shift+R replay, Z / X next play style
for blue / red, N / M next formation for blue / red, V vision, A anchors, F formations, 1-5 highlight, Q quit.
"""
from __future__ import annotations
import argparse
import math
import os
import numpy as np
import pygame
import sys

from football import FootballEnv, load_config
from football.geometry import to_frame, dist
from football.mechanics import shot_probability, sure_goal
from agents import RandomAgent


# ─────────────────────────────────────────────  font shim
#
# pygame.font AND pygame.freetype are both broken on Python 3.14:
# freetype.py → sysfont.py → font.py → sysfont.py  (circular import)
#
# Fix: use pygame._freetype, the raw C extension that underlies both wrappers.
# It has zero Python-level imports, so the circular issue never triggers.
# We load a real monospace TTF from known macOS/Linux paths for aesthetics;
# if none is found we fall back to pygame's built-in font (None path).

import pygame._freetype as _ft   # C extension — no Python circular imports

_MONO_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Monaco.ttf",
    # Linux
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
_MONO_PATH = next((p for p in _MONO_CANDIDATES if os.path.exists(p)), None)


class _FTFont:
    """Wraps pygame._freetype.Font to expose a pygame.font-style .render()."""
    def __init__(self, ft: _ft.Font):
        self._ft = ft

    def render(self, text: str, antialias, color) -> pygame.Surface:
        surf, _ = self._ft.render(str(text), fgcolor=color)
        return surf.convert_alpha()   # ensure transparent background, no black bbox

    def get_height(self) -> int:
        return int(self._ft.size)


def _make_font(size: int, bold: bool = False) -> _FTFont:
    font = _ft.Font(_MONO_PATH, size)   # None → built-in font if no path found
    font.strong = bold
    return _FTFont(font)


# Module-level font cache — populated once by _init_fonts() after pygame.init()
_FONTS: dict = {}


def _init_fonts(ui: float = 1.0):
    """(Re)build every font at the current UI scale, so text stays sharp at any window size."""
    z = lambda n: max(7, int(round(n * ui)))
    _FONTS["hud"]    = _make_font(z(30), bold=True)   # score
    _FONTS["team"]   = _make_font(z(15), bold=True)   # team names
    _FONTS["small"]  = _make_font(z(13))
    _FONTS["med"]    = _make_font(z(18), bold=True)
    _FONTS["big"]    = _make_font(z(52), bold=True)
    _FONTS["player"] = _make_font(z(12), bold=True)
    _FONTS["slot"]   = _make_font(z(9))
    _FONTS["tiny"]   = _make_font(z(10))


# ─────────────────────────────────────────────  layout
# Everything is laid out for the CURRENT window size by set_layout(), which is called at start and on every
# resize. The pitch keeps its true proportions and is drawn at native resolution (no blurry scaling).
WIN_W, WIN_H = 1280, 800
UI = 1.0                 # scale of text and bars relative to the 1280x800 design size
HUD_H = 72               # top bar: scoreboard, clock, status
FOOT_H = 64              # bottom bar: match stats, shot chance, key hints
PX0 = PY0 = 0            # top-left pixel of the pitch
PITCH_PX_W = PITCH_PX_H = 1


def set_layout(w: int, h: int, L: float = 100.0, W: float = 60.0):
    global WIN_W, WIN_H, UI, HUD_H, FOOT_H, PX0, PY0, PITCH_PX_W, PITCH_PX_H
    WIN_W, WIN_H = max(480, int(w)), max(320, int(h))
    UI = max(0.55, min(WIN_W / 1280, WIN_H / 800))
    HUD_H, FOOT_H = int(72 * UI), int(64 * UI)
    pad_x, pad_y = int(48 * UI), int(16 * UI)          # pad_x leaves room for the goals behind the goal lines
    avail_w, avail_h = WIN_W - 2 * pad_x, WIN_H - HUD_H - FOOT_H - 2 * pad_y
    PITCH_PX_W = int(max(50, min(avail_w, avail_h * L / W)))
    PITCH_PX_H = int(PITCH_PX_W * W / L)
    PX0 = (WIN_W - PITCH_PX_W) // 2
    PY0 = HUD_H + pad_y + max(0, (avail_h - PITCH_PX_H) // 2)
    _init_fonts(UI)


# colours
C_PITCH_DARK   = (34,  85, 34)
C_PITCH_LIGHT  = (40,  98, 40)
C_WHITE        = (255, 255, 255)
C_BLACK        = (0,   0,   0)
C_GOAL         = (220, 220, 220)
C_SHOOT_ARC    = (255, 200,  50, 60)
C_TEAM         = [(43, 108, 176), (197, 48, 48)]   # blue, red
C_GOLD         = (255, 200,  30)
C_FROZEN       = (120, 120, 120)
C_BALL         = (255, 255, 255)
C_BALL_OUTLINE = (30,  30,  30)
C_FLIGHT_PASS  = (255, 255, 255)
C_FLIGHT_SHOT  = (255, 140,  40)
C_HUD_BG       = (18,  20,  24)
C_PAGE_BG      = (12,  22,  16)     # around the pitch
C_MUTED        = (150, 156, 165)
C_HUD_TEXT     = (230, 230, 230)
C_ANCHOR       = [(80, 140, 220, 100), (220, 80, 80, 100)]
C_VISION       = (43, 108, 176, 28)
C_VISION_EDGE  = (43, 108, 176, 80)

SLOT_COLOURS = {"D": (100, 200, 255), "M": (100, 255, 150), "F": (255, 160, 60)}

FLASH_GOAL_FRAMES   = 14    # goal / goal-kick badge stays for 14 frames
FLASH_EVENT_FRAMES  = 25
SHOT_HOLD_FRAMES    = 6     # shot replay: frames the ball rests where it ended up (it flies there at pass speed)
GOAL_PAUSE_FRAMES   = 75   # 3-2-1 countdown: 25 frames per digit at 20fps = 3.75s


# ─────────────────────────────────────────────  coordinate helpers

def w2s(wx: float, wy: float, L: float = 100.0, W: float = 60.0):
    """World coords (m) → screen pixel (int, int). y flipped: higher y = higher on screen."""
    sx = PX0 + int(wx / L * PITCH_PX_W)
    sy = PY0 + int((1.0 - wy / W) * PITCH_PX_H)
    return sx, sy


def w2r(metres: float, L: float = 100.0) -> int:
    """World length (m) → pixel radius."""
    return max(1, int(metres / L * PITCH_PX_W))


# ─────────────────────────────────────────────  static background

def build_background(env: FootballEnv) -> pygame.Surface:
    L, W = env.L, env.W
    surf = pygame.Surface((WIN_W, WIN_H))
    surf.fill(C_PAGE_BG)

    # pitch stripes (every 10 m along x)
    n_stripes = int(L / 10)
    for i in range(n_stripes):
        x0 = PX0 + int(i / n_stripes * PITCH_PX_W)
        x1 = PX0 + int((i + 1) / n_stripes * PITCH_PX_W)
        y0 = PY0
        y1 = PY0 + PITCH_PX_H
        c = C_PITCH_DARK if i % 2 == 0 else C_PITCH_LIGHT
        pygame.draw.rect(surf, c, (x0, y0, x1 - x0, y1 - y0))

    # pitch border
    bx = PX0; by = PY0
    bw = PITCH_PX_W; bh = PITCH_PX_H
    pygame.draw.rect(surf, C_WHITE, (bx, by, bw, bh), 2)

    # centre line
    cx = PX0 + PITCH_PX_W // 2
    pygame.draw.line(surf, C_WHITE, (cx, PY0), (cx, PY0 + bh), 1)

    # centre circle
    cy_world = W / 2
    cx_world = L / 2
    cc = w2s(cx_world, cy_world, L, W)
    pygame.draw.circle(surf, C_WHITE, cc, w2r(9.15, L), 1)
    pygame.draw.circle(surf, C_WHITE, cc, w2r(0.6, L))   # centre spot

    gw = env.cfg.pitch.goal_width
    px_y = lambda m: max(1, int(m / W * bh))      # world length (m) -> pixels along y

    def box(x_goal, facing, depth, width, line=1):
        """Rectangle of the given depth/width standing on a goal line (facing = +1 into the pitch from x=0)."""
        (xa, ya), (xb, yb) = w2s(x_goal, W / 2 + width / 2, L, W), w2s(x_goal + facing * depth, W / 2 - width / 2, L, W)
        pygame.draw.rect(surf, C_WHITE, (min(xa, xb), ya, abs(xb - xa), yb - ya), line)

    pitch_rect = pygame.Rect(bx, by, bw, bh)
    arc_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    shoot_r = env.cfg.mechanics.shoot_max
    for x_goal, facing in [(0.0, 1), (L, -1)]:
        pen_d, pen_w = 16.5, min(gw + 33.0, W - 4.0)       # penalty area
        box(x_goal, facing, pen_d, pen_w)
        box(x_goal, facing, 5.5, min(gw + 11.0, W - 8.0))  # goal area
        spot = w2s(x_goal + facing * 11.0, W / 2, L, W)
        pygame.draw.circle(surf, C_WHITE, spot, 2)         # penalty spot
        # penalty arc: the part of the 9.15 m circle around the spot that lies outside the penalty area
        r = w2r(9.15, L)
        half = math.acos((pen_d - 11.0) / 9.15)
        mid = 0.0 if facing == 1 else math.pi
        pygame.draw.arc(surf, C_WHITE, (spot[0] - r, spot[1] - r, 2 * r, 2 * r), mid - half, mid + half, 1)

        # goal: posts and net stand OUTSIDE the pitch, behind the goal line
        depth = 3.0
        (fx, fy0), (_, fy1) = w2s(x_goal, W / 2 + gw / 2, L, W), w2s(x_goal, W / 2 - gw / 2, L, W)
        bxp = w2s(x_goal - facing * depth, 0, L, W)[0]
        net = pygame.Rect(min(fx, bxp), fy0, abs(bxp - fx), fy1 - fy0)
        pygame.draw.rect(surf, (28, 60, 28), net)
        for gx in range(net.left, net.right, 6):
            pygame.draw.line(surf, (90, 110, 90), (gx, net.top), (gx, net.bottom), 1)
        for gy in range(net.top, net.bottom, 6):
            pygame.draw.line(surf, (90, 110, 90), (net.left, gy), (net.right, gy), 1)
        pygame.draw.lines(surf, C_GOAL, False, [(fx, fy0), (bxp, fy0), (bxp, fy1), (fx, fy1)], 3)
        pygame.draw.circle(surf, C_WHITE, (fx, fy0), 4)    # posts
        pygame.draw.circle(surf, C_WHITE, (fx, fy1), 4)

        # shooting zone (radius D_SHOOT_MAX around the goal centre), faint and clipped to the pitch
        cxp, cyp = w2s(x_goal, W / 2, L, W)
        rx, ry = w2r(shoot_r, L), px_y(shoot_r)
        arc_surf.set_clip(pitch_rect)
        pygame.draw.ellipse(arc_surf, C_SHOOT_ARC, (cxp - rx, cyp - ry, 2 * rx, 2 * ry), 1)
    surf.blit(arc_surf, (0, 0))

    # corner arcs (1 m)
    cr = w2r(1.0, L)
    for (cxw, cyw, a0) in [(0, 0, 0.0), (L, 0, math.pi / 2), (L, W, math.pi), (0, W, 3 * math.pi / 2)]:
        cpx, cpy = w2s(cxw, cyw, L, W)
        pygame.draw.arc(surf, C_WHITE, (cpx - cr, cpy - cr, 2 * cr, 2 * cr), a0, a0 + math.pi / 2, 1)

    return surf


# ─────────────────────────────────────────────  drawing helpers

def draw_dashed_line(surface: pygame.Surface, colour, start, end, dash=8, gap=5, width=2):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash if drawing else gap
        x0 = int(start[0] + ux * pos)
        y0 = int(start[1] + uy * pos)
        end_pos = min(pos + seg, length)
        x1 = int(start[0] + ux * end_pos)
        y1 = int(start[1] + uy * end_pos)
        if drawing:
            pygame.draw.line(surface, colour, (x0, y0), (x1, y1), width)
        pos += seg
        drawing = not drawing


def draw_anchors(screen: pygame.Surface, env: FootballEnv):
    for team in (0, 1):
        r, g, b, a = C_ANCHOR[team]
        for i in range(5):
            ax, ay = w2s(*env.anchors[team][i], env.L, env.W)
            size = 6
            pygame.draw.line(screen, (r, g, b), (ax - size, ay), (ax + size, ay), 1)
            pygame.draw.line(screen, (r, g, b), (ax, ay - size), (ax, ay + size), 1)


def draw_formation_ghosts(screen: pygame.Surface, env: FootballEnv):
    """Show anchor positions of all formations as outlined ghost circles."""
    font_tiny = _FONTS["tiny"]
    for f_idx, f_name in enumerate(env.book.names):
        anchors_tf = env.book.anchors_team_frame(f_name)  # (5,2) team frame
        is_current = [env.formation[t] == f_name for t in (0, 1)]
        for team in (0, 1):
            alpha = 180 if is_current[team] else 40
            r, g, b = C_TEAM[team]
            for i in range(5):
                a_w = to_frame(team, anchors_tf[i], env.L, env.W)
                px, py = w2s(*a_w, env.L, env.W)
                pygame.draw.circle(screen, (r, g, b, alpha), (px, py), 8, 1)
                lbl = font_tiny.render(f_name[:3], True, (r, g, b))
                screen.blit(lbl, (px + 9, py - 5))


def draw_vision(screen: pygame.Surface, env: FootballEnv):
    surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    for i in range(5):
        rho = env.vision_radius(0, i)
        if not math.isfinite(rho):
            continue
        px, py = w2s(*env.pos[0][i], env.L, env.W)
        r_px = w2r(rho, env.L)
        pygame.draw.circle(surf, C_VISION, (px, py), r_px)
        pygame.draw.circle(surf, C_VISION_EDGE, (px, py), r_px, 1)
    screen.blit(surf, (0, 0))


def draw_flight(screen: pygame.Surface, env: FootballEnv):
    if env.ball_state != "flight" or env.flight is None:
        return
    f = env.flight
    start_px = w2s(*f["start"], env.L, env.W)
    end_px   = w2s(*f["end"],   env.L, env.W)
    draw_dashed_line(screen, C_FLIGHT_PASS, start_px, end_px, dash=7, gap=5)
    # ball dot along the line
    t = min(1.0, f["k"] / max(1, f["n"]))
    bx = int(start_px[0] + (end_px[0] - start_px[0]) * t)
    by = int(start_px[1] + (end_px[1] - start_px[1]) * t)
    r = w2r(0.9, env.L)
    pygame.draw.circle(screen, C_BALL, (bx, by), r)
    pygame.draw.circle(screen, C_BALL_OUTLINE, (bx, by), r, 1)


def _tri_pts(cx: int, cy: int, r: int) -> list:
    """Equilateral triangle pointing up, circumradius r, centred at (cx, cy)."""
    h  = int(r * 0.866)   # half-base  (r * sin 60°)
    ht = int(r * 1.0)     # top offset (r)
    hb = int(r * 0.5)     # bottom offset (r * cos 60°)
    return [(cx, cy - ht), (cx - h, cy + hb), (cx + h, cy + hb)]


def draw_players(screen: pygame.Surface, env: FootballEnv, highlight: int = -1):
    """Team 0 (ours) = upward triangle; team 1 (opponent) = circle."""
    font      = _FONTS["player"]
    font_slot = _FONTS["slot"]
    slot_types = {t: env.book[env.formation[t]].slot_types for t in (0, 1)}

    for team in (0, 1):
        for i in range(5):
            px, py = w2s(*env.pos[team][i], env.L, env.W)
            r = w2r(1.5, env.L)

            is_holder      = env.holder == (team, i)
            is_playmaker   = (team == 1 and env.sigma_idx == i)
            is_frozen      = env.frozen[team][i] > 0
            is_highlighted = (team == 0 and i == highlight)

            fill = C_FROZEN if is_frozen else C_TEAM[team]

            # choose border colour and width
            if is_playmaker:
                edge, ew = C_GOLD,          3
            elif is_holder:
                edge, ew = C_WHITE,         3
            elif is_highlighted:
                edge, ew = (255, 255, 100), 2
            else:
                edge, ew = C_WHITE,         1

            if team == 0:
                # ── upward triangle ──
                pts = _tri_pts(px, py, r)
                pygame.draw.polygon(screen, fill, pts)
                pygame.draw.polygon(screen, edge, pts, ew)
            else:
                # ── circle ──
                pygame.draw.circle(screen, fill, (px, py), r)
                pygame.draw.circle(screen, edge, (px, py), r, ew)

            # player number centred inside the shape
            ns = font.render(str(i + 1), True, C_WHITE)
            screen.blit(ns, (px - ns.get_width() // 2,
                              py - ns.get_height() // 2))

            # slot type label just below the shape
            st   = slot_types[team][i]
            st_s = font_slot.render(st, True, SLOT_COLOURS.get(st, C_WHITE))
            screen.blit(st_s, (px - st_s.get_width() // 2, py + r + 2))


def draw_ball(screen: pygame.Surface, env: FootballEnv):
    if env.ball_state == "flight":
        return   # drawn by draw_flight
    bx, by = w2s(*env.ball_pos, env.L, env.W)
    r = w2r(0.9, env.L)
    pygame.draw.circle(screen, C_BALL, (bx, by), r)
    pygame.draw.circle(screen, C_BALL_OUTLINE, (bx, by), r, 1)
    # simple pentagon patches
    for k in range(5):
        angle = k * 2 * math.pi / 5 - math.pi / 2
        dx = int(r * 0.55 * math.cos(angle))
        dy = int(r * 0.55 * math.sin(angle))
        pygame.draw.circle(screen, (80, 80, 80), (bx + dx, by + dy), max(1, r // 4))


def draw_shot_indicator(screen: pygame.Surface, env: FootballEnv, font_small):
    if env.holder is None:
        return
    ht, hi = env.holder
    me_w = env.pos[ht][hi]
    me_tf = to_frame(ht, me_w, env.L, env.W)
    goal_tf = np.array([env.L, env.W / 2])
    if dist(me_tf, goal_tf) > env.cfg.mechanics.shoot_max:
        return
    defs_tf = [to_frame(ht, env.pos[1 - ht][j], env.L, env.W) for j in range(5)]
    p = shot_probability(me_tf, goal_tf, defs_tf,
                         env.control(ht, hi),
                         env.cfg.mechanics.p_max,
                         env.cfg.mechanics.d0,
                         env.cfg.mechanics.shot_block_scale, sure_goal(env.cfg.mechanics))
    bar_w, bar_h = int(170 * UI), int(12 * UI)
    bar_x = WIN_W - bar_w - int(20 * UI)
    bar_y = WIN_H - FOOT_H + int(30 * UI)
    colour = (60, 200, 60) if p > 0.5 else (220, 200, 0) if p > 0.25 else (200, 60, 60)
    pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(screen, colour, (bar_x, bar_y, int(bar_w * p), bar_h))
    pygame.draw.rect(screen, (150, 150, 150), (bar_x, bar_y, bar_w, bar_h), 1)
    lbl = font_small.render(f"Shot chance  {p:.0%}", True, C_WHITE)
    screen.blit(lbl, (bar_x, bar_y - lbl.get_height() - int(6 * UI)))


def _clamp_color(r, g, b, a) -> tuple:
    """Return an RGBA tuple with each component clamped to 0–255."""
    return (max(0, min(255, int(r))),
            max(0, min(255, int(g))),
            max(0, min(255, int(b))),
            max(0, min(255, int(a))))


def _pill(screen: pygame.Surface, cx: int, cy: int, text: str,
          text_col, bg_col, alpha: int, font, pad_x=22, pad_h=14):
    """Draw a rounded pill (dark bg + coloured text) centred at (cx, cy)."""
    alpha = max(0, min(255, int(alpha)))
    txt_surf = font.render(text, True, text_col)
    tw, th = txt_surf.get_size()
    pw, ph = max(1, tw + pad_x * 2), max(1, th + pad_h)
    radius  = min(ph // 2, pw // 2)          # never exceed half the shorter side
    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    bg      = _clamp_color(*bg_col,   min(alpha, 200))
    outline = _clamp_color(*text_col, min(alpha, 220))
    pygame.draw.rect(surf, bg,      (0, 0, pw, ph), border_radius=radius)
    pygame.draw.rect(surf, outline, (0, 0, pw, ph), width=2, border_radius=radius)
    surf.blit(txt_surf, (pad_x, pad_h // 2))
    screen.blit(surf, (cx - pw // 2, cy - ph // 2))


def draw_flashes(screen: pygame.Surface, flashes: list):
    """Draw and tick all active flash animations. Mutates the list in-place."""
    font_big = _FONTS["big"]
    font_med = _FONTS["med"]
    to_remove = []

    for fl in flashes:
        fl["frames"] -= 1
        if fl["frames"] <= 0:
            to_remove.append(fl)
            continue
        t   = fl["frames"]
        kind = fl["kind"]

        if kind == "goal":
            # Simple linear fade-out from full opacity; pill sits just below HUD
            alpha   = int(255 * t / FLASH_GOAL_FRAMES)
            team_c  = C_TEAM[fl["team"]]
            cy_pill = PY0 + int(30 * UI)  # top of the pitch
            _pill(screen, WIN_W // 2, cy_pill,
                  f"GOAL  {fl['score'][0]}–{fl['score'][1]}", team_c, (10, 10, 10),
                  alpha, font_med, pad_x=18, pad_h=10)

        elif kind == "tackle":
            px, py = fl["pos"]
            for ang in range(0, 360, 45):
                rad = math.radians(ang)
                fade = int(255 * t / FLASH_EVENT_FRAMES)
                length = int(8 + 6 * (1.0 - t / FLASH_EVENT_FRAMES))
                ex = px + int(length * math.cos(rad))
                ey = py + int(length * math.sin(rad))
                pygame.draw.line(screen, (255, 220, 0), (px, py), (ex, ey), 2)

        elif kind == "interception":
            alpha = int(255 * t / FLASH_EVENT_FRAMES)
            _pill(screen, fl["pos"][0], fl["pos"][1],
                  "INT", (255, 160, 40), (20, 20, 20), alpha, font_med, pad_x=10, pad_h=8)

        elif kind == "turnover":
            alpha = int(255 * t / FLASH_EVENT_FRAMES)
            _pill(screen, fl["pos"][0], fl["pos"][1],
                  "LOST", (220, 60, 60), (20, 20, 20), alpha, font_med, pad_x=10, pad_h=8)

        elif kind == "pass_ok":
            pygame.draw.circle(screen, (60, 220, 60), fl["pos"], 7, 2)

        elif kind == "clear":
            alpha = int(255 * t / FLASH_EVENT_FRAMES)
            _pill(screen, fl["pos"][0], fl["pos"][1] - 22,
                  "CLEARED", C_TEAM[fl["team"]], (240, 240, 240), alpha, font_med, pad_x=10, pad_h=8)

        elif kind == "goal_kick":
            alpha = int(255 * min(1.0, 2.0 * t / FLASH_GOAL_FRAMES))
            _pill(screen, fl["pos"][0], fl["pos"][1] - 26,
                  "GOAL KICK", C_TEAM[fl["team"]], (240, 240, 240), alpha, font_med, pad_x=10, pad_h=8)

    for fl in to_remove:
        flashes.remove(fl)


def draw_kickoff_overlay(screen: pygame.Surface, env: FootballEnv, frames_left: int,
                         _kicking_team: int):
    """Just a big 3-2-1 digit at the centre. No overlays, no color tinting."""
    cx, cy = w2s(env.L / 2, env.W / 2, env.L, env.W)

    third = max(1, GOAL_PAUSE_FRAMES // 3)
    digit = max(1, min(3, frames_left // third + 1))  # 74→3, 49→2, 24→1

    # Short fade-out in the last few frames before each digit switches
    pos_in_third = frames_left % third
    alpha = 255 if pos_in_third > 4 else int(255 * pos_in_third / 4)

    font  = _FONTS["big"]
    # Drop-shadow (dark, 2 px offset) for readability on the green pitch
    sh = font.render(str(digit), True, (15, 15, 15))
    screen.blit(sh, (cx - sh.get_width() // 2 + 2,
                     cy - sh.get_height() // 2 + 2))
    # White digit
    ds = font.render(str(digit), True, (240, 240, 240))
    if alpha < 255:
        tmp = pygame.Surface(ds.get_size(), pygame.SRCALPHA)
        tmp.blit(ds, (0, 0))
        tmp.set_alpha(alpha)
        screen.blit(tmp, (cx - ds.get_width() // 2, cy - ds.get_height() // 2))
    else:
        screen.blit(ds, (cx - ds.get_width() // 2, cy - ds.get_height() // 2))


def make_shot_anim(event: dict, env: FootballEnv, pre_pos) -> dict:
    """Replay of a shot: from the shooter to inside the net (goal) or past the goal line wide of the post (miss)."""
    team, idx = event["by"]
    L, W, gw = env.L, env.W, env.cfg.pitch.goal_width
    start = np.array(pre_pos[team][idx], float)
    out = 1.0 if team == 0 else -1.0              # team 0 shoots at x = L, team 1 at x = 0
    gx = L if team == 0 else 0.0
    side = 1.0 if start[1] >= W / 2 else -1.0     # the ball stays on the shooter's side of the goal
    if event["scored"]:
        end = np.array([gx + out * 1.8, W / 2 + side * min(abs(start[1] - W / 2) * 0.3, gw / 2 - 1.5)])
    else:
        end = np.array([gx + out * 4.0, W / 2 + side * (gw / 2 + 4.0)])
    fly = max(1, math.ceil(dist(start, end) / env.cfg.speeds["pass"]))   # same ball speed as in play, one step per frame
    return {"start": start, "end": end, "scored": bool(event["scored"]), "team": team,
            "pre_pos": pre_pos, "fly": fly, "frames": fly + SHOT_HOLD_FRAMES}


def draw_shot_anim(screen: pygame.Surface, env: FootballEnv, an: dict):
    """Draw the ball of a shot replay and its label. Ticks the animation."""
    an["frames"] -= 1
    k = min(1.0, (an["fly"] + SHOT_HOLD_FRAMES - an["frames"]) / an["fly"])
    pos = an["start"] + k * (an["end"] - an["start"])
    s_px, b_px = w2s(*an["start"], env.L, env.W), w2s(*pos, env.L, env.W)
    draw_dashed_line(screen, C_FLIGHT_SHOT, s_px, b_px, dash=7, gap=5)
    r = w2r(0.9, env.L)
    pygame.draw.circle(screen, C_BALL, b_px, r)
    pygame.draw.circle(screen, C_BALL_OUTLINE, b_px, r, 1)
    if k >= 1.0:
        e_px = w2s(*an["end"], env.L, env.W)
        tx = min(max(e_px[0], 90), WIN_W - 90)
        if an["scored"]:
            _pill(screen, tx, e_px[1] - 30, "GOAL!", C_TEAM[an["team"]], (255, 255, 255), 255, _FONTS["med"],
                  pad_x=12, pad_h=8)
        else:
            _pill(screen, tx, e_px[1] - 30, "MISS - OUT", (40, 40, 40), (255, 200, 60), 255, _FONTS["med"],
                  pad_x=12, pad_h=8)


def make_flash(event: dict, env: FootballEnv) -> dict | None:
    kind = event.get("type")
    if kind == "goal":
        return {"kind": "goal", "team": event["team"],
                "score": (env.score[0], env.score[1]),
                "frames": FLASH_GOAL_FRAMES}
    if kind == "goal_kick":
        tt, ti = event["by"]
        px, py = w2s(*env.pos[tt][ti], env.L, env.W)
        return {"kind": "goal_kick", "team": tt, "pos": (px, py), "frames": FLASH_GOAL_FRAMES}
    if kind == "clear":
        tt, ti = event["by"]
        px, py = w2s(*env.pos[tt][ti], env.L, env.W)
        return {"kind": "clear", "team": tt, "pos": (px, py), "frames": FLASH_EVENT_FRAMES}
    if kind == "tackle" and event.get("success"):
        tt, ti = event["by"]
        px, py = w2s(*env.pos[tt][ti], env.L, env.W)
        return {"kind": "tackle", "pos": (px, py), "frames": FLASH_EVENT_FRAMES}
    if kind == "pass_result":
        result = event.get("result")
        if result == "intercepted":
            dt, di = event["by"]
            px, py = w2s(*env.pos[dt][di], env.L, env.W)
            return {"kind": "interception", "pos": (px - 12, py - 10), "frames": FLASH_EVENT_FRAMES}
        if result == "completed":
            dt, di = event["by"]
            px, py = w2s(*env.pos[dt][di], env.L, env.W)
            return {"kind": "pass_ok", "pos": (px, py), "frames": FLASH_EVENT_FRAMES}
    return None


def _text(screen, font, text, colour, x, y, anchor="l"):
    """Blit text with its left / centre / right edge at x. Returns the rect it covers."""
    surf = font.render(text, True, colour)
    w = surf.get_width()
    x0 = x if anchor == "l" else x - w // 2 if anchor == "c" else x - w
    screen.blit(surf, (x0, y))
    return pygame.Rect(x0, y, w, surf.get_height())


def _tag(screen, font, text, fg, bg, x, y, anchor="l"):
    """Small filled label (status chip). Returns its rect."""
    surf = font.render(text, True, fg)
    pw, ph = surf.get_width() + int(16 * UI), surf.get_height() + int(8 * UI)
    x0 = x if anchor == "l" else x - pw // 2 if anchor == "c" else x - pw
    pygame.draw.rect(screen, bg, (x0, y, pw, ph), border_radius=ph // 2)
    screen.blit(surf, (x0 + int(8 * UI), y + int(4 * UI)))
    return pygame.Rect(x0, y, pw, ph)


def draw_hud(screen: pygame.Surface, env: FootballEnv, ui: dict):
    """Top bar. Left: clock and status. Centre: scoreboard (the team with the ball is underlined).
    Right: seed and speed. A thin bar along the bottom shows match progress with a half-time tick."""
    u = lambda n: int(n * UI)
    pygame.draw.rect(screen, C_HUD_BG, (0, 0, WIN_W, HUD_H))
    f_score, f_team, f_small = _FONTS["hud"], _FONTS["team"], _FONTS["small"]
    cx = WIN_W // 2
    names = ["BLUE", "RED"]

    # scoreboard
    sc = f_score.render(f"{env.score[0]}  -  {env.score[1]}", True, C_WHITE)
    sy = u(10)
    screen.blit(sc, (cx - sc.get_width() // 2, sy))
    gap = sc.get_width() // 2 + u(22)
    for team, sign in ((0, -1), (1, 1)):
        r = _text(screen, f_team, names[team], C_WHITE, cx + sign * gap, sy + u(2), "r" if sign < 0 else "l")
        chip_x = r.left - u(16) if sign < 0 else r.right + u(16)
        if team == 0:
            pygame.draw.polygon(screen, C_TEAM[0], _tri_pts(chip_x, r.centery + u(1), u(8)))
        else:
            pygame.draw.circle(screen, C_TEAM[1], (chip_x, r.centery), u(7))
        sub = f"{env.formation[team]}  {env.style[team]}"
        if team == 1 and env.sigma_idx is not None:
            sub += f"  playmaker {env.sigma_idx + 1}"
        _text(screen, f_small, sub, C_MUTED, cx + sign * gap, r.bottom + u(6), "r" if sign < 0 else "l")
        if env.holder is not None and env.holder[0] == team:      # possession underline
            pygame.draw.rect(screen, C_TEAM[team], (r.left, r.bottom + u(2), r.width, max(2, u(3))))

    # left: clock + status
    half = "1st half" if env.t < env.cfg.time.T_half else "2nd half"
    r = _text(screen, f_team, f"{env.t} / {env.T}", C_WHITE, u(18), u(12))
    _text(screen, f_small, "full time" if env.done else half, C_MUTED, u(18), r.bottom + u(6))
    if env.done:
        _tag(screen, f_small, "FULL TIME", (20, 20, 20), (235, 235, 235), r.right + u(14), u(10))
    elif ui["paused"]:
        _tag(screen, f_small, "PAUSED", (20, 20, 20), (240, 190, 60), r.right + u(14), u(10))
    else:
        _tag(screen, f_small, "LIVE", C_WHITE, (190, 50, 50), r.right + u(14), u(10))

    # right: seed, speed, overlays that are on
    r = _text(screen, f_team, f"seed {ui['seed']}", C_WHITE, WIN_W - u(18), u(12), "r")
    on = [n for n, k in (("vision", "show_vision"), ("anchors", "show_anchors"), ("formations", "show_ghosts")) if ui[k]]
    _text(screen, f_small, f"{ui['fps']} fps" + ("   showing: " + ", ".join(on) if on else ""), C_MUTED,
          WIN_W - u(18), r.bottom + u(6), "r")

    # progress bar with half-time tick
    bar_h = max(3, u(4))
    pygame.draw.rect(screen, (44, 48, 56), (0, HUD_H - bar_h, WIN_W, bar_h))
    pygame.draw.rect(screen, (90, 150, 230), (0, HUD_H - bar_h, int(env.t / env.T * WIN_W), bar_h))
    hx = int(env.cfg.time.T_half / env.T * WIN_W)
    pygame.draw.line(screen, (200, 200, 200), (hx, HUD_H - bar_h - u(3)), (hx, HUD_H), 1)


def draw_footer(screen: pygame.Surface, env: FootballEnv):
    """Bottom bar. Top row: live match stats (blue value | label | red value). Bottom row: key hints."""
    u = lambda n: int(n * UI)
    y0 = WIN_H - FOOT_H
    pygame.draw.rect(screen, C_HUD_BG, (0, y0, WIN_W, FOOT_H))
    pygame.draw.line(screen, (44, 48, 56), (0, y0), (WIN_W, y0), 1)
    f_small, f_team = _FONTS["small"], _FONTS["team"]
    st = env.stats
    poss = st["possession_steps"]
    tot = max(1, poss[0] + poss[1])
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"
    rows = [("possession", f"{100 * poss[0] / tot:.0f}%", f"{100 * poss[1] / tot:.0f}%"),
            ("shots", str(st["shots"][0]), str(st["shots"][1])),
            ("passes", str(st["passes"][0]), str(st["passes"][1])),
            ("pass accuracy", pct(st["passes_completed"][0], st["passes"][0]), pct(st["passes_completed"][1], st["passes"][1])),
            ("tackles won", str(st["tackles_won"][0]), str(st["tackles_won"][1]))]
    x = u(18)
    y = y0 + u(10)
    for label, a, b in rows:
        ra = _text(screen, f_team, a, C_TEAM[0], x, y)
        rl = _text(screen, f_small, label, C_MUTED, ra.right + u(8), y + u(2))
        rb = _text(screen, f_team, b, (225, 90, 90), rl.right + u(8), y)
        x = rb.right + u(30)
    keys = ("Space pause  Right step  Up/Down speed  R new match  Shift+R replay  Z/X blue/red style  "
            "N/M blue/red formation  V vision  A anchors  F formations  1-5 highlight  Q quit")
    _text(screen, f_small, keys, (110, 116, 126), u(18), WIN_H - f_small.get_height() - u(10))


def draw_full_time(screen: pygame.Surface, env: FootballEnv):
    """Result card in the middle of the pitch once the match is over."""
    u = lambda n: int(n * UI)
    a, b = env.score
    head = "DRAW" if a == b else ("BLUE WINS" if a > b else "RED WINS")
    col = (235, 235, 235) if a == b else C_TEAM[0 if a > b else 1]
    cx, cy = w2s(env.L / 2, env.W / 2, env.L, env.W)
    w, h = u(360), u(150)
    card = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(card, (14, 16, 20, 225), (0, 0, w, h), border_radius=u(14))
    pygame.draw.rect(card, (*col, 255), (0, 0, w, h), width=2, border_radius=u(14))
    screen.blit(card, (cx - w // 2, cy - h // 2))
    _text(screen, _FONTS["med"], "FULL TIME  -  " + head, col, cx, cy - h // 2 + u(16), "c")
    _text(screen, _FONTS["big"], f"{a} - {b}", C_WHITE, cx, cy - u(26), "c")
    _text(screen, _FONTS["small"], "press R for a new match", C_MUTED, cx, cy + h // 2 - u(28), "c")


# ─────────────────────────────────────────────  main

def main():
    ap = argparse.ArgumentParser(description="Top-down 2D Pygame renderer")
    ap.add_argument("--mode",     choices=["random", "scripted"], default="scripted")
    ap.add_argument("--T",        type=int,   default=1200)
    ap.add_argument("--fps",      type=int,   default=20)
    ap.add_argument("--radius",   type=float, default=None)
    ap.add_argument("--sigma",    type=int,   default=0)
    ap.add_argument("--opp-mode", default="NONE")
    ap.add_argument("--seed",     type=int,   default=0)
    styles = list(load_config().styles.presets.to_dict())
    ap.add_argument("--style",     choices=styles, default=None, help="play style of the blue team (default: config)")
    ap.add_argument("--opp-style", choices=styles, default=None, help="play style of the red team (default: config)")
    forms = list(load_config().formations.anchors.to_dict())
    ap.add_argument("--formation",     choices=forms, default=None, help="formation of the blue team (default: config)")
    ap.add_argument("--opp-formation", choices=forms, default=None, help="formation of the red team (default: config)")
    ap.add_argument("--no-anchors", action="store_true")
    args = ap.parse_args()

    overrides = {"time.T": args.T, "opponent.switching.mode": args.opp_mode}
    if args.radius:
        overrides["vision.radius"] = args.radius
    cfg  = load_config(overrides=overrides)
    env  = FootballEnv(cfg, scripted_ours=(args.mode == "scripted"))
    style = [args.style or cfg.styles.ours, args.opp_style or cfg.styles.opponent]   # blue, red (Z / X change them)
    formation = [args.formation or cfg.formations.ours, args.opp_formation or cfg.formations.opponent]   # N / M
    ag   = RandomAgent(args.seed) if args.mode == "random" else None

    pygame.init()
    _ft.init()   # initialise the C-level freetype engine
    pygame.display.set_caption(f"Football Agent — Top-down 2D (seed {args.seed})")
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    set_layout(WIN_W, WIN_H, env.L, env.W)   # after set_mode so convert_alpha() works when fonts render
    clock  = pygame.time.Clock()

    fps                  = args.fps
    paused               = False
    show_vision          = False
    show_anchors         = not args.no_anchors
    show_ghosts          = False
    highlight            = -1
    seed                 = args.seed
    flashes: list        = []
    goal_pause_remaining = 0     # frames left in pre-kickoff freeze (countdown)
    kicking_off_team     = 0     # team taking the next kickoff
    match_over_flashed   = False  # prevent repeated end-of-match flash
    shot_anim            = None   # active shot replay (sim is frozen while it plays)

    def do_reset(s):
        nonlocal flashes, goal_pause_remaining, kicking_off_team, match_over_flashed, shot_anim
        flashes = []
        shot_anim = None
        match_over_flashed = False
        obs, _ = env.reset(seed=s, sigma=args.sigma, our_style=style[0], opp_style=style[1],
                           our_formation=formation[0], opp_formation=formation[1])
        # Start with a 3-2-1 countdown before the first kick
        goal_pause_remaining = GOAL_PAUSE_FRAMES
        kicking_off_team     = env.holder[0] if env.holder else 0
        return obs

    obs = do_reset(seed)
    bg  = build_background(env)

    step_once = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_RIGHT:
                    step_once = True
                elif k in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                    fps = min(60, fps + 5)
                elif k in (pygame.K_DOWN, pygame.K_MINUS):
                    fps = max(1, fps - 5)
                elif k == pygame.K_v:
                    show_vision = not show_vision
                elif k == pygame.K_a:
                    show_anchors = not show_anchors
                elif k == pygame.K_f:
                    show_ghosts = not show_ghosts
                elif k == pygame.K_r:                      # R = new match with a random seed; Shift+R = replay this one
                    if not (event.mod & pygame.KMOD_SHIFT):
                        seed = int.from_bytes(os.urandom(2), "big")   # 0..65535, shown in the top bar
                    obs = do_reset(seed)
                    bg  = build_background(env)
                    paused = False
                    pygame.display.set_caption(f"Football Agent — Top-down 2D (seed {seed})")
                elif k in (pygame.K_z, pygame.K_x):        # Z / X = next play style for blue / red, mid-match
                    team = 0 if k == pygame.K_z else 1
                    style[team] = styles[(styles.index(style[team]) + 1) % len(styles)]
                    env.set_style(team, style[team])
                elif k in (pygame.K_n, pygame.K_m):        # N / M = next formation for blue / red, mid-match
                    team = 0 if k == pygame.K_n else 1     # (from the one on the pitch: the team may have switched itself)
                    formation[team] = forms[(forms.index(env.formation[team]) + 1) % len(forms)]
                    env.set_formation(team, formation[team])
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    idx = k - pygame.K_1
                    highlight = -1 if highlight == idx else idx

        # ── advance simulation ──────────────────────────────────────
        # Count down the pre-kickoff freeze (sim paused while this is > 0)
        if shot_anim is not None and shot_anim["frames"] <= 0:
            shot_anim = None
        if goal_pause_remaining > 0 and not paused and shot_anim is None:
            goal_pause_remaining -= 1

        should_step = ((not paused and not env.done and goal_pause_remaining == 0) or step_once) and shot_anim is None
        step_once = False

        if should_step and not env.done:
            actions = ag.act(obs) if ag else None
            pre_pos = env.pos.copy()
            obs, _, done, info = env.step(actions)
            for ev in info["events"]:
                if ev.get("type") == "shot":
                    shot_anim = make_shot_anim(ev, env, pre_pos)
                fl = make_flash(ev, env)
                if fl:
                    flashes.append(fl)
                    if fl["kind"] == "goal":
                        goal_pause_remaining = GOAL_PAUSE_FRAMES
                        kicking_off_team     = 1 - fl["team"]  # opposite team kicks off

        # ── draw ────────────────────────────────────────────────────
        screen = pygame.display.get_surface()            # the window's surface (a resize may replace it)
        if screen.get_size() != (WIN_W, WIN_H):          # window resized: lay everything out again, sharp
            set_layout(*screen.get_size(), env.L, env.W)
            bg = build_background(env)
            flashes = []                                 # their pixel positions are stale
        screen.blit(bg, (0, 0))

        if show_anchors:
            draw_anchors(screen, env)
        if show_ghosts:
            draw_formation_ghosts(screen, env)
        if show_vision:
            draw_vision(screen, env)

        if shot_anim is not None:
            # replay the shot over the positions the players had when it was taken
            live_pos, live_holder = env.pos, env.holder
            env.pos, env.holder = shot_anim["pre_pos"], None
            draw_players(screen, env, highlight=highlight)
            env.pos, env.holder = live_pos, live_holder
            draw_shot_anim(screen, env, shot_anim)
        else:
            if goal_pause_remaining == 0:
                draw_flight(screen, env)   # no flight possible during freeze
            draw_players(screen, env, highlight=highlight)
            draw_ball(screen, env)
            draw_flashes(screen, flashes)
            if goal_pause_remaining > 0:
                draw_kickoff_overlay(screen, env, goal_pause_remaining, kicking_off_team)
        draw_footer(screen, env)
        if shot_anim is None and goal_pause_remaining == 0:
            draw_shot_indicator(screen, env, _FONTS["small"])
        draw_hud(screen, env, dict(fps=fps, paused=paused, seed=seed, show_vision=show_vision,
                                   show_anchors=show_anchors, show_ghosts=show_ghosts))
        if env.done and shot_anim is None:
            draw_full_time(screen, env)

        clock.tick(fps)
        pygame.display.flip()


if __name__ == "__main__":
    main()
