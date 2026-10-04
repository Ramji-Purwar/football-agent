"""Top-down 2D Pygame renderer for the football game.

The game is strictly 2D — ball and players have only (x, y) coordinates, no height.
This renderer faithfully represents that with a top-down pitch view.

Run from football_adapt/:
    .venv/bin/python -m scripts.pygame_render --mode scripted --fps 20
    .venv/bin/python -m scripts.pygame_render --mode random --sigma 3 --opp-mode SCHEDULED
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
from football.mechanics import shot_probability
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
        return surf

    def get_height(self) -> int:
        return int(self._ft.size)


def _make_font(size: int, bold: bool = False) -> _FTFont:
    font = _ft.Font(_MONO_PATH, size)   # None → built-in font if no path found
    font.strong = bold
    return _FTFont(font)


# Module-level font cache — populated once by _init_fonts() after pygame.init()
_FONTS: dict = {}


def _init_fonts():
    _FONTS["hud"]    = _make_font(26, bold=True)
    _FONTS["small"]  = _make_font(13)
    _FONTS["med"]    = _make_font(18, bold=True)
    _FONTS["big"]    = _make_font(52, bold=True)
    _FONTS["player"] = _make_font(11, bold=True)
    _FONTS["slot"]   = _make_font( 8)
    _FONTS["tiny"]   = _make_font( 9)


# ─────────────────────────────────────────────  layout constants
WIN_W, WIN_H = 1080, 720
MARGIN = 50
PITCH_PX_W = WIN_W - 2 * MARGIN   # 980
PITCH_PX_H = WIN_H - 2 * MARGIN   # 620 – extra headroom is used by HUD at the bottom

HUD_H = 52          # px reserved at the very top for score / match info

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
C_HUD_BG       = (18,  18,  18)
C_HUD_TEXT     = (230, 230, 230)
C_ANCHOR       = [(80, 140, 220, 100), (220, 80, 80, 100)]
C_VISION       = (43, 108, 176, 28)
C_VISION_EDGE  = (43, 108, 176, 80)

SLOT_COLOURS = {"D": (100, 200, 255), "M": (100, 255, 150), "F": (255, 160, 60)}

FLASH_GOAL_FRAMES   = 60
FLASH_EVENT_FRAMES  = 25


# ─────────────────────────────────────────────  coordinate helpers

def w2s(wx: float, wy: float, L: float = 100.0, W: float = 60.0):
    """World coords (m) → screen pixel (int, int). y flipped: higher y = higher on screen."""
    sx = MARGIN + int(wx / L * PITCH_PX_W)
    sy = HUD_H + MARGIN + int((1.0 - wy / W) * (PITCH_PX_H - HUD_H))
    return sx, sy


def w2r(metres: float, L: float = 100.0) -> int:
    """World length (m) → pixel radius."""
    return max(1, int(metres / L * PITCH_PX_W))


# ─────────────────────────────────────────────  static background

def build_background(env: FootballEnv) -> pygame.Surface:
    L, W = env.L, env.W
    surf = pygame.Surface((WIN_W, WIN_H))

    # HUD bar
    surf.fill(C_HUD_BG, (0, 0, WIN_W, HUD_H))

    # pitch stripes (every 10 m along x)
    n_stripes = int(L / 10)
    for i in range(n_stripes):
        x0 = MARGIN + int(i / n_stripes * PITCH_PX_W)
        x1 = MARGIN + int((i + 1) / n_stripes * PITCH_PX_W)
        y0 = HUD_H + MARGIN
        y1 = HUD_H + MARGIN + (PITCH_PX_H - HUD_H)
        c = C_PITCH_DARK if i % 2 == 0 else C_PITCH_LIGHT
        pygame.draw.rect(surf, c, (x0, y0, x1 - x0, y1 - y0))

    # pitch border
    bx = MARGIN; by = HUD_H + MARGIN
    bw = PITCH_PX_W; bh = PITCH_PX_H - HUD_H
    pygame.draw.rect(surf, C_WHITE, (bx, by, bw, bh), 2)

    # centre line
    cx = MARGIN + PITCH_PX_W // 2
    pygame.draw.line(surf, C_WHITE, (cx, HUD_H + MARGIN), (cx, HUD_H + MARGIN + bh), 1)

    # centre circle
    cy_world = W / 2
    cx_world = L / 2
    cc = w2s(cx_world, cy_world, L, W)
    pygame.draw.circle(surf, C_WHITE, cc, w2r(9.15, L), 1)
    pygame.draw.circle(surf, C_WHITE, cc, w2r(0.6, L))   # centre spot

    # goals (depth 3 m, width GOAL_W)
    gw = env.cfg.pitch.goal_width
    goal_depth = 3.0
    for (x_world, facing) in [(0.0, 1), (L, -1)]:
        y0_w = W / 2 - gw / 2
        y1_w = W / 2 + gw / 2
        gx0 = w2s(x_world, y0_w, L, W)
        gx1 = w2s(x_world, y1_w, L, W)
        gx_back0 = w2s(x_world + facing * goal_depth, y0_w, L, W)
        gx_back1 = w2s(x_world + facing * goal_depth, y1_w, L, W)
        # goal box outline
        pts = [gx0, gx_back0, gx_back1, gx1]
        pygame.draw.lines(surf, C_GOAL, False, pts, 3)

    # shooting zone arcs (centered on each goal mouth, radius D_SHOOT_MAX)
    shoot_r = env.cfg.mechanics.shoot_max
    arc_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    for (x_goal, start_ang, stop_ang) in [
        (0.0,  -math.pi / 2, math.pi / 2),   # left goal — arc faces right
        (L,     math.pi / 2, 3 * math.pi / 2),   # right goal — arc faces left
    ]:
        gcy = W / 2
        cx_px, cy_px = w2s(x_goal, gcy, L, W)
        r_px = w2r(shoot_r, L)
        rect = pygame.Rect(cx_px - r_px, cy_px - r_px, r_px * 2, r_px * 2)
        pygame.draw.arc(arc_surf, C_SHOOT_ARC, rect, start_ang, stop_ang, 1)
    surf.blit(arc_surf, (0, 0))

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


def draw_players(screen: pygame.Surface, env: FootballEnv, highlight: int = -1):
    font = _FONTS["player"]
    font_slot = _FONTS["slot"]
    slot_types = {
        team: env.book[env.formation[team]].slot_types
        for team in (0, 1)
    }

    for team in (0, 1):
        for i in range(5):
            px, py = w2s(*env.pos[team][i], env.L, env.W)
            r = w2r(1.5, env.L)
            is_holder = env.holder == (team, i)
            is_playmaker = (team == 1 and env.sigma_idx == i)
            is_frozen = env.frozen[team][i] > 0
            is_highlighted = (team == 0 and i == highlight)

            fill_colour = C_FROZEN if is_frozen else C_TEAM[team]

            # shadow circle for depth feel
            shadow = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(shadow, (0, 0, 0, 40), (r + 2, r + 2), r + 2)
            screen.blit(shadow, (px - r - 2, py - r - 2))

            # main circle
            pygame.draw.circle(screen, fill_colour, (px, py), r)

            # border: gold for playmaker, thick for ball holder, white normally
            if is_playmaker:
                pygame.draw.circle(screen, C_GOLD, (px, py), r, 3)
            elif is_holder:
                pygame.draw.circle(screen, C_WHITE, (px, py), r, 3)
            elif is_highlighted:
                pygame.draw.circle(screen, (255, 255, 100), (px, py), r + 2, 2)
            else:
                pygame.draw.circle(screen, C_WHITE, (px, py), r, 1)

            # frozen overlay
            if is_frozen:
                fsurf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(fsurf, (100, 100, 100, 120), (r, r), r)
                screen.blit(fsurf, (px - r, py - r))

            # player number
            num_surf = font.render(str(i + 1), True, C_WHITE)
            screen.blit(num_surf, (px - num_surf.get_width() // 2,
                                   py - num_surf.get_height() // 2))

            # slot type label (below circle)
            st = slot_types[team][i]
            st_col = SLOT_COLOURS.get(st, C_WHITE)
            st_surf = font_slot.render(st, True, st_col)
            screen.blit(st_surf, (px - st_surf.get_width() // 2, py + r + 1))


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
                         env.cfg.mechanics.shot_block_scale)
    bar_x = WIN_W - 185
    bar_y = WIN_H - 38
    bar_w = 140
    bar_h = 14
    colour = (60, 200, 60) if p > 0.5 else (220, 200, 0) if p > 0.25 else (200, 60, 60)
    pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(screen, colour, (bar_x, bar_y, int(bar_w * p), bar_h))
    pygame.draw.rect(screen, (150, 150, 150), (bar_x, bar_y, bar_w, bar_h), 1)
    lbl = font_small.render(f"Shoot  {p:.0%}", True, C_WHITE)
    screen.blit(lbl, (bar_x, bar_y - 16))


def draw_flashes(screen: pygame.Surface, flashes: list):
    """Draw and tick all active flash animations. Mutates the list."""
    font_big  = _FONTS["big"]
    font_med  = _FONTS["med"]
    to_remove = []
    for fl in flashes:
        fl["frames"] -= 1
        if fl["frames"] <= 0:
            to_remove.append(fl)
            continue
        t = fl["frames"]
        kind = fl["kind"]
        if kind == "goal":
            alpha = min(220, int(220 * t / FLASH_GOAL_FRAMES))
            overlay = pygame.Surface((WIN_W, WIN_H - HUD_H), pygame.SRCALPHA)
            team_c  = (*C_TEAM[fl["team"]], alpha // 3)
            overlay.fill(team_c)
            screen.blit(overlay, (0, HUD_H))
            col = (*C_TEAM[fl["team"]], alpha)
            txt = font_big.render("GOAL!", True, col[:3])
            screen.blit(txt, (WIN_W // 2 - txt.get_width() // 2, WIN_H // 2 - 26))
        elif kind == "tackle":
            alpha = int(255 * t / FLASH_EVENT_FRAMES)
            px, py = fl["pos"]
            spark_c = (255, 220, 0)
            for ang in range(0, 360, 45):
                rad = math.radians(ang)
                ex = px + int(14 * math.cos(rad))
                ey = py + int(14 * math.sin(rad))
                pygame.draw.line(screen, spark_c, (px, py), (ex, ey), 2)
        elif kind == "interception":
            alpha = int(255 * t / FLASH_EVENT_FRAMES)
            txt = font_med.render("INT", True, (255, 150, 0))
            screen.blit(txt, fl["pos"])
        elif kind == "turnover":
            txt = font_med.render("LOST", True, (220, 50, 50))
            screen.blit(txt, fl["pos"])
        elif kind == "pass_ok":
            alpha = int(200 * t / FLASH_EVENT_FRAMES)
            pygame.draw.circle(screen, (60, 220, 60), fl["pos"], 7, 2)
    for fl in to_remove:
        flashes.remove(fl)


def make_flash(event: dict, env: FootballEnv) -> dict | None:
    kind = event.get("type")
    if kind == "goal":
        return {"kind": "goal", "team": event["team"], "frames": FLASH_GOAL_FRAMES}
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


def draw_hud(screen: pygame.Surface, env: FootballEnv, fps: int,
             paused: bool, show_vision: bool, show_anchors: bool,
             show_ghosts: bool, font_hud, font_small):
    pygame.draw.rect(screen, C_HUD_BG, (0, 0, WIN_W, HUD_H))
    pygame.draw.line(screen, (60, 60, 60), (0, HUD_H), (WIN_W, HUD_H), 1)

    s0, s1 = env.score
    score_str = f"{s0}  –  {s1}"
    score_surf = font_hud.render(score_str, True, C_WHITE)
    screen.blit(score_surf, (WIN_W // 2 - score_surf.get_width() // 2, 6))

    # left: formations
    our_f = env.formation[0]
    opp_f = env.formation[1]
    f_str = f"Ours: {our_f}   Opp: {opp_f}"
    if env.sigma_idx is not None:
        f_str += f"   ★PM:p{env.sigma_idx + 1}"
    fl = font_small.render(f_str, True, (180, 180, 180))
    screen.blit(fl, (8, 8))

    # right: step + fps + toggles
    t_frac = env.t / env.T
    r_str = f"t {env.t}/{env.T}  {t_frac:.0%}  {fps}fps"
    flags = []
    if paused:       flags.append("PAUSED")
    if show_vision:  flags.append("[V]vision")
    if show_anchors: flags.append("[A]anchors")
    if show_ghosts:  flags.append("[F]ghosts")
    if flags:        r_str += "  " + "  ".join(flags)
    rs = font_small.render(r_str, True, (160, 160, 160))
    screen.blit(rs, (WIN_W - rs.get_width() - 8, 8))

    # possession dot
    poss_colours = [C_TEAM[0], C_TEAM[1], (140, 140, 140)]
    if env.holder is not None:
        poss_c = poss_colours[env.holder[0]]
    else:
        poss_c = poss_colours[2]
    pygame.draw.circle(screen, poss_c, (WIN_W // 2 - score_surf.get_width() // 2 - 18, 20), 7)

    # step bar
    bar_y = HUD_H - 4
    bar_w = int((env.t / env.T) * WIN_W)
    pygame.draw.rect(screen, (40, 40, 40), (0, bar_y, WIN_W, 4))
    pygame.draw.rect(screen, (80, 140, 220), (0, bar_y, bar_w, 4))


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
    ap.add_argument("--no-anchors", action="store_true")
    args = ap.parse_args()

    overrides = {"time.T": args.T, "opponent.switching.mode": args.opp_mode}
    if args.radius:
        overrides["vision.radius"] = args.radius
    cfg  = load_config(overrides=overrides)
    env  = FootballEnv(cfg, scripted_ours=(args.mode == "scripted"))
    ag   = RandomAgent(args.seed) if args.mode == "random" else None

    pygame.init()
    _ft.init()   # initialise the C-level freetype engine
    _init_fonts()
    pygame.display.set_caption("Football Agent — Top-down 2D")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    font_hud   = _FONTS["hud"]
    font_small = _FONTS["small"]

    fps          = args.fps
    paused       = False
    show_vision  = False
    show_anchors = not args.no_anchors
    show_ghosts  = False
    highlight    = -1
    seed         = args.seed
    flashes: list = []

    def do_reset(s):
        nonlocal flashes
        flashes = []
        obs, _ = env.reset(seed=s, sigma=args.sigma)
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
                elif k == pygame.K_r:
                    obs = do_reset(seed)
                    bg  = build_background(env)
                    paused = False
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    idx = k - pygame.K_1
                    highlight = -1 if highlight == idx else idx

        # ── advance simulation ──────────────────────────────────────
        should_step = (not paused and not env.done) or step_once
        step_once = False

        if should_step and not env.done:
            actions = ag.act(obs) if ag else None
            obs, _, done, info = env.step(actions)
            for ev in info["events"]:
                fl = make_flash(ev, env)
                if fl:
                    flashes.append(fl)

        if env.done and not any(fl["kind"] == "done" for fl in flashes):
            # show final score flash
            flashes.append({"kind": "goal", "team": 0 if env.score[0] > env.score[1] else 1,
                            "frames": 120})

        # ── draw ────────────────────────────────────────────────────
        screen.blit(bg, (0, 0))

        if show_anchors:
            draw_anchors(screen, env)
        if show_ghosts:
            draw_formation_ghosts(screen, env)
        if show_vision:
            draw_vision(screen, env)

        draw_flight(screen, env)
        draw_players(screen, env, highlight=highlight)
        draw_ball(screen, env)
        draw_shot_indicator(screen, env, font_small)
        draw_flashes(screen, flashes)
        draw_hud(screen, env, fps, paused, show_vision, show_anchors, show_ghosts,
                 font_hud, font_small)

        clock.tick(fps)
        pygame.display.flip()


if __name__ == "__main__":
    main()
