"""FormationManager: decides the current formation g_t of the scripted team (opponent.md, Section 4).

Modes: NONE | SCHEDULED | RANDOM | SCORE_REACTIVE. A minimum dwell time DWELL_MIN is respected between switches.
The manager does not move players. It only says which formation is active. Players then WALK to the new
anchors (no teleporting), see ScriptedTeam."""
from __future__ import annotations


class FormationManager:
    def __init__(self, cfg, rng, formation: str, formation_names: list, mode_override: str | None = None):
        sw = cfg.opponent.switching
        self.mode = (mode_override or sw.mode).upper()
        assert self.mode in ("NONE", "SCHEDULED", "RANDOM", "SCORE_REACTIVE"), self.mode
        self.sw = sw
        self.rng = rng
        self.names = list(formation_names)
        self.formation = formation
        self.dwell = 10 ** 9          # start "settled", so the first switch is not blocked
        self.dwell_min = sw.dwell_min

    def reset(self, formation: str):
        self.formation = formation
        self.dwell = 10 ** 9

    def update(self, t: int, own_goals: int, other_goals: int):
        """Called once per step. Returns the NEW formation name if a switch happens now, else None."""
        self.dwell += 1
        target = None
        if self.mode == "SCHEDULED":
            if t == self.sw.scheduled.t_switch:
                target = self.sw.scheduled.target
        elif self.mode == "RANDOM":
            if self.dwell >= self.dwell_min and self.rng.random() < self.sw.random.h_switch:
                choices = [n for n in self.names if n != self.formation]
                target = choices[int(self.rng.integers(0, len(choices)))]
        elif self.mode == "SCORE_REACTIVE":
            sr = self.sw.score_reactive
            if t > 0 and t % sr.check_interval == 0 and self.dwell >= self.dwell_min:
                d = own_goals - other_goals            # goals for the scripted team minus goals against
                if d <= -sr.trail_thresh:
                    target = sr.f_att
                elif d >= sr.lead_thresh:
                    target = sr.f_def
        if target is not None and target != self.formation:
            self.formation = target
            self.dwell = 0
            return target
        return None
