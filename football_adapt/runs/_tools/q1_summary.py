"""Per-setting averages over seeds for the Q1 report: test results, W per opponent formation, learning speed.
Run from football_adapt/:  python runs/_tools/q1_summary.py  -> prints JSON (one object per setting)."""
import json
import re
from pathlib import Path
import numpy as np

FORMS = ["2-2-1", "2-1-2", "1-3-1", "1-2-2", "3-1-1"]
TEST = ["W", "win", "draw", "goals_for", "goal_diff", "passes", "passes_completed", "pass_completion", "shots",
        "chance_per_shot", "poor_shots", "through_balls", "turnovers", "possession", "ball_own_third",
        "ball_mid_third", "ball_their_third"]


def curve(run: Path):
    ev = [json.loads(l) for l in open(run / "log.jsonl")]
    ev = [r for r in ev if r["kind"] == "eval"]
    m = np.array([round(r["step"] / 1200) for r in ev])
    w = np.array([r["W"] for r in ev])
    return m, w


def first_reach(m, w, level):
    s = np.convolve(w, np.ones(3) / 3, mode="valid")          # 3-evaluation moving average
    i = np.argmax(s >= level) if (s >= level).any() else None
    return float(m[i + 2]) if i is not None else float("nan")


def main():
    out = {}
    for setting in sorted(Path("runs/q1").iterdir()):
        if not setting.is_dir():
            continue
        runs = sorted(p.parent for p in setting.glob("*/result.json"))
        res = [json.loads((r / "result.json").read_text()) for r in runs]
        d = {"setting": setting.name, "seeds": len(runs), "zeta": res[0]["args"]["zeta"],
             "radius": res[0]["args"]["radius"]}
        for k in TEST:
            v = [r[k] for r in res]
            d[k] = float(np.mean(v))
            if k == "W":
                d["W_spread"] = float(np.max(v) - np.min(v))
                d["W_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        d["goals_against"] = d["goals_for"] - d["goal_diff"]
        for who, col in (("vs", 2), ("scripted", 3)):
            per = {f: [] for f in FORMS}
            for r in runs:
                for f, wt, ws in re.findall(r"^(\S+)(?: NEW)?\s+W ([\d.]+)\s+goals \S+\s+W ([\d.]+)",
                                            (r / "vs_formations.txt").read_text(), re.M):
                    per[f].append(float(wt if col == 2 else ws))
            d[who] = {f: float(np.mean(v)) for f, v in per.items()}
        curves = [curve(r) for r in runs]
        n = min(len(c[0]) for c in curves)
        mean_w = np.mean([c[1][:n] for c in curves], axis=0)
        d["curve_matches"] = curves[0][0][:n].tolist()
        d["curve_W"] = mean_w.tolist()
        for lv in (0.3, 0.5, 0.7):
            d[f"reach_{lv}"] = [first_reach(*c, lv) for c in curves]
        d["last10_W"] = float(np.mean([c[1][-10:].mean() for c in curves]))
        d["best_at"] = float(np.mean([int(re.search(r"\d+", x["best_ckpt"]).group()) for x in res]))
        out[setting.name] = d
    print(json.dumps(out))


if __name__ == "__main__":
    main()
