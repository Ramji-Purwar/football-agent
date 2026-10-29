"""Write runs/<q>/README.md: every finished run of a question, its old name, setting, test result and W against
each opponent formation. Run from football_adapt/:  python runs/_tools/index.py q1 [q2 ...]"""
import json
import re
import sys
from pathlib import Path

OLD = {"q1/z1_r25": "v1", "q1/z1_r12": "v2", "q1/z1_r120": "v3", "q1/z0_r25": "v4", "q1/z0_r12": "v5",
       "q1/z0_r120": "v6"}                    # the names the Q1 runs had before 2026-10-20
FORMS = ["2-2-1", "2-1-2", "1-3-1", "1-2-2", "3-1-1"]


def row(run: Path, q: str) -> str:
    r = json.loads((run / "result.json").read_text())
    a = r["args"]
    seed = run.name
    old = OLD.get(f"{q}/{run.parent.name}", "")
    old = (old if seed == "s0" else f"{old}_{seed}") if old else ""
    vf = run / "vs_formations.txt"
    w = dict(re.findall(r"^(\S+)(?: NEW)?\s+W ([\d.]+)", vf.read_text(), re.M)) if vf.exists() else {}
    seen = a.get("opp_formations") or []
    cells = [(w.get(f, "-") + ("" if f in seen else "*")) for f in FORMS]
    return (f"| {run.parent.name}/{seed} | {old} | {a['zeta']} | {a['radius']:g} | {a['seed']} | {r['best_ckpt']} | "
            f"**{r['W']:.2f}** | {r['goals_for']:.2f} - {r['goals_for'] - r['goal_diff']:.2f} | "
            f"{r['possession'] * 100:.0f}% | " + " | ".join(cells) + " |")


def main():
    for q in sys.argv[1:]:
        root = Path("runs") / q
        runs = sorted(p.parent for p in root.glob("*/*/result.json"))
        lines = [f"# {q.upper()} runs", "",
                 "One folder per setting, one sub-folder per training seed (`s<seed>`). Each run folder holds",
                 "`ckpts/` (a checkpoint per evaluation), `log.jsonl`, `config.json`, `result.json` (the best checkpoint",
                 "on the evaluation seeds, tested on 300 new matches), `vs_formations.txt` and `train.log`.", "",
                 "Test W = wins + draws/2 over 300 matches (about +-0.03). Formation columns: W over 50 matches;",
                 "`*` = a formation the run never trained against.", "",
                 "| run | old name | zeta | radius | seed | best checkpoint | test W | goals | possession | "
                 + " | ".join(FORMS) + " |",
                 "|" + "---|" * (9 + len(FORMS))]
        lines += [row(r, q) for r in runs]
        (root / "README.md").write_text("\n".join(lines) + "\n")
        print(f"{root}/README.md: {len(runs)} runs")


if __name__ == "__main__":
    main()
