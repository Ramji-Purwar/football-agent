"""Print the command that trains a run again, rebuilt from the arguments stored in its result.json.

Run from football_adapt/:
    python -m scripts.train_command runs/q1/z1_r25/s0 [--name NEW_NAME]
Only arguments that differ from the defaults of training/train.py are printed. A full run is 40,000 matches and took
13-16 hours on a 40-thread CPU server (--workers runs one process per game); it is not needed to check the results."""
from __future__ import annotations
import argparse
import json
import shlex
from pathlib import Path

from training.train import parse_args

# the Q3 mixture teams were warm-started (runs/_tools/warm_start.py): from the weights of the Q2 GRU seed 0, with the
# shapes of their own from-scratch run, then trained with this command (which resumes from the warm-start ckpt.pt)
WARM = {"q3/mixw_r25_z0/s0": "q3/mix_r25_z0/s0", "q3/mixw_ns_r25_z0/s0": "q3/mix_ns_r25_z0/s0",
        "q3/mixw_told_r25_z0/s0": "q3/oracle_r25_z0/s0"}


def flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--name", default=None, help="folder for the new run (default: the run's own name + '_again')")
    a = ap.parse_args()
    stored = json.load(open(Path(a.run) / "result.json"))["args"]
    defaults = vars(parse_args([]))
    rel = Path(a.run).resolve().relative_to(Path("runs").resolve()).as_posix()
    new = a.name or rel + "_again"
    parts = ["python", "-m", "training.train", "--name", new]
    for k, v in stored.items():
        if k in ("name", "runs_dir") or k not in defaults or v == defaults[k]:
            continue
        if stored.get("matches") and k in ("total_steps", "eval_interval"):
            continue                       # with --matches, train() sets these itself from the match length
        if stored.get("matches") and k == "rollout_len" and defaults["rollout_len"] % stored["seq_len"] == 0:
            continue                       # likewise (given only where --seq-len needs it to parse)
        if isinstance(v, bool):
            parts.append(flag(k) if v else "--no-" + k.replace("_", "-"))
        elif k == "set":
            for item in v:
                parts += ["--set", str(item)]
        elif isinstance(v, list):
            parts += [flag(k), ",".join(map(str, v))]
        else:
            parts += [flag(k), str(v)]
    if rel in WARM:
        print(f"python runs/_tools/warm_start.py --src runs/q2/gru_r25_z0/s0/ckpt.pt --like runs/{WARM[rel]}/ckpt.pt "
              f"--out runs/{new}")
    print(" ".join(shlex.quote(p) for p in parts))


if __name__ == "__main__":
    main()
