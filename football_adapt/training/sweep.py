"""Run a grid of training runs side by side, one process (one core) per run.

Run from football_adapt/. Everything after `--` is passed to every training run unchanged:

   python -m training.sweep --jobs 28 --formations 2-2-1,2-1-2,1-3-1,1-2-2,3-1-1 --radii 12,25,40,100 \\
          --zetas 0,1,2 --seeds 0,1,2,3,4 -- --total-steps 5000000                     # the Q1 grid

Each run writes to runs/<name>/ (its console output goes to stdout.log there). Finished runs are skipped and
interrupted ones resume, so after a node expires just start the same command again.
"""
from __future__ import annotations
import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.train import csv


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        extra = argv[argv.index("--") + 1:]
        argv = argv[:argv.index("--")]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, required=True, help="runs at the same time (use the number of cores)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--actor", default="mlp")
    ap.add_argument("--formations", type=csv, default=["2-2-1"], help="OUR formation, one run each")
    ap.add_argument("--radii", type=csv, default=["25"])
    ap.add_argument("--zetas", type=csv, default=["2"])
    ap.add_argument("--seeds", type=csv, default=["0"])
    args = ap.parse_args(argv)

    def one(cell):
        f, r, z, s = cell
        name = f"{args.actor}_f{f}_r{r}_z{z}_s{s}"
        run = Path(args.runs_dir) / name
        run.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "training.train", "--name", name, "--runs-dir", args.runs_dir,
               "--actor", args.actor, "--our-formation", f, "--radius", r, "--zeta", z, "--seed", s] + extra
        with open(run / "stdout.log", "a") as out:
            code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
        print(f"{'done  ' if code == 0 else 'FAILED'} {name}", flush=True)
        return code

    cells = list(itertools.product(args.formations, args.radii, args.zetas, args.seeds))
    print(f"{len(cells)} runs, {args.jobs} at a time", flush=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(one, cells))
    sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
