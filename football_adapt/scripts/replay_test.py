"""Replay the test of trained runs and check that the stored result comes out again.

Every run's result.json says which checkpoint was its best (best_ckpt) and what it scored on the test matches
(seeds 50000, 50001, ...). This script loads that checkpoint, plays the same test matches again with the same seeds
and the configuration stored in the checkpoint, and compares the win score W (and goal difference) with result.json.
No training happens here. W must come out exactly; the goal difference can differ in the last digit, because on
another machine or torch version it can change by a goal in a few hundred matches (seen: one goal in 300).

Run from football_adapt/:
    python -m scripts.replay_test runs/q1/z1_r25/s0 runs/q2/mlp_r25_z0/s0 ...   [--procs P]
Exit code 0 if W of every run reproduces exactly (or within --tolerance), 1 otherwise."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from agents.mappo import MAPPOAgent
from training.train import evaluate, TEST_SEED0, EVAL_SEED0


def load_actor(path: Path):
    """The actor and the config values stored in a checkpoint, exactly as the viewers and evaluations load them."""
    ag = MAPPOAgent.load(path)
    return ag.actor, ag.overrides


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders, e.g. runs/q1/z1_r25/s0")
    ap.add_argument("--quick", action="store_true",
                    help="replay the 50 validation matches of the best checkpoint (seeds 10000+) instead of the 300 test "
                         "matches; compared with the validation W stored in result.json")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="accept a replayed W that differs from the stored one by at most this much (default 0: exact)")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    ok_all, counts = True, {}
    for run in map(Path, a.runs):
        res = json.load(open(run / "result.json"))
        ckpt = run / "ckpts" / res["best_ckpt"]
        actor, overrides = load_actor(ckpt)
        if a.quick:
            ref, seed0, what = res["validation"], EVAL_SEED0, "validation"
        else:
            ref, seed0, what = res, TEST_SEED0, "test"
        n = int(ref["matches"])
        t0 = time.time()
        ev = evaluate(actor, overrides, res["scenario"], n, a.procs, seed0=seed0)
        d = ev["W"] - ref["W"]
        tag = "EXACT" if abs(d) < 1e-9 else ("CLOSE" if abs(d) <= a.tolerance + 1e-9 else "DIFF ")
        ok_all &= tag != "DIFF "
        counts[tag.strip()] = counts.get(tag.strip(), 0) + 1
        print(f"   {tag} {run}  ({res['best_ckpt']}, {n} {what} matches, {time.time() - t0:.0f}s): "
              f"W {ev['W']:.4f} (stored {ref['W']:.4f}); goal difference {ev['goal_diff']:+.3f} "
              f"(stored {ref['goal_diff']:+.3f})", flush=True)
    print("   replayed W: " + ", ".join(f"{v} {k.lower()}" for k, v in sorted(counts.items()))
          + (f" (close = within {a.tolerance} of the stored W)" if "CLOSE" in counts else ""), flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
