#!/bin/bash
# Train one run, then play its best checkpoint against every opponent formation (runs/NAME/vs_formations.txt).
# NAME may be a path, e.g. q2/f221_r25_z0/s0 (one folder per question and setting, one sub-folder per seed).
# Run from football_adapt/ on the server:
#   bash runs/_tools/run_then_check.sh NAME [training.train arguments]      (the console output goes to runs/NAME/train.log)
name=$1; shift
PY=~/fa-venv/bin/python
export PYTHONPATH=.
mkdir -p "runs/$name"
$PY -m training.train --name "$name" "$@" >> "runs/$name/train.log" 2>&1 || exit 1
best=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['best_ckpt'])" "runs/$name/result.json")
$PY runs/_tools/vs_formations.py "runs/$name/ckpts/$best" --matches 50 --procs 50 > "runs/$name/vs_formations.txt" 2>&1
