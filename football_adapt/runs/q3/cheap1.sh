export PYTHONPATH=.
# Checkpoint paths are those of this repository (on the training servers the same files sat under runs/v4, runs/f122_r25_z0, runs/q2pick).
PY=$HOME/fa-venv/bin/python
$PY runs/_tools/ckpt_vs_form.py --matches 200 --seed0 70000 --out runs/q3/other_slots_matches.jsonl runs/q1/z0_r25/s0/ckpts/m038016.pt:2-2-1:1 runs/q1/z0_r25/s0/ckpts/m038016.pt:2-2-1:2 runs/q1/z0_r25/s0/ckpts/m038016.pt:2-2-1:5 runs/q1/z0_r25/s0/ckpts/m038016.pt:2-1-2:1 runs/q1/z0_r25/s0/ckpts/m038016.pt:2-1-2:2 runs/q1/z0_r25/s0/ckpts/m038016.pt:2-1-2:4 runs/q1/z0_r25/s0/ckpts/m038016.pt:2-1-2:5 runs/q1/z0_r25/s0/ckpts/m038016.pt:1-3-1:1 runs/q1/z0_r25/s0/ckpts/m038016.pt:1-3-1:5 runs/q1/z0_r25/s0/ckpts/m038016.pt:3-1-1:1 runs/q1/z0_r25/s0/ckpts/m038016.pt:3-1-1:2 runs/q1/z0_r25/s0/ckpts/m038016.pt:3-1-1:3 runs/q1/z0_r25/s0/ckpts/m038016.pt:3-1-1:5 > runs/q3/other_slots.txt 2>&1
$PY runs/_tools/phase0_full.py --matches 300 --out runs/phase0/phase0.txt > runs/phase0/phase0.log 2>&1
