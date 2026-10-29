export PYTHONPATH=.
PY=$HOME/fa-venv/bin/python
$PY runs/_tools/ckpt_vs_form.py --matches 200 --seed0 70000 --out runs/q3/other_slots_matches.jsonl runs/v4/ckpts/m038016.pt:2-2-1:1 runs/v4/ckpts/m038016.pt:2-2-1:2 runs/v4/ckpts/m038016.pt:2-2-1:5 runs/v4/ckpts/m038016.pt:2-1-2:1 runs/v4/ckpts/m038016.pt:2-1-2:2 runs/v4/ckpts/m038016.pt:2-1-2:4 runs/v4/ckpts/m038016.pt:2-1-2:5 runs/v4/ckpts/m038016.pt:1-3-1:1 runs/v4/ckpts/m038016.pt:1-3-1:5 runs/v4/ckpts/m038016.pt:3-1-1:1 runs/v4/ckpts/m038016.pt:3-1-1:2 runs/v4/ckpts/m038016.pt:3-1-1:3 runs/v4/ckpts/m038016.pt:3-1-1:5 > runs/q3/other_slots.txt 2>&1
$PY runs/_tools/phase0_full.py --matches 300 --out runs/phase0/phase0.txt > runs/phase0/phase0.log 2>&1
