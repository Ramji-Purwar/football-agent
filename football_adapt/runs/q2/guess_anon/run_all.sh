#!/bin/bash
# Q2 guesser WITHOUT player numbers, both radii: maps -> tune (on the practice matches of runs/q2/guess) -> test
cd ~/fa/football_adapt
export PYTHONPATH=.
P="taskset -c 0-55 $HOME/fa-venv/bin/python -m scripts.q2"
run() {  # name general specialist starts c_late c_false
  D=runs/q2/guess_anon/$1
  $P maps   --general $2 --specialist $3 --out $D/maps.npz &&
  $P tune   --maps $D/maps.npz --traces runs/q2/guess/$1/traces.npz --c-late $5 --c-false $6 --anonymous --out $D/filter.json > $D/tune.txt &&
  $P test   --general $2 --specialist $3 --maps $D/maps.npz --filter $D/filter.json --starts $4 --out $D/test.jsonl > /dev/null
}
run r25 runs/v4/ckpts/m030000.pt runs/f122_r25_z0/ckpts/m005040.pt 2-2-1 0.47 2-2-1=0.39
run r12 runs/q2pick/v2/m030000.pt runs/q2pick/f122_r12_z1/m016032.pt 2-2-1,3-1-1 0.44 2-2-1=0.34,3-1-1=0.27
echo DONE
