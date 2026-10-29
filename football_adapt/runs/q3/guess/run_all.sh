#!/bin/bash
# Q3 playmaker guesser: record what v4 sees in every case (maps / tuning / test seeds), then evaluate
cd ~/fa/football_adapt
export PYTHONPATH=.
P="taskset -c 0-55 $HOME/fa-venv/bin/python -m scripts.q3"
G=runs/v4/ckpts/m038016.pt
CASES=2-2-1:0,2-2-1:3,2-2-1:4,2-1-2:0,2-1-2:3,1-3-1:0,1-3-1:2,1-3-1:3,1-3-1:4,3-1-1:0,3-1-1:4
D=runs/q3/guess
$P record --general $G --cases $CASES --matches 100 --seed0 30000 --out $D/maps_rec.npz &&
$P record --general $G --cases $CASES --matches 50 --seed0 40000 --out $D/tune_rec.npz &&
$P record --general $G --cases $CASES --matches 100 --seed0 60000 --out $D/test_rec.npz &&
$P evaluate --maps $D/maps_rec.npz --tune $D/tune_rec.npz --test $D/test_rec.npz --out $D/guess.txt > /dev/null
echo DONE
