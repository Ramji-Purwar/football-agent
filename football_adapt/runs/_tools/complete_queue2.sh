#!/bin/bash
# After complete_queue.sh: Q1's best formation by radius (idea.pdf f*(r)). Our team plays 1-3-1 (best on average in
# the Phase 0 win matrix) and 1-2-2 (maximin), zeta 1 (the level whose seeds agree in Q1), at the three radii, as the
# Q1 runs. Run from football_adapt/ on the server:   bash runs/_tools/complete_queue2.sh
while pgrep -f "[c]omplete_queue.sh" > /dev/null; do sleep 60; done
while pgrep -f "[r]un_then_check.sh" > /dev/null; do sleep 60; done
for f in 1-3-1 1-2-2; do
  for r in 12 25 120; do
    bash runs/_tools/run_then_check.sh "q1/f${f//-/}_z1_r$r/s0" --recipe configs/study.yaml --our-formation $f \
        --opp-formations 2-2-1,2-1-2,1-3-1,3-1-1 --radius $r --zeta 1 --seed 0 --opp-eps-start 0.5 --workers \
        --n-envs 48 --batch-size 1024 --torch-threads 8 --matches 40000 --eval-every 1000 --eval-matches 50 \
        --eval-procs 50 --test-matches 300
  done
done
