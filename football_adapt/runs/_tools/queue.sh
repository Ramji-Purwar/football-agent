#!/bin/bash
# Q1 runs one after another on the whole server, each followed by its formation check (run_then_check.sh). Waits first
# until no other run_then_check.sh is going. Run from football_adapt/:
#   bash runs/_tools/queue.sh SEED NAME:ZETA:RADIUS [NAME:ZETA:RADIUS ...]      (NAME may be a path: q1/z0_r12/s7)
seed=$1; shift
while pgrep -f "[r]un_then_check.sh" > /dev/null; do sleep 60; done
for spec in "$@"; do
  IFS=: read -r name zeta radius <<< "$spec"
  bash runs/_tools/run_then_check.sh "$name" --recipe configs/study.yaml --our-formation 2-2-1 \
      --opp-formations 2-2-1,2-1-2,1-3-1,3-1-1 --radius "$radius" --zeta "$zeta" --seed "$seed" --opp-eps-start 0.5 \
      --workers --n-envs 48 --batch-size 1024 --torch-threads 8 --matches 40000 --eval-every 1000 \
      --eval-matches 50 --eval-procs 50 --test-matches 300
done
