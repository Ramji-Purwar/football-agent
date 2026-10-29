#!/bin/bash
# Q3 oracle specialists: one team per playmaker number, trained (like v4: zeta 0, radius 25, 40k matches, seed 0)
# against every formation in which that number is a midfielder, with him as the playmaker (opponent.playmaker in
# configs/default.yaml). Waits until no other run is going. Run from football_adapt/ on the server:
#   bash runs/_tools/q3_specialists.sh
while pgrep -f "[r]un_then_check.sh|[s]cripts.q3" > /dev/null; do sleep 60; done
for spec in 4:2-2-1,1-3-1,3-1-1 3:2-2-1,2-1-2,1-3-1 2:1-3-1; do
  IFS=: read -r sigma forms <<< "$spec"
  bash runs/_tools/run_then_check.sh "q3/pm${sigma}_r25_z0/s0" --recipe configs/study.yaml --our-formation 2-2-1 \
      --opp-formations "$forms" --sigmas "$sigma" --radius 25 --zeta 0 --seed 0 --opp-eps-start 0.5 \
      --workers --n-envs 48 --batch-size 1024 --torch-threads 8 --matches 40000 --eval-every 1000 \
      --eval-matches 50 --eval-procs 50 --test-matches 300
done
