#!/bin/bash
# The runs idea.pdf asks for that were still missing (2026-10-24), one after another on the whole server, each like
# the Q1 runs (study recipe, our 2-2-1, 40k matches, seed 0, easy start) and each followed by its formation check
# (run_then_check.sh). Waits until no other run is going. Run from football_adapt/ on the server:
#   bash runs/_tools/complete_queue.sh
while pgrep -f "[r]un_then_check.sh|[s]cripts.q[23]|[p]hase0_full|[c]kpt_vs_form" > /dev/null; do sleep 60; done
COMMON="--recipe configs/study.yaml --our-formation 2-2-1 --seed 0 --opp-eps-start 0.5 --workers --n-envs 48
        --batch-size 1024 --torch-threads 8 --matches 40000 --eval-every 1000 --eval-matches 50 --eval-procs 50
        --test-matches 300"
GRU="--actor gru --seq-len 40 --rollout-len 1200"
ALL=$(for f in 2-2-1 2-1-2 1-3-1 3-1-1; do for s in 0 1 2 3 4 5; do printf "%s:%s," $f $s; done; done)
ALL=${ALL%,}
run() { bash runs/_tools/run_then_check.sh "$@"; }
# Q2: one team over the switching opponent (the test's switch process: one switch to 1-2-2 at a random step)
run q2/gru_r25_z0/s0 $COMMON $GRU --opp-formations 2-2-1 --switching WINDOW --radius 25 --zeta 0
run q2/gru_r12_z1/s0 $COMMON $GRU --opp-formations 2-2-1,3-1-1 --switching WINDOW --radius 12 --zeta 1
run q2/mlp_r25_z0/s0 $COMMON --opp-formations 2-2-1 --switching WINDOW --radius 25 --zeta 0
run q2/mlp_r12_z1/s0 $COMMON --opp-formations 2-2-1,3-1-1 --switching WINDOW --radius 12 --zeta 1
# Q3: one recurrent team over every playmaker position and none (theta_mix), the same told who he is (theta_or),
# and theta_mix without the shadow action (theta_ns)
run q3/mix_r25_z0/s0 $COMMON $GRU --cases "$ALL" --radius 25 --zeta 0
run q3/oracle_r25_z0/s0 $COMMON $GRU --cases "$ALL" --radius 25 --zeta 0 --set actions.show_playmaker=true
run q3/mix_ns_r25_z0/s0 $COMMON $GRU --cases "$ALL" --radius 25 --zeta 0 --set actions.shadow=false
# Q1: structure as a constraint (zeta 2) at the three radii
for r in 12 25 120; do
  run q1/z2_r$r/s0 $COMMON --opp-formations 2-2-1,2-1-2,1-3-1,3-1-1 --radius $r --zeta 2
done
