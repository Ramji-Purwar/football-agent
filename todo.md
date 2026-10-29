# What is left (as of 2026-10-24)

Checked against `idea.pdf`. Already done: Phase 0 (`runs/phase0/phase0.txt`), the Q1 report (`docs/reportQ1.md`,
ζ 0 and ζ 1), Q2 at a basic level (memoryless team + guessing function + switch to the 1-2-2 checkpoint), and the Q3
report (`docs/reportQ3.md`: specialists for player 2 and 4, guess, unaware/adaptive/oracle).

Nothing is committed yet. All the code below is in the working tree (branch `arpan`).

## 0. Server state right now

- **Nothing is running on er011** (everything was stopped at 06:34 UTC on 24 Oct).
- `q2/gru_r25_z0/s0` (the Q2 recurrent team, radius 25) was stopped at about match 23,000 of 40,000. The last
  checkpoint is `m023040.pt`. The whole run folder, including `ckpt.pt`, is on the PC in `football_adapt/runs/q2/gru_r25_z0/s0/`.
  **Running the same command again resumes from `ckpt.pt`**, so copy that folder to the new server and rerun the line
  for priority 1 below. It needs about 4 more hours.
- Every result from er011 is on the PC. **er011 goes away on 26 Oct.**

## 1. Training runs still to do

Run each from `football_adapt/` on a server (code synced with
`rsync -a --exclude __pycache__ football agents opponent configs scripts tests training runs/_tools SERVER:fa/football_adapt/`).
Each line is `bash runs/_tools/run_then_check.sh NAME $COMMON ...`, where

```
COMMON="--recipe configs/study.yaml --our-formation 2-2-1 --seed 0 --opp-eps-start 0.5 --workers --n-envs 48
        --batch-size 1024 --torch-threads 8 --matches 40000 --eval-every 1000 --eval-matches 50 --eval-procs 50
        --test-matches 300"
GRU="--actor gru --seq-len 40 --rollout-len 1200"
ALL = the 24 cases 2-2-1:0,...,2-2-1:5,2-1-2:0,...,3-1-1:5 (see complete_queue.sh)
```

Times are on er011 (56 threads): memoryless (MLP) about 3.5 h, recurrent (GRU) about 8.5 h.

| Priority | Run | Arguments after `$COMMON` | Time | Why |
|---|---|---|---|---|
| 1 | `q2/gru_r25_z0/s0` (half done, resumes) | `$GRU --opp-formations 2-2-1 --switching WINDOW --radius 25 --zeta 0` | ~4 h left | Q2: does a team with memory notice the switch by itself |
| 2 | `q1/z2_r25/s0` | `--opp-formations 2-2-1,2-1-2,1-3-1,3-1-1 --radius 25 --zeta 2` | 3.5 h | Q1: ζ 2 was never trained |
| 3 | `q1/z2_r12/s0`, `q1/z2_r120/s0` | same, `--radius 12` / `--radius 120` | 2 × 3.5 h | Q1: ζ 2 at the other radii |
| 4 | `q2/mlp_r25_z0/s0` | `--opp-formations 2-2-1 --switching WINDOW --radius 25 --zeta 0` | 3.5 h | Q2: fair memoryless baseline for the recurrent team |
| 5 | `q2/gru_r12_z1/s0` | `$GRU --opp-formations 2-2-1,3-1-1 --switching WINDOW --radius 12 --zeta 1` | 8.5 h | Q2 at radius 12 |
| 6 | `q2/mlp_r12_z1/s0` | `--opp-formations 2-2-1,3-1-1 --switching WINDOW --radius 12 --zeta 1` | 3.5 h | its memoryless baseline |
| 7 | `q1/f131_z1_r{12,25,120}/s0`, `q1/f122_z1_r{12,25,120}/s0` | replace `--our-formation 2-2-1` by `1-3-1` / `1-2-2`; `--opp-formations 2-2-1,2-1-2,1-3-1,3-1-1 --radius R --zeta 1` | 6 × 3.5 h | Q1: best formation for each radius (1-3-1 = best on average in Phase 0, 1-2-2 = maximin) |
| 8 | `q3/mix_r25_z0/s0` | `$GRU --cases "$ALL" --radius 25 --zeta 0` | 8.5 h | Q3: one team for all 24 cases (idea.pdf's θmix) |
| 9 | `q3/oracle_r25_z0/s0` | `$GRU --cases "$ALL" --radius 25 --zeta 0 --set actions.show_playmaker=true` | 8.5 h | Q3: the same team told who the playmaker is (θor) |
| 10 | `q3/mix_ns_r25_z0/s0` | `$GRU --cases "$ALL" --radius 25 --zeta 0 --set actions.shadow=false` | 8.5 h | Q3: value of the shadow action (θns) |

`runs/_tools/complete_queue.sh` and `complete_queue2.sh` have exactly these lines. Remove a run's line first if it is
already done. Unfinished runs resume.
To launch: `setsid nohup taskset -c 0-55 bash runs/_tools/complete_queue.sh > runs/complete_queue.log 2>&1 < /dev/null &`

If time is short: 8–10 can be dropped, because Q3 is already answered with the specialists. 4 and 6 can be dropped by
using the existing memoryless team (`runs/q1/z0_r25/s0/ckpts/m030000.pt`) as the baseline (less fair, since it never saw a switch).
With one free day on er011 (about 24 h): 1 to 5 fit (about 27 h with 5), so drop 5 or start it last.

## 2. Evaluations (after the runs; on a server, from `football_adapt/`, `PYTHONPATH=.`)

Best checkpoint of a run: `best_ckpt` in `runs/NAME/result.json`.

**Q2**
1. Probe the recurrent team's memory ("is the opponent in 1-2-2 now"):
   `python -m scripts.q2 probe --team runs/q2/gru_r25_z0/s0/ckpts/BEST.pt --matches 200 --out runs/q2/probe_r25.npz`
2. Test (300 matches, seeds 60000+), named guess, then the anonymous one (`runs/q2/guess_anon/r25/filter.json`):
   ```
   python -m scripts.q2 test --general runs/q1/z0_r25/s0/ckpts/m030000.pt \
          --specialist runs/q2/f122_r25_z0/s0/ckpts/m005040.pt --starts 2-2-1 \
          --filter runs/q2/guess/r25/filter.json \
          --team gru=runs/q2/gru_r25_z0/s0/ckpts/BEST.pt --team mlp=runs/q2/mlp_r25_z0/s0/ckpts/BEST.pt \
          --probe runs/q2/probe_r25.npz --mode window --out runs/q2/test_r25
   ```
   Radius 12: general `runs/q1/z1_r12/s0/ckpts/m030000.pt`, specialist `runs/q2/f122_r12_z1/s0/ckpts/m016032.pt`,
   `--starts 2-2-1,3-1-1`, filter `runs/q2/guess/r12/filter.json`.
3. The same with `--mode score` (the opponent switches when it is losing).
4. Read off: regret against the oracle, the guess's delay and false alarms, the adaptation lag Δad, goal difference
   over time, and whether the recurrent team (alone, or through the probe) notices the switch as fast as the guess.

**Q1**
5. ζ 2 results next to ζ 0 and ζ 1 (same tables as `docs/reportQ1.md`).
6. Best formation f*(r): compare the 2-2-1 / 1-3-1 / 1-2-2 teams' test W at each radius, against the Phase 0
   best response and maximin.

**Q3** (only if runs 8–10 were trained)
7. Probe: `python -m scripts.q3 probe --team runs/q3/mix_r25_z0/s0/ckpts/BEST.pt --cases "$ALL" --out runs/q3/probe_mix.npz`
8. Play all 24 cases with the extra teams added to the command in `docs/reportQ3.md` §6:
   `--team mix=... --team oracle=... --team mix_ns=... --probe runs/q3/probe_mix.npz --cases "$ALL"`.
   Note: the guess maps (`runs/q3/guess_c4/maps_rec.npz`) and the specialists only cover the midfield cases.
   For defender/forward cases, either `record` the maps over all 24 cases first, or run those cases with the
   single teams only. Check how `scripts/q3.py play` handles a case that has no specialist before relying on it.
9. Read off: W and κ per case, shadow value (mix vs mix_ns), the probe's accuracy and how long it takes to identify the playmaker,
   and how often the playmaker gets the ball.

## 3. Writing

- `docs/reportQ2.md`: does not exist yet. Q2 design (pairs of teams with similar strength, the guess with and without
  player numbers), results from step 2, recurrent vs guess vs memoryless, adaptation lag, limitations.
- `docs/reportQ1.md`: add a Phase 0 section (`runs/phase0/phase0.txt`), ζ 2, best formation by radius.
- `docs/reportQ3.md`: add all-positions results if 8–10 were run. Otherwise list them as not done (defenders/forwards
  also beat v4: `runs/q3/other_slots.txt`).
- `docs/questions.md`: bring it in line with what was actually done. Deviations to state:
  - per-player specialists instead of θmix/θor (unless 8–10 are run)
  - no belief-augmented network (filter output as input)
  - no held-out switch pairs
  - Q3 only at radius 25
- Commit everything (nothing from this study is committed yet).

## 4. Known issues

- clnode148 refuses the SSH key ("Permission denied (publickey)"). Re-add the key to use it as a second server.
- The PC's `.venv` has no torch. Use it (`../.venv/bin/python`) only for light scripts, never for training.
