# Structure, Information and Inference in 5-a-Side Football

CS329 course project of Arpan Gupta, Deep Buha and Ramji Purwar (IIT Gandhinagar).

We built a small 2D 5-a-side football game in which every player sees opponents only within a limited radius,
trained our team with parameter-shared MAPPO against a hand-written zonal opponent, and asked three questions:

| | Question | Answer (95% intervals over training seeds or paired over matches) |
|---|---|---|
| **Q1** | Can a formation compensate for limited vision? | **No, not measurably.** Formation as information: W +0.09 ± 0.15; enforced formation: −0.12 ± 0.16 (pooled over radii, 33 runs). The training seed matters more (one setting: W 0.18 to 0.76 across seeds). |
| **Q2** | How fast can an opponent's formation switch be inferred, and what is it worth? | **About 30 steps, almost for free.** A Bayesian filter finds the switch in a median 28 steps and keeps 97% of the oracle's gain (regret 0.01 ± 0.04 at radius 25). End-to-end GRU/MLP teams learn robustness instead of identification. |
| **Q3** | Can the team identify and adapt to an opponent playmaker? | **Partly.** The playmaker costs the unaware team 0.54–0.80 W. Identify-then-act recovers 2–47% of that, an oracle 12–81%, one recurrent team trained on all cases 53–81%. |

W = wins + ½ draws on 300 held-out test matches per trained team. 65 teams were trained (about 2.5 million training
matches).

**Report:** [`paper/CS329_Group26_Final_Report.pdf`](paper/CS329_Group26_Final_Report.pdf) (LaTeX source in
[`paper/`](paper/)). Longer per-question reports: [`docs/reportQ1.md`](docs/reportQ1.md),
[`docs/reportQ2.md`](docs/reportQ2.md), [`docs/reportQ3.md`](docs/reportQ3.md).

## Reproducing the results (one command)

Needs Python 3.10 or newer on Linux or macOS (on Windows, use WSL). No GPU.

```bash
git clone https://github.com/Ramji-Purwar/football-agent.git
cd football-agent
bash reproduce.sh
```

The first run creates `.venv/` and installs [`requirements.txt`](requirements.txt) (CPU-only torch, about 1 GB of
downloads). Then it checks, in about 5–10 minutes:

1. **Unit tests** of the game, the opponent, the rewards and the filters (63 tests).
2. **Every table, figure and number of the report**, recomputed from the stored run files and compared with the
   committed ones: `docs/results_final.md`, the tables in `docs/reportQ{1,2,3}.md`, and `paper/numbers.txt` (each number
   quoted in the paper with the file it comes from). The recomputed figures are in `reproduce_out/paper/media/`.
3. **Trained teams replayed**: one stored team per question is loaded and plays its 50 validation matches again with
   the same seeds; its win score is compared with the one recorded during training.

It ends with `ALL CHECKS PASSED` or lists what failed. Nothing is trained, and no committed file is changed (all
outputs go to `reproduce_out/`).

| Option | What it adds | Extra time (8 cores) |
|---|---|---|
| `bash reproduce.sh --full` | the 3 teams also replay their full 300-match test (the W reported in the paper) | 10–30 min |
| `bash reproduce.sh --all` | all 65 trained teams replay their validation matches | 30–60 min |

**About exactness of replays.** On the same machine a replay is identical every time. On a different CPU or torch
version, tiny floating-point differences in the network can change one sampled action, and from then on that match
plays out differently, so occasionally a match ends differently. Replaying all 65 server-trained teams on our PC (50 validation
matches each) gave the identical W for 64 of them; one (`q3/pm4_r25_z0/s0`) scored 0.39 instead of 0.41, one match
result out of 50; `reproduce.sh` therefore accepts a
replayed W within 0.05 (50 matches) or 0.03 (300 matches) and labels each team EXACT or CLOSE. The tables, figures and
numbers of step 2 do not depend on this and must match exactly.

## Watching a trained team

```bash
cd football_adapt
../.venv/bin/python -m scripts.pygame_render --ckpt runs/q1/z1_r25/s0/ckpts/m039024.pt      # blue = our trained team
```

Every run folder's `result.json` names its best checkpoint (`best_ckpt`), which is in that folder's `ckpts/`.

## Training a team again (optional, slow)

Training is not needed to check the results. Each run stores all its arguments in `result.json`; this prints the exact
command that trained it:

```bash
cd football_adapt
PYTHONPATH=. ../.venv/bin/python -m scripts.train_command runs/q1/z1_r25/s0
```

A full run is 40,000 matches (48 million game steps) and took 13–16 hours on a 40-thread CPU server, so it is meant for
a server, not a laptop. Training is stochastic and the seed-to-seed spread is large (that is one of our findings), so
a retrained team is expected to land within the spread of its setting, not on the same W. For the three warm-started
Q3 teams (`q3/mixw_*`) the command also prints the warm-start step.

## Repository layout

```
reproduce.sh                 the one-command check described above
requirements.txt
idea.pdf                     the project proposal
paper/                       the final report: main.tex, references.bib, PRIMEarxiv.sty, media/ (figures),
                             make_figs.py (draws every figure, writes numbers.txt), CS329_Group26_Final_Report.pdf
docs/                        per-question reports (reportQ1-3.md), results_final.md, FINAL_REPORT.md, and the
                             design documents: game.md, opponent.md, agents.md, training.md, experiments.md, ...
football_adapt/              all code (run everything from inside this folder, with PYTHONPATH=.)
  football/                  the game: config, geometry, formations, actions, mechanics, observation, env
  opponent/                  the scripted zonal opponent, formation switching, the playmaker
  agents/                    MAPPO actor/critic (MLP or GRU), the formation filter (Q2), the playmaker filter (Q3)
  training/train.py          one training run: rollouts, PPO update, evaluation, checkpoints, final test
  configs/                   default.yaml (every parameter) and study.yaml (the settings used in the study)
  scripts/                   q2.py and q3.py (the Q2/Q3 experiments), replay_test.py, train_command.py, viewers
  tests/                     unit tests
  runs/                      the results of every run (see below)
    _tools/                  final_report.py (all tables) and the evaluation helpers used during the study
    phase0/                  the Phase 0 checks of the game and the opponent
    q1/ q2/ q3/              one folder per setting, one sub-folder per training seed
```

Each trained run folder (e.g. `football_adapt/runs/q1/z1_r25/s0/`) holds `result.json` (arguments, best checkpoint,
test result), `log.jsonl` (training and evaluation log, used for the learning curves), `config.json`, `train.log`,
`progress.txt`, `vs_formations.txt` (Q1 teams against every opponent formation) and `ckpts/` with the checkpoints the
results use: the best one of every run, plus the specific checkpoints the Q2 and Q3 experiments used. The other
intermediate checkpoints (one per 1,000 training matches) and the optimiser states were left out to keep the
repository small; the four optimiser states needed for the Q3 warm start are included.

The Q2 and Q3 experiment files (`runs/q2/test/`, `runs/q2/guess*/`, `runs/q3/play*`, `runs/q3/guess_c4/`, ...) are
the raw outputs that `final_report.py` and `paper/make_figs.py` read. The commands that produced them are listed in
the section "Files and how to reproduce" of each report in `docs/`.

## Simplifications (stated in the report)

2D ground passes only; teammates and the ball are always visible, opponents only within the vision radius; opponents
are identified by number (relaxed in Q2's "no player IDs" filter); one fixed scripted opponent.
