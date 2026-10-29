# Q2 runs

One folder per setting, one sub-folder per training seed (`s<seed>`). Each run folder holds
`ckpts/` (a checkpoint per evaluation), `log.jsonl`, `config.json`, `result.json` (the best checkpoint
on the evaluation seeds, tested on 300 new matches), `vs_formations.txt` and `train.log`.

Test W = wins + draws/2 over 300 matches (about +-0.03). Formation columns: W over 50 matches;
`*` = a formation the run never trained against.

| run | old name | zeta | radius | seed | best checkpoint | test W | goals | possession | 2-2-1 | 2-1-2 | 1-3-1 | 1-2-2 | 3-1-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f122_r12_z1/s0 |  | 1 | 12 | 0 | m032016.pt | **0.97** | 12.36 - 0.56 | 7% | 0.01* | 0.00* | 0.02* | 0.98 | 0.00* |
| f122_r25_z0/s0 |  | 0 | 25 | 0 | m039024.pt | **0.99** | 8.73 - 3.80 | 29% | 0.04* | 0.13* | 0.09* | 0.99 | 0.03* |
