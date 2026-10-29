# Q2 team pairs (picked 2026-10-20)

Before the switch the general team plays (trained against 2-2-1, 2-1-2, 1-3-1, 3-1-1). Once the switch to 1-2-2 is
detected, the 1-2-2 specialist takes over. Both teams of a pair are saved checkpoints of finished runs, chosen so that
they play at the same moderate level (similar W and goal difference). The final checkpoints were too strong
(specialists W 0.97-0.99, +5 to +12 goals a match): a team that strong still wins after a late detection, so a slow
detection would not show in the result.

Picked on 100 matches per checkpoint (seeds 20000+, `stage1.txt`), confirmed on 300 matches (test seeds 50000+,
`stage2.txt`). The general team plays 2-2-1, the specialist plays 1-2-2. Goals are for - against, then the difference.

| radius | zeta | before the switch | vs 2-2-1 | after detection | vs 1-2-2 |
|---|---|---|---|---|---|
| 25 | 0 | `q1/z0_r25/s0/ckpts/m030000.pt` (v4) | W 0.75, 3.72 - 1.86 (+1.86) | `q2/f122_r25_z0/s0/ckpts/m005040.pt` | W 0.84, 5.38 - 3.52 (+1.86) |
| 12 | 1 | `q1/z1_r12/s0/ckpts/m030000.pt` (v2) | W 0.64, 3.44 - 2.54 (+0.90) | `q2/f122_r12_z1/s0/ckpts/m016032.pt` | W 0.62, 3.27 - 2.69 (+0.58) |

With 300 matches a W is good to about +-0.05 and a goal difference to about +-0.3.

## Which starting formations Q2 uses (300 matches each, test seeds 50000+, `stage3.txt`)

The opponent may start in any of the four formations the general team trained against, then switches to 1-2-2. The
pair stays as above, and a starting formation is used only if the general team plays it at the specialist's level
against 1-2-2: W within 0.10 and goal difference within 0.6 (the 2-2-1 pairs pass this).

| radius | specialist vs 1-2-2 | general team vs 2-2-1 | vs 2-1-2 | vs 1-3-1 | vs 3-1-1 |
|---|---|---|---|---|---|
| 25 | W 0.84, +1.86 | **W 0.75, +1.86** | W 0.68, +1.07 | W 0.51, +0.17 | W 0.84, +3.88 |
| 12 | W 0.62, +0.58 | **W 0.64, +0.90** | W 0.48, -0.46 | W 0.20, -2.56 | **W 0.62, +0.61** |

Bold = used. Radius 25: 2-2-1 only. Radius 12: 2-2-1 and 3-1-1.

## The wrong team (300 matches, test seeds 50000+)

| radius | general team vs 1-2-2 (switch not yet detected) | specialist vs 2-2-1 (false alarm) | vs 2-1-2 | vs 1-3-1 | vs 3-1-1 |
|---|---|---|---|---|---|
| 25 | W 0.04, 0.60 - 4.43 (-3.83) | W 0.11, 0.86 - 3.68 (-2.82) | W 0.21, -2.08 | W 0.22, -2.12 | W 0.05, -3.10 |
| 12 | W 0.03, 1.35 - 6.02 (-4.67) | W 0.05, 0.45 - 3.63 (-3.18) | W 0.02, -3.97 | W 0.04, -4.08 | W 0.07, -2.61 |

So every 100 steps (a match is 1200) of late detection costs about 0.47 goals at radius 25 and 0.44 at radius 12.
A false alarm costs, per 100 steps, about 0.39 (radius 25, from 2-2-1), 0.34 (radius 12, from 2-2-1) and 0.27
(radius 12, from 3-1-1).

## Notes

- Radius 25: the specialist jumped from W 0.24 (m004032) to 0.84 (m005040) to 0.95 (m006000), so m005040 is the
  only moderate checkpoint. Same goal difference as v4 m030000, but it scores and concedes more.
- The radius-25 pair is stronger (+1.9 a match) than the radius-12 pair (+0.6 to +0.9), so at radius 25 the delay has to
  be longer before a win turns into a loss. The goal difference after the switch compares the two radii fairly.
- Tool: `runs/_tools/ckpt_vs_form.py --matches N --seed0 S --out FILE CKPT:FORMATION ...` (run it on a server).
