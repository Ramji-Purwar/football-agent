# Q2: the formation guesser against the oracle (2026-10-21)

Teams: the pairs of `../pairs/README.md`. The opponent starts in 2-2-1 (radius 25) or in 2-2-1 / 3-1-1 (radius 12,
drawn per match) and switches to 1-2-2 at a random step between 100 and 1100. Every test match (300 per radius, seeds
60000+) is played three times:

- none: the general team plays on after the switch
- oracle: the 1-2-2 specialist comes on at the switch
- guess: the specialist comes on while the guesser says 1-2-2 (`agents/belief_filter.py`, `scripts/q2.py`)

The guesser's maps were counted on 400 practice matches (seeds 30000+), its settings tuned on 200 practice switch
matches (seeds 40000+): beta 0.1, h 0.0001, m 1 at both radii (`tune.txt`, `filter.json`).

| radius | team | W | goals | goal difference | after the switch |
|---|---|---|---|---|---|
| 25 | none | 0.46 | 2.60 - 2.97 | -0.37 | -1.61 |
| 25 | oracle | 0.75 | 4.06 - 2.63 | +1.42 | +0.18 |
| 25 | guess | **0.74** | 4.02 - 2.59 | +1.43 | +0.19 |
| 12 | none | 0.25 | 2.20 - 4.28 | -2.07 | -2.56 |
| 12 | oracle | 0.53 | 2.54 - 2.69 | -0.15 | -0.63 |
| 12 | guess | **0.46** | 2.40 - 2.89 | -0.49 | -0.97 |

Loss of the guess against the oracle (same matches): radius 25 W 0.01 +-0.04, goal difference -0.01 +-0.20;
radius 12 W 0.07 +-0.06, goal difference 0.34 +-0.32 (from 2-2-1: W 0.12 +-0.07; from 3-1-1: W 0.02 +-0.08).
Not switching at all loses W 0.29 (radius 25) and 0.28 (radius 12).

The guesser found every switch. Median delay 28 steps at radius 25, 36 at radius 12 (the opponent players need up to
60 steps to walk to their new spots). False alarms: none at radius 25, in 1% of the matches at radius 12. Steps on the
wrong team after the switch, a match: 31 at radius 25, 59 at radius 12.

## Notes

- Radius 12: after the specialist comes on, the guess sometimes drops back to the general team for a while (59 steps on
  the wrong team after the switch against a mean delay of 38). The tuning matches had the general team play on after
  the switch, so they could not show this, and m = 1 was picked.
- After a mid-match handover the specialist plays below its kick-off level: the oracle gets +0.18 after the switch at
  radius 25 (about half a match, where its kick-off level would give about +0.9) and -0.63 at radius 12 (about +0.3).
- Files per radius: `maps.npz`, `traces.npz` (tuning matches), `tune.txt` and `filter_grid.json` (every setting tried),
  `filter.json`, `test.jsonl` (one line per match and team), `test.txt`. `run_all.sh` ran all of it on clnode148.
