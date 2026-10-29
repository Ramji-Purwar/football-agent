# Q2: the guesser without player numbers (2026-10-21)

Same teams, matches and switches as `../guess/README.md`, but the guesser is not told which opponent is which
(`FormationFilter(..., anonymous=True)`): its maps say where AN opponent stands (all five pooled) and how many of them
we see at once, so it has to recognise the shape. Maps: the same 400 practice matches (seeds 30000+; the per-player
counts came out identical to `../guess`). Tuned on the same 200 practice switch matches: beta 0.1 (radius 25) / 0.2
(radius 12), h 0.0001, m 1. Test: the same 300 matches (seeds 60000+); the none and oracle matches are identical to
`../guess` (1200 of 1200 scores).

| | radius 25, numbers known | radius 25, no numbers | radius 12, numbers known | radius 12, no numbers |
|---|---|---|---|---|
| switches found | 100% | 97% | 100% | 96% |
| median delay (steps) | 28 | 68 | 36 | 66 |
| matches with a false alarm | 0% | 23% | 1% | 59% |
| steps a match on the wrong team after the switch | 31 | 128 | 59 | 208 |
| W guess (oracle, none) | 0.74 (0.75, 0.46) | 0.68 (0.75, 0.46) | 0.46 (0.53, 0.25) | 0.39 (0.53, 0.25) |
| loss against the oracle, W | 0.01 +-0.04 | 0.07 +-0.05 | 0.07 +-0.06 | 0.14 +-0.07 |
| loss against the oracle, goal difference | -0.01 +-0.20 | 0.42 +-0.24 | 0.34 +-0.32 | 0.80 +-0.41 |
| share of the oracle's gain kept | 97% | 76% | 75% | 50% |

Radius 12 without numbers, by starting formation: from 2-2-1 a loss of W 0.17 +-0.09 (delay 73, false alarms in 47%
of matches), from 3-1-1 W 0.11 +-0.10 (delay 58.5, false alarms in 72%).

## Notes

- Without numbers the delay is longer than the opponent's walk to its new spots (up to 60 steps): now what the team
  sees limits the guess, and radius 12 loses about twice as much as radius 25.
- The guess often drops back to the general team after the specialist comes on (wrong-team steps after the switch far
  above the delay). The tuning matches have the general team play on after the switch, so they cannot show this.
