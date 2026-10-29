# The three research questions

Theme: **structure versus information** in 5-a-side football with limited vision.
Our team is one parameter-shared learned policy. The opponent is a fixed, hand-coded zonal team.
Each question is run from scratch (no warm start from earlier trained teams such as v3).

Common conventions:

- The vision radius `r` applies to **our team only**. The opponent's behaviour is never changed by `r`.
- Score `W = Pr[win] + 0.5 * Pr[draw]`, with confidence intervals, 300+ evaluation matches per condition, 3 seeds (5 on headline conditions).
- Always run a random-action baseline first.
- Structure level `zeta`:
  - `zeta = 0`: the formation only sets kickoff positions. No anchor observation, no hold-shape action, no control rating.
  - `zeta = 2`: the player sees its anchor, has the hold-shape action, and tackle/pass/shot success drops with distance from the anchor (control rating).
  - `zeta = 1` (anchor visible and hold-shape, no penalty) is left as a later ablation.

## Phase 0 (before any question)

- Formation x formation win matrix `M(f, g)` with the scripted team, plus best-response and maximin formations.
- Check that formations are distinguishable from where opponents stand.
- Measure how many opponents are visible on average at each candidate `r`.
- Check that a team that loses to a formation (like v3 off its training formation) loses because of the formation, not noise.

---

## Q1. Can formation structure compensate for limited vision?

**Question.** Does giving the team a formation help, does the benefit grow as `r` shrinks, and does the best formation depend on `r`?

**Setup.**

| Item | Choice |
|---|---|
| Opponent | zonal, formation randomized per match in training and unknown to our team, fixed within a match, no switching |
| Our structure | `zeta = 0` (baseline, team finds its own shape) vs `zeta = 2` (formation given) |
| Our formation | 3 formations (e.g. 2-2-1, 1-3-1, 3-1-1) |
| Radius `r` | 3 values plus full vision, one model per value, never mixed in training |
| Training | from scratch, one model per (zeta, formation, r) cell |

**Measures.**

- `B(r) = W(zeta=2) - W(zeta=0)` at each radius.
- Best formation `f*(r)`, best against the mix of opponent formations. Compared with the formation with the best mean row of `M(f, g)` and the maximin formation from Phase 0.
- Behaviour: spread between teammates, distance from anchor, possession, passes, shots.
  For `zeta = 0` also read the positions the team learns and compare them with the imposed formations.

**Hypothesis.** Structure helps more at small `r`, where a default position gives useful behaviour when a player sees little. Spread is expected to be larger at large `r`.

**Notes.**

- `zeta = 2` bundles anchor information, hold-shape and the control-rating penalty, so a positive `B(r)` does not say which part helps.
- The opponent formation is random and unknown, so `B(r)` is measured against an unknown opponent. Q1 does not produce the Q2 specialists; those need their own fixed-formation runs.

---

## Q2. How quickly can the team infer an opponent formation change?

**Question.** When the opponent changes formation mid-match and our team is not told, how fast does it infer the change, and how much does vision matter?

**Setup.**

| Item | Choice |
|---|---|
| Opponent | zonal, switches formation mid-match (scheduled single switch first; random and score-reactive as extensions) |
| Radius `r` | fixed within a match, 2 values (small, large), one model set per `r` |
| Our structure | `zeta = 0` (the team learns its own shape) |
| Specialists | one per opponent formation, each trained against a fixed formation (the only place the opponent formation is fixed in training) |
| Guesser | Bayesian filter over the opponent positions the players currently see (pooled across teammates); switch specialist only after the same guess holds for `m` steps |
| Test split | held-out switch times and formation pairs |

**Teams compared.**

1. Specialist with no switching (the cost of a formation change).
2. Specialist with oracle switch (true formation told): upper bound.
3. Specialist with guess switch: the identify-then-act team.
4. One recurrent (GRU) policy trained over all formations and switch schedules: does learning find inference by itself?

Test matches always switch formation, whatever the training of each team.

**Measures.**

- `W` and regret against the oracle.
- Detection delay `Delta_det`: steps after the switch until the guess is correct and stays correct for `m` steps.
- Adaptation lag `Delta_ad`: steps until goal difference recovers, averaged over many matches; shot and possession rates as a denser signal.

**Hypothesis.** The gap between oracle switch and guess switch is larger at small `r`. The recurrent policy may or may not match the explicit filter.

**Notes.**

- Pooling teammates' observations is a deliberate choice (mild communication); report single-view if cheap.
- Specialists are trained from kickoff; handover mid-match may be off-distribution, which the GRU comparison covers.
- Filter parameters are tuned on validation matches from the training distribution only.

---

## Q3. Can the team identify and neutralize an opponent playmaker?

**Question.** One opponent player has full-pitch vision. Can our team work out who he is and adapt, and how much of the lost win rate does it recover compared with an unaware team and an oracle?

**Setup.**

| Item | Choice |
|---|---|
| Opponent | zonal, one player is the playmaker (infinite vision radius), the others keep the coded limited vision |
| Opponent formation | randomized per match in training, unknown to our team |
| Radius `r` | fixed, our team only |
| Our structure | `zeta = 0` |
| Specialists | one per playmaker index (5), each trained against that player as playmaker; fallback: one per role (defender, midfielder, forward) |
| Identification | guess who the playmaker is from behaviour (long passes, interceptions from afar), then play the specialist for the guess |
| Extra action | shadow: move to the point at distance `d_sh` from the opponent on the line to our goal, only available when he is visible |

**Teams compared.**

1. Unaware team (trained with no playmaker).
2. Specialist with oracle (true playmaker told).
3. Specialist with guess: the identify-then-act team.
4. Same as 3 without the shadow action (ablation).
5. One recurrent policy trained over all playmaker positions including none.

**Measures.**

- Cost `E(sigma) = W(unaware, no playmaker) - W(unaware, playmaker)`.
- Recovery `kappa(sigma) = (W(adaptive) - W(unaware)) / E(sigma)`, reported only when the interval of `E` excludes zero.
- Regret against the oracle, shadow value `A_sh`, identification accuracy and delay, shadow rate.

**Hypothesis.** The playmaker costs the unaware team measurable win rate, more at small `r`. The adaptive team recovers part of it but stays below the oracle. Identification is easier when his long passes and distant interceptions reveal him.

**Open points.**

- Check in Phase 0 that his behaviour leaves evidence a classifier can use; otherwise there is nothing to identify.
- Confirm the coded opponent really has limited vision, so the playmaker is a real contrast.
- With random formations, a player index maps to different roles; per-role specialists are the cheaper alternative.
- Include "no playmaker" as a class for the guess so the team does not always shadow someone.

---

## Order and priorities

Phase 0, then Q1, then Q2, then Q3 (the heaviest). If time runs short: drop the no-shadow ablation, then reduce playmaker positions (index to role), then formations, then cut headline runs to 3 seeds.
