# Q3 report: Can the team identify and neutralize an opponent playmaker?

All numbers come from **one training run per team** (seed 0). Every comparison is **paired by seed**: the same seed
gives the same match up to the moment the teams act differently, so the unaware, oracle and adaptive teams are
compared match by match. 500 matches per case, seeds 70000+, never used in training or tuning. W is good to about
±0.04, a difference of two W on the same seeds to about ±0.05.

## 1. Answer in short

| Q3 asks | Answer |
|---|---|
| Does the playmaker cost the unaware team? | **Yes, a lot.** Whichever midfielder he is, v4 (the best Q1 team) goes from W 0.82–0.90 to 0.02–0.41: a cost E of 0.46–0.80 W, 6–10 goals of goal difference a match. |
| Can our team tell who he is? | **Usually, but not quickly.** From where the opponents stand relative to the ball, the guess names the right player in 99–100% of matches in three of the four cases, after 140–195 of 1,200 steps (median). With 1-3-1 and player 4 it is right in only 68%. Matches without a playmaker are the weak point: in 13–38% the guess wrongly names someone. |
| How much of the cost does the adaptive team win back, against the unaware team and the oracle? | **Part of it, always less than the oracle.** The oracle (specialist from the kick-off) wins back κ 0.12–0.81 of the cost; the identify-then-act team 0.02–0.47. Regret against the oracle 0.08–0.20 W. |

Other findings:

- **The specialists do not neutralize him by marking him.** They shadow the playmaker no more often than v4 does (about one SHADOW step in four is on him, for both teams, close to the 1 in 5 of picking an opponent at random).
- **Specialists are much worse than v4 against an opponent without a playmaker** (W 0.11–0.60 against 0.82–0.90), so a false alarm is expensive: when the guess wrongly names player 4 in a 2-2-1 match without a playmaker, W falls from 0.63 to 0.23 in those matches. Over all no-playmaker matches the adaptive team loses 0.01–0.09 W.
- **Training from scratch against this playmaker is hard.** The player 3 specialist stayed below v4 (test W 0.25 against v4's 0.33 in its cases) and was dropped. The player 4 specialist was still improving when training ended.
- **Switching late costs more than the lost time.** With 1-3-1 and player 2 the adaptive team, switching at a median step of 141 (12% of the match), gets 0.50 against the oracle's 0.71.

| case | W unaware | W adaptive | W oracle | cost E | κ adaptive | κ oracle |
|---|---|---|---|---|---|---|
| 1-3-1, player 2 | 0.22 | **0.50** | 0.71 | 0.60 | 0.47 | 0.81 |
| 3-1-1, player 4 | 0.36 | **0.56** | 0.64 | 0.54 | 0.36 | 0.52 |
| 2-2-1, player 4 | 0.21 | **0.32** | 0.46 | 0.65 | 0.17 | 0.38 |
| 1-3-1, player 4 | 0.02 | **0.04** | 0.12 | 0.80 | 0.02 | 0.12 |

## 2. What was run

### 2.1 The playmaker

The plan's playmaker (one opponent with infinite vision, everything else the same) made no difference: against v4 only
player 4 in 2-2-1 cost anything (W 0.88 → 0.66, 300 matches) and the defenders and the forward cost nothing. The user
then asked that **every midfielder** must make an impact, in **several formations**, and finally that v4 must **lose in
every case** (W < 0.5). The abilities were added one at a time and each was screened against v4:

| playmaker | v4 W in the 7 midfielder cases | kept? |
|---|---|---|
| none | 0.82–0.90 | |
| full vision (+ teammates look for him) | 0.55–0.90; v4 wins all 7 | yes |
| + free role (shows for the ball next to the carrier) | 0.21–0.78; v4 still ahead in 6 of 7 | yes |
| + his passes 1.5 times faster | lower, but his fast passes gave him away within ~25 steps | **no** (it made identification trivial) |
| + **control 4** | **0.02–0.41: v4 loses all 7** | yes |

The final playmaker (`configs/default.yaml`, `opponent.playmaker`): sees the whole pitch; his teammates prefer passing to
him (`attack.hub_bonus` 1); in possession he leaves his zone and shows for the ball; and his control rating is 4
everywhere (the others' is 0.5–1 depending on distance from their zone). Control 4 means v4 wins about 1 tackle in 5
on him, his passes have no aiming error, and his open shots inside 35 m score. He scores 1.5–1.9 goals a match himself.
Control 4 was preferred over giving the whole team full vision because his whole effect goes through his own touches,
so a team that marks him can in principle take it away.

### 2.2 Cases

Only midfielders are playmakers: 2-2-1 players 3 and 4, 2-1-2 player 3, 1-3-1 players 2, 3 and 4, 3-1-1 player 4.
The player 3 specialist failed (Section 3.4), so the study keeps **four cases**: 1-3-1 p2, 2-2-1 p4, 1-3-1 p4,
3-1-1 p4, plus their three formations without a playmaker (for false alarms). The player 3 cases were still played,
as a check: the guess may name player 3, in which case the adaptive team keeps v4.

### 2.3 Teams

| team | what it is |
|---|---|
| **unaware** | v4 = Q1's zeta 0, radius 25, seed 0 (`runs/q1/z0_r25/s0/ckpts/m038016.pt`), trained against 2-2-1, 2-1-2, 1-3-1, 3-1-1 without a playmaker |
| **specialist p2** | trained like v4 but against 1-3-1 with player 2 as the playmaker. Best checkpoint m024000 |
| **specialist p4** | trained like v4 but against 2-2-1, 1-3-1 and 3-1-1 with player 4 as the playmaker. Best checkpoint m039024 |
| (specialist p3) | against 2-2-1, 2-1-2, 1-3-1 with player 3. Dropped: worse than v4 |
| **oracle** | the specialist for the true playmaker from the kick-off (v4 if there is none) |
| **adaptive** | v4 until the guess commits, then the specialist for the guess; no going back. A guess of "no playmaker" or "player 3" keeps v4 |

Every specialist: zeta 0, radius 25, our formation 2-2-1, the study recipe, 40,000 training matches, seed 0, easy start
from opponent randomness 0.5, best of 40 evaluations, exactly as v4 (Q1 report, Section 2.2).

### 2.4 The guess

`agents/playmaker_filter.py`. Each step it pools what our five players see: for each opponent number, how far he
stands from the ball (7 distance bins, or "nobody sees him"), in each ball state (theirs, ours, loose). Practice
matches of v4 against every case (100 per case, seeds 30000+) give, per number, how often a playmaker and an ordinary
player stand in each bin. A Bayes filter over "no playmaker", "player 2", "player 3", "player 4" multiplies in each
step's evidence raised to β (successive steps show almost the same picture). The guess commits when one hypothesis
reaches probability 0.95. β = 0.05 and the threshold were tuned on 50 other matches per case (seeds 40000+). The
filter only knows the opponent numbers, not their formation.

### 2.5 Measures

- Cost `E = W(unaware, no playmaker) − W(unaware, playmaker)`, same formation and seeds.
- Recovery `κ = (W(team) − W(unaware)) / E`: 0 = nothing won back, 1 = all of it. Intervals by bootstrap over seeds.
- Regret `W(oracle) − W(adaptive)`.
- Guess: share of matches whose commit is right, wrong, or never comes; median step of a right commit.
- Shadow rate: share of our SHADOW steps that shadow the true playmaker.

### 2.6 Changes from the plan in `questions.md`

| Plan | What was run | Why |
|---|---|---|
| playmaker = infinite vision only | + free role, hub bonus, control 4 | vision alone cost v4 nothing in most cases (user: every midfielder must matter, v4 must lose) |
| playmaker positions 1–5, decision 16's roles 1/3/5 | **midfielders only** (players 2, 3, 4), 4 formations | the user's choice; defenders and the forward made no difference |
| one specialist per playmaker number (5) | **2** (players 2 and 4) | only midfielders are playmakers; the player 3 specialist failed (user: keep players 2 and 4) |
| guess from long passes and interceptions from afar | from positions relative to the ball | his passes are no longer faster, and positions carry the evidence |
| small and large `r` | **radius 25 only** | v4 is the unaware team; the "more at small r" part of the hypothesis is not tested |
| team 4: no-shadow ablation | **not run** | the switch exists (`actions.shadow: false`), but the specialists do not use SHADOW on him (Section 3.5), so its value would be ~0; this was the first item to drop if time ran short |
| team 5: one recurrent policy over all cases | **not run** | |

## 3. Results

### 3.1 Cost of the playmaker

| case | v4 W, no playmaker → playmaker | E | goal difference a match |
|---|---|---|---|
| 1-3-1, player 2 | 0.82 → 0.22 | 0.60 ± 0.05 | +5.0 → −3.2 |
| 2-2-1, player 4 | 0.87 → 0.21 | 0.65 ± 0.04 | +5.4 → −3.0 |
| 1-3-1, player 4 | 0.82 → 0.02 | 0.80 ± 0.04 | +5.0 → −4.8 |
| 3-1-1, player 4 | 0.90 → 0.36 | 0.54 ± 0.05 | +5.1 → −1.3 |
| *2-2-1, player 3* | *0.87 → 0.41* | *0.46 ± 0.05* | *+5.4 → −1.5* |
| *2-1-2, player 3* | *0.83 → 0.37* | *0.46 ± 0.05* | *+4.4 → −2.3* |
| *1-3-1, player 3* | *0.82 → 0.22* | *0.60 ± 0.05* | *+5.0 → −3.2* |

Every interval excludes zero by a wide margin, so κ is reported for every case.

### 3.2 Recovery and regret

| case | W unaware | W adaptive | W oracle | κ adaptive [95%] | κ oracle [95%] | regret | GD unaware / adaptive / oracle |
|---|---|---|---|---|---|---|---|
| 1-3-1, player 2 | 0.22 | 0.50 | 0.71 | 0.47 [0.40, 0.54] | 0.81 [0.75, 0.88] | 0.20 ± 0.04 | −3.2 / −0.2 / +0.5 |
| 2-2-1, player 4 | 0.21 | 0.32 | 0.46 | 0.17 [0.10, 0.24] | 0.38 [0.31, 0.45] | 0.14 ± 0.05 | −3.0 / −1.1 / −0.4 |
| 1-3-1, player 4 | 0.02 | 0.04 | 0.12 | 0.02 [0.00, 0.04] | 0.12 [0.09, 0.15] | 0.08 ± 0.03 | −4.8 / −3.8 / −2.0 |
| 3-1-1, player 4 | 0.36 | 0.56 | 0.64 | 0.36 [0.27, 0.44] | 0.52 [0.43, 0.60] | 0.09 ± 0.05 | −1.3 / +0.4 / +0.9 |

- The specialists **turn two losing cases into winning ones** (1-3-1 p2 and 3-1-1 p4, oracle W 0.71 and 0.64), and
  halve or more the goal-difference loss everywhere.
- The adaptive team is **always between the unaware team and the oracle**, as the hypothesis says. It keeps 44–71% of
  the oracle's gain where the guess is quick and reliable, and almost nothing with 1-3-1 p4.
- 1-3-1 p4 stays lost even for the oracle (0.12): the p4 specialist's weakest formation, and the hardest case to
  identify.

### 3.3 Identification

Played live in the adaptive matches (500 per case; v4 plays until the commit, so the evidence is the one the maps were
made from).

| case | right | wrong | never | median step of a right commit |
|---|---|---|---|---|
| 1-3-1, player 2 | 100% | 0% | 0% | 141 |
| 3-1-1, player 4 | 100% | 0% | 0% | 169 |
| 2-2-1, player 4 | 99% | 1% | 0% | 194 |
| 1-3-1, player 4 | 68% | 18% | 14% | 507 |
| *2-1-2, player 3* | *100%* | *0%* | *0%* | *258* |
| *1-3-1, player 3* | *97%* | *3%* | *0%* | *346* |
| *2-2-1, player 3* | *89%* | *6%* | *5%* | *516* |
| no playmaker, 3-1-1 | 87% | 12% | 0% | 227 |
| no playmaker, 2-2-1 | 73% | 22% | 5% | 485 |
| no playmaker, 1-3-1 | 42% | 38% | 20% | 700 |
| no playmaker, 2-1-2 | 29% | 29% | 42% | 708 |

For a match without a playmaker, "never commits" has the same effect as a right commit (v4 keeps playing).

The only evidence is where each opponent stands relative to the ball, and with the free role the playmaker is near
the ball more often than an ordinary player of his number. Likely reason for the hard cases (not checked): in 1-3-1 and
2-1-2 the midfielders are often near the ball anyway, so a playmaker there stands out less, and an ordinary midfielder
can look like one. Earlier, when his passes were 1.5 times faster, he was named in 99–100% of matches within 20–28 steps; that
ability was dropped for being a giveaway.

### 3.4 What the regret is made of

Adaptive against oracle W, by when the right commit came (same seeds):

| case | commit before step 150 | 150–300 | 300–600 | 600+ | wrong or never |
|---|---|---|---|---|---|
| 1-3-1, player 2 | 0.49 / 0.74 (279) | 0.54 / 0.66 (200) | 0.42 / 0.70 (20) | — | (1) |
| 2-2-1, player 4 | 0.31 / 0.44 (99) | 0.31 / 0.46 (327) | 0.40 / 0.45 (68) | (2) | 0.12 / 0.75 (4) |
| 1-3-1, player 4 | 0.00 / 0.00 (7) | 0.07 / 0.09 (67) | 0.04 / 0.13 (139) | 0.06 / 0.12 (128) | 0.01 / 0.12 (159) |
| 3-1-1, player 4 | 0.57 / 0.63 (197) | 0.56 / 0.66 (272) | 0.48 / 0.55 (30) | (1) | — |

(adaptive / oracle, number of matches in brackets)

- In the three easy cases almost all the regret comes from **the time v4 plays before the commit**. Even a commit
  before step 150 (the first eighth of the match) costs 0.06–0.25 W. In those steps v4 concedes against a team the
  oracle would already be beating, and in close matches a goal down swings W a lot.
- In 1-3-1 p4 a third of the matches never get the right commit, and those keep v4's 0.01.

**False alarms** (no playmaker, the guess names one):

| formation | guess | matches | W adaptive | W unaware (same seeds) |
|---|---|---|---|---|
| 2-2-1 | player 4 | 111 | 0.23 | 0.63 |
| 3-1-1 | player 4 | 53 | 0.25 | 0.71 |
| 1-3-1 | player 2 | 16 | 0.50 | 0.94 |
| 1-3-1, 2-1-2 | player 3 | 173, 140 | same as v4 | (no specialist: no cost) |

The player 4 false alarms come mostly in matches v4 was already finding hard (W 0.63–0.71 against 0.87–0.90 overall). The
specialist then loses most of them. Over all no-playmaker matches the adaptive team loses 0.09 W in 2-2-1, 0.06 in
3-1-1, 0.02 in 1-3-1 and 0.01 in 2-1-2.

### 3.5 How the specialists play him: not by shadowing

| case | SHADOW steps a match, unaware / oracle | share of them on the playmaker, unaware / oracle |
|---|---|---|
| 1-3-1, player 2 | 1,289 / 1,198 | 19% / 25% |
| 2-2-1, player 4 | 1,219 / 1,091 | 28% / 31% |
| 1-3-1, player 4 | 1,338 / 1,131 | 24% / 21% |
| 3-1-1, player 4 | 1,027 / 1,031 | 33% / 34% |

Both teams use SHADOW about as much, and on the playmaker about as often (an opponent picked at random would be 20%).
Even with every match against the same playmaker, the specialists did not learn to mark him. Whatever they do
differently, it is not man-marking. Finding out how they win was not part of this work. So the "neutralize" in Q3 is
measured (W won back), but **not by the mechanism the plan expected**. For the same reason the no-shadow ablation
(team 4) was not run: it would remove an action the specialists do not aim at him.

### 3.6 The specialists' training

| specialist | test W (300, its cases) | evaluation W over training (every 4,000 matches) | full-strength opponent in the 2nd half |
|---|---|---|---|
| player 2 (1-3-1) | **0.69** | 0.00, 0.16, 0.14, 0.35, 0.77, 0.78, 0.70, 0.72, 0.61, 0.70 | 100% of the time |
| player 4 (3 formations) | **0.43** | 0.00, 0.03, 0.02, 0.05, 0.03, 0.18, 0.16, 0.19, 0.25, 0.33 | 9% |
| player 3 (3 formations) | **0.25** | 0.01, 0.08, 0.14, 0.18, 0.28, 0.25, 0.20, 0.23, 0.29, 0.36 | 8% |

The easy start lowers the opponent's randomness only while we win 60% of evaluation matches. The player 3 and 4
specialists rarely reached the full-strength opponent and spent the second half against randomness 0.05. Both were
still improving at 40,000 matches. The player 2 specialist, with one formation to learn, got there by 24,000 matches.

Against opponents **without** a playmaker (50 matches per formation, seeds 20000+): player 4 specialist 2-2-1 0.16,
1-3-1 0.11, 3-1-1 0.48; player 2 specialist 1-3-1 0.60 (v4: 0.82–0.90). Trained only against the playmaker, they
lost v4's way of beating an ordinary opponent.

## 4. Conclusions for Q3

1. **A playmaker can cost the unaware team the match**, but only once he has real abilities. Full vision alone did
   almost nothing; free role and control made every midfielder decisive (cost 0.46–0.80 W).
2. **He can be identified from behaviour alone,** from where he stands relative to the ball, in 99–100% of matches
   in three of four cases. But it takes 140–200 steps, and one case (1-3-1 p4) is only 68%.
3. **Identify-then-act wins back part of the cost** (κ 0.02–0.47) and **stays below the oracle** (κ 0.12–0.81,
   regret 0.08–0.20). Both parts of the hypothesis hold at r 25.
4. **The adaptive team's limit is mostly the specialists, and then the delay.** Where the specialist is strong and the
   guess quick (1-3-1 p2, 3-1-1 p4), the adaptive team wins. Where the specialist itself is weak (1-3-1 p4, oracle 0.12),
   identification cannot help.
5. **A wrong guess is expensive** because the specialists are poor against ordinary teams. A false alarm in 2-2-1
   turns a 0.63 match into 0.23.
6. **The specialists win without marking him.** The plan's shadow action is not how this team adapts.

## 5. Limitations

- **One run per team.** The player 3 failure and the player 4 specialist's weakness may be partly bad luck; with one
  seed only large differences count. The comparisons between teams are paired and tight; between training runs they
  are not.
- **Training length and the easy start.** The player 3 and 4 specialists were still improving and mostly trained
  against a slightly weakened opponent. Longer runs, or starting the specialists from v4's weights instead of from
  scratch, would probably make them stronger. That would raise both κ values, and a specialist that stays good against
  ordinary teams would also make false alarms cheaper.
- **Radius.** Only r 25. The "cost is larger at small r" part of the hypothesis is not tested.
- **The playmaker was tuned against v4.** His abilities were raised until v4 lost every case. The costs are large by
  design, and they are measured against the team they were tuned on.
- **The guess learns from v4's matches.** After the switch the evidence would look different, but the guess stops at
  the commit, so this does not matter here.
- **Not run:** team 4 (no shadow) and team 5 (one recurrent policy over all cases).
- **Best-checkpoint pick.** As in Q1, the specialists' best checkpoint is the best of 40 evaluations. All W in this
  report come from new seeds.

## 6. Files and how to reproduce

| What | Where (under `football_adapt/`) |
|---|---|
| Playmaker settings | `configs/default.yaml`: `opponent.playmaker`, `attack.hub_bonus`; SHADOW switch `actions.shadow` |
| Screens that chose the playmaker | `runs/q3/playmaker_v4.txt`, `midfield_speed.txt`, `midfield_hub.txt`, `midfield_free.txt`, `impact*.txt`, `control*.txt` |
| Cost of the final playmaker | `runs/q3/control4_test.txt`, `control4_cost.txt` (`runs/_tools/pm_cost.py`) |
| Specialists | `runs/q3/pm2_r25_z0/s0/`, `pm4_r25_z0/s0/`, `pm3_r25_z0/s0/` (`runs/_tools/q3_specialists.sh`) |
| Guess | `agents/playmaker_filter.py`; maps and offline test `runs/q3/guess_c4/` (`guess.txt`) |
| Unaware / oracle / adaptive | `runs/q3/play.txt`, `play_matches.jsonl` (one line per match) |

The guess (on a server, from `football_adapt/`; `runs/q3/guess_c4/run_all.sh` does all three steps):

```bash
python -m scripts.q3 record --general runs/v4/ckpts/m038016.pt --cases 2-2-1:0,2-2-1:3,2-2-1:4,2-1-2:0,2-1-2:3,1-3-1:0,1-3-1:2,1-3-1:3,1-3-1:4,3-1-1:0,3-1-1:4 \
       --matches 100 --seed0 30000 --out runs/q3/guess_c4/maps_rec.npz       # then tune 40000+ (50), test 60000+ (100)
python -m scripts.q3 evaluate --maps runs/q3/guess_c4/maps_rec.npz --tune runs/q3/guess_c4/tune_rec.npz \
       --test runs/q3/guess_c4/test_rec.npz --out runs/q3/guess_c4/guess.txt
```

The three teams on the same seeds (about one hour on er011):

```bash
python -m scripts.q3 play --general runs/v4/ckpts/m038016.pt \
       --specialist 2:runs/q3/pm2_r25_z0/s0/ckpts/m024000.pt --specialist 4:runs/q3/pm4_r25_z0/s0/ckpts/m039024.pt \
       --maps runs/q3/guess_c4/maps_rec.npz --beta 0.05 --commit 0.95 \
       --cases 1-3-1:2,2-2-1:4,1-3-1:4,3-1-1:4,1-3-1:0,2-2-1:0,3-1-1:0,2-2-1:3,2-1-2:3,1-3-1:3,2-1-2:0 \
       --matches 500 --seed0 70000 --out runs/q3/play
```

On the PC v4's checkpoint is `runs/q1/z0_r25/s0/ckpts/m038016.pt` (on the servers `runs/v4/ckpts/m038016.pt`).
