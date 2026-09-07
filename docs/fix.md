# Loss analysis: rounds 42-48 and what to fix

Analysis of every game in `pgns/` from round 42 onward, where the current model has been
playing. Six games: 1 win (R44), 5 losses (R42, R43, R45, R46, R48) — **every loss ended in
checkmate**, never on time and never on adjudication. That shape drives the whole analysis: we
are not losing to the clock or to grinding endgames, we are getting our king killed.

| R | Opponent | Col | Result | How | Material swing |
|---|---|---|---|---|---|
| 42 | ChessNotCheckers | W | **Loss** | mated m43 | equal → −1.25 after `13.Nxg5??` |
| 43 | The Huxley Knights | B | **Loss** | mated m78 | equal → −3.25 by m20, bled in pawns |
| 44 | Baryon | W | Win | mated m67 | — |
| 45 | Blunder Buss | W | **Loss** | mated m59 | **+2.00** at m22 → −6 by m40 |
| 46 | Loki 3.0 | B | **Loss** | mated m48 | equal at m23 → pawn queened m31 |
| 48 | Nakamura | W | **Loss** | mated m66 | equal → −2 after `21.b3??` |

## Part 1 — Panic mode: the direct answer

**Panic mode never fired in any of these games. Not once.**

The gate is `agent.py:124`, `time_left_ms <= timeman.PANIC_MS` where `PANIC_MS = 150`
(`timeman.py:32`). Our lowest clock across all six games:

```
R42: 39.93s   R43:  6.89s   R44:  7.55s
R45: 12.63s   R46: 10.97s   R48:  5.76s
```

The closest we ever came was 5.76s — 38x above the threshold. When it does fire it skips the
tree search entirely and returns `sr.quick_best_move`, a pure MVV-LVA/promotion pick with no
search at all (a tablebase hit still overrides it). That path is dead code in practice.

### What actually looks like panic, and isn't

In all five losses the clock goes *up* by ~0.497s/move at the end — we were moving in 0-20ms
with 40+ seconds banked. That is not panic. It is `agent.py:188`:

```python
if over_time or abs(score) >= MATE_THRESHOLD or depth >= MAX_DEPTH:
    break
```

Once any mate score comes back the iterative-deepening loop breaks. With a game-long persistent
TT, depth 1 gets an immediate mate-scored TT hit from the previous move's deep search, so we
break at depth 1 and play instantly. Demonstrated directly:

```
we MATE in 2: d1 score=+1000000 move=a1a8  <== BREAK, plays this, 0 further thought
```

## Part 2 — Root cause #1: mate scores carry no distance (search.py)

`search.py:816` and `search.py:1019`:

```python
return -MATE if is_check(bb, meta) else 0
```

Flat `±1_000_000`. No `- ply`. Mate-in-1 and mate-in-9 score identically. Consequences, all
visible in the PGNs:

1. **Being mated, we walk into the fastest mate.** Mated-in-1 and mated-in-9 are the same
   number, so among losing moves the tie-break is move-ordering order — arbitrary. Every loss
   was a clean checkmate rather than a long resistance.
2. **Mating, we can't find the shortest mate.** R44 moves 58-63: `Qe7+ Kh6, Qh4+ Kg7, Qe7+ Kh6,
   b6 c4, Qh4+ Kg7, Qe7+ Kh6` — queen + rook + two passers against a bare king, shuffling. We
   nearly drew a completely won game.
3. **The TT gets poisoned.** `search.py:904-906` does `raw_score - ply` / `+ ply` on
   store/retrieve, an adjustment that only makes sense for distance-encoded mate scores. Fed a
   flat `±MATE` it writes values that are still above threshold but numerically meaningless, and
   those spread.
4. Combined with the depth-1 break, we stop thinking the instant a stale mate score surfaces
   anywhere.

**Fix:** return `-MATE + ply` at both sites, thread `ply` into `quiescence`, and delete
`abs(score) >= MATE_THRESHOLD` from the agent break condition — or gate it to
`score >= MATE_THRESHOLD` (winning) only, and even then keep iterating until the mate distance
stops shrinking. This is the cheapest high-value fix in the list.

## Part 3 — Root cause #2: the evaluation is miscalibrated (the big one)

Every eval term measured over all 634 positions from rounds 42-48:

```
term            mean|v|    stdev   p90|v|   max|v|  nonzero%
material+pst      342.4    496.8     1005     1565       99%
pawn_struct        44.3     70.6      148      259       76%
unstoppable        38.3    126.7        0      874        9%
threats            33.9     46.8       72      202       75%
pin_xray           33.4     57.0       93      174       55%
mobility           20.6     26.3       46       88       96%
piece_feat         10.9     14.9       30       44       69%
king_safety         5.6     16.5       17      127       20%   <-- effectively off
```

Non-material eval reaches ±277cp at p90 and ±1088 at max. A knight is 320. The positional noise
floor is nearly a piece, so the search will trade a piece for positional fluff all day.

### 3a. `pin_and_xray_score` is 25% of all non-material eval and it measures nothing

Take R42 after the sacrifice, once Black has consolidated:

```
rbb2rk1/pp3p2/3q1n2/3p2B1/8/2PQ3P/PP1N1PP1/R4RK1 w
  material+pst  -125     <- we are a knight down for two pawns
  pawn_struct    +41
  pin_xray       +86     <- ???
  ...
  SUM            -9      <- the engine thinks this is DEAD EQUAL
```

Attributing that +86: it is entirely `Qd3` looking down the d-file *through Black's own d5 pawn*
at the queen on d6. python-chess confirms there is no pin anywhere on the board. The arithmetic
is `evaluate.py:618-621`: `XRAY_FLAT_BONUS(6) + (900 - 100) // XRAY_HEAVY_DIVISOR(10)` = +86 for
a rook/queen pointing at a heavier piece behind a pawn, which is one of the most ordinary
configurations in chess and worth approximately zero.

The absolute-pin branch is just as inflated: `PIECE_VALUE[pt] // 6` (`evaluate.py:614`) = +150
for pinning a queen, +83 for a rook, with no check that the pinned piece is actually attackable
or winnable, no defender count, and no cap. Queens get scored twice (bishop rays *and* rook
rays, `evaluate.py:698-707`), so they accumulate several of these.

There is also a latent correctness bug: `revealed` can contain squares from several rays at
once, and `revealed_sq = _bit_scan(revealed)` takes the lowest one, pairing the wrong blocker
with the wrong revealed piece.

**Fix (highest value in the file):** raise `XRAY_HEAVY_DIVISOR` to ~40 and `XRAY_FLAT_BONUS` to
~2, raise `PIN_KING_DIVISOR` to ~16, cap the term at ~±40 total, and require the pinned/x-rayed
piece to be attacked more times than it is defended before paying anything. Gating the whole
term to zero and re-measuring would be a legitimate first experiment — expect it to gain Elo on
its own.

### 3b. King safety is switched off — which is why we get mated

Same table: `king_safety` fires in 20% of positions with mean 5.6cp. Direct probes:

```
White Q+R+B+N all bearing on a bare black king  -> king_safety = +31
Black king stripped, White Q+R on the 7th       -> king_safety = +30
```

The meaningless x-ray term (+62 in the first position) outweighs a full mating attack by 2x.

Two causes:
- `evaluate.py:736`: `zone = KING_ATTACKS[king_square(enemy)]` — the 8-square ring only. No
  forward extension, no king square itself. Real engines use a 3x3 plus the two/three squares in
  front. So `attacker_count` rarely reaches 2.
- `evaluate.py:215`: `KING_ATTACK_COUNT_PERCENT = [0, 0, 50, ...]` — with a ring that small, 0
  and 1 attackers are the common case, and both multiply the whole thing by zero.

The head commit `a484b46` ("superlinear king-safety attacker scaling") made this worse, not
better: it took a term that was already too small and multiplied most of its instances by 0.
Rounds 46 and 48 are the two games dated 2026-09-07 and are the most likely to have played that
build; they are also the two most one-sided losses.

The consequence in the games is stark. Here is what the eval thinks about our king in R48 at
move 27, with Black about to play g4/f5-f4-f3 into our king:

```
Kg1: eval -90   king_safety +0
Kf1: eval -113  king_safety +0
Ke2: eval -119  king_safety +0
Kd1: eval -132  king_safety +0
```

`king_safety` is zero for every square. The only thing distinguishing them is the king PST,
worth 8-24cp — while `pin_xray` swings ±93 elsewhere on the board. So we played `27.Kf1 …
29.Ke1 … 31.Ke2 … 35.Kd1` and marched the king to the queenside while Black built the mating
net. Same story in R45: `Kg2 → Kg1 → Kf2 → Kh3`, mated on b1.

**Fix:** widen the zone to 3x3 + the three squares in front, change the percent table to
something like `[0, 20, 55, 78, 90, 96, 99, 100, 100]`, and add the two terms this eval has no
concept of at all — pawn-shelter/storm in front of our own king and open/semi-open files
bearing on the king. Those are what actually stop `f5-f4-f3`.

### 3c. `unstoppable_passed_pawn_score` is a 874cp landmine

`UNSTOPPABLE_PASSER_BONUS = 500`, max observed 874, fires in 9% of positions with stdev 127. A
wrong verdict here is instantly game-losing, and it also has a perverse effect: once it declares
a pawn unstoppable the engine stops trying to stop it. R46 is exactly this — 25...Nxa4 grabbing
a pawn while `c5-c6-c7-b8=Q` ran, promoting on move 31. Given how new this term is and how large
it is, verify it against a test suite of rule-of-the-square positions before trusting it, or
reduce it to ~250 pending that.

### 3d. Everything else is too loud too

`pawn_struct` p90 = 148 / max 259, `threats` p90 = 72 / max 202. Both are 2-3x what they should
be. The commit log says Texel tuning was "attempted, not shipped" (`32ad9e9`) — that is the
missing piece and it is the single highest-expected-value item on this list. Ship the tuner
against a real labelled set and let it set these constants; hand-picked weights of this
magnitude are what produce a ±277cp positional noise floor.

## Part 4 — Root cause #3: time is spent in the wrong places

Totals are not the problem — we spend as much as or more than opponents:

```
R42 us 93.6s / opp 78.5s   R43 144.7 / 130.1   R44 136.0 / 129.5
R45 128.1 / 130.1          R46 123.3 / 101.6   R48 139.8 / 112.4
```

The distribution is the problem.

**The volatility extension fires on almost everything.** `agent.py:224` marks a position
volatile if `piece_count < prev_piece_count` — i.e. any capture just happened, which is usually
a forced recapture requiring no thought at all:

```
R42: 16/36 of our moves (44%) get the 2.5x budget  [in-check 8, capture 11, low-material 0]
R45: 26/51 (50%)                                   [in-check 9, capture 17, low-material  9]
R46: 28/40 (70%)                                   [in-check 7, capture 15, low-material 16]
R48: 25/59 (42%)                                   [in-check 12, capture 15, low-material 10]
```

So we burn 10.1s on `R42 8.Nf3` (a quiet developing move, extended because a knight was just
traded) and 9.5s on `R45 13.f3` — and then have only 1.43s for `R42 26.Rxe4` and 1.37s for
`R48 22.bxc4`, the two moves that lost those games. A cold 20s search on the R42 position never
picks `Rxe4` at any depth (it prefers `a5`, scoring the played move's line at −338).

**Fixes:**
- Drop `piece_count < prev_piece_count` as a volatility trigger, or restrict it to "the capture
  was not a recapture we can answer in kind."
- Add the trigger that actually matters and is missing: fail-low at the root — if the score
  drops below the previous iteration's by a margin, extend. That is the classic signal and it
  fires exactly when we are about to blunder.
- `ASSUMED_MOVES_LEFT = 30` against games that ran 36-72 of our moves back-loads too little;
  consider a phase-aware curve that spends more in moves 15-35 where these games were decided.
- `BRANCHING_ESTIMATE = 2.5` cut R42 move 26 to 1.43s of a 2.67s budget. With mate-distance
  fixed and a fail-low extension in place, re-measure whether it is still paying for itself.

## Part 5 — Per-game, where it actually went wrong

- **R42** — `13.Nxg5??` (knight for two pawns, no attack). This reproduces deterministically and
  the engine plays it at every depth 4 through 13, scoring it −22 to −64, i.e. "roughly equal".
  Pure eval failure; more depth does not help. Then `26.Rxe4?` (exchange sac) on 1.43s of
  thought.
- **R43** — no single blunder, a bleed: `11...Qb5?` drops c7, `16...Rd8?` drops e4, `19.Nxe6`
  wins the bishop pair. −3.25 by move 20, then 30 moves of shuffling (`Ne6-Ng5-Ne6`,
  `Nd5-Nf6-Nd4-Ne6`) with no plan. The shuffling is the flat-eval symptom: no move improves
  anything measurable, so it treads water.
- **R45** — we were +2.00 at move 22 and lost. `24.a5?` drops a pawn, `31.Kf2?` walks into the
  centre (the engine's own d9-d10 search prefers `Ra2`/`Qc6`), `32.Qxc4??` grabs a pawn and
  drops the d4 knight to `Qxe3+`/`Rxd4`. Greedy + king-safety-blind.
- **R46** — `23...Bxc5?` chosen at every depth 1-12 while its own score craters +160 → −222; the
  position was already lost by then. The real loss is `25...Nxa4` letting the c-pawn queen on
  move 31.
- **R48** — `21.b3??` (wins a knight, loses a rook on b1, net −2), then the king march
  `Kf1-Ke1-Ke2-Kd1` while Black played `g4, f5, f4, f3`. The engine's own deep search never
  picks `27.Kf1` at any depth — it wants `Bd2`, `Ne2`, `g4`, `Rd1`, `Ba3`. That one is a genuine
  in-game miss, consistent with a warm-TT/low-time interaction.

## Part 6 — What to change, in order

| # | Change | Where | Cost | Expected |
|---|---|---|---|---|
| 1 | Mate scores get ply distance (`-MATE + ply`); remove/narrow the `MATE_THRESHOLD` break | `search.py:816`, `search.py:1019`, `agent.py:188` | ~10 lines | Stops losing faster than necessary; stops nearly drawing won games |
| 2 | Defang `pin_and_xray_score` (divisors up, cap ±40, require attackers>defenders) — or zero it and measure | `evaluate.py:583-709` | small | Removes the largest source of the ±277cp noise floor that funds bad sacs |
| 3 | Make king safety real: 3x3+front zone, retune percent table, add pawn shelter/storm and open-files-to-king | `evaluate.py:713-769` | medium | Directly targets 5 mate losses and the king marches |
| 4 | Ship Texel tuning; let it set `pawn_struct`, `threats`, king-safety and pin/xray weights | `tools/texel_tune.py` (`32ad9e9`) | large | The systematic fix for #2 and #3; biggest single Elo item |
| 5 | Fix volatility triggers: drop bare "a capture happened", add root fail-low extension | `agent.py:198-226` | small | Moves seconds from forced recaptures to the moves that decide games |
| 6 | Verify or halve `UNSTOPPABLE_PASSER_BONUS` against a rule-of-the-square suite | `evaluate.py:229`, `evaluate.py:404` | small | Removes an 874cp landmine |
| 7 | Fix the `revealed_sq = _bit_scan(revealed)` mis-pairing | `evaluate.py:616`, `evaluate.py:653` | tiny | Correctness; moot if #2 zeroes the term |

Items 1 and 2 are a couple of hours and should be worth more than everything shipped since
`2a2e381`. Item 3 is the one that addresses "we get checkmated in every loss."

**Methodology caveat:** replaying the games through `agent.get_move` on this dev box only
reproduced 26/36 (R42) and 34/51 (R45) of the actual moves played, so this box is not
clock-identical to the EPYC 9V74 the platform runs on. Validate changes with `make arena`
head-to-head against the current build, not with single-position spot checks.

## Part 7 — Status and next step

Items 1, 2, 3, 5, 6, and 7 are shipped (`docs/STATUS.md`'s Tier 19 section has the full writeup per
item). Verified against the full test suite plus a new `tests/test_unstoppable_passer.py` for item
6, and against a real 8-game 120s+0.5s head-to-head vs the pre-this-pass build (`a484b46`, the exact
commit this file's own analysis was run against): **+3 =4 -1, 62.5%**, no crashes or illegal moves,
4 checkmates and 4 threefold-repetition draws.

**Item 4 (Texel tuning) is the one open item, deliberately not attempted this pass.** Not an
oversight -- it had already been tried once (`32ad9e9`, before this file was written) and abandoned:
453 independently-tuned parameters overfit noisy self-play win/loss/draw labels, confirmed by a real
head-to-head where the untuned build led 3-2-0 through 5 of 8 games before the run was stopped.
Re-running the existing tuner as-is would very likely reproduce that exact failure, not fix items 2
and 3's remaining hand-picked constants (which #2 and #3 above have now made considerably less wrong
by hand, so the marginal value of tuning them has also dropped some).

**Next step, if this is picked back up:** don't re-run `tools/texel_tune.py` unchanged. The
overfitting diagnosis from the first attempt names the fix directly:
- **Fewer tunable parameters.** 453 (full PST tables per piece plus every positional constant) is
  too many for the label volume available. Start with the scalar constants this file already
  touched by hand (`XRAY_HEAVY_DIVISOR`/`PIN_KING_DIVISOR`/`PIN_XRAY_CAP`, `KING_ATTACK_COUNT_PERCENT`,
  `KING_OPEN_FILE_BONUS`/`KING_SEMI_OPEN_FILE_BONUS`, `UNSTOPPABLE_PASSER_BONUS`, `PAWN_THREAT_BONUS`,
  `DOUBLED_PAWN_PENALTY`/`ISOLATED_PAWN_PENALTY`) -- a few dozen values, not hundreds, and each one
  already has a hand-reasoned direction from this file to sanity-check the tuner's output against.
- **Filter to quiet positions.** Label noise from tactical positions (where the eval score and the
  eventual game result diverge for reasons that have nothing to do with the static evaluation being
  tuned) is a standard, well-documented Texel-tuning failure mode; `tools/texel_gen_data.py` does not
  currently filter for this.
- **A held-out validation split.** The first attempt had no way to detect overfitting except a full
  real head-to-head after the fact -- expensive and slow to iterate on. A validation set (positions
  not used for tuning) that the tuner reports loss against on every iteration would catch it early.
- **Re-verify with `make arena` / `tools/head_to_head.py` before trusting any result**, exactly as the
  first attempt's own `verify_parity()` check and the real arena test caught problems a purely
  numerical convergence check would have missed.

Separately: once this pass's fixes have played real rated games, re-run this same PGN-driven
analysis against the new losses (if any) the way this file itself was produced -- the six games
analysed here are now a stale sample once rounds 49+ start reflecting the mate-distance, pin/xray,
and king-safety changes.
