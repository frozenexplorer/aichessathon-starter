# Status

Handoff snapshot as of the Tier 9 pass (2026-09-05), on top of Tiers 1-8 (Tier 2 commit `02ec418`,
Tier 1 commit `12a38f9`), all documented in their own sections below. Read this instead of
replaying the whole build history.
Competition context: uploads close 2026-09-11 11:00 London; the rated ladder runs hourly
08:00-22:00; Daily Five runs 2026-09-06 through 2026-09-10.

Init time is no longer tracked here as a gating concern: this dev machine's numbers (which drift
into the 55-70s range depending on what's landed in `search.py`, and have caused real init
failures in local `harness.play`/`harness.arena` runs) are not assumed to reflect the actual
competition hardware, on the user's explicit direction. Each Tier 3+ item below was still verified
with the full correctness gate (`ruff`, `mypy --strict`, `tests/perft.py`,
`tests/test_repetition.py`, `tests/test_see.py`, `tests/test_magic_attacks.py`, from Tier 5 onward
`tests/test_threats.py`, and from Tier 6 onward `tests/test_quiescence_check.py`) plus functional
games (checkmates from the start position and the KBN-vs-K tablebase endgame, both colours) before
moving to the next.

## Architecture

`agent.py` does not use python-chess at runtime for movegen or search -- the numba baseline in
this repo shows that jitting only the evaluation barely helps when the tree walk is still pure
Python, so the whole hot path was rebuilt on a custom bitboard representation instead:

```
bitboard.py     position as (bb: uint64[12], meta: int8[6]); FEN parsing via python-chess
attacks.py      precomputed knight/king/pawn tables; sliding pieces (bishop/rook/queen) via
                magic bitboards -- a per-square lookup table indexed by
                ((occupied & mask) * magic) >> shift, magic numbers precomputed offline and
                hardcoded (see the module docstring), no runtime magic search
movegen.py      pseudo-legal + legal generation, copy-make (not incremental make/unmove)
evaluate.py     tapered material + PST (midgame/endgame king blend by game phase), pawn
                structure (doubled/isolated/passed, the passed-pawn bonus itself phase-blended),
                bishop pair, rook open/semi-open files, king pawn-shield, differential piece
                mobility, king proximity to passed-pawn promotion squares, tactical threat
                awareness (hanging pieces, pawn threats, forks, absolute pins, x-rays/skewers),
                king safety (attacker-weighted pressure on each king's own ring) -- all jitted
zobrist.py      position hashing, for repetition detection and the transposition table
search.py       negamax/alpha-beta, iterative deepening, a two-tier transposition table,
                killer-move + history (with malus) + counter-move ordering, principal variation
                search (PVS) with aspiration windows, null-move pruning, late move reductions and
                pruning, check extensions, futility and reverse-futility/static-null pruning,
                internal iterative deepening, static exchange evaluation (move ordering +
                quiescence SEE/delta pruning), quiescence (including a bounded full-width search
                of check evasions, not just captures, when the side to move is in check),
                repetition-draw scoring, panic-mode fallback
tablebase.py    Syzygy WDL + DTZ, root-only -- WDL filters to won/drawn-safe moves, DTZ narrows
                further to the ones that actually make progress
timeman.py      per-move budget (adaptive: a volatile position -- score swinging between
                iterations, in check, a capture just played, or few pieces left -- can spend up
                to timeman.EXTENSION_FACTOR times the base budget) + a hard low-clock panic
                threshold
tests/perft.py               movegen verified against python-chess: differential testing over
                              random games (6 position types incl. Kiwipete) + perft node counts
tests/test_repetition.py     direct unit tests of the repetition-draw and claim-eligibility logic
tests/test_see.py            static exchange evaluation against hand-computed winning/losing/
                              equal/x-ray exchanges
tests/test_magic_attacks.py  magic-bitboard lookups differentially tested against the plain
                              ray-cast reference over random occupancies, all 64 squares
tests/test_threats.py        fork/hanging-piece/pin/x-ray eval terms against hand-constructed
                              positions, isolated from the rest of evaluate() by calling
                              threats_score/pin_and_xray_score directly
tests/test_quiescence_check.py  quiescence's in-check evasion search against an old-vs-fixed
                              check_budget comparison, plus a genuine checkmate control case
tests/test_king_safety.py    king_safety_score against a hand-built king-ring-pressure position,
                              its sign-mirrored counterpart, a phase-0 fade-out, and a quiet control
```

Everything under `weights/syzygy/` and every `.py` file above ships in the zip (`make zip`);
`harness/` does not (platform-side, mirrors the real protocol, never edited).
`baselines/pre_tier1/` is a frozen, standalone copy of the engine exactly as it stood before this
pass (tag `pre-tier1-baseline` in git) -- kept as a fixed comparison point, not part of the
submission.

## Tier 1: what changed and why

Five additions, aimed at strength-per-day-of-work with the time remaining (see the conversation
that scoped this: transposition table + move ordering first, since that's the highest-confidence
gain per hour; DTZ next, since it closes a known bug; eval last, since it's free size-wise):

1. **Transposition table** (`search.py`) -- fixed-size, always-replace, parallel-array hash table
   (numba nopython mode has no fast dict), ~20MB, persists across the whole game. Caches a
   mate-distance-adjusted score, a bound type, and the best move found, so a re-visited node can
   short-circuit or at minimum reuse the stored move as the first one tried. Deliberately never
   consulted for the repetition-forced draw score -- that check runs first and returns before the
   table is touched, so a cached score can never mask a real repeat, and the root-level
   claim-eligibility safety net (see below) never reads it either.
2. **Killer moves + history heuristic** -- two killer moves per ply plus a from/to history table,
   both rebuilt fresh per real move decision, layered on top of the existing hash-move-first /
   MVV-LVA ordering for quiet moves that caused cutoffs elsewhere in the tree.
3. **Principal variation search (PVS)** -- the first move at each node gets a full-window search;
   every other move gets a cheap null-window probe first, re-searched with the full window only
   if it looks like it might beat alpha. Applied at both `search_root` and `negamax`.
4. **DTZ tablebase** -- `weights/syzygy/*.rtbz`, same 3-4-man scope as the WDL set already shipped
   (35 more files, ~3.1MB). Fixes the KBN-vs-K conversion gap from the previous pass: WDL alone
   only says a position is won, not which move makes progress, so the engine could drift between
   different won-but-not-shrinking lines forever. DTZ narrows WDL-tied candidates to the smallest
   DTZ magnitude while winning (fastest forced progress) or the largest while lost (best
   resistance). Verified directly: replaying the exact `8/8/8/4k3/8/3BN3/8/4K3 w - - 0 1` position
   against `baselines/random`, which used to draw ~5/5 by repetition, now converts 3/3.
5. **Richer, tapered evaluation** (`evaluate.py`) -- game-phase-blended king PST (safety early,
   centralization late), doubled/isolated/passed pawns, bishop pair, rook on open/semi-open
   files, a king pawn-shield term that fades out with the same phase blend as the king PST. All
   in the same white-minus-black, precomputed-mask style as the rest of the module.

Not done (considered, explicitly deferred, see the original Tier list from planning): null-move
pruning, aspiration windows, static exchange evaluation, an opening book, late move reductions,
magic bitboards, search extensions, a curated 5-man WDL subset, NN evaluation.

## Tier 2: what changed and why

Everything from `docs/FUTURE.md`'s prioritized list that Tier 1 explicitly deferred, done in that
same priority order, each checked against `ruff`, `mypy --strict`, `tests/perft.py`,
`tests/test_repetition.py`, and a functional game/engine check before moving to the next:

1. **Adaptive time management** (`timeman.py`, `agent.py`) -- the flat `time_left/30 + increment`
   formula is now the *quiet-position* budget; a volatile position (score swinging >= 60cp between
   the last two completed iterative-deepening depths, in check, a capture just landed us here, or
   few enough pieces left to matter -- see `agent._is_volatile`) can spend up to
   `timeman.EXTENSION_FACTOR` times that, capped well short of the whole clock
   (`timeman.MAX_EXTENSION_FRACTION`) so a run of sharp positions in a row cannot flag us.
2. **Aspiration windows** (`search.py`) -- `search_root`'s first pass at depth > 2 centers a narrow
   window on the previous depth's score instead of always searching -INF..INF, re-searching with
   the full window on a fail-high/fail-low (`_search_root_pass`, `NO_PREV_SCORE`).
3. **Null-move pruning** (`search.py`) -- give the side to move a free pass
   (`movegen.make_null_move`) and search the rest at `depth - 1 - NULL_MOVE_REDUCTION`; a cutoff
   there prunes the whole subtree. Two zugzwang guards: skipped in check, and skipped when the
   side to move has no non-pawn material (`movegen.has_non_pawn_material`) -- exactly the
   king+pawn endgames zugzwang matters in, the same class DTZ was added to get right. `allow_null`
   forbids two null moves in a row.
4. **Static exchange evaluation** (`search.py`, `see()`) -- a real least-valuable-attacker-first
   swap-list calculation replaces MVV-LVA's rough heuristic for capture ordering in `_move_score`,
   and quiescence now skips searching a capture outright once its SEE turns negative rather than
   exploring every capture regardless of whether it's obviously bad. Directly tested against known
   winning/losing/equal/x-ray exchanges (`tests/test_see.py`) -- the x-ray case specifically
   exercises that sliding attacks are recomputed against live occupancy each step of the exchange,
   not tracked incrementally, so discovered attackers fall out for free.
5. **Late move reductions** (`search.py`) -- a quiet, non-capture, non-check move late in the
   ordering (`LMR_MIN_MOVE_INDEX`) gets searched one ply shallower first; only promotes to a
   full-depth re-search if the reduced probe still beats alpha, then falls into the same
   full-window PVS re-search as any other move that beats alpha. Root moves are never reduced.
6. **Search extensions** (`search.py`) -- a move that gives check gets its child searched at the
   same depth instead of `depth - 1`. `ext_budget` (`MAX_CHECK_EXTENSIONS`) bounds how many of
   these one line can stack, so a long forcing check sequence cannot inflate one line's effective
   depth at every sibling line's expense within the same time budget.
7. **Magic bitboards** (`attacks.py`) -- bishop/rook/queen attacks are now a per-square lookup
   table (`((occupied & mask) * magic) >> shift`) instead of ray-casting one step at a time. Magic
   numbers for all 64 squares (rook and bishop) were found offline by random search, verified
   against the ray-cast reference for every occupancy subset of each square's relevant-occupancy
   mask, and hardcoded -- nothing at runtime searches for a magic. The ray-cast functions are kept
   (renamed `_rook_ray_attacks` / `_bishop_ray_attacks`) purely to build the lookup tables at
   import and for a differential test (`tests/test_magic_attacks.py`, 25,600 random-occupancy
   queries across all 64 squares, both piece types) -- the correctness rigor this item's own
   FUTURE.md entry called for before trusting a rewrite of the movegen core's attack generation.
   As a side benefit, the much simpler compiled lookup functions measurably *reduced* init time
   (see below) rather than adding to it, unlike every other Tier 2 item.

Not done, each for a reason specific to it rather than time pressure: a **curated 5-man Syzygy
subset** (item 8) is blocked on file acquisition -- this environment has no outbound network
access to fetch the actual `.rtbw`/`.rtbz` files (the existing 3-4-man set's own source,
`tablebase.sesse.net/syzygy/3-4-5/`, is unreachable from here); the code-side change
(`tablebase.MAX_PIECES = 5`) is a one-line follow-up once files are supplied. An **opening book**
(item 9) fails its own stated precondition -- no curated starting positions used by the platform
are known or obtainable, and FUTURE.md itself calls a book keyed on the standard start position
likely-wasted effort without them. **NN evaluation** (item 10) was explicitly declined: a
fundamentally different, multi-day scope (data/self-play, training, ONNX export, integration) that
FUTURE.md itself gates on "everything above done with real days to spare."

## Tier 3: what changed and why

A second round beyond `docs/FUTURE.md`'s original list, proposed as further optimizations once
that list was exhausted and picked up all at once on request. Same discipline as Tier 2: each
checked against the full correctness gate plus functional games before the next.

1. **Mobility eval term** (`evaluate.py`) -- replaced the previous crude proxy (the side to move's
   total legal move count, added as a flat bonus in `evaluate()`'s old signature) with a real
   differential term: for each knight/bishop/rook/queen, the squares it attacks that aren't held
   by an own piece, white minus black, one flat weight per square (`mobility_score()`). Affordable
   per node now that sliding attacks are magic-bitboard lookups rather than ray-casts (Tier 2 item
   7) -- the previous proxy was partly a workaround for how expensive a real per-piece mobility
   scan would have been before that. `evaluate()` dropped its `mobility` parameter entirely, only
   ever called from quiescence's stand-pat.
2. **Futility pruning** (`search.py`) -- at a shallow (`FUTILITY_MAX_DEPTH`), non-check node, a
   quiet, non-check move is skipped outright once the static eval plus a depth-scaled margin
   (`FUTILITY_MARGIN`) still can't reach alpha. `oi == 0` (the hash/best-ordered move) is never
   skipped, so the node always has at least one fully-searched move to report a score from.
3. **Bigger, two-tier transposition table** (`search.py`) -- grown from 2^20 always-replace entries
   (~20MB) to 2^21 buckets, two slots each (~80MB): a depth-preferred slot (kept unless a same-key
   refresh or a search that went at least as deep wants it) plus an always-replace slot, so a deep
   result isn't evicted by shallow, plentiful ones, while a node from the current search is never
   simply dropped for lack of a slot. `_tt_resolve` factors the shared probe-and-cutoff logic
   (mate-distance adjustment, bound comparison) out to one place, checked against both slots.
4. **Internal iterative deepening** (`search.py`) -- a node deep enough (`IID_MIN_DEPTH`) with a
   genuine TT miss (no hash move at all, not merely one too shallow for a cutoff) searches itself
   at a reduced depth purely to seed move ordering, implemented by re-invoking `negamax` on the
   exact same position and re-probing the TT for whatever move its own store just wrote -- no
   second return value threaded through `negamax` just for this.
5. **Counter-move heuristic** (`search.py`) -- a table indexed by the move that led to a node
   (`parent_from`/`parent_to`, now threaded through `negamax` and every recursive call site)
   records whatever quiet move most recently caused a cutoff in reply to it, giving move ordering
   a more targeted signal than the existing from/to history table alone. Ranked between killers and
   history in `_score_moves2`. Rebuilt fresh per real move decision, like killers and history.
6. **Delta pruning** (`search.py`, quiescence) -- complements SEE-pruning (Tier 2 item 4): a
   capture whose SEE, added to stand-pat, still can't reach alpha within `DELTA_MARGIN` is skipped,
   catching the case SEE-pruning alone misses (a real but too-small material gain against a large
   existing deficit). Same SEE-descending order makes both conditions one monotonic `break`.

Threading `parent_from`/`parent_to` and the counter-move table through `negamax`,
`_search_root_pass`, `search_root`, `agent._search_restricted`, and the `_warm_up` and
`test_repetition.py` call sites (item 5) was the widest-blast-radius change of this pass --
verified specifically by running `tests/test_repetition.py` (calls `negamax` directly) before the
broader suite, since a wrong argument count there would surface immediately as a `TypeError`
rather than silently.

## Tier 4: what changed and why

One further item, picked up from the "further ideas raised but not yet picked up" note at the
bottom of this doc (the other, a locally-generated retrograde-analysis endgame tablebase, stays
unpicked -- multi-day scope not worth it against the time remaining):

1. **Endgame-specific pawn/king eval terms** (`evaluate.py`) -- two additions, both phase-blended
   the same way as the king PST (`material_and_pst`) so they fade in as material comes off rather
   than applying at full strength in the middlegame where they don't belong:
   - The existing rank-scaled passed-pawn bonus (`PASSED_PAWN_BONUS`, now split into
     `_MID`/`_END` arrays) is blended by phase instead of applied flatly -- a passed pawn on rank
     6 is worth far more with two rooks and a queen off the board than at move 10.
   - A new king-proximity term (`passed_pawn_king_distance`): for each passed pawn, Chebyshev
     distance from each king to that pawn's promotion square, rewarded when the friendly king is
     closer and penalised when the enemy king is (the "rule of the square" -- whichever king gets
     there first tends to decide whether the pawn queens or gets caught). Zero-cost in the
     middlegame (`endgame_weight == 0` short-circuits before the board scan).

## Tier 5: what changed and why

User-reported: the engine was still missing middlegame tactics -- forks, pins, x-rays/skewers --
that a shallow, depth-limited search cannot see until it searches far enough ahead to reach the
actual capture, even though search already extends check sequences and orders captures by SEE.
Two new eval terms give the *static* evaluation, at every leaf node, awareness of these patterns
one ply after the threatening move is made, not just once the capture itself is on the board:

1. **`threats_score`** (`evaluate.py`) -- attacker-centric (loop over our own pieces, look at what
   each attacks, not the reverse): a bonus for a pawn attacking a more valuable piece, a bonus for
   any piece attacking an enemy piece with no defender (`movegen.attacked_by`), and a fork bonus
   when one piece attacks two or more enemy pieces at once -- the enemy king counts toward the
   fork count (attacking it is still only one of the two-plus pieces the opponent must choose
   between saving) but is excluded from the hanging-piece bonus itself, since "check" is already
   search's job, not eval's.
2. **`pin_and_xray_score`** (`evaluate.py`) -- for each of our sliding pieces, the nearest enemy
   piece it attacks under full occupancy is a pin/x-ray candidate; recomputing the same ray with
   just that candidate removed reveals whatever sits directly behind it (if anything) -- the same
   "discovered attacker falls out for free" trick `search.see` already uses for the same reason,
   rather than tracking rays incrementally. An enemy king revealed behind the candidate is an
   absolute pin (scaled by the pinned piece's own value, since a more valuable pinned piece is a
   bigger constraint); another enemy piece revealed is a skewer/x-ray, scaled up further when the
   revealed piece is worth more than the candidate in front of it.

Caught one real implementation bug during its own verification, not in gameplay: the first version
of `_ray_pin_score` intersected the recomputed ray against *all* remaining pieces on the board
rather than only the newly-revealed squares beyond the candidate, so an unrelated blocker on a
different ray direction (e.g. our own king sitting on the same rank) could be picked up by
`_bit_scan` instead of the real piece behind the candidate -- caught by `tests/test_threats.py`'s
skewer case scoring 0 when hand-computation said it should be positive; fixed by intersecting
against `extended & ~full` (squares reachable only after removing the candidate) before looking
for a piece there.

Both terms are cheap relative to the rest of `evaluate.py`: `threats_score` iterates each piece's
own bitboard directly (bit-scan-and-clear, not a 64-square scan) and reuses the same attack-table
lookups mobility scoring already does; `pin_and_xray_score` only does extra ray recomputation for
the (typically 0-2) enemy pieces a slider directly touches, not every square on the board. A fixed
8-second search on the exact blunder-position FEN from Tier 1's investigation (`docs/STATUS.md`'s
own earlier note) still reached depth 6 complete / depth 7 partial (133K nodes) after adding both
terms, confirming no meaningful throughput collapse from the extra per-node work.

New dedicated test: `tests/test_threats.py`, same style as `tests/test_see.py` -- five
hand-constructed positions (a king+queen fork with an undefended queen, a defended-attacker/
undefended-target case built so a naive symmetric read would net to zero, an absolute pin, an
x-ray/skewer, and a bare-kings control case expecting exactly zero) calling `threats_score` and
`pin_and_xray_score` directly rather than through the full `evaluate()`, so each case isolates the
mechanism under test from PST/mobility/pawn-structure noise.

## Tier 6: what changed and why

A real search bug, found while investigating why middlegame tactics were still being missed after
Tier 5's eval additions -- not a new feature, a fix:

1. **Quiescence did not handle being in check** (`search.py`) -- `quiescence()` always computed a
   static `stand_pat` and only ever searched captures (plus queen promotions), with no special
   case for the side to move being in check. But there is no "decline to respond" option while in
   check, and a legal evasion that is not itself a capture (a king step, a block) was never even
   looked at -- if the only escapes from check were non-captures, `ncap == 0` and the function
   returned the parent node's own stand-pat unconditionally, treating a position where the side to
   move might be getting mated as if it were quiet. Captures routinely give check, so this
   triggered on any quiescence-depth capture sequence that landed on a checking position, which is
   common -- very plausibly a bigger contributor to "missed tactics" than any single eval term.
   Fixed: when in check, `quiescence` now skips the stand-pat/beta-cutoff shortcut and searches
   every legal move (the movegen already restricts to legal evasions when in check), the same
   posture as a normal negamax node. Bounded by a new `check_budget` parameter (mirroring
   negamax's own `ext_budget`/`MAX_CHECK_EXTENSIONS`, new constant `QSEARCH_CHECK_BUDGET = 6`)
   rather than `qdepth`, since quiescence carries no history array and so cannot detect a
   perpetual-check line via repetition the way negamax can -- once the budget is exhausted, a
   still-in-check node falls back to the old stand-pat/captures-only path so recursion still
   terminates.

New dedicated test: `tests/test_quiescence_check.py` -- calls `quiescence` on the same in-check,
zero-legal-captures position with `check_budget=0` (reproducing the old behaviour exactly, since
the budget that would trigger the new evasion search is already spent) versus the real
`QSEARCH_CHECK_BUDGET`, and asserts the two disagree (proving the fixed path actually searches the
forced king move rather than reusing the parent's stand-pat), plus a genuine checkmate control case
(fool's mate) confirming the pre-existing `count == 0` handling is unaffected. A fixed 8s search on
the Tier 1 blunder-position FEN reached the same depth 6 complete / depth 7 partial as before the
fix (123K vs. 133K nodes, well within normal run-to-run variance) -- no throughput regression.

## Tier 7: what changed and why

1. **History malus** (`search.py`) -- at a beta cutoff, the from/to history table previously only
   rewarded the cutoff move itself (`history_table[f*64+t] += depth*depth`). It now also penalises,
   by that same magnitude, every other quiet move already tried at that node before the cutoff
   (they had the same chance to cut off and didn't) -- not merely withheld a bonus, but pushed
   below an untried, `history == 0` move the next time `_score_moves2` consults the table. Doubles
   the resolution of the heuristic (a move's score now reflects both how often it cuts off and how
   often it was tried and failed to), which matters most in exactly the sharp, many-candidate
   tactical positions this and Tier 5/6 exist for. One extra pass over the moves already tried
   before the cutoff, bounded by the same move list the node already generated -- no new movegen.

No new dedicated test: this changes move-ordering quality only, never correctness (a cutoff still
requires the real search to prove alpha >= beta; history only affects which order moves are tried
in), so it is covered by the existing full gate plus functional games rather than a new unit test.
A fixed 8s search on the Tier 1 blunder-position FEN reached the same depth 6 complete / depth 7
partial as Tier 6 (127K nodes, consistent with normal run-to-run variance) -- no regression.

## Tier 8: what changed and why

1. **King safety eval term** (`evaluate.py`, `king_safety_score`) -- the only existing king-safety
   signal was the pawn-shield bonus (a king's own cover), with nothing valuing how much of the
   enemy's actual reach lands on the squares around each king. Added: for every enemy non-king
   piece, how many squares of the king's own ring (`KING_ATTACKS[king_sq]`, its up-to-eight
   neighbours) it currently attacks, weighted per piece type (`KING_ZONE_ATTACK_WEIGHT` -- a queen
   in the ring is far more dangerous than a knight), phase-blended like the king PST so it carries
   full weight with material on the board to attack with and fades to exactly zero at phase 0 (an
   exposed king in a bare endgame is an asset, not a liability -- the king PST's own blend already
   covers that side of it). Deliberately linear in attacker count rather than a nonlinear
   safety-table lookup (as e.g. Stockfish's classical eval used) to keep the addition small and
   low-risk; a nonlinear table is a plausible later refinement if there's time to tune it properly.

New dedicated test: `tests/test_king_safety.py` -- a hand-built position with a queen and rook
bearing down on one king's ring (asserted positive, and exactly zero once `phase` is forced to 0),
its file-mirrored counterpart onto the other king (asserted to be the exact negation, checking the
sign convention both ways), and a bare-kings control case (asserted exactly zero).

## Tier 9: what changed and why

Two more pruning layers in `search.py`, on the same "ordering/eval is already trustworthy enough
to act on without a full search" premise as futility pruning and LMR:

1. **Reverse futility / static null-move pruning** -- at a shallow node (`depth <= RFP_MAX_DEPTH`)
   not in check, if the static eval already clears beta by more than a depth-scaled margin
   (`RFP_MARGIN`), the whole node returns that eval outright, before move generation -- the
   opponent's best defense is assumed unable to drag a position this good back down to beta.
   Different target from the existing futility pruning below it: this prunes the entire node,
   futility prunes individual moves inside a node already being searched. Shares its static-eval
   computation with futility pruning (`STATIC_EVAL_MAX_DEPTH` gates a single `evaluate()` call
   reused by both, computed at most once per node, replacing what were two separate calls before).
2. **Late move pruning (LMP)** -- inside the move loop, a quiet, non-check move at a shallow node
   (`depth <= LMP_MAX_DEPTH`) whose position in the ordering has reached `LMP_THRESHOLD[depth]` is
   skipped outright rather than even given LMR's reduced-depth probe, on the premise that ordering
   has already put anything worth searching ahead of it. Threshold grows with depth, so a more
   expensive (and more trustworthy) node tolerates more late moves before pruning. `oi == 0` stays
   unreachable here, same guarantee futility pruning already gives -- a node always has at least
   one fully-searched move to report a real score from.

No new dedicated test: both are heuristic pruning (can, rarely, discard a real line -- the
accepted tradeoff every pruning technique in this engine already makes, from null-move pruning
onward), not correctness changes, so covered by the existing full gate plus functional games.
Substantial depth gain on the Tier 1 blunder-position FEN: an 8s search that previously reached
depth 6 complete / depth 7 partial (Tier 8: 182K nodes) now reaches **depth 8 complete / depth 9
partial** with far fewer nodes per depth (e.g. depth 7: 90K nodes vs. Tier 8's depth-7-partial
182K) -- the two prunes compounding with the existing futility/null-move/LMR stack rather than
duplicating their effect.

## What's implemented and verified

- `ruff` / `mypy --strict` clean. `tests/perft.py` (movegen, unaffected by Tier 1, differentially
  re-verified against the magic-bitboard rewrite in Tier 2) and `tests/test_repetition.py`
  (updated for the new `negamax`/`search_root` signatures across both tiers) both pass.
- Numba bounds-checking (`NUMBA_BOUNDSCHECK=1`) run across several games surfaced zero
  IndexErrors -- no memory-safety issue in the new TT/killer/history array indexing.
- **Head-to-head vs the pre-Tier1 baseline** (`baselines/pre_tier1`, 10 games, 20s+0.2s clock):
  **+9 =1 -0 (95%)**. No crashes, illegal moves, or init failures in that batch. The one draw was
  by threefold repetition, not a loss.
- Clean wins in spot checks against `random`/`greedy`/`minimax`/`numba` baselines, both colours.
- KBN vs K vs `baselines/random`: 3/3 checkmates post-DTZ (previously ~5/5 draws -- see Tier 1
  item 4 above); re-verified 2/2 after the full Tier 2 pass on top, including null-move pruning's
  zugzwang guard and the magic-bitboard movegen rewrite.
- Tier 2's own dedicated correctness tests: `tests/test_see.py` (5 hand-computed exchanges,
  including an x-ray case) and `tests/test_magic_attacks.py` (25,600 differential queries against
  the ray-cast reference, all 64 squares, both sliding piece types) both pass.
- Functional in-process games (bypassing the harness subprocess so init time doesn't gate the
  check) after every Tier 2 item: no crashes, no illegal moves, checkmates both colours from the
  standard start position throughout.
- `baselines/pre_tier2/` is a frozen, standalone copy of the engine as it stood at the end of
  Tier 2 (commit `02ec418`) -- created this session as a real, on-disk comparison point (unlike
  the `baselines/pre_tier1/` referenced above, which is not actually present in this checkout).
  Not part of the submission (`harness.package` only ships root-level `.py` files).
- Tier 3: same full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks tests) plus
  functional in-process games (checkmates both colours from the start position, and 2/2 KBN-vs-K
  tablebase-endgame checkmates exercising `agent._search_restricted`) after every item, all clean.
  The counter-move heuristic's signature threading (negamax, `_search_root_pass`, `search_root`,
  `_search_restricted`, `_warm_up`, and `test_repetition.py`) was verified by running
  `tests/test_repetition.py` first specifically, since it calls `negamax` directly and a wrong
  argument count there surfaces immediately as a `TypeError`.
- Tier 4: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks) clean, plus
  functional in-process games (checkmates both colours from the start position, 2/2 KBN-vs-K
  tablebase-endgame checkmates) after the eval change.
- Tier 5: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks) plus the new
  `tests/test_threats.py` (5 hand-constructed positions, including the skewer case that caught the
  ray-isolation bug described above) all clean, plus the same functional in-process games. A fixed
  8s search on the Tier 1 blunder-position FEN still reached depth 6 complete / depth 7 partial
  (133K nodes) with both new terms active, confirming no meaningful throughput regression.
- Tier 6: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats) plus the
  new `tests/test_quiescence_check.py` all clean, plus the same functional in-process games. Same
  8s-search throughput check as Tier 5, unaffected (depth 6 complete / depth 7 partial, 123K
  nodes).
- Tier 7: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats,
  quiescence-check) all clean, plus the same functional in-process games. Same 8s-search
  throughput check as Tier 6, unaffected (depth 6 complete / depth 7 partial, 127K nodes).
- Tier 8: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats,
  quiescence-check) plus the new `tests/test_king_safety.py` all clean, plus the same functional
  in-process games. Same 8s-search throughput check reached depth 6 complete faster than Tier 7
  (2.80s vs. 4.30s) and depth 7 partial with more nodes (182K) -- no regression, within normal
  run-to-run variance from eval-driven pruning decisions shifting slightly.
- Tier 9: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats,
  quiescence-check, king-safety) all clean, plus the same functional in-process games. Same
  8s-search throughput check reached depth 8 complete / depth 9 partial, a real jump from Tier 8's
  depth 6 complete / depth 7 partial, with far fewer nodes per depth (e.g. depth 7: 90K vs. Tier
  8's depth-7-partial 182K) -- the two new prunes compounding with the existing stack as intended.
- Investigated a user-reported "blundered a winning position" game at
  `r2qr2k/pp5p/8/3nPbp1/3B1P1b/1BPp4/PQ4P1/R5KR b - - 3 22` (75s on the clock). Not a bug: the
  ~2.89s budget that position gets (see the init-time and time-budget notes below) only reaches
  depth 4 complete / depth 5 partial, and at that depth `Nxf4` (the pre-Tier1 baseline's actual
  choice in this exact position) looks like a clean pawn grab (+350 to +420) but is objectively
  weak once you search deeper (+13 at depth 5, -25 at depth 7 in an unhurried re-check). The
  current engine's partial depth-5 search, via PVS/TT/killer ordering, had already moved off
  `Nxf4` onto `Bc8` before time ran out, which holds up much better deeper (+221 at depth 5, +172
  at depth 7). Direct evidence Tier 1 handles this specific class of shallow-search trap better
  than before, not worse.

## Known risk: init-time margin

Tracked through Tier 2 rather than gated on every single item (an explicit call: check the actual
engine at each step, only flag init time if it climbs and holds above a much higher ceiling than
the harness's own 60s budget, since this dev machine's absolute numbers are not assumed to match
organizer hardware -- see the Tier 1 decision below, which this follows). The trajectory on this
dev machine: ~52-63s after Tier 1 (unchanged); climbed to **~67-68s** after null-move pruning, SEE,
and LMR/search extensions (each added branches to `negamax`/`_move_score`/quiescence, already the
most expensive functions to JIT); dropped back to **~59s** once magic bitboards (Tier 2 item 7)
replaced the ray-cast sliding-attack functions used throughout movegen/evaluate/search with much
simpler compiled lookups -- a side benefit, not the reason that item was done.

Confirmed directly, not just estimated: at ~67-68s, `harness.play`/`harness.arena` runs on this
machine failed init outright (the harness's `INIT_BUDGET_S = 60.0` is a hard subprocess-level
cutoff, unlike a plain `import agent` timing). That failure went away once magic bitboards brought
the number back under 60s here. The number is expected to keep moving as more of Tier 2 lands
elsewhere in the codebase; the standing decision is to keep shipping and re-check with real games
rather than pre-emptively cut a working feature on an unverified hardware assumption -- unchanged
from the original Tier 1 call below, just with a higher, explicitly-set re-check ceiling this
time.

Original Tier 1 profiling, still the two largest fixed costs: `movegen.generate_legal`'s first
compile (~16-18s, unaffected by any Tier 2 item except magic bitboards' own contribution to it)
and `negamax`/`search_root`'s first compile (~23s baseline, larger now with Tier 2's added
branches). Tried and ruled out for Tier 1: `NUMBA_OPT=1` (no measurable effect -- the cost is in
type inference/lowering, not LLVM optimization passes) and collapsing small helper functions into
fewer, larger ones (also no measurable effect -- the bottleneck is control-flow complexity, not
per-function overhead). **Decision (Tier 1, still standing): accept and ship** -- organizer
hardware is unknown and plausibly comparable to or better than this dev machine, and the
alternative (cutting a working feature pre-emptively on an unverified hardware assumption) was
judged worse than the risk. Revisit if real competition results show init failures.

The earlier KBN-vs-K "Known issue" from before Tier 1 is closed by DTZ (Tier 1 item 4 above) and
was folded into the Tier 1 section rather than kept as an open issue.

**Final call for Tier 3 and beyond (explicit, superseding the re-check-ceiling approach above):**
stop tracking this dev machine's init-time number as a gating concern at all. Confirmed
empirically that numba's on-disk caching (`cache=True` + `NUMBA_CACHE_DIR`) is not a viable
workaround -- every jitted function in this codebase touches a module-level global array (attack
tables, PSTs, magic tables), and numba explicitly refuses to cache any function that does
("Cannot cache compiled function ... uses dynamic globals"), so there is no cheap fix available
here even if it mattered. Going forward: verify with the correctness gate and functional games,
same as every item above; do not re-run local `harness.play`/`harness.arena` init timing as a
check step.

## Future

See `docs/FUTURE.md` -- items 1-7 (adaptive time management, aspiration windows, null-move
pruning, SEE, LMR, search extensions, magic bitboards) are the Tier 2 work above. Items 8-10
(curated 5-man tablebase subset, opening book, NN evaluation) remain undone for the
item-specific reasons in the Tier 2 section above, not for lack of time -- see there before
picking any of them back up. Tier 3 (mobility eval, futility pruning, a bigger two-tier TT, IID,
counter-move heuristic, delta pruning) is a second round of optimizations beyond FUTURE.md's
original list -- not itself tracked in FUTURE.md, see the Tier 3 section above instead. Tier 4
(phase-blended passed-pawn bonus, king-distance-to-passed-pawn) picked up one of the two further
ideas noted here previously -- see the Tier 4 section above. Tier 5 (fork/hanging-piece/pawn-
threat and pin/x-ray/skewer eval terms) was picked up on a direct user report that middlegame
tactics were still being missed -- see the Tier 5 section above. Tier 6 (the quiescence-in-check
fix) came from the same report, found while looking for further eval-side tactical terms to add --
see the Tier 6 section above. From the same "further ideas" discussion: history malus (penalize
quiet moves that fail to cause a cutoff, not just reward the ones that do) and a king-safety eval
term (attacker count/weight near the enemy king, not just the existing pawn-shield bonus) are
next in line if picked back up, then reverse futility/static-null pruning, late move pruning, and
singular extensions as further search-side levers. Tier 7 (history malus), Tier 8 (king safety),
and Tier 9 (reverse futility/static-null pruning, late move pruning) picked up everything on that
list except singular extensions -- see their sections above. Singular extensions is next if
continuing this list (higher risk/effort, interacts with the existing extension budget), alongside
two correctness/robustness gaps raised in the same discussion: fifty-move-rule and
insufficient-material draw detection inside search (currently absent -- tablebase probing is
root-only, so internal search nodes have no such awareness), and a move-overhead safety margin in
`timeman.py` for real subprocess/IO latency in competition play. Still not picked up: generating a
small custom endgame tablebase locally via retrograde analysis instead of downloading one (item
8's blocker) -- multi-day scope, not worth it against the time remaining unless everything else is
done early.
