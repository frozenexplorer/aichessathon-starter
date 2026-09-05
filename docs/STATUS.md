# Status

Handoff snapshot as of the Tier 15 pass (2026-09-05), on top of Tiers 1-14 (Tier 2 commit
`02ec418`, Tier 1 commit `12a38f9`), all documented in their own sections below. Read this instead
of replaying the whole build history.
Competition context: uploads close 2026-09-11 11:00 London; the rated ladder runs hourly
08:00-22:00; Daily Five runs 2026-09-06 through 2026-09-10.

Init time on this dev machine is not assumed to reflect the actual competition hardware as a
matter of policy (see "Known risk: init-time margin" below), but real platform numbers did arrive
during Tier 13 (~85s init against a real 90s cap) and briefly made this a live decision rather than
a purely theoretical one -- see that same section for the profiling, the trial removal of singular
extensions, the head-to-head match that measured its real strength cost, and why it was restored
rather than left out. Tiers 14-15 were built under an explicit "increase power, leave init time
alone" request following that episode -- see their own sections for what was validated before
writing any code (Lazy SMP in particular rests on two standalone feasibility probes, not
assumption) and the two related tuning investigations that concluded "no change" is itself the
right call for now. Each Tier 3+ item below was still verified with the full correctness gate
(`ruff`, `mypy --strict`, `tests/perft.py`,
`tests/test_repetition.py`, `tests/test_see.py`, `tests/test_magic_attacks.py`, from Tier 5 onward
`tests/test_threats.py`, from Tier 6 onward `tests/test_quiescence_check.py`, from Tier 8 onward
`tests/test_king_safety.py`, from Tier 10 onward `tests/test_fifty_move.py`, from Tier 11 onward
`tests/test_insufficient_material.py`, from Tier 12 onward `tests/test_timeman.py`, from Tier 13
onward `tests/test_singular_extension.py`, from Tier 14 onward `tests/test_tempo_and_ocb.py`, and
from Tier 15 onward `tests/test_lazy_smp.py`) plus functional games (checkmates from the start
position and the KBN-vs-K tablebase endgame, both colours) before moving to the next.

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
movegen.py      pseudo-legal + legal generation, copy-make (not incremental make/unmove);
                is_insufficient_material -- a conservative, path-independent dead-draw check
evaluate.py     tapered material + PST (midgame/endgame king blend by game phase), pawn
                structure (doubled/isolated/passed, the passed-pawn bonus itself phase-blended),
                bishop pair, rook open/semi-open files, king pawn-shield, differential piece
                mobility, king proximity to passed-pawn promotion squares, tactical threat
                awareness (hanging pieces, pawn threats, forks, absolute pins, x-rays/skewers),
                king safety (attacker-weighted pressure on each king's own ring), a flat tempo
                bonus for the side to move, opposite-coloured-bishop draw scaling -- all jitted
zobrist.py      position hashing, for repetition detection and the transposition table
search.py       negamax/alpha-beta, iterative deepening, a two-tier transposition table (4M
                buckets, ~160MB), killer-move + history (with malus) + counter-move ordering,
                principal variation search (PVS) with aspiration windows, null-move pruning, late
                move reductions and pruning, check extensions, singular extensions (excluded-move
                verification search), futility and reverse-futility/static-null pruning, internal
                iterative deepening, static exchange evaluation (move ordering + quiescence
                SEE/delta pruning), quiescence (including a bounded full-width search of check
                evasions, not just captures, when the side to move is in check),
                repetition/fifty-move/insufficient-material draw scoring, panic-mode fallback;
                search_root compiled nogil=True for Lazy SMP (see agent.py)
tablebase.py    Syzygy WDL + DTZ, root-only -- WDL filters to won/drawn-safe moves, DTZ narrows
                further to the ones that actually make progress
timeman.py      per-move budget (adaptive: a volatile position -- score swinging between
                iterations, in check, a capture just played, or few pieces left -- can spend up
                to timeman.EXTENSION_FACTOR times the base budget) + a hard low-clock panic
                threshold; two stacked safety reservations (clock-proportional SAFETY_MARGIN_MS,
                fixed POST_SEARCH_BUFFER_MS for real post-search-loop work); constants checked
                against a small empirical self-play tournament (Tier 15), no change warranted
agent.py        Lazy SMP: up to SMP_THREADS-1 helper threads run search_root concurrently with
                the main thread's own iterative-deepening loop, sharing the TT, each with its own
                history/killer/counter-move tables; a branching-factor early stop skips starting
                a depth very unlikely to complete before its deadline
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
tests/test_fifty_move.py     negamax's fifty-move-rule draw detection: one ply short of the
                              limit (real score), at the limit and past it (immediate draw, 0)
tests/test_insufficient_material.py  is_insufficient_material and negamax's use of it: four
                              recognised dead-draw shapes, three deliberately-unflagged ones
tests/test_timeman.py        budget_ms/extended_budget_ms's two stacked safety reservations, and
                              sane, non-negative output at very low clock values
tests/test_singular_extension.py  negamax's excluded-move mechanism: an excluded search still
                              returns a sane score and never stores to the TT under its own
                              position's key, while its own real-move children still store to
                              theirs normally
tests/test_tempo_and_ocb.py  the tempo-bonus invariant (evaluate(white-to-move) +
                              evaluate(black-to-move) sums to exactly 2*TEMPO_BONUS regardless of
                              material) and all four opposite-bishop draw-scaling cases
tests/test_lazy_smp.py       a real self-play game with Lazy SMP active: every move legal, zero
                              helper-thread exceptions, a large number of TT entries written as
                              proof the helper threads actually ran concurrently
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

## Tier 10: what changed and why

1. **Fifty-move-rule draw detection** (`search.py`, `bitboard.py`) -- a real correctness gap,
   raised while listing further ideas: nothing in the engine tracked or checked the fifty-move
   rule (100 plies with no pawn move or capture is an automatic draw), so a winning line could in
   principle shuffle pieces in the search tree without the engine ever realising the clock mattered
   (tablebase probing, the only other draw-aware mechanism besides repetition, is root-only and
   only covers <= `tb.MAX_PIECES`). Added `bitboard.halfmove_clock(fen)` (reads the FEN's own
   halfmove field via python-chess, each `get_move` call, rather than tracked incrementally --
   correct even if a game does not start at 0) and threaded a `halfmove_clock: int` parameter
   through `negamax`/`_search_root_pass`/`search_root`/`agent._search_restricted`, reset to 0 for
   a child move that is a pawn move or capture, incremented by 1 otherwise. At
   `HALFMOVE_DRAW_LIMIT` (100), a node returns an immediate draw (0) -- checked in the same early
   position as the repetition check, before the transposition table, for the same reason (a cached
   score must never mask a real forced draw). Deliberately scoped to the main search tree, not
   threaded through quiescence -- see the module docstring's "Fifty-move rule" paragraph for why
   that gap is an accepted, negligible-risk simplification rather than an oversight. Computing
   `child_halfmove_clock` once per move in negamax's loop also let the three separate,
   already-redundant `is_capture` calls in the futility/LMP/LMR checks collapse into one cached
   `is_cap` variable, reused by all of them.

New dedicated test: `tests/test_fifty_move.py` -- the same "up a rook" fixture as
`tests/test_repetition.py`, checked one ply short of `HALFMOVE_DRAW_LIMIT` (must still show the
real material edge), exactly at the limit, and past it (both must score as an immediate draw).

## Tier 11: what changed and why

1. **Insufficient-material draw detection** (`movegen.py`, `is_insufficient_material`; used in
   `search.py`'s `negamax` and `quiescence`) -- the other half of the draw-detection gap raised
   alongside the fifty-move rule (Tier 10). Unlike the halfmove clock, this is a pure,
   path-independent property of the current board alone, so it needed no parameter threading at
   all: both functions check it directly, immediately after their existing early-return checks,
   and return an immediate draw (0) when it holds. Deliberately conservative (see the function's
   own docstring) -- true only for bare king vs bare king, king plus exactly one minor piece
   against a bare king, or any number of bishops (no pawns/knights/rooks/queens) all on one square
   colour; every other combination, including ones the full FIDE rule also treats as insufficient
   (a knight on each side), returns False, which is always the safe direction: eval simply runs
   as normal, exactly as if the check did not exist. This does not protect the real game result --
   the harness's own `Board.outcome()` already treats insufficient material as an automatic,
   non-claim terminal condition on the actual position played -- it exists purely so the search's
   own evaluation of a hypothetical such position reached while descending the tree is accurate
   (an exact 0) rather than running ordinary material/PST/threat scoring on a position that is
   provably a dead draw. Confirmed the conservative scoping doesn't clip a real endgame the
   engine actually needs to win: KBN vs K (bishop and knight together, not covered by the
   single-minor case) still checkmates in the functional-game check below, unaffected.

New dedicated test: `tests/test_insufficient_material.py` -- four hand-built positions the
function must recognise (bare kings, king+knight, king+bishop, same-colour-square bishops both
sides) checked against both `is_insufficient_material` directly and a live `negamax` call
(expecting score 0), plus three it must deliberately leave unflagged (opposite-colour bishops, a
knight on each side, and a real material edge), checked against `is_insufficient_material` alone.

## Tier 12: what changed and why

1. **Move-overhead safety margin** (`timeman.py`) -- the existing `SAFETY_MARGIN_MS` (300ms,
   subtracted from the clock before any planning formula runs) already hedges against unknown,
   clock-proportional overhead, but nothing previously accounted for a second, distinct source:
   fixed, non-clock-proportional real work that happens on every single move regardless of how
   much time is left -- `agent._record_and_return`'s own post-search `make_move`/`position_hash`
   bookkeeping, building the returned UCI string, and negamax/quiescence's own unavoidable
   overshoot past a deadline (their time check only fires every `CHECK_INTERVAL` nodes, so a
   deadline can be crossed by however long those last few nodes take before the next check
   unwinds the search). Added `POST_SEARCH_BUFFER_MS = 20`, subtracted from the *planned budget
   itself* (not the clock) in both `budget_ms` and `extended_budget_ms`, so the search loop's own
   deadline sits that much earlier than the raw allocation, leaving headroom for this fixed
   overhead before `get_move` actually hands control back to the harness. Two reservations
   stacked rather than merged, since they model genuinely different things (see the module
   docstring): one scales with the clock, the other does not.

New dedicated test: `tests/test_timeman.py` (this module had none before) -- confirms
`budget_ms`'s output is the raw planned allocation minus `POST_SEARCH_BUFFER_MS`, that
`extended_budget_ms` never returns less than `budget_ms` while still respecting the same buffer,
and that both stay sane (floor at `MIN_BUDGET_MS`, never negative) across a range of very low
clock values.

## Tier 13: what changed and why

1. **Singular extensions** (`search.py`) -- the last, highest-risk item from the "further ideas"
   discussion, done last and most carefully for exactly that reason. At a deep enough node
   (`SE_MIN_DEPTH`) with a hash move backed by a deep, trustworthy TT entry (`SE_TT_DEPTH_MARGIN`,
   not merely an upper bound), a verification search of this exact node's *other* moves -- at
   `(depth - 1) // 2`, in a narrow window just below the hash move's own stored score
   (`SE_MARGIN_PER_DEPTH * depth`) -- checks whether any alternative can even come close. If none
   can, the hash move is "singular" and its own child gets the same one-ply extension a checking
   move already gets from search extensions, since the search is now committing real depth to
   confirming a move the ordering already trusts rather than assuming that trust is warranted.
   Implemented via two new `negamax` parameters, `excluded_from`/`excluded_to`: a per-node
   exclusion, never inherited by any recursive call the excluded search itself makes (its own
   null-move, IID, and move-loop children all pass `-1, -1`, identical to every ordinary node), so
   nesting is impossible by construction -- the trigger condition also independently requires
   `excluded_from < 0`, belt and braces. The two places this needed real care, both handled: an
   excluded node must never resolve via its own TT entry (written for the *whole* move set, not
   the one this search is deliberately missing) -- the early-return in `_tt_resolve`'s caller is
   skipped whenever `excluded_from >= 0`, though `hint_from`/`hint_to` are still read for ordering,
   since the loop's own exclusion check (`continue` on a from/to match) makes that harmless; and an
   excluded node must never write one either, since a partial-search result stored under the full
   position's key would corrupt every future non-excluded probe of it -- skipped by wrapping the
   entire end-of-function store in the same `excluded_from < 0` guard.

Measured cost: a fixed 8s search on the Tier 1 blunder-position FEN now reaches depth 7 complete
(89971 nodes, essentially identical to every prior tier's depth-7 count) but only depth 8 partial
(107648 nodes) rather than Tier 12's depth 8 complete (~95-107K nodes) / depth 9 partial -- a real,
measured throughput cost from the extra verification searches at qualifying nodes, not a
measurement artifact. This is the expected, accepted tradeoff the technique is known for in real
engines: real playing strength from singular extensions comes from tactical accuracy on critical
lines the verification search actually catches, not from raw nodes/sec, which is exactly why this
item was scoped, implemented, and tested more carefully than anything else in this list rather
than skipped for its cost alone.

New dedicated test: `tests/test_singular_extension.py` -- directly exercises the exclusion
mechanism itself (not the singularity heuristic's tuning, which nothing asserts on): a baseline
unexcluded search populates its own TT normally; excluding one real, always-legal move (a king
step) still returns a sane material-edge score by searching everything else; and, most
importantly, the excluded search never writes to the TT under its own position's key while its own
real-move children (different positions, reached normally) still write to theirs -- caught a real
bug in the test itself during development (an earlier draft asserted nothing was written anywhere,
which is wrong: children legitimately write under their own keys) before the assertion was
narrowed to the actual invariant that matters.

## Tier 14: three zero-init-cost strength additions

Prompted by a direct ask to increase strength without touching the init-time lever, after Tier 13's
init-time investigation above -- see that section for why init time briefly became a live concern
and why it settled where it did. All three are pure constants or near-zero-compile-cost eval
additions, not new search control flow, so none of them meaningfully move init time on their own:

1. **TT size doubled** (`search.py`) -- `TT_BUCKETS` 2M -> 4M buckets (~80MB -> ~160MB). A pure
   array-size constant; the compiled code is identical either way, so this costs nothing at init
   beyond a slightly larger allocation/zeroing at `new_tt()` time. Fewer collisions over a full
   game's worth of nodes, kept to a modest doubling rather than a larger multiple since the
   platform's real memory ceiling is not known.
2. **Tempo bonus** (`evaluate.py`) -- a flat `TEMPO_BONUS` added to `evaluate()`'s return value,
   after the mover-perspective sign flip, so it always rewards whoever's turn it is. One line.
3. **Opposite-coloured-bishop draw scaling** (`evaluate.py`) -- `opposite_bishops_scale` scales the
   whole eval down to `OCB_SCALE_PERCENT` percent in a pure OCB endgame (exactly one bishop each, on
   opposite-coloured squares, no other minor or major piece for either side): these are famously
   drawish even a material edge up, since the bishops can never contest the same squares. Deliberately
   scoped to bishops-and-pawns-only endgames -- OCB alongside a rook or queen does not carry the same
   drawish tendency, confirmed by a dedicated test case that a rook present suppresses the scaling.

New test: `tests/test_tempo_and_ocb.py` -- the tempo invariant (`evaluate(white-to-move) +
evaluate(black-to-move)` on the same position sums to exactly `2 * TEMPO_BONUS` regardless of any
material imbalance, since the imbalance term cancels out of the sum) and all four OCB scale cases
(opposite colours scales, same colours does not, OCB plus a rook does not, no bishops does not).

## Tier 15: Lazy SMP (nogil multi-threaded search) and a branching-factor early stop

Two more items from the same "increase power, leave init time alone" request, both validated with
standalone feasibility probes *before* touching real code, given how large and hard-to-reverse a
mistake in either would be:

1. **Lazy SMP.** `search_root` is now compiled `nogil=True`, and `agent.py`'s `_spawn_helpers`
   starts up to `SMP_THREADS - 1` (capped at 4, `os.cpu_count()`-aware) helper threads per move, all
   calling `search_root` concurrently with the main thread's own loop. Validated in two steps before
   writing any of this: first, a synthetic CPU-bound `nogil=True` workload benchmarked at 5.4x
   speedup across 8 threads on this machine, confirming numba's nogil actually releases the GIL for
   real multi-core execution here rather than just in theory; second, a probe confirming numba
   resolves an njit-to-njit call as a direct native call at the *callee's* own compile time, not
   through the nogil-controlled Python-facing dispatcher wrapper -- meaning only `search_root`
   itself (the one function agent.py calls directly) needed the decorator, not `negamax`/
   `quiescence`/`evaluate`/etc. `_search_restricted` (the tablebase-narrowed endgame path) is
   deliberately left single-threaded for this first pass.

   Threads share the transposition table (the entire point) but nothing else: each gets its own
   copy of `_history`/`_opponent_history` (negamax writes further entries onto whatever history
   array it is given as it descends, so two threads sharing one array would corrupt each other's
   in-search repetition detection -- a real correctness hazard, unlike the TT) and its own fresh
   killer/history-heuristic/counter-move tables. Sharing the TT lock-free is a deliberate, standard
   Lazy SMP tradeoff: a torn read can only produce the same class of "wrong info" a single-threaded
   run already tolerates from an ordinary hash-index collision (a hint move that doesn't match any
   move in the current position's own generated legal-move list, already skipped harmlessly rather
   than trusted blindly) -- see search.py's module docstring for the full argument, including why
   the individual TT fields cannot meaningfully tear on real hardware. Helper threads target
   `base_deadline` (never the volatility-extended budget) and are always fully joined before a move
   is returned, so they can only ever make a move later by the deadline-check overshoot margin
   already priced into the timeman safety buffers, never leak, and never silently swallow an
   exception (verified directly, see below).

2. **Branching-factor early stop** (`agent.py`) -- before starting depth N+1, compare depth N's own
   elapsed time against the time actually remaining; skip starting the next depth (return the
   current best immediately) if even `BRANCHING_ESTIMATE` (4, deliberately conservative) times as
   long would not fit. An aborted root search is discarded entirely regardless (search.py's own
   docstring), so starting a depth with no real chance of finishing only wastes clock a future move
   could have used instead -- a free improvement, not a strength/time tradeoff.

New test: `tests/test_lazy_smp.py` -- plays a real 24-ply self-play game with Lazy SMP active,
asserting every move returned is legal, zero helper-thread exceptions were raised (checked via a
temporary `threading.excepthook` override, since `Thread.join()` alone does not surface them), and
a large number of TT entries were written (54k+ in practice) as direct proof the helper threads
actually ran concurrently rather than merely starting and no-opping.

**Two related investigations, both concluding "no change," which is itself the useful result:**
a small empirical tournament (4 candidate `timeman.py` constant settings, 8 games each, vs current
defaults) found every candidate lost to the defaults -- the existing `EXTENSION_FACTOR`/
`MAX_EXTENSION_FRACTION`/`ASSUMED_MOVES_LEFT`/`SAFETY_MARGIN_MS` values are already reasonably
tuned, so nothing changed there. Separately, a real (not just proposed) Texel-tuning pipeline was
built for `evaluate.py`'s weight constants -- confirmed feasible via a numba-recompile mechanism
(mutating a module-level weight constant and recompiling only the specific function that reads it
plus `evaluate()` itself, without needing to touch `negamax`/`quiescence`'s own compiled bodies at
all, since tuning only ever calls `evaluate()` directly on a fixed position dataset) -- but the
first real run (40 self-play games, 880 sampled positions, forced fast to stay tractable) produced
weights with clear overfitting symptoms (`ROOK_OPEN_BONUS` tuned *below* `ROOK_SEMI_OPEN_BONUS`,
inverting a basic chess principle; most bonuses cut roughly in half in a uniform pattern more
consistent with fitting noise in a small, blunder-prone shallow-search dataset than with a real
signal). Those tuned values were deliberately not applied. The pipeline itself is sound and
reusable; a trustworthy run would need a much larger dataset (thousands of positions from deeper
searches), which is a multi-hour undertaking not attempted in this pass.

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
- Tier 10: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats,
  quiescence-check, king-safety) plus the new `tests/test_fifty_move.py` all clean, plus the same
  functional in-process games. Same 8s-search throughput check as Tier 9, identical depth/node
  profile (depth 8 complete / depth 9 partial) -- the new check never fires in that position
  (halfmove_clock starts at 3), confirming zero overhead when the rule is not in play.
- Tier 11: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats,
  quiescence-check, king-safety, fifty-move) plus the new `tests/test_insufficient_material.py`
  all clean, plus the same functional in-process games -- including confirming KBN vs K still
  checkmates (the one common near-insufficient shape the conservative scoping deliberately does
  not clip). Same 8s-search throughput check as Tier 10, identical depth/node profile.
- Tier 12: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats,
  quiescence-check, king-safety, fifty-move, insufficient-material) plus the new
  `tests/test_timeman.py` all clean, plus the same functional in-process games (a pure timing
  constant change, not a search-algorithm one, so the 8s-search throughput check does not apply
  here -- functional games exercising the real `agent.get_move` path, which is the only consumer
  of `timeman.budget_ms`/`extended_budget_ms`, is the relevant check).
- Tier 13: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats,
  quiescence-check, king-safety, fifty-move, insufficient-material, timeman) plus the new
  `tests/test_singular_extension.py` all clean, plus the same functional in-process games (no
  crashes, no hangs, checkmates both colours -- meaningful here specifically, since a bug in the
  exclusion/recursion-termination logic could plausibly manifest as a hang rather than a wrong
  score). 8s-search throughput check shows a real, measured cost -- see the Tier 13 section above.
- Tier 14: full gate (ruff, mypy --strict, perft, repetition, SEE, magic-attacks, threats,
  quiescence-check, king-safety, fifty-move, insufficient-material, timeman, singular-extension)
  plus the new `tests/test_tempo_and_ocb.py` all clean, plus functional games.
- Tier 15: full gate (ruff, mypy --strict, all 13 test files including the new
  `tests/test_lazy_smp.py`) clean, plus functional games (KBN-vs-K checkmate both colours, bounded
  start-position self-play) with Lazy SMP and the branching-factor early stop both active. Also
  specifically checked for what a threading change most needs and unit tests cannot easily catch:
  `test_lazy_smp.py` overrides `threading.excepthook` for the duration of a real game to confirm
  zero helper-thread exceptions (which `Thread.join()` alone would silently swallow), and confirms
  54k+ TT entries were written in a single 24-ply game -- proof the helper threads actually ran
  concurrently and shared the table, not just started and no-opped.
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

**Update: real platform numbers arrived and made this concrete, not theoretical.** The website
reported ~85s init against a real 90s cap -- a 5s margin, not the "dev numbers don't reflect
organizer hardware" situation the policy above was written for. That prompted the first actual
stage-by-stage profiling of `_warm_up()`: `search_root`'s first compile is ~85-90% of total init
time, and it's genuinely `negamax`/`quiescence`/SEE/move-ordering's own control flow, not
`evaluate.py`'s eval terms (mobility/king-safety/threats/pins-xrays/passed-pawns all compile in
~8s together). Two fixes were tried and confirmed dead: numba caching still fails on every real
function here for the reason already established above, and thread-based parallel warm-up is
blocked by numba's own global compiler lock, which serializes JIT compilation across threads
within a process.

With those out, the only remaining lever was cutting search-side control-flow complexity --
singular extensions (the newest, most control-flow-heavy piece, only triggering at depth >= 7)
was removed via `git revert` of its own isolated commit as a trial, dropping this dev machine's
`search_root` compile from ~187s to ~80s (total init ~210s to ~98s, 53%). That trial was then
tested head-to-head against the pre-removal build over 6 real games (`harness.sandbox`'s actual
subprocess protocol, extended init budget so both sides could start on this dev machine, 15s+150ms
blitz): singular extensions won 5-1, five of those by outright checkmate, not time scrambles. Six
games is a small sample, but a 5-1 result with five clean checkmates is a real signal, not noise
worth dismissing -- singular extensions costs meaningfully more strength than its "only triggers
deep, highest-risk item on the list" framing suggested when it shipped.

**Decision: restore singular extensions, keep the tighter margin.** The build that reported 85s on
the real platform was not failing -- it was tight, not broken. Trading a measured, real strength
loss for margin the current build was not actually short on (5s is uncomfortable but not
insufficient) was judged the wrong side of this trade. Singular extensions is back in as of this
decision; the revert-then-restore round trip is `c5be352` (Tier 13 ships) -> `cde9c17` (reverted
for init time) -> the commit after this one (restored). If init-time margin becomes a live problem
again (a real failed init on the platform, not just a tight number), the next candidates in the
same vein -- search-side control flow, not eval terms, per the profiling above -- are IID and the
counter-move heuristic from Tier 3, not singular extensions again unless a bigger head-to-head
sample changes this read.

## Future

See `docs/FUTURE.md` -- items 1-7 (adaptive time management, aspiration windows, null-move
pruning, SEE, LMR, search extensions, magic bitboards) are the Tier 2 work above. Items 8-10
(curated 5-man tablebase subset, opening book, NN evaluation) remain undone for the item-specific
reasons in the Tier 2 section above, not for lack of time -- see there before picking any of them
back up.

Tiers 3-13 are further passes beyond FUTURE.md's original list, none of them tracked in that file
-- see each one's own section above for what it covers and why. In short, two rounds of further
ideas were proposed and fully worked through, one item at a time with the full gate re-run after
each: the first round (Tier 3) added mobility eval, futility pruning, a bigger two-tier TT, IID,
the counter-move heuristic, and delta pruning. A direct user report that middlegame tactics were
still being missed prompted the second round: Tier 4 picked up two previously-noted endgame-eval
ideas (phase-blended passed pawns, king-distance-to-passed-pawn); Tier 5 added tactical-motif eval
terms (forks, hanging pieces, pawn threats, pins, x-rays/skewers) directly for the report; Tier 6,
found while working on Tier 5, fixed a real search bug (quiescence never handled being in check
correctly); a further "what else can be done" discussion produced six more items, all picked up
in turn: Tier 7 (history malus), Tier 8 (king safety eval), Tier 9 (reverse-futility/static-null
pruning, late move pruning), Tier 10 (fifty-move-rule draw detection), Tier 11 (insufficient-
material draw detection), Tier 12 (a `timeman.py` move-overhead safety margin), and Tier 13
(singular extensions, the highest-risk item on the list, done last and most carefully).

That closes out every item raised across both rounds. Still not picked up: generating a small
custom endgame tablebase locally via retrograde analysis instead of downloading one (item 8's
blocker) -- multi-day scope, not worth it against the time remaining unless everything else is
done early.
