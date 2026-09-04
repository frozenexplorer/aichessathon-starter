# Status

Handoff snapshot as of the Tier 1 pass (2026-09-04), on top of commit `aba6079`. Read this
instead of replaying the whole build history. Competition context: uploads close 2026-09-11
11:00 London; the rated ladder runs hourly 08:00-22:00; Daily Five runs 2026-09-06 through
2026-09-10.

## Architecture

`agent.py` does not use python-chess at runtime for movegen or search -- the numba baseline in
this repo shows that jitting only the evaluation barely helps when the tree walk is still pure
Python, so the whole hot path was rebuilt on a custom bitboard representation instead:

```
bitboard.py     position as (bb: uint64[12], meta: int8[6]); FEN parsing via python-chess
attacks.py      precomputed knight/king/pawn tables; sliding pieces via ray-cast, not magic
                bitboards -- slower per query, far simpler to get right
movegen.py      pseudo-legal + legal generation, copy-make (not incremental make/unmove)
evaluate.py     tapered material + PST (midgame/endgame king blend by game phase), pawn
                structure (doubled/isolated/passed), bishop pair, rook open/semi-open files,
                king pawn-shield -- all jitted
zobrist.py      position hashing, for repetition detection and the transposition table
search.py       negamax/alpha-beta, iterative deepening, a transposition table, killer-move +
                history move ordering, principal variation search (PVS), quiescence,
                repetition-draw scoring, panic-mode fallback
tablebase.py    Syzygy WDL + DTZ, root-only -- WDL filters to won/drawn-safe moves, DTZ narrows
                further to the ones that actually make progress
timeman.py      per-move budget + a hard low-clock panic threshold
tests/perft.py          movegen verified against python-chess: differential testing over random
                         games (6 position types incl. Kiwipete) + perft node counts
tests/test_repetition.py  direct unit tests of the repetition-draw and claim-eligibility logic
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

## What's implemented and verified

- `ruff` / `mypy --strict` clean. `tests/perft.py` (movegen, unaffected by this pass) and
  `tests/test_repetition.py` (updated for the new `negamax`/`search_root` signatures) both pass.
- Numba bounds-checking (`NUMBA_BOUNDSCHECK=1`) run across several games surfaced zero
  IndexErrors -- no memory-safety issue in the new TT/killer/history array indexing.
- **Head-to-head vs the pre-Tier1 baseline** (`baselines/pre_tier1`, 10 games, 20s+0.2s clock):
  **+9 =1 -0 (95%)**. No crashes, illegal moves, or init failures in that batch. The one draw was
  by threefold repetition, not a loss.
- Clean wins in spot checks against `random`/`greedy`/`minimax`/`numba` baselines, both colours.
- KBN vs K vs `baselines/random`: 3/3 checkmates post-DTZ (previously ~5/5 draws -- see Tier 1
  item 4 above).
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

Init time is now **~52-63s against the 60s budget** (measured on this dev machine; 2 of several
cold-start runs exceeded 60s outright before other games in the same session ran clean). Profiled
where it goes: `movegen.generate_legal`'s first compile (~16-18s, unchanged by this pass) and
`negamax`/`search_root`'s first compile (~23s, the main new cost -- TT + PVS branching + killer
logic all live in one recursive, heavily-typed function, which numba's type inference handles
disproportionately slowly). Tried and ruled out: `NUMBA_OPT=1` (no measurable effect -- the cost
is in type inference/lowering, not LLVM optimization passes) and collapsing the small new helper
functions into fewer, larger ones (also no measurable effect -- the bottleneck is negamax's own
control-flow complexity, not per-function overhead). The one lever that would reliably reclaim
several seconds is dropping PVS specifically (the single biggest driver of negamax's compile
complexity). **Decision (explicit, made when this was flagged): accept as-is and ship all of
Tier 1 including PVS** -- organizer hardware is unknown and plausibly comparable to or better
than this dev machine, and the alternative (cutting a working Tier 1 feature pre-emptively on an
unverified hardware assumption) was judged worse than the risk. Revisit if real competition
results show init failures.

The earlier KBN-vs-K "Known issue" from the previous pass is now closed by DTZ (Tier 1 item 4
above) and has been folded into the Tier 1 section rather than kept as an open issue.

## Future

See `docs/FUTURE.md` for the full prioritized list (adaptive time management, aspiration windows,
null-move pruning, SEE, LMR, search extensions, magic bitboards, a curated 5-man tablebase subset,
an opening book, NN evaluation) with rationale, effort, and risk for each.
