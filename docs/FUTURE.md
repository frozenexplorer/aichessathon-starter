# Future work, in priority order

Everything here is scoped work not yet done, ranked by strength-gained-per-day-of-effort given
the 2026-09-11 11:00 London deadline -- not by raw impact alone. Tier 1 (transposition table,
killer/history move ordering, PVS, DTZ tablebase, tapered/richer evaluation) is already shipped;
see `docs/STATUS.md` for what that covers and the current verified state. Re-run
`tests/perft.py`, `tests/test_repetition.py`, `ruff`, and `mypy --strict` after any of these, and
re-check init time (see the risk noted in `docs/STATUS.md`) after anything touching `search.py`
or `movegen.py`.

1. **Adaptive time management** (`timeman.py`) -- the flat `time_left/30 + increment` formula
   gives a razor-sharp middlegame the same budget as a simplified endgame. This is not
   theoretical: the investigation into a user-reported "blundered a winning position" game
   (`r2qr2k/pp5p/8/3nPbp1/3B1P1b/1BPp4/PQ4P1/R5KR b`, see `docs/STATUS.md`) traced it directly to
   this -- 2.89s from a 75s clock wasn't enough depth for that position regardless of search
   quality, and a deeper look changed the answer completely. Fix: spend more of the budget when
   the position looks volatile -- e.g. widen the budget when the score swings between the last
   two or three iterative-deepening depths, or when the position isn't quiet (in check, recent
   capture, few pieces left). Low risk (touches only `timeman.py` and the depth loop in
   `agent.py`, not movegen/search correctness), moderate effort, and directly backed by a real
   observed failure -- highest priority.

2. **Aspiration windows** (`search.py`) -- narrow `search_root`'s alpha-beta window around the
   previous iteration's score instead of always searching `-INF..INF`, re-searching wider only on
   a fail-high/fail-low. Cheap given iterative deepening already exists; the only complexity is
   handling the re-search. Was on the original Tier 1/2 list but not part of what shipped. Low
   risk, low-to-moderate effort, tightens pruning further on top of the TT/PVS/killers already in
   place.

3. **Null-move pruning** (`search.py`) -- give the opponent a free move and see if the position is
   still good enough to cause a cutoff; when it is, skip searching that branch further. The
   biggest remaining raw speed lever (lets the same time budget reach real extra depth). Needs a
   zugzwang guard -- skip it in king+pawn endgames and while in check, or it risks misplaying
   exactly the kind of tightly-boxed endgame DTZ was just added to fix (a null move can look safe
   in a zugzwang position when it is not). Moderate risk, moderate effort.

4. **Static Exchange Evaluation (SEE)** (`search.py`) -- replace MVV-LVA's rough capture ordering
   with a real exchange evaluation, and let quiescence prune clearly-losing captures instead of
   exploring them. Directly relevant to node counts actually observed: the blunder-position
   investigation above saw 500K-2M+ nodes at depth 7-9 in a single sharp middlegame, and
   quiescence explores every capture regardless of whether it is obviously bad. Moderate effort,
   moderate risk (a wrong SEE implementation silently misorders moves rather than crashing, so it
   needs its own test cases, e.g. known winning/losing/equal exchanges).

5. **Late move reductions (LMR)** (`search.py`) -- search moves late in the ordering at reduced
   depth first, re-searching at full depth only if they look promising. Real speed win, but wants
   to sit on top of what's already in place (TT for verification, killers/history for trustworthy
   ordering) rather than replace it -- doing this before items 2-4 would be searching a less
   reliable move order at reduced depth, compounding the risk of missing real tactics. Moderate
   effort, higher risk than the items above (subtly wrong reduction conditions cost real tactical
   accuracy, not just speed).

6. **Search extensions** (`search.py`) -- extend the search by a ply in forcing lines (e.g. when
   in check) instead of treating every ply as equally worth a full node budget. Helps tactical
   accuracy in exactly the sharp, forcing lines that are hardest for a depth-limited search.
   Adds complexity to depth/time bookkeeping (an extension must not let a line dodge the time
   budget or the mate-distance-adjusted TT scoring added in Tier 1). Moderate effort and risk.

7. **Magic bitboards** (`attacks.py`, `movegen.py`) -- sliding-piece attacks currently ray-cast per
   query, a deliberate simplicity-over-speed tradeoff from the initial build. This is the single
   largest remaining node/sec lever, and also currently the single largest fixed cost in init time
   (`generate_legal`'s first compile alone measured ~16-18s of the ~52-63s total -- see
   `docs/STATUS.md`'s init-time risk note). Worth it for both reasons, but it is the
   highest-correctness-risk item on this list: touches the movegen core directly and needs the
   full perft/differential rigor `movegen.py` already has re-earned from scratch before it can be
   trusted. Do this only with enough runway to re-verify properly, not under tight time pressure.

8. **Curated 5-man Syzygy WDL subset** (`tablebase.py`, `weights/syzygy/`) -- the complete 5-man
   set is ~378MB, far past what's worth spending out of the 50MB cap alongside everything else
   already shipped. A curated subset (e.g. pawnless endings, common rook endgames) could extend
   tablebase coverage past the current 3-4-man scope within a few more MB. Needs research into
   which 5-man combinations are both small and likely to actually occur.

9. **Opening book** -- lower priority than it looks: `docs/IDEAS.md` (the original platform
   guidance) explicitly notes that rated games start from curated positions rather than the
   standard starting position, so a book keyed on move one is often already out of book by the
   time it would matter. Worth doing only if the actual curated starting positions used by the
   platform can be obtained or inferred in advance -- otherwise the effort is better spent on
   search depth (items above), which helps in every position, not just known ones.

10. **NN evaluation** -- still deprioritized. The numba baseline's own evidence in this repo shows
    depth beats eval quality at these node counts, and training risk (data, time, correctness) is
    high against the competition timeline. Only worth revisiting if everything above is done with
    real days to spare before 2026-09-11 11:00.
