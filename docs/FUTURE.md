# Future work, in priority order

Ranked by strength-gained-per-day-of-effort given the 2026-09-11 11:00 London deadline -- not by
raw impact alone. Tier 1 (transposition table, killer/history move ordering, PVS, DTZ tablebase,
tapered/richer evaluation) and Tier 2 (items 1-7 below) are both shipped; see `docs/STATUS.md`
for what each covers and the current verified state, including why items 8-10 remain undone for
reasons specific to each, not time pressure. Tiers 3-13 are further passes shipped on top of this
list, none of them part of the original prioritization here -- see `docs/STATUS.md`'s own
Tier 3 through Tier 13 sections for what each covers (in short: Tier 3 mobility/futility/bigger
TT/IID/counter-move/delta pruning; Tier 4 endgame-specific eval; Tier 5 threat/pin/x-ray eval;
Tier 6 a quiescence-in-check search bug fix; Tier 7 history malus; Tier 8 king safety eval;
Tier 9 reverse-futility and late-move pruning; Tier 10 fifty-move-rule draw detection; Tier 11
insufficient-material draw detection; Tier 12 a timeman move-overhead safety margin; Tier 13
singular extensions). Re-run the full gate -- `tests/perft.py`, `tests/test_repetition.py`,
`tests/test_see.py`, `tests/test_magic_attacks.py`, `tests/test_threats.py`,
`tests/test_quiescence_check.py`, `tests/test_king_safety.py`, `tests/test_fifty_move.py`,
`tests/test_insufficient_material.py`, `tests/test_timeman.py`, `tests/test_singular_extension.py`,
`ruff`, and `mypy --strict` -- after any change to `search.py`, `movegen.py`, `attacks.py`,
`evaluate.py`, or `timeman.py`. Do not gate on local init-time measurements (see
`docs/STATUS.md`'s final call on this).

1. **[DONE, Tier 2] Adaptive time management** (`timeman.py`) -- the flat `time_left/30 + increment` formula
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

2. **[DONE, Tier 2] Aspiration windows** (`search.py`) -- narrow `search_root`'s alpha-beta window around the
   previous iteration's score instead of always searching `-INF..INF`, re-searching wider only on
   a fail-high/fail-low. Cheap given iterative deepening already exists; the only complexity is
   handling the re-search. Was on the original Tier 1/2 list but not part of what shipped. Low
   risk, low-to-moderate effort, tightens pruning further on top of the TT/PVS/killers already in
   place.

3. **[DONE, Tier 2] Null-move pruning** (`search.py`) -- give the opponent a free move and see if the position is
   still good enough to cause a cutoff; when it is, skip searching that branch further. The
   biggest remaining raw speed lever (lets the same time budget reach real extra depth). Needs a
   zugzwang guard -- skip it in king+pawn endgames and while in check, or it risks misplaying
   exactly the kind of tightly-boxed endgame DTZ was just added to fix (a null move can look safe
   in a zugzwang position when it is not). Moderate risk, moderate effort.

4. **[DONE, Tier 2] Static Exchange Evaluation (SEE)** (`search.py`) -- replace MVV-LVA's rough capture ordering
   with a real exchange evaluation, and let quiescence prune clearly-losing captures instead of
   exploring them. Directly relevant to node counts actually observed: the blunder-position
   investigation above saw 500K-2M+ nodes at depth 7-9 in a single sharp middlegame, and
   quiescence explores every capture regardless of whether it is obviously bad. Moderate effort,
   moderate risk (a wrong SEE implementation silently misorders moves rather than crashing, so it
   needs its own test cases, e.g. known winning/losing/equal exchanges).

5. **[DONE, Tier 2] Late move reductions (LMR)** (`search.py`) -- search moves late in the ordering at reduced
   depth first, re-searching at full depth only if they look promising. Real speed win, but wants
   to sit on top of what's already in place (TT for verification, killers/history for trustworthy
   ordering) rather than replace it -- doing this before items 2-4 would be searching a less
   reliable move order at reduced depth, compounding the risk of missing real tactics. Moderate
   effort, higher risk than the items above (subtly wrong reduction conditions cost real tactical
   accuracy, not just speed).

6. **[DONE, Tier 2] Search extensions** (`search.py`) -- extend the search by a ply in forcing lines (e.g. when
   in check) instead of treating every ply as equally worth a full node budget. Helps tactical
   accuracy in exactly the sharp, forcing lines that are hardest for a depth-limited search.
   Adds complexity to depth/time bookkeeping (an extension must not let a line dodge the time
   budget or the mate-distance-adjusted TT scoring added in Tier 1). Moderate effort and risk.

7. **[DONE, Tier 2] Magic bitboards** (`attacks.py`, `movegen.py`) -- sliding-piece attacks currently ray-cast per
   query, a deliberate simplicity-over-speed tradeoff from the initial build. This is the single
   largest remaining node/sec lever, and also currently the single largest fixed cost in init time
   (`generate_legal`'s first compile alone measured ~16-18s of the ~52-63s total -- see
   `docs/STATUS.md`'s init-time risk note). Worth it for both reasons, but it is the
   highest-correctness-risk item on this list: touches the movegen core directly and needs the
   full perft/differential rigor `movegen.py` already has re-earned from scratch before it can be
   trusted. Do this only with enough runway to re-verify properly, not under tight time pressure.

8. **[BLOCKED: no network access] Curated 5-man Syzygy WDL subset** (`tablebase.py`,
   `weights/syzygy/`) -- attempted during the Tier 2 pass; this environment has no outbound
   network access, so the actual `.rtbw`/`.rtbz` files (the existing 3-4-man set's own source,
   `tablebase.sesse.net/syzygy/3-4-5/`, is unreachable from here) cannot be fetched. The code-side
   change is a one-line follow-up (`tablebase.MAX_PIECES = 5` -- python-chess's Syzygy reader
   already handles 5-man files generically) once files are supplied by some other means. The
   complete 5-man set is ~378MB, far past what's worth spending out of the 50MB cap alongside
   everything else already shipped. A curated subset (e.g. pawnless endings, common rook
   endgames) could extend tablebase coverage past the current 3-4-man scope within a few more MB.
   Starting shortlist based on general tablebase domain knowledge, not verified file sizes (no
   network access to check real listings): pawnless minor-piece-heavy endings (KBBvKN, KBNvKN,
   KNNvKB, KRvKBN, KRvKNN -- low piece mobility means few reachable positions, so small files) and
   KRBvKR/KRNvKR (moderate size, common from simplification). KRPvKR is probably the single most
   practically important 5-man endgame but likely among the larger files, alongside other
   rook/queen-heavy combinations -- verify actual sizes before committing budget to any of these.

9. **[NOT WORTH DOING as scoped] Opening book** -- checked during the Tier 2 pass: no curated
   starting positions used by the platform are known or obtainable from here, so this item's own
   stated precondition (below) is unmet, not merely deprioritized. Lower priority than it looks:
   `docs/IDEAS.md` (the original platform
   guidance) explicitly notes that rated games start from curated positions rather than the
   standard starting position, so a book keyed on move one is often already out of book by the
   time it would matter. Worth doing only if the actual curated starting positions used by the
   platform can be obtained or inferred in advance -- otherwise the effort is better spent on
   search depth (items above), which helps in every position, not just known ones.

10. **[DECLINED for this pass] NN evaluation** -- explicitly declined when checked during Tier 2:
    a fundamentally different, multi-day scope (data/self-play, training, ONNX export,
    integration and testing) than every other item on this list, against one week of runway.
    Revisit only with real days to spare. Still deprioritized on the merits too. The numba
    baseline's own evidence in this repo shows
    depth beats eval quality at these node counts, and training risk (data, time, correctness) is
    high against the competition timeline. Only worth revisiting if everything above is done with
    real days to spare before 2026-09-11 11:00.
