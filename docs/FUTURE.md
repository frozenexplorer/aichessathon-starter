# Future work, in priority order

Ranked by strength-gained-per-day-of-effort given the 2026-09-11 11:00 London deadline -- not by
raw impact alone. Tier 1 (transposition table, killer/history move ordering, PVS, DTZ tablebase,
tapered/richer evaluation) and Tier 2 (items 1-7 below) are both shipped; see `docs/STATUS.md`
for what each covers and the current verified state, including why items 8-10 remain undone for
reasons specific to each, not time pressure. Tiers 3-18 are further passes shipped on top of this
list, none of them part of the original prioritization here -- see `docs/STATUS.md`'s own
Tier 3 through Tier 18 sections for what each covers (in short: Tier 3 mobility/futility/bigger
TT/IID/counter-move/delta pruning; Tier 4 endgame-specific eval; Tier 5 threat/pin/x-ray eval;
Tier 6 a quiescence-in-check search bug fix; Tier 7 history malus; Tier 8 king safety eval;
Tier 9 reverse-futility and late-move pruning; Tier 10 fifty-move-rule draw detection; Tier 11
insufficient-material draw detection; Tier 12 a timeman move-overhead safety margin; Tier 13
singular extensions; Tier 14 a TT size doubling plus a tempo bonus and opposite-bishop draw
scaling; Tier 15 Lazy SMP multi-threaded search and a branching-factor early stop; Tier 16
re-verifying the live contract directly -- which found Tier 15's `SMP_THREADS` trusting
`os.cpu_count()` on hardware the contract already documents as one core, fixed by hardcoding it --
plus another TT doubling, multi-cut pruning, adaptive late move reductions, and two eval terms
(knight outposts, space/pawn-storm); Tier 17 init-time recovery -- committed `bench_init.py`/
`bench_nodes.py` benchmarks, found and fixed the real mechanism behind Tier 16's regression
(duplicate numba specialisations from `Literal[int]`/`int8`-vs-`int64` argument-type drift, not
function size as earlier tiers assumed), deleted Lazy SMP entirely now that `SMP_THREADS = 1`
(Tier 16) had made it fully dead code, and shipped precomputed magic attack tables, then confirmed
the fix cost no strength with a 4-game real-contract (120s+0.5s) head-to-head vs Tier 13, +2 =1 -1
-- see `docs/STATUS.md`'s Tier 17 section for why this reverses the earlier blitz-clock batch's
direction; Tier 18, `docs/plan.md`'s Phase 2 (node rate, zero init cost, gated purely on
`tests/bench_nodes.py` plus the correctness suite): skipped a redundant leaf `generate_legal` call
in `negamax`, real `llvm.ctpop`/`cttz` popcount/bitscan intrinsics, set-bits-only Zobrist hashing,
an allocation-free occupancy-based SEE rewrite, staged move-ordering selection in place of a full
`argsort`, deferring `make_move`'s board copy past the futility/LMP prune checks via a cheap
direct-check-only test, and a handful of ordering fixes (a promotion-scoring bug, deduped
`is_capture` calls, a tighter branching-factor estimate, a widening aspiration-window ladder) --
node rate up ~30-55% cumulatively with zero init-time cost, see `docs/STATUS.md`'s Tier 18 section
for the full breakdown and numbers. Tier 13 was also trial-reverted and restored in between
shipping and Tier 14 -- see `docs/STATUS.md`'s "Known risk: init-time margin" section for that
whole round trip and why it ended back where it started; that same section now also carries Tier
16's confirmation that the real init cap really is 90s, not the 60s this repo's own
`AGENTS.md`/`harness/rules.py` still say (the latter deliberately left unedited regardless, per
`AGENTS.md`'s own standing rule against ever changing `harness/`), and Tier 17's own numbers on
where init time actually landed after the fix.
Re-run the full gate -- `tests/perft.py`, `tests/test_repetition.py`, `tests/test_see.py`,
`tests/test_magic_attacks.py`, `tests/test_threats.py`, `tests/test_quiescence_check.py`,
`tests/test_king_safety.py`, `tests/test_fifty_move.py`, `tests/test_insufficient_material.py`,
`tests/test_timeman.py`, `tests/test_singular_extension.py`, `tests/test_tempo_and_ocb.py`,
`tests/test_bit_ops.py`, `tests/test_zobrist_hash.py` (both Tier 18), `ruff`, and `mypy --strict`
(explicitly against `movegen.py`/`evaluate.py`/`search.py` too, not just the `agent.py`/`harness`
the `[tool.mypy]` `files` config covers by default -- still an open gap, see `docs/plan.md`) --
after any change to `search.py`, `movegen.py`, `attacks.py`, `evaluate.py`, `timeman.py`,
`zobrist.py`, or `agent.py`. `tests/test_lazy_smp.py` no longer exists (Tier 17 deleted it along
with the feature it tested). Init time is a real, checked concern again as of Tier 13 (see
`docs/STATUS.md`'s header), not something to gate on blindly but not something to ignore either --
re-profile with `tests/bench_init.py` (Tier 17) if a change might meaningfully grow
`negamax`/`quiescence`/`search_root`'s own compiled complexity or reintroduce a duplicate
specialisation, and `tests/bench_nodes.py` (Tier 18's own acceptance test) for anything touching
search/movegen speed.

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

11. **[NOT NEEDED so far, keep in reserve] Split `generate_pseudo_legal`/`negamax`, consolidate
    negamax's three near-identical recursive call sites** (`docs/plan.md` Phase 1.2/1.3) --
    Tier 17's Phase 1.1 specialisation collapse alone cut `TOTAL_IMPORT_SEC` 44% (162.61s ->
    ~91s on this dev machine) with the signature audit at zero duplicates, which `docs/plan.md`'s
    own sequencing treats as sufficient without touching either function's control flow. Revisit
    only if a future change grows `negamax`/`generate_pseudo_legal` enough that `bench_init.py`
    shows real margin pressure again -- splitting a 391-line, 83-branch, 7-recursive-call-site
    function is real correctness risk (mutual recursion between njit functions is fragile) for a
    win not currently needed.

12. **[DONE, Tier 18] `docs/plan.md` Phase 2 (node-rate)** -- seven independent, individually
    revertable search/movegen speed items (redundant leaf `generate_legal` in quiescence, real
    popcount/bitscan via `llvm.ctpop`/`cttz`, set-bits-only Zobrist hashing, an allocation-free
    occupancy-based SEE rewrite, staged move-ordering selection instead of a full `argsort`,
    deferring `make_move`'s board copy past the futility/LMP prune checks, and a handful of small
    ordering/correctness fixes), each accepted only on `tests/bench_nodes.py` plus the correctness
    suite. Node rate up ~30-55% cumulatively, zero init-time cost -- see `docs/STATUS.md`'s Tier 18
    section for the full breakdown and numbers, `docs/plan.md` for the original writeup.

13. **[NOT STARTED] `docs/plan.md` Phase 3 (trained NNUE)** -- explicitly deferred behind Phase 2
    per the plan's own sequencing. Spends the ~43MB of zip headroom left after `weights/syzygy/`
    and `weights/attacks.npz` (Tier 17) -- the biggest strength ceiling available, but multi-day
    (self-play or externally-annotated training data, PyTorch training offline, quantised `int16`
    export, an njit inference forward pass -- never torch/onnxruntime at inference time, per-node
    call overhead would be fatal inside a search doing 100k+ nodes) and shipped only if it beats the
    handcrafted eval over a real arena run (>=40 games). See `docs/plan.md` for the full writeup.
