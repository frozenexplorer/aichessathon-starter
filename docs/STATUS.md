# Status

Handoff snapshot as of commit `aba6079` (2026-09-04). Read this instead of replaying the whole
build history. Competition context: uploads close 2026-09-11 11:00 London; the rated ladder runs
hourly 08:00-22:00; Daily Five runs 2026-09-06 through 2026-09-10.

## Architecture

`agent.py` does not use python-chess at runtime for movegen or search -- the numba baseline in
this repo shows that jitting only the evaluation barely helps when the tree walk is still pure
Python, so the whole hot path was rebuilt on a custom bitboard representation instead:

```
bitboard.py     position as (bb: uint64[12], meta: int8[6]); FEN parsing via python-chess
attacks.py      precomputed knight/king/pawn tables; sliding pieces via ray-cast, not magic
                bitboards -- slower per query, far simpler to get right
movegen.py      pseudo-legal + legal generation, copy-make (not incremental make/unmove)
evaluate.py     material + piece-square tables, jitted
zobrist.py      position hashing, for repetition detection only
search.py       negamax/alpha-beta, iterative deepening, MVV-LVA + PV-first ordering,
                quiescence, repetition-draw scoring, panic-mode fallback
tablebase.py    Syzygy WDL, root-only
timeman.py      per-move budget + a hard low-clock panic threshold
tests/perft.py          movegen verified against python-chess: differential testing over random
                         games (6 position types incl. Kiwipete) + perft node counts
tests/test_repetition.py  direct unit tests of the repetition-draw and claim-eligibility logic
```

Everything under `weights/syzygy/` and every `.py` file above ships in the zip (`make zip`);
`harness/` does not (platform-side, mirrors the real protocol, never edited).

## What's implemented and verified

- Negamax + alpha-beta, iterative deepening, MVV-LVA/PV move ordering, quiescence search.
- Time management: per-move budget from remaining clock, in-search deadline checks every 128
  nodes via `objmode` (numba can't call `time.perf_counter()` directly), and a `PANIC_MS` (150ms)
  floor below which search is skipped entirely for a non-recursive move pick. Swept 4 positions
  across clocks from 120s down to 1ms: zero overruns.
- Repetition avoidance -- three mechanisms, all unit-tested in `tests/test_repetition.py`,
  because the harness's referee checks `Board.outcome(claim_draw=True)` *before* asking either
  side for a move, and python-chess's `can_claim_threefold_repetition()` fires the instant a
  repeating option merely *exists*, not only once it's played:
  1. Our own move directly creating our 3rd occurrence of a position (negamax, ply-indexed
     `history` array, even plies only since we only ever see our own turn).
  2. `claim_eligible_for_opponent`: our move handing the opponent a position that either already
     recurred twice itself, or from which they have a reply that would make something recur.
  3. A bounded one-level recursive lookahead on (2), since the same condition can resurface on
     our *own* next turn depending on which (uncontrolled) reply the opponent makes -- caught by
     reusing (2) symmetrically with the two history arrays swapped.
- Syzygy WDL tablebase (`tablebase.py`), complete 3-4-man set, 35 files / 1.3MB, root-only:
  narrows the root move list to WDL-optimal candidates, then the normal search (eval +
  repetition-avoidance) picks among those -- WDL says whether a position is won, not how to
  make progress, so filtering rather than picking outright is what converts an endgame.
- `ruff` / `mypy --strict` clean. Beats `greedy`/`minimax`/`numba` baselines comfortably as both
  colours; direct games and 10-game arena batches show no crashes, flags, illegal moves, or init
  failures against normal opponents.
- Init time: ~40-46s uncontended (60s budget). Margin has been shrinking as features are added
  (zobrist hashing + wider negamax/search_root signatures cost real compile time) -- worth
  watching, not yet a problem.

## Known issue: KBN vs K conversion against a random opponent

Replaying `8/8/8/4k3/8/3BN3/8/4K3 w - - 0 1` (the classic bishop+knight mate, notoriously hard
for naive engines) against `baselines/random` repeatedly still draws more often than not, mixed
threefold and fifty-move. This is not a bug in the repetition-avoidance mechanisms above, which
were verified directly and are working correctly -- it's a structural gap: WDL tells us whether
a position is *objectively won*, never whether a given move makes *progress* toward winning it.
Without DTZ (distance-to-zero), the tablebase can't distinguish a move that shortens the
conversion from one that doesn't, so avoiding any one repeating shape doesn't stop the engine
drifting into a different one over a long, technique-heavy, tightly-boxed sequence. Deeper
lookahead in the repetition mechanisms only pushes the same failure further out, not away from
it -- confirmed empirically (lookahead=1 changed some draws from threefold to fifty-move, not to
wins). Every other tested scenario (normal play, other endgames) is unaffected.

## Future

- **DTZ tables** -- the direct fix for the KBN-vs-K gap above. Distance-to-zero data lets move
  selection always shorten the path to conversion, which by construction can't get stuck
  repeating. Complete 3-4-man DTZ from the same source (`tablebase.sesse.net/syzygy/3-4-5/`) is
  ~2.9MB -- small, same scope as what's already shipped, deliberately cut from this pass for time.
- **Transposition table** -- deliberately cut from the very first working version (documented in
  `search.py`). Ordering + quiescence mattered more at these node counts per `docs/IDEAS.md`, but
  a TT is a safe, well-scoped next addition (a plain dict keyed on position hash, cleared or
  capped per move).
- **5-man Syzygy WDL** -- the full set is ~378MB, too big to ship complete; would need a curated
  subset (e.g. pawnless endings, common rook endgames) to fit alongside everything else.
- **Magic bitboards** -- sliding-piece attacks currently ray-cast per query (a deliberate
  simplicity-over-speed tradeoff for the initial build). This is the actual node/sec lever per
  `docs/IDEAS.md`; worth it once there's time to re-earn the same perft/differential rigor
  `movegen.py` already has, since it's the highest correctness-risk change available.
- **Richer evaluation** -- king safety, passed pawns, bishop pair, per-piece mobility instead of
  a flat move-count term. Free size-wise, no architectural change, straightforward to add
  incrementally to `evaluate.py`.
- **Killer-move / history heuristics** -- on top of the existing MVV-LVA ordering, for cheaper
  additional pruning.
- **Compile-time margin** -- init time has been trending up (~35s -> ~41-46s) as features were
  added; TT and DTZ will add more. Worth a dedicated pass if the 60s budget ever starts to feel
  tight rather than comfortable.
- **NN evaluation** -- considered and deliberately deprioritized early on (see conversation
  history / original planning): depth beats eval quality at these node counts per the numba
  baseline's own evidence, and training risk is high against the competition timeline. Only
  worth revisiting if the above are done with days to spare before 2026-09-11 11:00.
