# Init-time recovery + strength plan (AI Chessathon)

## Context

The current build **fails on the platform: import exceeds the 90 s init budget**. `docs/STATUS.md:904-913`
records the last good measurement as **~85 s on the real platform** — a 5 s margin held deliberately.
Tier 16 (`95d24e7`, `b53b5f6`) then added multi-cut and an adaptive LMR table, documented as
"+2-3% compile cost" (`STATUS.md:748-751`), and that pushed it over.

The reference point you gave — **Tier 13+ ran ~70-85 s on their machine** — is the target to get back
under, with real margin this time.

At the same time the zip uses **4.5 MB of 50 MB** (`submission.zip`: 4,514,476 B unzipped, 9% of cap).

Three things are being asked for, and this plan separates them because they have very different risk:

1. **Get init back under budget** — must ship, must be safe.
2. **Make the engine faster inside the same budget** — free strength, no init cost.
3. **Spend the 45 MB of headroom on a trained NNUE** — the biggest strength ceiling available,
   and a multi-day project that must not block (1) and (2).

### Decisions already taken (yours)
- 50 MB goes to a **trained NNUE** + **precomputed attack tables**. No 5-man Syzygy expansion.
- **No shipped numba cache** — `.nbc` files are native machine code and collide with "Native
  binaries in the zip are rejected."
- Init time is fixed **structurally only** — no playing feature gets deleted to buy seconds.

---

## Measured current state

Measured on this dev machine (`.venv`, numba 0.67.0, Python 3.12.1) this session:

| Measurement | Value |
|---|---|
| `import agent` wall time, cold | **203 s** |
| `import agent` wall time, warm, instrumented | **153.4 s** |
| Of which `search_root`'s first compile | **147.8 s (96%)** |
| `NUMBA_OPT=1` + vectorizers off | **139.9 s (−8.8%)** |
| `NUMBA_OPT=0` + vectorizers off | **166.4 s (worse — reject)** |
| Dev : platform ratio implied by `STATUS.md:916-918` | **~2.4×** (dev 210 s ↔ platform 85 s) |

Compile time is essentially **one number**: the first call to `search_root`, which drags in the whole
njit call graph. Within it, by inclusive compile time: `negamax` ≫ `quiescence` (52 s) >
`generate_legal` (15 s) > `generate_pseudo_legal` (12 s) > `evaluate` (9 s) > everything else.
**All 17 `evaluate.py` terms together are ~9 s** — eval is *not* the init problem.

Size ranking of the njit bodies (docstrings stripped, AST-measured):

| lines | branches | function |
|---|---|---|
| **391** | **83** | `negamax` — `search.py:773`, 31 parameters, **7 recursive call sites** |
| **150** | **56** | `generate_pseudo_legal` — `movegen.py:215` |
| 78 | 24 | `quiescence` — `search.py:656` |
| 57 | 21 | `make_move` — `movegen.py:155` |
| 60 | 18 | `see` — `search.py:497` |

### Two facts that break the local workflow today

- `harness/rules.py:3 INIT_BUDGET_S = 60.0` and `AGENTS.md:68` forbids editing `harness/`. With dev
  init at 140-200 s, **`make play`, `make arena` and `make gate` all fail with `AgentFailure("init")`
  right now.** Every game-based measurement in this plan is blocked until Phase 1 lands.
- **There is no committed benchmark script of any kind** — no init timer, no node-rate bench. Every
  number in `STATUS.md` came from throwaway runs. That is why the regression was invisible.

---

## Phase 0 — make the problem measurable (do this first)

Nothing else is trustworthy without it.

- **`tests/bench_init.py`** — imports `agent` in a subprocess, reports total wall time, a per-function
  compile breakdown (patch `numba.core.dispatcher.Dispatcher.compile`), and **a table of distinct
  signatures per dispatcher**. Exits non-zero if any function has more than one specialisation or if
  total exceeds a configurable budget. This is the regression guard that was missing.
- **`tests/bench_nodes.py`** — fixed-time search on a fixed position set (reuse the Tier-1 blunder FEN
  `r2qr2k/pp5p/8/3nPbp1/3B1P1b/1BPp4/PQ4P1/R5KR b - - 3 22` that every tier was benched on, plus a
  quiet middlegame and an endgame), reporting nodes, node rate and depth reached. This is how every
  Phase 2 item gets accepted or rejected.
- Record a baseline from both before touching anything.

---

## Phase 1 — init time, structural only

Ordered by expected payoff. **Re-run `bench_init.py` after each item** — several of these are
hypotheses with a measured mechanism, not certainties.

### 1.1 Collapse duplicate specialisations *(measured — this is the main event)*

**The big functions are each being compiled several times over.** Dumping `.signatures` off every
dispatcher after a real import gives **211 specialisations across 51 functions**:

| function | body lines | **specialisations** | inclusive compile |
|---|---|---|---|
| **`negamax`** | 391 | **2** | the bulk of `search_root`'s 147.8 s |
| **`quiescence`** | 78 | **4** | 52.1 s |
| **`attacked_by`** | 11 | **11** | 8.1 s |
| **`bishop_attacks` / `rook_attacks`** | 6 / 5 | **12 each** | 3.1 s / 2.2 s |
| `make_move` | 57 | **2** | 1.9 s |
| `piece_type_at`, `king_square`, `occ_color` | ~5 | **4 each** | — |
| `_move_score`, `_score_moves`, `claim_eligible_for_opponent`, `_ray_pin_score` | — | **2 each** | — |

Two independent mechanisms produce this, and both are one-line-per-site fixes:

**(a) `Literal[int]` from module-level constants.** Numba types a *plain Python int constant* passed
across an njit call boundary as `Literal[int](v)`, a distinct type per value. So
`quiescence(..., QUIESCENCE_MAX_PLIES, QSEARCH_CHECK_BUDGET)` at `search.py:868-869` compiles a
*different* `quiescence` than the recursive calls at `search.py:681` and `724` which pass
`check_budget - 1` / `qdepth - 1` (plain `int64`). Same mechanism gives `attacked_by` 11 copies and
`bishop_attacks`/`rook_attacks` 12 each — `WHITE`/`BLACK` and the castling squares `2,3,4,5,6` and
`58..62` each mint their own `Literal[int]`.

Fix: define these module constants as `np.int64(...)` instead of bare ints — `QUIESCENCE_MAX_PLIES`,
`QSEARCH_CHECK_BUDGET`, `MAX_CHECK_EXTENSIONS`, `WHITE`/`BLACK` (`bitboard.py:19-20`) — and wrap the
literal squares at the castling call sites in `movegen.py`. Keep `Literal` specialisation only where
it is measurably worth it (a folded magic lookup in a 6-line function may be; four copies of a
78-line `quiescence` is not).

**(b) `int8` vs `int64` drift on `negamax`.** `from_arr`/`to_arr`/`promo_arr` are `dtype=np.int8`
(`movegen.py:374-376`), so the in-loop recursive sites (`search.py:1028`, `1071`, `1082`, `1090`,
`1288`, `1295`, `1302`) pass `parent_from, parent_to` as **`int8`**, while the singular-extension site
(`search.py:960`) passes `int64`. That alone duplicates the single largest compile unit in the engine.

Fix: normalise at the call sites (`np.int64(f)`, `np.int64(t)`). Optionally follow with an **explicit
eager signature** on `negamax`, `quiescence`, `search_root`, `_search_root_pass` and `make_move`,
which makes a second specialisation structurally impossible rather than merely absent today. Use
`[::1]` (C-contiguous) array types — every array here is contiguous, and `[:]` would generate slower
indexing.

**(c) The on-the-clock hazard.** `agent.py:329` `_search_restricted` calls `negamax` **from Python
with plain Python ints** — a third signature that `_warm_up` never exercises. This is worse than slow:
**the 391-line `negamax` can compile from scratch mid-game, on the clock, on the first
tablebase-restricted move.** That is a latent flag loss, not just an init cost. Same for
`sr.is_capture` and `mg.piece_type_at` on the line above it.

Expected payoff: halving `negamax` and quartering `quiescence` addresses the two largest entries in
the compile profile. This is the item most likely to solve the whole problem on its own.

### 1.2 Split the two oversized njit functions

LLVM's cost is superlinear in function size, so splitting is the structural counterpart to 1.1.
(Note: `STATUS.md:883-885` records that *collapsing* helpers had no effect — that is the opposite
direction and does not predict this.)

- **`generate_pseudo_legal`** (`movegen.py:215`, 150 lines / 56 branches, 9-12 s): split into
  `_gen_pawn_moves`, `_gen_knight_moves`, `_gen_slider_moves`, `_gen_king_moves`, `_gen_castling`
  called from a thin driver. Non-recursive, so this is mechanical and safe.
- **`negamax`** (391 lines / 83 branches): extract the **non-recursive** blocks only —
  - TT probe (`search.py:831-861`, repeated at `934-948`) → `_tt_probe`
  - TT store (`search.py:1131-1161`) → `_tt_store`
  - cutoff bookkeeping: killers + history bonus + counter-move + history malus (`search.py:1105-1128`) → `_on_cutoff`
  - the futility / LMP / LMR decision block (`search.py:1036-1069`) → `_prune_and_reduce` returning a small tuple

  **Do not extract anything containing a recursive `negamax` call.** Numba handles self-recursion;
  mutual recursion between two njit functions is fragile. Singular extensions therefore stay inline —
  only the *predicate* at `search.py:951-958` moves out.

### 1.3 Consolidate the recursive call sites

Seven `negamax(...)` call sites × 31 arguments is a large amount of argument-marshalling IR inside
the function whose size is the whole problem. `search.py:1071`, `1082` and `1090` differ only in
`depth` and the window; fold them into one call site with pre-computed `child_depth_eff`, `lo`, `hi`
locals. Behaviour-identical, and it removes two full 31-argument call expansions.

### 1.4 Numba environment flags *(measured: −8.8%)*

Set at the very top of `agent.py`, **before `import bitboard`** (numba reads its config at import):

```python
os.environ.setdefault("NUMBA_OPT", "1")
os.environ.setdefault("NUMBA_LOOP_VECTORIZE", "0")
os.environ.setdefault("NUMBA_SLP_VECTORIZE", "0")
```

Measured 153.4 s → 139.9 s. `NUMBA_OPT=0` is *worse* (166.4 s) — do not use it. **Accept only if
`bench_nodes.py` shows no meaningful node-rate loss**; this trades LLVM optimisation for compile time
and the runtime side must be checked, not assumed.

### 1.5 Ship the magic attack tables precomputed *(also your chosen 50 MB item)*

`attacks.py:237-242` rebuilds `ROOK_ATTACK_TABLE` (2.0 MB) and `BISHOP_ATTACK_TABLE` (0.26 MB) at every
import via **107,648 Python→njit dispatched calls**, and that is the only reason
`_rook_ray_attacks` / `_bishop_ray_attacks` are `@njit` at all.

Ship `weights/attacks.npz` (~2.3 MB) and `np.load` it. Keep the ray-cast builders as **plain Python**
(un-jitted) for `tests/test_magic_attacks.py` and for regenerating the file. Removes two compiles and
the build loop; add a checksum assert at import so a corrupt/stale file fails loudly rather than
silently producing wrong attacks.

### 1.6 `_warm_up` cleanup — `agent.py:354-382`

- Lines 371, 374, 378 (`generate_legal`, `quiescence`, `claim_eligible_for_opponent`) are **already
  compiled** by the `search_root` call on line 365. Dead cost.
- Line 373's `int(...)` casts force a **second `make_move` specialisation** — pass `np.int8(...)`.
- Add the `_search_restricted` path (`sr.is_capture`, `mg.piece_type_at`, `sr.negamax` from Python) so
  nothing is left to compile on the clock. With 1.1's explicit signatures this becomes a cheap assert.
- The docstring still says "60s init budget" — it is 90 s (`STATUS.md:642-653`).

### 1.7 Delete dead code

No init win, but it shrinks what a judge reads and removes live hazards:

- **All of Lazy SMP**: `agent.py:231-280` (`_helper_worker`, `_spawn_helpers`), the `helper.join()` loop
  at `agent.py:200-201`, `SMP_THREADS` at `agent.py:95`, `nogil=True` at `search.py:1323`, the 13-line
  and 27-line docstring sections at `agent.py:39-51` and `search.py:252-278`, and
  `tests/test_lazy_smp.py`. `SMP_THREADS = 1` makes every line of it unreachable.
- `evaluate.warm_up()` — `evaluate.py:784-790`, never called by anything.
- The three unused `pv_from/pv_to/pv_promo` parameters of `_search_root_pass` (`search.py:1242-1244`).
- Duplicate `_bit_scan` (`search.py:456` / `evaluate.py:438`) and `_popcount64` (`movegen.py:91` /
  `evaluate.py:256`) — collapse into one home each (see 2.2, which replaces both anyway).

### 1.8 TT allocation — `search.py:1421-1436`

`TT_BUCKETS = 1 << 23` allocates 335 MB, of which **117 MB is eagerly written** because `tt_depth`,
`tt_from`, `tt_to`, `tt_promo` use `np.full(-1)`. Store `depth + 1` so `0` means empty and all seven
arrays can be `np.zeros` (lazy calloc). Small init win, ~117 MB less RSS, no strength change.

**Phase 1 exit criterion:** dev init low enough that `make play` clears the harness's 60 s gate — which
also puts the platform comfortably under 90 s at the ~2.4× ratio, with margin this time.

One thing to be clear-eyed about: **leftover init budget is not strength.** Once the numbers fit,
extra seconds under the 90 s cap buy only safety margin — the compile is a fixed one-off and there is
nothing useful to spend the remainder on (the opponent and the opening position are unknown, so
there is nothing to precompute). The currency that actually converts into rating is *per-move* search
speed, which is what Phase 2 buys, and evaluation quality, which is what Phase 3 buys. Aim for a
comfortable margin, then stop optimising init and move on.

---

## Phase 2 — node rate (strength, at zero init cost)

Each item is independent, individually revertable, and accepted only on `bench_nodes.py` plus an
arena run. Ordered by expected payoff.

Everything here is **search- and movegen-side on purpose**. Since Phase 3 may delete the handcrafted
eval terms outright, do not spend effort optimising or Texel-tuning `pin_and_xray_score`,
`threats_score`, `king_safety_score` and friends — that work would be thrown away. The one exception
is 2.2, which speeds up eval, movegen and hashing alike and survives either outcome.

### 2.1 Redundant legal-move generation at every leaf *(biggest free win)*

`search.py:863` calls `generate_legal(bb, meta)` and checks `count == 0`; then `search.py:867-870`
returns into `quiescence`, which at `search.py:673-675` **does exactly the same generation and the same
mate/stalemate check again**. `generate_legal` is not cheap — it runs a full `make_move` board copy
plus `attacked_by` per pseudo-legal move and allocates three 256-byte arrays (`movegen.py:374-383`).

Move the `depth <= 0` return **above** line 863. Leaves are the majority of nodes, and this is
behaviour-identical because quiescence repeats both checks itself.

### 2.2 Real popcount and bitscan

`_popcount64` is a Kernighan loop and `_bit_scan` is a **linear 0→63 scan** (~32 iterations per bit
extracted). They sit in the hot loops of `threats_score`, `pin_and_xray_score`, `king_safety_score`,
`outpost_score`, `space_and_storm_score`, `mobility_score`, `game_phase` and `position_hash`.

Replace both with `llvm.ctpop.i64` / `llvm.cttz.i64` via `numba.extending.intrinsic` — about 20 lines
of ordinary Python source, no native binary, no new dependency. Verify against the current
implementations over random inputs in a test.

### 2.3 `zobrist.position_hash` — `zobrist.py:26-41`

A **768-iteration** scan (12 bitboards × 64 squares) called on **every** `negamax` node
(`search.py:816`) and twice per candidate move in `claim_eligible_for_opponent` (`search.py:1207`, `1218`).

Two steps, in order: first iterate **set bits only** with the 2.2 bitscan (~32 iterations, a ~20×
cut, trivially safe); then, if the bench still justifies it, make it **incremental** — have
`make_move` return the updated hash alongside `(new_bb, new_meta)` and thread `parent_hash` into
`negamax`. Keep the full-scan version as the differential-test oracle.

### 2.4 `see()` allocates a board copy per capture — `search.py:522`

`bb.copy()` for every capture scored at every node (`_move_score:581`) and again per capture in
quiescence (`search.py:705`). Rewrite as the standard occupancy-based swap-off: one `occupied`
bitboard mutated in place, attacker sets recomputed by clearing bits. `tests/test_see.py`'s five
hand-computed exchanges are the acceptance test.

### 2.5 Move ordering: full sort → staged selection

`np.argsort(-scores)` at `search.py:1004`, `710` and `1370` fully sorts every move list at every node,
with a fresh `np.empty`/`np.full` per node. Replace with pick-best-remaining selection: most nodes
fail high on the first few moves and never need the rest ordered.

### 2.6 Work done for moves that are then pruned — `search.py:1013-1023`

`make_move` (a full board copy) and `is_check(new_bb, new_meta)` run at `search.py:1016-1017`, *before*
the futility test at `1036` and the LMP test at `1043`. Reorder so a pruned move costs no board copy.
`gives_check` is needed by both tests, so this needs a cheap "does this move give check" test that
does not build the child position — or move only the LMP test (which does not need `gives_check`
until after its index/depth conditions) above the `make_move`.

### 2.7 Small, cheap correctness/ordering fixes

- `is_capture` is recomputed **three times** per cutoff move (`search.py:1013`, `1105`, `1127`) — the
  value from `1013` is already in `is_cap`.
- **Promotion ordering bug** — `_move_score:574-575` scores a non-capture queen promotion at
  `500 + 900 = 1400`, which is *below* killers (5002/5001), the counter-move (5000), and any quiet move
  with history ≥ 1400. And `_score_moves2:641` short-circuits on `base != 0`, so a promotion can never
  pick up a killer/history bonus either. Promotions belong just under captures.
- `BRANCHING_ESTIMATE = 4` (`agent.py:84`) is conservative enough that it almost never fires; the real
  effective branching factor with this ordering is 2-3. Try 2.5 and measure over an arena run.
- Aspiration windows (`search.py:1372-1392`) jump straight from a 50 cp window to full `-INF..INF` on
  any fail. A widening ladder (50 → 200 → full) is a few lines and saves whole re-search passes.

---

## Phase 3 — trained NNUE evaluation

This is the strength ceiling and the only thing that genuinely needs 40 MB. It is also **multi-day**,
so it is built **behind a flag, after Phases 1 and 2 have shipped**, and only replaces the handcrafted
eval if it wins games.

A useful second-order effect: if NNUE wins, `mobility_score`, `threats_score`, `pin_and_xray_score`,
`king_safety_score`, `outpost_score`, `space_and_storm_score`, `pawn_structure`, `piece_features` and
`passed_pawn_king_distance` all get deleted — **~477 njit lines and ~110 branches of compile surface
gone**, so it pays back into the init budget too.

### 3.0 Budget arithmetic

The 50 MB cap is on **unzipped** size, and `harness/package.py:50-59` measures it as the sum of the
shipped files' on-disk sizes. A `.npz` is itself a compressed archive, so it counts at its own
compressed size — use `np.savez_compressed`. Current occupancy and what's left:

| | bytes |
|---|---|
| Source (`*.py` at root) | 168,396 |
| `weights/syzygy/` (unchanged) | 4,346,080 |
| `weights/attacks.npz` (Phase 1.5) | ~2,300,000 |
| **Available for `weights/nnue.npz`** | **~43,100,000** |

**Careful:** `harness/package.py:14` ships **every** `*.py` at the repo root. Training and data-generation
scripts must live in `tools/` or `tests/`, never at the root, or they end up inside the submission.

### 3.1 Architecture — staged

- **Stage A (ship-safe, ~0.8 MB):** `768 → 512 → 32 → 1`, side-to-move perspective, piece-square
  features, `int16` weights. Evaluated from scratch per node: ~32 non-zero features × 512 accumulator
  adds is comparable to the current handcrafted eval's cost. Simple enough to get right.
- **Stage B (uses the budget, ~21-40 MB):** king-bucketed HalfKP-style features
  (`40960 × 256` `int16` = 20.9 MB per perspective), with an **incremental accumulator** — `make_move`
  already copies the board arrays, so it copies and updates a `2 × 256` `int16` accumulator alongside.
  Stage B is where both the strength and the 40 MB actually live.

### 3.2 Inference in numba, not torch/onnxruntime

`torch` and `onnxruntime` are preinstalled and are in `pyproject.toml`, but neither is imported today.
**Keep it that way for inference** — per-node torch/ORT call overhead is fatal inside a search doing
100k+ nodes. Train in PyTorch offline; export quantised `int16` weights to `weights/nnue.npz`; run the
forward pass in an njit function. `.npz` is plain zipped numpy arrays, not a native binary.

### 3.3 Training data — your call, and it matters a lot

`AGENTS.md:61-64` is explicit: **"Training on data an engine annotated is allowed; the ban covers what
ships and runs inside the zip."** So there are two routes:

- **Self-bootstrapped** — positions from our own self-play, labelled by our own search at fixed depth.
  No external tools, but the net's ceiling is our own eval plus search.
- **Externally annotated** — positions labelled offline by a strong engine, or a public evaluated
  position set. Explicitly permitted, and produces a substantially stronger net. Nothing from that
  engine ships.

I'd recommend the second, since the rules allow it outright. Either way the model shipped is one we
trained, as required.

### 3.4 Acceptance

NNUE ships **only** if it beats the handcrafted eval over a real arena run (≥ 40 games, alternating
colours, real time control). Keep `evaluate()` intact behind the flag until then. This is the item
most likely to run out of runway — the deadline in `docs/FUTURE.md` is **2026-09-11 11:00 London**,
six days out — and Phases 1 and 2 must be shippable without it.

---

## Phase 3 result (for the record)

Ran, on real self-play data, not a toy: Stage A (`768 → 512 → 32 → 1`), trained on 57,045 positions
from 500 self-play games (checkpointed every 10 games after an earlier run lost 294 games of
progress to an unrelated machine restart — the generator writes incrementally now), labelled with
a fixed-depth (depth 4) `search.negamax` score, called standalone the same way
`tests/test_singular_extension.py` already does (fresh killers/history/counter tables per position,
one shared `new_tt()` across the run). MSE went 0.290 → 0.072 over 200 epochs of training — real
learning, not noise. Wired behind exactly the flag 3.4 called for (`evaluate.USE_NNUE`, an env var,
off by default, confirmed via `bench_init.py` to cost zero extra compile time when off since numba's
own dead-branch pruning drops the unreached arm — `extract_features`/`nnue_forward` show **0**
specialisations in the default build).

**Result: lost 0-5 to the handcrafted eval in a real-contract arena** (`tools/head_to_head.py`,
120000 ms + 500 ms, alternating colours, stopped early — a 5-0 shutout at real time control is
decisive enough not to need the rest of a 20-game budget). Per 3.4's own gate, **it does not ship.**
Read as a *label-quality* problem, not a data-quantity one: depth-4 `negamax` is shallow, and a net
can never exceed what labelled it. More self-play games at the same depth would likely just be more
of the same signal, not better signal — the lever that matters is deeper labels, which is slower per
position, not more positions at the same depth. Stage B (3.1) was never attempted; Stage A alone
already needed more runway than was available before the deadline.

---

## Phase 4 — what the freed-up margin is actually worth spending on

Two numbers changed since this plan was written: real-platform init is now **~40-50 s against the
90 s budget** (Phase 1 landed with real margin, not just enough to scrape by), and the zip is at
**~4.6 MB of 50 MB** (`submission.zip`: 4,595,151 B unzipped). Both budgets have real headroom now.
The temptation is to spend them because they're there — resist that and rank by expected payoff per
hour of runway left, not by which budget looks emptiest.

### 4.1 Texel-tune the existing eval *(top pick)*

The self-play + position-extraction pipeline built for Phase 3 (`tools/nnue_selfplay_fast.py`,
checkpointing, in-process bulk generation) is reusable for something with much better odds: tuning
the **existing** handcrafted eval's constants — PST values, mobility weights, threat/pin/king-safety
bonuses, `ROOK_OPEN_BONUS` and friends in `evaluate.py` — via logistic regression against real game
outcomes (classic Texel tuning), not training a new architecture from nothing. `docs/STATUS.md`
already flags some of these constants as hand-picked with visible overfitting symptoms (a source
comment on `ROOK_OPEN_BONUS` notes it was tuned *below* `ROOK_SEMI_OPEN_BONUS`, backwards from what
the term is meant to reward). Tens of parameters via regression is a fundamentally easier problem
than the ~400K-parameter net Phase 3 just lost with, and it improves code already proven to work
instead of competing against it. Costs no init time (pure offline tuning) and no zip space (same
constants, better values). Given Phase 3's result, this is the highest-probability way to convert
the data-generation investment already made into real Elo before the deadline.

### 4.2 Spend the init-time margin (~40-50 s of runway) on search-side additions

- **Continuation history** — a second move-ordering signal indexed by (piece type, to-square)
  across the last 1-2 plies, not just the from/to counter-move table Tier 3 already has. A proven
  Elo source in real engines, pure move-ordering, no correctness risk to verify beyond the existing
  differential-test pattern.
- **Internal Iterative Deepening** for nodes with no TT hash move — `docs/STATUS.md`'s own "Known
  risk" section already names this (alongside the counter-move heuristic, which is done) as the
  next candidate if init margin opened up again. It has.
- **Double the TT again** — `search.py:415` `TT_BUCKETS = 1 << 23` (~320 MB) → `1 << 24` (~640 MB).
  Zero init cost, pure RAM, and this exact move has been made twice already (Tier 14, Tier 16) with
  measured-neutral cost both times. Diminishing returns this far up, but cheap to check.
- **Eager numba signatures** on `negamax`/`quiescence`/`search_root`/`make_move` (mentioned, never
  done, in 1.1). Not strength — insurance. Makes the exact duplicate-specialisation regression that
  cost Tier 16 its margin structurally impossible instead of merely absent, and there is now enough
  spare margin to afford being defensive about it.

### 4.3 Spend the zip-space margin (~45 MB free)

- **A small opening book.** A few hundred KB to low single-digit MB of known theory for the first
  6-10 plies removes early-game risk entirely and costs nothing at runtime — a flat lookup, same
  shape as `weights/attacks.npz`. Not explicitly addressed by `AGENTS.md:61-64` (the ban is on
  shipping/running another *engine*; static opening data is the same category Syzygy tables and
  NNUE weights already occupy), but check the live rules page before investing time in it, the same
  "fetch before you rely on a number" instinct that caught the stale 60 s init figure earlier.
- **NNUE, retried with deeper labels, if there is runway left.** The 0-5 result points at label
  depth as the bottleneck, not sample count — a retry would mean depth 6+ labels at fewer games, not
  depth 4 at more of them. Highest cost, least certain payoff of anything on this list this close to
  the deadline; do 4.1 first.
- **5-man Syzygy — deprioritised, not just declined again.** The full 5-man set is ~950 MB, nowhere
  near the 50 MB cap regardless of how much margin exists; only a hand-picked subset of endings
  could ever fit, which is a lot of curation for endgames that rarely come up. Lower expected payoff
  than 4.1 or 4.2 for the effort.

**If only one thing gets done from this phase: 4.1.** It is the best risk-adjusted use of both the
time remaining and the data-generation work already sunk into Phase 3.

---

## What gets removed (summary)

| Item | Where | Why |
|---|---|---|
| Lazy SMP, entire path | `agent.py:39-51,95,200-201,231-280`, `search.py:252-278,1323`, `tests/test_lazy_smp.py` | Unreachable at `SMP_THREADS = 1` |
| `evaluate.warm_up()` | `evaluate.py:784-790` | Never called |
| `pv_from/pv_to/pv_promo` params | `search.py:1242-1244` | Unreferenced in the body |
| Duplicate `_bit_scan`, `_popcount64` | `search.py:456`, `evaluate.py:438`, `movegen.py:91`, `evaluate.py:256` | Two copies each; replaced by 2.2 |
| Redundant `_warm_up` calls | `agent.py:371,374,378` | Already compiled by line 365 |
| Redundant leaf `generate_legal` | `search.py:863` | Repeated verbatim in `quiescence` |
| Magic-table build loop | `attacks.py:237-242` | Shipped precomputed instead |
| `np.full(-1)` TT init | `search.py:1429-1435` | 117 MB of eager writes for a sentinel |
| Handcrafted eval terms | `evaluate.py` | **Only if Phase 3 NNUE wins its arena run** |

Not touched: `baselines/`, `docs/`, `tests/` (except `test_lazy_smp.py`) — none of it ships
(`harness/package.py:12-27` takes only root `*.py` plus `weights/`).

---

## Verification

Run after **every** item, not just at the end:

1. `.venv/Scripts/python.exe tests/bench_init.py` — total init, per-function compile breakdown, and
   **the signature table must show exactly one specialisation per function**.
2. `.venv/Scripts/python.exe tests/bench_nodes.py` — nodes, node rate, depth reached on the fixed
   position set. Any Phase 2 item that does not improve this gets reverted.
3. The existing 13 test scripts in `tests/` (they are plain `main()` scripts, not pytest):
   `perft.py` first — it is the differential movegen oracle and the only thing that will catch a
   `generate_pseudo_legal` split going wrong.
4. `make gate` — `ruff check .`, `mypy`, then 2 games vs `baselines/random`. **This is currently
   failing on the 60 s harness init gate and is the first thing that should start passing again.**
5. `make arena` — 20 games vs `baselines/greedy` for anything that changes play, and a longer run
   (≥ 40 games) for the aspiration/branching/ordering changes in 2.7 and for the NNUE decision.
6. `make zip` — confirm `agent.py` at the root and the unzipped total (with `attacks.npz` and, later,
   `nnue.npz`) stays under 50,000,000 B. Note `harness/package.py:50-59` only **warns** and still exits
   0, so the number has to be read, not trusted.

Two workflow notes worth fixing while in here: `pyproject.toml:37` sets `files = ["agent.py", "harness"]`,
so a bare `mypy` never directly checks `search.py`, `evaluate.py`, `movegen.py` and the rest despite the
docs treating `mypy --strict` as gating them; and `.github/workflows/ci.yml` never runs `tests/` at all.

---

## Sequencing

1. **Phase 0** — benchmarks committed, baseline recorded. *(half a day)*
2. **Phase 1.1** — signature collapse. Measure. This is the item most likely to solve the problem alone.
3. **Phase 1.4, 1.5, 1.6, 1.7, 1.8** — cheap, independent, low risk.
4. **Phase 1.2, 1.3** — the splits, only if 1.1 + the cheap items have not bought enough margin.
5. **Ship.** A build that passes `make gate` and clears the budget with margin is worth more than a
   stronger build that fails validation.
6. **Phase 2** — one item at a time, benched.
7. **Phase 3** — NNUE, flagged, shipped only on arena evidence. **Done: ran, lost 0-5, did not
   ship** — see "Phase 3 result" above.
8. **Phase 4** — rank by expected payoff on the margin actually available now, not by which budget
   looks emptiest. Texel-tuning (4.1) first.

Also worth doing once Phase 1 lands: update `docs/STATUS.md` with the real per-function compile
numbers, since the current entry (`STATUS.md:880-885`) attributes the cost to function size alone and
misses the duplicate-specialisation mechanism entirely.
