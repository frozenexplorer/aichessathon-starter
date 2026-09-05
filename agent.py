"""The submission entrypoint. The platform imports this file and calls get_move.

Movegen, evaluation and search run on a bitboard representation (bitboard.py / movegen.py /
attacks.py / evaluate.py / search.py), jitted with numba, not on python-chess objects -- see
docs/IDEAS.md and the differential tests in tests/perft.py for why and how that was verified.

Two arrays persist across get_move calls in the same game (the process stays alive between our
own moves, never between games), tracking two independent ways the harness's referee can
auto-draw a won game before we are ever asked to move again -- see search.py's module docstring
for the mechanism and the two real games that exposed each one:

_history: our own turn recurring, or the opponent having a reply available that would make it
recur. Grows by 1 per call (rules.PLY_CAP caps a game at 300 plies, nowhere near
search.HISTORY_CAPACITY) and search only ever writes at indices up to hist_len + MAX_DEPTH, so
this never runs off the end of the array.

_opponent_history: the position we hand the opponent recurring on its own, independent of
anything on our side of the board. One appended per move we actually play (not extended during
search, since only the real, played sequence matters for what has actually occurred) -- so it
grows at exactly the same rate as _history and is bounded the same way.

Once few enough pieces remain, tablebase.best_moves narrows the root to WDL/DTZ-optimal
candidates (see tablebase.py) and _search_restricted picks among just those with the normal
search -- still eval- and repetition-aware, just guaranteed never to concede a win or a drawable
position once the tablebase has a definitive read. Any failure there (missing files, a position
out of range) returns None and this falls straight back to the unrestricted search below, so a
bug in the tablebase path can only cost the optimization, never the game.

_tt_*: the transposition table (search.py), a fixed-size array-based hash table. Persists across
the whole game like the history arrays above, since positions can transpose across our own move
choices even without repeating outright -- but unlike them, it never affects correctness: a stale
or colliding entry can only cost some search accuracy, never mask a real repetition (negamax
checks that first and never touches the table for a forced-draw node) or destabilize the
claim-eligibility safety net above (which never consults it at all). Killer moves, the from/to
history table, and the counter-move table are cheap move-ordering aids that encode this search's
own cutoff history, not the position, so they are rebuilt fresh every call rather than carried
across moves.

Lazy SMP: alongside the main thread's own iterative-deepening loop below (unchanged from before,
still the only thread whose move is ever returned), `_spawn_helpers` starts up to
SMP_THREADS - 1 helper threads each running the same search_root against the same _tt_* arrays --
real concurrent execution, not just interleaved, since search.py's search_root is compiled
nogil=True (see its module docstring for why that is safe and confirmed to work on this numba
version before relying on it here). Helper threads exist purely to enrich the shared TT while the
main thread searches; their own return values are discarded. Each helper gets its own COPY of
_history/_opponent_history (search.py's negamax writes further entries onto whatever history array
it is given as it descends, so two threads sharing one array would corrupt each other's
in-search repetition detection -- a real correctness hazard, unlike the TT's tolerable-by-design
races) and its own fresh killer/history/counter-move tables and counters, never shared. Helpers
target `base_deadline` (never the volatility-extended budget), so joining them after the main
thread's own loop finishes is always a short, bounded wait, never the reason a move is late.

Branching-factor early stop: before starting depth N+1, the main loop compares the time depth N
itself took against the time actually remaining under the deadline that iteration would use --
skip it, and return the current best immediately, if even BRANCHING_ESTIMATE times as long would
not fit. An aborted root search is discarded entirely regardless (search.py's own docstring), so
starting a depth with no real chance of finishing only wastes clock that a future move could have
used instead; BRANCHING_ESTIMATE is deliberately conservative so this only ever skips depths that
were very unlikely to complete anyway.
"""

import os
import threading
import time

import numpy as np

import bitboard as bbm
import movegen as mg
import search as sr
import tablebase as tb
import timeman
import zobrist as zb

MATE_THRESHOLD = sr.MATE - 64
MAX_DEPTH = 64

# A search_root call that hits its deadline mid-depth is discarded entirely (search.py's own
# docstring: "a root search that aborts before finishing is discarded... never returned as a
# partial, possibly-misordered result"), so starting a depth almost certain to abort wastes that
# time for nothing -- better to stop one depth early and bank the time for a future move. 4x is a
# deliberately conservative multiple (real effective branching factor with TT/killers/history
# ordering is usually well under this), so this only ever skips a depth that is very unlikely to
# have completed anyway, not one with a real chance.
BRANCHING_ESTIMATE = 4

# Capped modestly rather than at os.cpu_count() itself: the platform's real core count and load
# are unknown, TT contention gives diminishing returns past a handful of threads anyway (see the
# nogil-parallelism benchmark this was validated against before adding), and a lower number leaves
# headroom for the harness/OS on whatever hardware this actually lands on.
SMP_THREADS = max(1, min(4, os.cpu_count() or 1))

_history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
_history_len = 0
_opponent_history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
_opponent_history_len = 0

(_tt_key, _tt_depth, _tt_score, _tt_flag, _tt_from, _tt_to, _tt_promo) = sr.new_tt()

# Piece count as of the previous get_move call's root, for the adaptive-time volatility check
# below (_is_volatile) -- None until the first real move decision.
_prev_piece_count: int | None = None


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    global _history_len, _opponent_history_len, _prev_piece_count

    start = time.perf_counter()
    bb, meta = bbm.from_fen(fen)

    from_arr, to_arr, promo_arr, count = mg.generate_legal(bb, meta)
    f, t, p = sr.quick_best_move(bb, meta, from_arr, to_arr, promo_arr, count)
    best_from, best_to, best_promo = int(f), int(t), int(p)

    _history[_history_len] = zb.position_hash(bb, meta)

    allowed: list[tuple[int, int, int]] | None = None
    if tb.piece_count(bb) <= tb.MAX_PIECES:
        try:
            allowed = tb.best_moves(fen)
        except Exception:
            allowed = None
        if allowed:
            best_from, best_to, best_promo = allowed[0]

    if time_left_ms <= timeman.PANIC_MS:
        # No tree search: even depth 1 can cascade through thousands of quiescence nodes on a
        # tactical position before the first in-search time check, which is more than we can
        # spend when the clock itself is this low. A tablebase pick needs no search at all, so
        # it is still safe to use here.
        best_from, best_to, best_promo = _record_and_return(
            bb, meta, best_from, best_to, best_promo
        )
        return bbm.move_uci(best_from, best_to, best_promo)

    base_deadline = start + timeman.budget_ms(time_left_ms) / 1000.0
    max_deadline = start + timeman.extended_budget_ms(time_left_ms) / 1000.0
    hist_len = _history_len + 1
    killer_from, killer_to, killer_promo = sr.new_killers()
    history_table = sr.new_history_table()
    counter_from, counter_to, counter_promo = sr.new_counter_table()
    piece_count = tb.piece_count(bb)
    halfmove_clock = bbm.halfmove_clock(fen)

    if allowed:
        best_from, best_to, best_promo = _search_restricted(
            bb, meta, allowed, max_deadline, hist_len, killer_from, killer_to, killer_promo,
            history_table, counter_from, counter_to, counter_promo, halfmove_clock,
        )
    else:
        helper_threads = _spawn_helpers(
            bb, meta, base_deadline, hist_len, _opponent_history_len, halfmove_clock
        )
        pv_from, pv_to, pv_promo = -1, -1, -1
        prev_score: int | None = None
        prev_iteration_elapsed: float | None = None
        depth = 1
        while True:
            volatile = _is_volatile(bb, meta, prev_score, None, piece_count, _prev_piece_count)
            deadline = max_deadline if volatile else base_deadline
            if prev_iteration_elapsed is not None:
                remaining = deadline - time.perf_counter()
                if remaining < prev_iteration_elapsed * BRANCHING_ESTIMATE:
                    break
            counters = sr.new_counters()
            window_seed = sr.NO_PREV_SCORE if prev_score is None else prev_score
            iteration_start = time.perf_counter()
            f, t, p, score, completed = sr.search_root(
                bb, meta, depth, deadline, counters, pv_from, pv_to, pv_promo,
                _history, hist_len, _opponent_history, _opponent_history_len,
                _tt_key, _tt_depth, _tt_score, _tt_flag, _tt_from, _tt_to, _tt_promo,
                killer_from, killer_to, killer_promo, history_table,
                counter_from, counter_to, counter_promo, window_seed, halfmove_clock,
            )
            prev_iteration_elapsed = time.perf_counter() - iteration_start
            if f != -1:
                best_from, best_to, best_promo = int(f), int(t), int(p)
            if not completed:
                break
            pv_from, pv_to, pv_promo = f, t, p
            still_volatile = _is_volatile(
                bb, meta, prev_score, score, piece_count, _prev_piece_count
            )
            next_deadline = max_deadline if still_volatile else base_deadline
            over_time = time.perf_counter() >= next_deadline
            if over_time or abs(score) >= MATE_THRESHOLD or depth >= MAX_DEPTH:
                break
            prev_score = score
            depth += 1
        for helper in helper_threads:
            helper.join()

    _prev_piece_count = piece_count
    best_from, best_to, best_promo = _record_and_return(bb, meta, best_from, best_to, best_promo)
    return bbm.move_uci(best_from, best_to, best_promo)


def _is_volatile(
    bb: np.ndarray,
    meta: np.ndarray,
    prev_score: int | None,
    score: int | None,
    piece_count: int,
    prev_piece_count: int | None,
) -> bool:
    """Whether the position looks sharp enough to deserve more than the base time budget -- see
    docs/FUTURE.md item 1: a score swing between the last two completed iterative-deepening
    depths, a position not in a quiet state (in check, a capture just landed us here), or few
    enough pieces left that precise endgame play matters.
    """
    swing = None if prev_score is None or score is None else abs(score - prev_score)
    if swing is not None and swing >= timeman.SCORE_SWING_CP:
        return True
    if mg.is_check(bb, meta):
        return True
    if prev_piece_count is not None and piece_count < prev_piece_count:
        return True
    return piece_count <= timeman.LOW_PIECE_COUNT


def _helper_worker(
    bb: np.ndarray,
    meta: np.ndarray,
    deadline: float,
    hist_len: int,
    opponent_hist_len: int,
    halfmove_clock: int,
) -> None:
    """One Lazy SMP helper thread: its own iterative-deepening loop against the shared _tt_*
    arrays, own history copies and move-ordering tables (see this module's docstring), own return
    value discarded -- it exists only to enrich the shared TT while the main thread's own loop
    (unchanged, still the only source of the move actually played) runs concurrently.
    """
    history_copy = _history.copy()
    opponent_history_copy = _opponent_history.copy()
    killer_from, killer_to, killer_promo = sr.new_killers()
    history_table = sr.new_history_table()
    counter_from, counter_to, counter_promo = sr.new_counter_table()
    counters = sr.new_counters()
    depth = 1
    while True:
        _f, _t, _p, score, completed = sr.search_root(
            bb, meta, depth, deadline, counters, -1, -1, -1,
            history_copy, hist_len, opponent_history_copy, opponent_hist_len,
            _tt_key, _tt_depth, _tt_score, _tt_flag, _tt_from, _tt_to, _tt_promo,
            killer_from, killer_to, killer_promo, history_table,
            counter_from, counter_to, counter_promo, sr.NO_PREV_SCORE, halfmove_clock,
        )
        if not completed or abs(score) >= MATE_THRESHOLD or depth >= MAX_DEPTH:
            break
        depth += 1


def _spawn_helpers(
    bb: np.ndarray,
    meta: np.ndarray,
    deadline: float,
    hist_len: int,
    opponent_hist_len: int,
    halfmove_clock: int,
) -> list[threading.Thread]:
    threads = []
    for _ in range(SMP_THREADS - 1):
        thread = threading.Thread(
            target=_helper_worker,
            args=(bb, meta, deadline, hist_len, opponent_hist_len, halfmove_clock),
        )
        thread.start()
        threads.append(thread)
    return threads


def _record_and_return(
    bb: np.ndarray, meta: np.ndarray, best_from: int, best_to: int, best_promo: int
) -> tuple[int, int, int]:
    """Advance both history arrays for the move actually chosen, once, on every path out of
    get_move -- _history_len already counts the position just searched from (written above);
    _opponent_history records the position this move hands to the opponent.
    """
    global _history_len, _opponent_history_len
    new_bb, new_meta = mg.make_move(bb, meta, best_from, best_to, best_promo)
    _opponent_history[_opponent_history_len] = zb.position_hash(new_bb, new_meta)
    _opponent_history_len += 1
    _history_len += 1
    return best_from, best_to, best_promo


def _search_restricted(
    bb: np.ndarray,
    meta: np.ndarray,
    moves: list[tuple[int, int, int]],
    deadline: float,
    hist_len: int,
    killer_from: np.ndarray,
    killer_to: np.ndarray,
    killer_promo: np.ndarray,
    history_table: np.ndarray,
    counter_from: np.ndarray,
    counter_to: np.ndarray,
    counter_promo: np.ndarray,
    halfmove_clock: int,
) -> tuple[int, int, int]:
    """Iterative deepening restricted to `moves` (already filtered to WDL/DTZ-optimal by the
    tablebase) -- same shape as the main loop, just over a smaller candidate set, so eval and
    repetition-avoidance still pick the move that actually makes progress toward mate.
    """
    best = moves[0]
    depth = 1
    while True:
        counters = sr.new_counters()
        best_score = -sr.INF
        depth_best = moves[0]
        for f, t, p in moves:
            child_halfmove_clock = (
                0 if (sr.is_capture(bb, meta, f, t) or mg.piece_type_at(bb, meta[0], f) == bbm.PAWN)
                else halfmove_clock + 1
            )
            new_bb, new_meta = mg.make_move(bb, meta, f, t, p)
            score = -sr.negamax(
                new_bb, new_meta, depth - 1, -sr.INF, sr.INF, deadline, counters, 1, _history,
                hist_len, _tt_key, _tt_depth, _tt_score, _tt_flag, _tt_from, _tt_to, _tt_promo,
                killer_from, killer_to, killer_promo, history_table, True, sr.MAX_CHECK_EXTENSIONS,
                counter_from, counter_to, counter_promo, f, t, child_halfmove_clock, -1, -1,
            )
            if counters[1]:
                break
            if sr.claim_eligible_for_opponent(
                new_bb, new_meta, _history, hist_len, _opponent_history, _opponent_history_len, 1
            ):
                score = min(score, 0)
            if score > best_score:
                best_score = score
                depth_best = (f, t, p)
        if counters[1]:
            break
        best = depth_best
        over_time = time.perf_counter() >= deadline
        if over_time or abs(best_score) >= MATE_THRESHOLD or depth >= MAX_DEPTH:
            break
        depth += 1
    return best


def _warm_up() -> None:
    """Pay the numba compile cost here, inside the 60s init budget, not on the match clock."""
    bb, meta = bbm.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    counters = sr.new_counters()
    deadline = time.perf_counter() + 30.0
    history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
    history[0] = zb.position_hash(bb, meta)
    opponent_history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
    killer_from, killer_to, killer_promo = sr.new_killers()
    history_table = sr.new_history_table()
    counter_from, counter_to, counter_promo = sr.new_counter_table()
    sr.search_root(
        bb, meta, 2, deadline, counters, -1, -1, -1, history, 1, opponent_history, 0,
        _tt_key, _tt_depth, _tt_score, _tt_flag, _tt_from, _tt_to, _tt_promo,
        killer_from, killer_to, killer_promo, history_table,
        counter_from, counter_to, counter_promo, sr.NO_PREV_SCORE, 0,
    )
    from_arr, to_arr, promo_arr, count = mg.generate_legal(bb, meta)
    sr.quick_best_move(bb, meta, from_arr, to_arr, promo_arr, count)
    new_bb, new_meta = mg.make_move(bb, meta, int(from_arr[0]), int(to_arr[0]), int(promo_arr[0]))
    sr.quiescence(
        new_bb, new_meta, -sr.INF, sr.INF, time.perf_counter() + 5.0, counters, 4,
        sr.QSEARCH_CHECK_BUDGET,
    )
    sr.claim_eligible_for_opponent(new_bb, new_meta, history, 1, opponent_history, 0, 1)
    tb.best_moves("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")


_warm_up()
