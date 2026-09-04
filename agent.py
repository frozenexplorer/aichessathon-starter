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

Once few enough pieces remain, tablebase.best_moves narrows the root to WDL-optimal candidates
(see tablebase.py) and _search_restricted picks among just those with the normal search -- still
eval- and repetition-aware, just guaranteed never to concede a win or a drawable position once
the tablebase has a definitive read. Any failure there (missing files, a position out of range)
returns None and this falls straight back to the unrestricted search below, so a bug in the
tablebase path can only cost the optimization, never the game.
"""

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

_history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
_history_len = 0
_opponent_history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
_opponent_history_len = 0


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    global _history_len, _opponent_history_len

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

    deadline = start + timeman.budget_ms(time_left_ms) / 1000.0
    hist_len = _history_len + 1

    if allowed:
        best_from, best_to, best_promo = _search_restricted(bb, meta, allowed, deadline, hist_len)
    else:
        pv_from, pv_to, pv_promo = -1, -1, -1
        depth = 1
        while True:
            counters = sr.new_counters()
            f, t, p, score, completed = sr.search_root(
                bb, meta, depth, deadline, counters, pv_from, pv_to, pv_promo,
                _history, hist_len, _opponent_history, _opponent_history_len,
            )
            if f != -1:
                best_from, best_to, best_promo = int(f), int(t), int(p)
            if not completed:
                break
            pv_from, pv_to, pv_promo = f, t, p
            over_time = time.perf_counter() >= deadline
            if over_time or abs(score) >= MATE_THRESHOLD or depth >= MAX_DEPTH:
                break
            depth += 1

    best_from, best_to, best_promo = _record_and_return(bb, meta, best_from, best_to, best_promo)
    return bbm.move_uci(best_from, best_to, best_promo)


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
) -> tuple[int, int, int]:
    """Iterative deepening restricted to `moves` (already filtered to WDL-optimal by the
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
            new_bb, new_meta = mg.make_move(bb, meta, f, t, p)
            score = -sr.negamax(
                new_bb, new_meta, depth - 1, -sr.INF, sr.INF, deadline, counters, 1, _history,
                hist_len,
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
    sr.search_root(bb, meta, 2, deadline, counters, -1, -1, -1, history, 1, opponent_history, 0)
    from_arr, to_arr, promo_arr, count = mg.generate_legal(bb, meta)
    sr.quick_best_move(bb, meta, from_arr, to_arr, promo_arr, count)
    new_bb, new_meta = mg.make_move(bb, meta, int(from_arr[0]), int(to_arr[0]), int(promo_arr[0]))
    sr.quiescence(new_bb, new_meta, -sr.INF, sr.INF, time.perf_counter() + 5.0, counters, 4)
    sr.claim_eligible_for_opponent(new_bb, new_meta, history, 1, opponent_history, 0, 1)
    tb.best_moves("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")


_warm_up()
