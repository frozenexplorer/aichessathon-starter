"""The submission entrypoint. The platform imports this file and calls get_move.

Movegen, evaluation and search run on a bitboard representation (bitboard.py / movegen.py /
attacks.py / evaluate.py / search.py), jitted with numba, not on python-chess objects -- see
docs/IDEAS.md and the differential tests in tests/perft.py for why and how that was verified.
"""

import time

import bitboard as bbm
import movegen as mg
import search as sr
import timeman

MATE_THRESHOLD = sr.MATE - 64
MAX_DEPTH = 64


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    start = time.perf_counter()
    bb, meta = bbm.from_fen(fen)

    from_arr, to_arr, promo_arr, count = mg.generate_legal(bb, meta)
    f, t, p = sr.quick_best_move(bb, meta, from_arr, to_arr, promo_arr, count)
    best_from, best_to, best_promo = int(f), int(t), int(p)

    if time_left_ms <= timeman.PANIC_MS:
        # No tree search: even depth 1 can cascade through thousands of quiescence nodes on a
        # tactical position before the first in-search time check, which is more than we can
        # spend when the clock itself is this low.
        return bbm.move_uci(best_from, best_to, best_promo)

    deadline = start + timeman.budget_ms(time_left_ms) / 1000.0

    pv_from, pv_to, pv_promo = -1, -1, -1
    depth = 1
    while True:
        counters = sr.new_counters()
        f, t, p, score, completed = sr.search_root(
            bb, meta, depth, deadline, counters, pv_from, pv_to, pv_promo
        )
        if f != -1:
            best_from, best_to, best_promo = int(f), int(t), int(p)
        if not completed:
            break
        pv_from, pv_to, pv_promo = f, t, p
        if time.perf_counter() >= deadline or abs(score) >= MATE_THRESHOLD or depth >= MAX_DEPTH:
            break
        depth += 1

    return bbm.move_uci(best_from, best_to, best_promo)


def _warm_up() -> None:
    """Pay the numba compile cost here, inside the 60s init budget, not on the match clock."""
    bb, meta = bbm.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    counters = sr.new_counters()
    deadline = time.perf_counter() + 30.0
    sr.search_root(bb, meta, 2, deadline, counters, -1, -1, -1)
    from_arr, to_arr, promo_arr, count = mg.generate_legal(bb, meta)
    sr.quick_best_move(bb, meta, from_arr, to_arr, promo_arr, count)
    new_bb, new_meta = mg.make_move(bb, meta, int(from_arr[0]), int(to_arr[0]), int(promo_arr[0]))
    sr.quiescence(new_bb, new_meta, -sr.INF, sr.INF, time.perf_counter() + 5.0, counters, 4)


_warm_up()
