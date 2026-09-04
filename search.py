"""Negamax with alpha-beta, iterative deepening, MVV-LVA move ordering and quiescence search.

No transposition table yet -- deliberately cut for the first working version under time
pressure. Ordering (root PV move first, then MVV-LVA captures) and quiescence are the pieces
that matter most at Python-scale node counts, per docs/IDEAS.md, and a TT is a safe fast-follow
once this is proven correct and robust.

Time safety: numba cannot call time.perf_counter() directly in nopython mode, so the search
drops back to Python via `objmode` every 128 nodes to check a wall-clock deadline. On abort it
propagates a sentinel up immediately rather than unwinding cleanly -- cheap because the check is
frequent enough that the overrun past the deadline is small and bounded by remaining call depth.
A root search that aborts before finishing is discarded entirely by the caller (timeman-driven
iterative deepening in agent.py), never returned as a partial, possibly-misordered result.

Below timeman.PANIC_MS remaining, agent.py skips the tree search altogether and plays
quick_best_move instead: even a depth-1 search can cascade through thousands of quiescence
nodes on a tactical position before the first check point, and at a few tens of milliseconds of
clock left that alone can be enough to flag. quick_best_move never recurses, so it cannot.

Repetition: negamax and search_root take a shared `history` array of zobrist hashes plus
`hist_len`, the count of entries that precede the current node. get_move only ever sees
positions where it is our own turn, and a real repeating shuffle shows up on both sides in
lockstep, so only our-turn positions (even ply, root = ply 0) are ever recorded or checked --
the array is indexed by "how many our-turn positions have occurred so far on the current path",
which a plain depth-first walk keeps correct with no explicit push/pop: a node writes its own
slot once, and a later sibling at the same ply simply overwrites it when the search backtracks.
agent.py owns the persistent real-game prefix and writes the root's own slot before calling
search_root; everything at ply >= 1 is written by negamax itself as it descends. If a line would
make a position recur a third time, it scores as an immediate draw (0) instead of running eval
or search on it further -- alpha-beta then avoids it on its own when winning (0 loses to a
positive score) and walks into it when losing (0 beats a negative one).
"""

import time

import numpy as np
from numba import njit, objmode

from bitboard import PAWN, QUEEN
from evaluate import PIECE_VALUE, evaluate
from movegen import generate_legal, is_check, make_move, piece_type_at
from zobrist import position_hash

MATE = 1_000_000
INF = 2_000_000
CHECK_INTERVAL = 127
QUIESCENCE_MAX_PLIES = 24
ONE = np.uint64(1)

# Our-turn positions per game are bounded by rules.PLY_CAP / 2 (~150); in-search growth is
# bounded by MAX_DEPTH / 2 (agent.py caps depth at 64, so ~32 more). 512 leaves ample headroom.
HISTORY_CAPACITY = 512


@njit(cache=False)
def _time_up(deadline: float, counters: np.ndarray) -> bool:
    if counters[0] & CHECK_INTERVAL == 0:
        with objmode(now="float64"):
            now = time.perf_counter()
        if now >= deadline:
            counters[1] = 1
    return bool(counters[1] != 0)


@njit(cache=False)
def is_capture(bb: np.ndarray, meta: np.ndarray, from_sq: int, to_sq: int) -> bool:
    opponent = 1 - meta[0]
    to_bit = ONE << np.uint64(to_sq)
    for pt in range(6):
        if bb[opponent * 6 + pt] & to_bit:
            return True
    if to_sq == meta[5] and piece_type_at(bb, meta[0], from_sq) == PAWN:
        return (from_sq % 8) != (to_sq % 8)
    return False


@njit(cache=False)
def _move_score(
    bb: np.ndarray,
    meta: np.ndarray,
    from_sq: int,
    to_sq: int,
    promo: int,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
) -> int:
    if from_sq == pv_from and to_sq == pv_to and promo == pv_promo:
        return 1_000_000

    score = 0
    if promo >= 0:
        score += 500 + PIECE_VALUE[promo]

    opponent = 1 - meta[0]
    victim_pt = piece_type_at(bb, opponent, to_sq)
    if victim_pt >= 0:
        attacker_pt = piece_type_at(bb, meta[0], from_sq)
        score += 10_000 + PIECE_VALUE[victim_pt] * 10 - PIECE_VALUE[attacker_pt]
    elif to_sq == meta[5] and piece_type_at(bb, meta[0], from_sq) == PAWN:
        if (from_sq % 8) != (to_sq % 8):
            score += 10_000 + PIECE_VALUE[PAWN] * 10 - PIECE_VALUE[PAWN]
    return score


@njit(cache=False)
def _score_moves(
    bb: np.ndarray,
    meta: np.ndarray,
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    promo_arr: np.ndarray,
    count: int,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
) -> np.ndarray:
    scores = np.empty(count, dtype=np.int64)
    for i in range(count):
        scores[i] = _move_score(
            bb, meta, from_arr[i], to_arr[i], promo_arr[i], pv_from, pv_to, pv_promo
        )
    return scores


@njit(cache=False)
def quiescence(
    bb: np.ndarray,
    meta: np.ndarray,
    alpha: int,
    beta: int,
    deadline: float,
    counters: np.ndarray,
    qdepth: int,
) -> int:
    counters[0] += 1
    if _time_up(deadline, counters):
        return 0

    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    if count == 0:
        return -MATE if is_check(bb, meta) else 0

    stand_pat = evaluate(bb, meta, count)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth <= 0:
        return alpha

    scores = np.full(count, -1, dtype=np.int64)
    ncap = 0
    for i in range(count):
        f, t, p = from_arr[i], to_arr[i], promo_arr[i]
        if is_capture(bb, meta, f, t) or p == QUEEN:
            scores[i] = _move_score(bb, meta, f, t, p, -1, -1, -1)
            ncap += 1
    if ncap == 0:
        return alpha

    order = np.argsort(-scores)
    for oi in range(ncap):
        idx = order[oi]
        f, t, p = from_arr[idx], to_arr[idx], promo_arr[idx]
        new_bb, new_meta = make_move(bb, meta, f, t, p)
        score = -quiescence(new_bb, new_meta, -beta, -alpha, deadline, counters, qdepth - 1)
        if counters[1]:
            return 0
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def negamax(
    bb: np.ndarray,
    meta: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    deadline: float,
    counters: np.ndarray,
    ply: int,
    history: np.ndarray,
    hist_len: int,
) -> int:
    counters[0] += 1
    if _time_up(deadline, counters):
        return 0

    if ply % 2 == 0:
        h = position_hash(bb, meta)
        matches = 0
        for i in range(hist_len):
            if history[i] == h:
                matches += 1
        if matches >= 2:
            return 0
        history[hist_len] = h
        child_hist_len = hist_len + 1
    else:
        child_hist_len = hist_len

    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    if count == 0:
        return -MATE if is_check(bb, meta) else 0

    if depth <= 0:
        return quiescence(bb, meta, alpha, beta, deadline, counters, QUIESCENCE_MAX_PLIES)

    scores = _score_moves(bb, meta, from_arr, to_arr, promo_arr, count, -1, -1, -1)
    order = np.argsort(-scores)

    best = -INF
    for oi in range(count):
        idx = order[oi]
        f, t, p = from_arr[idx], to_arr[idx], promo_arr[idx]
        new_bb, new_meta = make_move(bb, meta, f, t, p)
        score = -negamax(
            new_bb, new_meta, depth - 1, -beta, -alpha, deadline, counters,
            ply + 1, history, child_hist_len,
        )
        if counters[1]:
            return 0
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


@njit(cache=False)
def search_root(
    bb: np.ndarray,
    meta: np.ndarray,
    depth: int,
    deadline: float,
    counters: np.ndarray,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
    history: np.ndarray,
    hist_len: int,
) -> tuple[int, int, int, int, bool]:
    """hist_len counts our-turn positions already recorded, including the root's own -- the
    caller (agent.py) writes history[hist_len - 1] = hash(bb, meta) before calling this, since it
    already needs that hash for the persistent cross-move history anyway.
    """
    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    if count == 0:
        return -1, -1, -1, 0, False

    scores = _score_moves(bb, meta, from_arr, to_arr, promo_arr, count, pv_from, pv_to, pv_promo)
    order = np.argsort(-scores)

    alpha = -INF
    beta = INF
    best_score = -INF
    best_from, best_to, best_promo = from_arr[order[0]], to_arr[order[0]], promo_arr[order[0]]

    for oi in range(count):
        idx = order[oi]
        f, t, p = from_arr[idx], to_arr[idx], promo_arr[idx]
        new_bb, new_meta = make_move(bb, meta, f, t, p)
        score = -negamax(
            new_bb, new_meta, depth - 1, -beta, -alpha, deadline, counters, 1, history, hist_len
        )
        if counters[1]:
            return best_from, best_to, best_promo, best_score, False
        if score > best_score:
            best_score = score
            best_from, best_to, best_promo = f, t, p
        if best_score > alpha:
            alpha = best_score

    return best_from, best_to, best_promo, best_score, True


@njit(cache=False)
def quick_best_move(
    bb: np.ndarray,
    meta: np.ndarray,
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    promo_arr: np.ndarray,
    count: int,
) -> tuple[int, int, int]:
    """The best-looking legal move by MVV-LVA/promotion score alone, no search. Never recurses,
    so it has no way to overrun a deadline -- the fallback for when there is no time to search.
    """
    scores = _score_moves(bb, meta, from_arr, to_arr, promo_arr, count, -1, -1, -1)
    best_idx = 0
    best_score = scores[0]
    for i in range(1, count):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    return from_arr[best_idx], to_arr[best_idx], promo_arr[best_idx]


def new_counters() -> np.ndarray:
    return np.zeros(2, dtype=np.int64)
