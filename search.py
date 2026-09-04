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

Repetition, mechanism 1 -- our own turn recurring: negamax and search_root take a shared
`history` array of zobrist hashes plus `hist_len`, the count of entries that precede the current
node. Only our-turn positions (even ply, root = ply 0) are ever recorded or checked here -- the
array is indexed by "how many our-turn positions have occurred so far on the current path",
which a plain depth-first walk keeps correct with no explicit push/pop: a node writes its own
slot once, and a later sibling at the same ply simply overwrites it when the search backtracks.
agent.py owns the persistent real-game prefix and writes the root's own slot before calling
search_root; everything at ply >= 1 is written by negamax itself as it descends. If a line would
make a position recur a third time, it scores as an immediate draw (0) instead of running eval
or search on it further.

That alone is not sufficient, for two separate reasons, both found by a won game that kept
drawing (see agent.py's module docstring for the two real games). First: the harness's referee
checks Board.outcome(claim_draw=True) *before* asking either side for a move, and python-chess's
can_claim_threefold_repetition() fires the instant the side to move *has some legal reply* that
would create a third occurrence -- not only once one is actually played. Second: it also fires
when the *current* position (the one about to be handed over, the opponent's turn) has itself
already occurred twice before -- a repeat of an opponent-turn position counts on its own, not
only when it happens to line up with one of ours.

claim_eligible_for_opponent checks both: condition two is a direct lookup against
`opponent_history`, the real game's own past opponent-turn positions (one appended per move we
actually play, in agent.py -- not extended during search, since only the real, played sequence
matters for what has actually occurred); condition one is one extra generate_legal on the
position we would hand over, matching each of the opponent's replies against `history`. Either
way, search_root / agent._search_restricted cap that candidate move's score at 0, so a move that
merely offers the position is treated the same as one that plays it. Root-only, not inside
negamax's own recursion: condition one is a full extra movegen per candidate move, affordable
once per real move decision but not throughout the tree.
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
def claim_eligible_for_opponent(
    bb: np.ndarray,
    meta: np.ndarray,
    history: np.ndarray,
    hist_len: int,
    opponent_history: np.ndarray,
    opponent_hist_len: int,
    lookahead: int,
) -> bool:
    """True if handing over the position `bb`/`meta` (the opponent's turn, right after one of
    our candidate moves) would let the harness's referee auto-draw before we are ever asked to
    move again -- python-chess's can_claim_threefold_repetition(), which Board.outcome(claim_draw
    =True) calls before asking either side for a move, matches on two conditions:

    (a) this exact position has already occurred twice before (this handoff would be the 3rd) --
        checked directly against opponent_history, the real game's own past opponent-turn
        positions (recorded once per move we actually play, in agent.py).
    (b) the opponent has some legal reply that would make a position recur a third time --
        checked by generating their replies and matching each against `history`, our own
        real-plus-in-search-so-far turn positions (see this module's docstring).

    Neither depends on what the opponent would actually choose to play, so a move of ours
    creating either condition must be scored as if it directly caused the draw.

    That is still not the whole story: the same auto-draw applies to *our* next position too, and
    it applies before we are ever consulted -- so even if handing over `bb`/`meta` looks safe, an
    opponent reply we do not control could land us on a position where the referee ends the game
    on our own next (still unconsulted) turn. `lookahead` recurses one call further per unit,
    with the two history arrays swapped, to check exactly that for each of the opponent's
    replies; two real games were needed to find these two conditions plus this recursive case
    between them (see agent.py's module docstring), so this is deliberately bounded rather than
    assumed complete for arbitrary depth -- lookahead=1, used at the call sites in search_root /
    agent._search_restricted, closes both games actually observed while keeping the cost bounded
    (branching^2 per candidate root move, still small at these endgame piece counts).

    Deliberately root-only (called from search_root / agent._search_restricted, not from inside
    negamax's own recursion): each level is a full extra generate_legal, affordable once per real
    move decision but not throughout the tree.
    """
    h = position_hash(bb, meta)
    matches = 0
    for j in range(opponent_hist_len):
        if opponent_history[j] == h:
            matches += 1
    if matches >= 2:
        return True

    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    for i in range(count):
        new_bb, new_meta = make_move(bb, meta, from_arr[i], to_arr[i], promo_arr[i])
        reply_hash = position_hash(new_bb, new_meta)
        reply_matches = 0
        for j in range(hist_len):
            if history[j] == reply_hash:
                reply_matches += 1
        if reply_matches >= 2:
            return True
        if lookahead > 0 and claim_eligible_for_opponent(
            new_bb, new_meta, opponent_history, opponent_hist_len, history, hist_len,
            lookahead - 1,
        ):
            return True
    return False


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
    opponent_history: np.ndarray,
    opponent_hist_len: int,
) -> tuple[int, int, int, int, bool]:
    """hist_len counts our-turn positions already recorded, including the root's own -- the
    caller (agent.py) writes history[hist_len - 1] = hash(bb, meta) before calling this, since it
    already needs that hash for the persistent cross-move history anyway. opponent_hist_len
    counts the real game's own past opponent-turn positions, one recorded per move we have
    actually played -- see claim_eligible_for_opponent.
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
        if claim_eligible_for_opponent(
            new_bb, new_meta, history, hist_len, opponent_history, opponent_hist_len, 1
        ):
            score = min(score, 0)
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
