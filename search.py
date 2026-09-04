"""Negamax with alpha-beta, iterative deepening, a transposition table, killer/history move
ordering, principal variation search (PVS), and quiescence search.

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

Transposition table: a fixed-size, two-tier hash table (parallel numpy arrays, not a dict -- numba
nopython mode has no fast Python dict), keyed on the low bits of the Zobrist hash with a full
64-bit key comparison to detect index collisions. Each bucket (TT_BUCKETS of them) has two raw
slots: a depth-preferred one, replaced only on a same-key refresh, an empty slot, or a search that
went at least as deep as what's already there, and an always-replace one that a fresh position
falls back to otherwise -- so a deep, expensive result isn't evicted by shallow, plentiful ones,
while a node from the current search is still never simply dropped for lack of a slot (see
_tt_resolve and the store logic at the end of negamax). Stores a depth, a score (bound-adjusted
for mate distance so a cached "mate in N from here" is still correct when reused at a different
ply), a bound type (exact/lower/upper), and the best move found, so a re-visited node can either
return immediately (sufficient stored depth, exact bound, or a bound that already causes a
cutoff) or at minimum reuse the stored move as the first one tried. Persists across the whole
game (agent.py owns the arrays, created once), since positions can transpose across our own
move choices even when they don't recur outright. It is deliberately not consulted for the
repetition-forced draw score (see below) -- that check runs first and returns before the table
is ever read or written for that node, so a cached score can never paper over an actual repeat.

Killer moves (two per ply), a from/to history table, and a counter-move table give cheap move
ordering for quiet moves that caused a beta cutoff elsewhere in the tree, on top of the
hash-move-first, then MVV-LVA capture ordering already in place. The counter-move table is indexed
by the move that led to a node (parent_from/parent_to, threaded through negamax) rather than by
ply or by the moving side's own from/to squares: it answers "what quiet reply most recently beat
this specific opponent move," a more targeted signal than the from/to history table's coarser
one. All three are rebuilt fresh per real move decision (agent.py allocates them per get_move
call), since they encode this search's own cutoff history, not anything about the position itself.

Principal variation search: the first move at each node (hash move or best-scoring by ordering)
is searched with the full alpha-beta window; every other move first gets a cheap null-window
probe (-alpha-1, -alpha) and is only re-searched with the full window if that probe suggests it
might actually beat alpha. Cuts the cost of nodes that ordering already got right, which the
hash-move-first and killer/history ordering above make the common case.

Aspiration windows: search_root's first pass at a given depth (depth > 2, and only once the
previous depth's score is a real, non-mate value) searches a narrow window centered on the
previous iterative-deepening depth's score rather than -INF..INF, on the premise that the score
rarely swings far in one more ply. A fail-high or fail-low re-searches the same depth with the
full window -- one extra pass in the rare case, never a correctness risk, since the final result
is only ever accepted once it comes from a pass whose window did not clip it.

Null-move pruning: at a non-leaf node deep enough (NULL_MOVE_MIN_DEPTH), negamax additionally
tries giving the side to move a free pass (make_null_move: same board, turn flipped, en passant
forfeited) and searching the rest at depth - 1 - NULL_MOVE_REDUCTION with a null window just above
beta. If even a free move for the opponent is not enough to drop the score below beta, the real
position is assumed to cut off too and the whole subtree is pruned. Two zugzwang guards, since a
free move can look safe in a position where every real move only makes things worse: skipped
while in check (the null move would leave an illegal, check-blind position), and skipped when the
side to move has no non-pawn material left (has_non_pawn_material) -- exactly the king+pawn
endgames zugzwang is common in, which is also the class of position DTZ was added to get right
(see tablebase.py). `allow_null` additionally forbids two null moves in a row (a null move whose
own child is itself another null-move probe proves nothing and just burns depth), threaded through
every real-move recursive call as True and through the null-move probe's own call as False.

Static exchange evaluation (see()): a real exchange calculation (least-valuable-attacker-first,
swap-list algorithm) replaces MVV-LVA's rough capture-ordering heuristic in _move_score, and
quiescence uses it to skip searching a capture outright once its SEE is negative -- a capture
that loses material cannot help quiescing the position, and exploring it anyway was the direct
cause of the 500K-2M+ node counts observed at depth 7-9 in sharp middlegames (see docs/STATUS.md).
Ignores pins, same simplification essentially every engine's SEE makes -- see() is a heuristic for
ordering and pruning, never a legality check, so a wrong answer there misorders or mis-prunes but
cannot make the search return an illegal move.

Delta pruning: quiescence additionally skips a capture once stand-pat plus its SEE, plus a safety
margin (DELTA_MARGIN), still can't reach alpha -- catching the case SEE-pruning alone misses, a
capture that genuinely wins material but not enough to close a large existing gap. Same
SEE-descending move order as SEE-pruning makes both a single monotonic break condition (see
quiescence's move loop): once either trips, every remaining, lower-scored capture trips it too.

Late move reductions: inside negamax's move loop (never at the root -- search_root explores every
root move at full depth), a quiet move late enough in the ordering (see LMR_MIN_MOVE_INDEX) gets
searched one ply shallower first, on the premise that TT/killer/history ordering has already
almost certainly put the moves worth full depth ahead of it. Only promotes to a full-depth
re-search if the reduced probe still beats alpha, and from there falls into the same full-window
PVS re-search as any other move that beats alpha -- so a reduction can only cost extra nodes
re-confirming a move, never silently accept a wrong score, since the final accepted score for any
move that raises alpha always comes from a full-depth search. Skipped entirely when the current
node or the resulting position is in check, or the move is a capture or promotion -- exactly the
tactical moves a shallower search is least equipped to judge.

Search extensions: a move that gives check gets its child searched at the same depth rather than
depth - 1 -- a full ply deeper than normal, since a forced reply to check is exactly the kind of
forcing line a depth-limited search most needs the extra ply for. ext_budget bounds how many of
these one line can stack (MAX_CHECK_EXTENSIONS), so a long forcing check sequence cannot inflate
one line's effective depth arbitrarily at every other line's expense within the same time budget.
Depth extends, never ply: the TT still stores against the exact depth value passed to that node,
so a cached entry is always compared like-for-like regardless of how much extension went into
reaching it, and the mate-distance adjustment (keyed on ply, not depth) is unaffected either way.

Futility pruning: at a shallow node (depth <= FUTILITY_MAX_DEPTH) not in check, a quiet, non-check
move is skipped outright once the static eval plus a depth-scaled margin still can't reach alpha
-- the position already looks bad enough that a quiet move is very unlikely to save it, so the
search cost is spent on captures/promotions/checks instead. oi == 0 is never skipped, so a node
this applies to always has at least one fully-searched move to report a real score from.

Internal iterative deepening: a node deep enough (IID_MIN_DEPTH) with no hash move to order by --
a genuine TT miss, not merely an entry too shallow for a cutoff, since that still yields a hint --
searches itself at depth - IID_REDUCTION purely to populate one before the real move ordering.
Implemented by re-invoking negamax on the exact same (bb, meta, ply, hist_len): the sub-search's
own TT store, keyed by the same position hash, is the return channel -- re-probing right after
picks up whatever move it found, avoiding a second return value threaded through negamax just for
this. Self-terminating (depth strictly decreases each nesting) and skipped in check, matching the
other depth-side heuristics' posture toward tactical nodes.

Repetition, mechanism 1 -- our own turn recurring: negamax and search_root take a shared
`history` array of zobrist hashes plus `hist_len`, the count of entries that precede the current
node. Only our-turn positions (even ply, root = ply 0) are ever recorded or checked here -- the
array is indexed by "how many our-turn positions have occurred so far on the current path",
which a plain depth-first walk keeps correct with no explicit push/pop: a node writes its own
slot once, and a later sibling at the same ply simply overwrites it when the search backtracks.
agent.py owns the persistent real-game prefix and writes the root's own slot before calling
search_root; everything at ply >= 1 is written by negamax itself as it descends. If a line would
make a position recur a third time, it scores as an immediate draw (0) instead of running eval
or search on it further -- and this check runs before the transposition table is touched, so a
position that is genuinely a repeat on this path is never short-circuited by a stale cached score
from a path where it wasn't.

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
once per real move decision but not throughout the tree. Untouched by the transposition table --
it always recomputes from the real history arrays, never from a cached score.
"""

import time

import numpy as np
from numba import njit, objmode

from attacks import KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, bishop_attacks, rook_attacks
from bitboard import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE
from evaluate import PIECE_VALUE, evaluate
from movegen import (
    generate_legal,
    has_non_pawn_material,
    is_check,
    make_move,
    make_null_move,
    occ_all,
    piece_type_at,
)
from zobrist import position_hash

MATE = 1_000_000
INF = 2_000_000
CHECK_INTERVAL = 127
QUIESCENCE_MAX_PLIES = 24
ONE = np.uint64(1)

# Delta pruning in quiescence: a capture whose SEE, added to the stand-pat eval, still can't
# reach alpha within this safety margin is skipped -- see quiescence's move loop. A generous
# margin (bigger than a minor piece) since this is a hard skip, not a reduction.
DELTA_MARGIN = 200

# Aspiration windows: search_root centers the first pass's alpha-beta window on the previous
# iterative-deepening depth's score instead of always searching -INF..INF, re-searching with the
# full window on a fail-high/fail-low. NO_PREV_SCORE (a value no real score or mate score can ever
# equal, since MATE < INF strictly) tells search_root there is no previous depth to center on --
# used for depth 1 and whenever the prior score was itself outside the mate threshold.
ASPIRATION_WINDOW = 50
NO_PREV_SCORE = INF

# Null-move pruning: give the side to move a free pass and search the rest at a reduced depth; if
# that is still enough to cause a beta cutoff, the real move would only do better, so the whole
# subtree is pruned. NULL_MOVE_MIN_DEPTH keeps depth - 1 - NULL_MOVE_REDUCTION non-negative (any
# smaller and it degrades to a quiescence call anyway, not a useful probe) and NULL_MOVE_REDUCTION
# is the standard R=2. Guarded against zugzwang two ways -- see negamax.
NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2

# Late move reductions: a quiet move searched late in the ordering (oi >= LMR_MIN_MOVE_INDEX,
# after the hash/killer/history-backed moves have already had their full-depth say) gets one ply
# less depth first; only a reduced-depth score that still beats alpha earns a full-depth
# re-search before PVS's own full-window re-search gets a chance. Deliberately conservative (a
# fixed one-ply reduction, not depth- or move-count-scaled) since this sits on top of the
# existing TT/killer/history ordering rather than replacing it -- see docs/FUTURE.md item 5.
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE_INDEX = 3
LMR_REDUCTION = 1

# Search extensions: a move that gives check gets its child searched a full ply deep (depth is
# not decremented) instead of the usual depth - 1, since a forced reply to check is exactly the
# kind of forcing, tactically-critical line a depth-limited search is otherwise most likely to
# misjudge. ext_budget, threaded through negamax and decremented once per extension granted,
# caps how many of these a single line can stack -- unbounded stacking (e.g. a long forcing check
# sequence) would let one line's effective depth grow arbitrarily, at the expense of every
# sibling line sharing the same time budget. MAX_CHECK_EXTENSIONS is deliberately small: enough
# for a real forcing sequence, far short of enough to meaningfully skew the time budget.
MAX_CHECK_EXTENSIONS = 8

# Futility pruning: at a shallow, non-check node, a quiet move whose best case (the static eval
# plus a depth-scaled margin) still can't reach alpha is skipped outright rather than searched --
# a quiet move rarely swings the score by more than a pawn or so, so if the position already
# looks this bad before the move, searching it is very unlikely to change the outcome. Margins
# widen with depth since more plies of search behind the static eval means more room for a quiet
# move to matter after all. Index 0 is never read (depth <= 0 returns via quiescence earlier).
FUTILITY_MAX_DEPTH = 3
FUTILITY_MARGIN = np.array([0, 150, 300, 450], dtype=np.int64)

# Internal iterative deepening: a node deep enough to matter but with no hash move to order by
# (a TT miss, or a stored entry too shallow to have one -- see hint_from in negamax) gets a
# reduced-depth search of itself first, purely to seed move ordering before the real search of
# it. IID_MIN_DEPTH keeps this to nodes where good ordering is worth the extra sub-search.
IID_MIN_DEPTH = 4
IID_REDUCTION = 2

# Our-turn positions per game are bounded by rules.PLY_CAP / 2 (~150); in-search growth is
# bounded by MAX_DEPTH / 2 (agent.py caps depth at 64, so ~32 more). 512 leaves ample headroom.
HISTORY_CAPACITY = 512

# Negamax's own recursion only deepens through ply while depth > 0, and depth only ever
# decreases from the root's initial value (agent.py caps that at 64), so 128 is ample headroom
# for indexing killer moves by ply, matching HISTORY_CAPACITY's own margin above its real need.
MAX_KILLER_PLY = 128

# Two-tier: each bucket has two slots, a depth-preferred one (kept unless a same-key refresh or a
# search that went at least as deep wants to replace it) and an always-replace one (a pure
# recency slot, so a fresh position from the current search is never simply dropped because the
# depth-preferred slot happens to be occupied by something deeper). TT_BUCKETS is the addressable
# bucket count (indexed by the low bits of the Zobrist hash); the arrays below are twice that many
# raw slots. 2M buckets * 2 slots * 20 bytes/slot (parallel arrays below) is a fixed ~80MB,
# independent of game length -- no growth, no eviction bookkeeping beyond the two-slot policy.
TT_BUCKETS = 1 << 21
TT_SIZE = TT_BUCKETS * 2
TT_MASK = np.uint64(TT_BUCKETS - 1)
TT_EXACT = np.int8(0)
TT_LOWER = np.int8(1)
TT_UPPER = np.int8(2)
MATE_STORE_THRESHOLD = MATE - 1000


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


# Exchange chains are bounded by total pieces on the board (32), so 32 gain-array slots is
# ample headroom -- see() never appends past one entry per capture in the chain.
SEE_MAX_DEPTH = 32


@njit(cache=False)
def _bit_scan(bits: np.uint64) -> int:
    for square in range(64):
        if bits & (ONE << np.uint64(square)):
            return square
    return -1


@njit(cache=False)
def _see_least_valuable_attacker(
    bb: np.ndarray, occ: np.uint64, color: int, square: int
) -> tuple[int, int]:
    """The lowest-value piece of `color` attacking `square` given `occ`, as (from_square,
    piece_type), or (-1, -1) if none. Sliding attacks are recomputed against the live `occ` on
    every call rather than tracked incrementally, which is what makes discovered ("x-ray")
    attackers fall out for free as pieces are removed during the exchange in see() below -- once
    a blocker is cleared from `occ`, bishop_attacks/rook_attacks simply see past it on the next
    call. Same ray-cast-over-magic-bitboards tradeoff as the rest of attacks.py.
    """
    attackers = PAWN_ATTACKS[1 - color, square] & bb[color * 6 + PAWN]
    if attackers:
        return _bit_scan(attackers), PAWN
    attackers = KNIGHT_ATTACKS[square] & bb[color * 6 + KNIGHT]
    if attackers:
        return _bit_scan(attackers), KNIGHT
    attackers = bishop_attacks(square, occ) & bb[color * 6 + BISHOP]
    if attackers:
        return _bit_scan(attackers), BISHOP
    attackers = rook_attacks(square, occ) & bb[color * 6 + ROOK]
    if attackers:
        return _bit_scan(attackers), ROOK
    attackers = (bishop_attacks(square, occ) | rook_attacks(square, occ)) & bb[color * 6 + QUEEN]
    if attackers:
        return _bit_scan(attackers), QUEEN
    attackers = KING_ATTACKS[square] & bb[color * 6 + KING]
    if attackers:
        return _bit_scan(attackers), KING
    return -1, -1


@njit(cache=False)
def see(bb: np.ndarray, meta: np.ndarray, from_sq: int, to_sq: int, promo: int) -> int:
    """Static exchange evaluation of playing (from_sq, to_sq, promo): the net material change on
    `to_sq` after both sides recapture there optimally (least-valuable-attacker first), not just
    the value of what this one move immediately wins. Standard swap-list algorithm (see
    chessprogramming.org "SEE - The Swap Algorithm"): gain[d] is what the side capturing at ply d
    nets (what they take, minus the previous ply's gain from their opponent's perspective), and
    the final backward min-max pass lets either side stop the exchange early if continuing it
    would lose more than declining to.

    Ignores pins and check-legality of intermediate recaptures (a standard SEE simplification --
    it can misjudge a recapture that is actually illegal because the recapturing piece is pinned),
    which is why this is a move-ordering and quiescence-pruning heuristic, never a legality check.
    """
    color = meta[0]
    opponent = 1 - color
    moving_pt = piece_type_at(bb, color, from_sq)
    ep_square = meta[5]
    is_ep = moving_pt == PAWN and to_sq == ep_square and (from_sq % 8) != (to_sq % 8)
    victim_pt = PAWN if is_ep else piece_type_at(bb, opponent, to_sq)

    gain = np.zeros(SEE_MAX_DEPTH, dtype=np.int64)
    gain[0] = PIECE_VALUE[victim_pt] if victim_pt >= 0 else 0
    if promo >= 0:
        gain[0] += PIECE_VALUE[promo] - PIECE_VALUE[PAWN]

    work = bb.copy()
    if is_ep:
        captured_sq = to_sq - 8 if color == WHITE else to_sq + 8
        work[opponent * 6 + PAWN] &= ~(ONE << np.uint64(captured_sq))
    elif victim_pt >= 0:
        work[opponent * 6 + victim_pt] &= ~(ONE << np.uint64(to_sq))
    work[color * 6 + moving_pt] &= ~(ONE << np.uint64(from_sq))
    on_square_pt = promo if (moving_pt == PAWN and promo >= 0) else moving_pt
    work[color * 6 + on_square_pt] |= ONE << np.uint64(to_sq)
    on_square_value = PIECE_VALUE[on_square_pt]

    side = opponent
    depth = 0
    while depth < SEE_MAX_DEPTH - 1:
        occ = occ_all(work)
        atk_sq, atk_pt = _see_least_valuable_attacker(work, occ, side, to_sq)
        if atk_sq < 0:
            break
        depth += 1
        gain[depth] = on_square_value - gain[depth - 1]

        work[(1 - side) * 6 + on_square_pt] &= ~(ONE << np.uint64(to_sq))
        work[side * 6 + atk_pt] &= ~(ONE << np.uint64(atk_sq))
        promotes = atk_pt == PAWN and (
            (side == WHITE and to_sq >= 56) or (side != WHITE and to_sq <= 7)
        )
        on_square_pt = QUEEN if promotes else atk_pt
        work[side * 6 + on_square_pt] |= ONE << np.uint64(to_sq)
        on_square_value = PIECE_VALUE[on_square_pt]
        side = 1 - side

    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return int(gain[0])


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
    is_ep = to_sq == meta[5] and piece_type_at(bb, meta[0], from_sq) == PAWN
    if victim_pt >= 0 or (is_ep and (from_sq % 8) != (to_sq % 8)):
        score += 10_000 + see(bb, meta, from_sq, to_sq, promo)
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
def _score_moves2(
    bb: np.ndarray,
    meta: np.ndarray,
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    promo_arr: np.ndarray,
    count: int,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
    k1_from: int,
    k1_to: int,
    k1_promo: int,
    k2_from: int,
    k2_to: int,
    k2_promo: int,
    cm_from: int,
    cm_to: int,
    cm_promo: int,
    history_table: np.ndarray,
) -> np.ndarray:
    """Same per-move scoring as _score_moves, but a plain quiet move (base score exactly 0: not
    the hash/PV move, not a promotion, not a capture) additionally gets a killer-move bonus, or
    failing that a counter-move bonus (cm_from/cm_to/cm_promo: whatever quiet move most recently
    caused a cutoff in reply to the move that led to this node -- see negamax's counter_from/to/
    promo table), or failing that its from/to history score -- ranked killers, then counter-move,
    then history, all below every capture and promotion, above zero. Inlined into one function
    (rather than calling a per-move helper in the loop, as an earlier version did) purely to keep
    numba's compile graph smaller -- this and negamax are the two most expensive functions to JIT
    in the whole engine.
    """
    scores = np.empty(count, dtype=np.int64)
    for i in range(count):
        from_sq, to_sq, promo = from_arr[i], to_arr[i], promo_arr[i]
        base = _move_score(bb, meta, from_sq, to_sq, promo, pv_from, pv_to, pv_promo)
        if base != 0:
            scores[i] = base
        elif from_sq == k1_from and to_sq == k1_to and promo == k1_promo:
            scores[i] = 5002
        elif from_sq == k2_from and to_sq == k2_to and promo == k2_promo:
            scores[i] = 5001
        elif from_sq == cm_from and to_sq == cm_to and promo == cm_promo:
            scores[i] = 5000
        else:
            h = history_table[from_sq * 64 + to_sq]
            scores[i] = h if h < 4999 else 4999
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

    stand_pat = evaluate(bb, meta)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth <= 0:
        return alpha

    scores = np.full(count, -INF, dtype=np.int64)
    ncap = 0
    for i in range(count):
        f, t, p = from_arr[i], to_arr[i], promo_arr[i]
        if is_capture(bb, meta, f, t) or p == QUEEN:
            scores[i] = see(bb, meta, f, t, p)
            ncap += 1
    if ncap == 0:
        return alpha

    order = np.argsort(-scores)
    for oi in range(ncap):
        idx = order[oi]
        if scores[idx] < 0 or stand_pat + scores[idx] + DELTA_MARGIN <= alpha:
            # SEE-descending order: everything from here on scores at least as low, so once
            # either condition trips it holds for the rest of the loop too. The first
            # (scores[idx] < 0) is SEE-pruning -- a capture that loses material outright cannot
            # help (see docs/FUTURE.md item 4). The second is delta pruning: even a *winning*
            # capture skipped here still can't close the gap to alpha by more than a safety
            # margin, catching the case SEE-pruning alone misses (a real but too-small gain
            # against a large existing deficit).
            break
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
def _tt_resolve(
    stored_depth: int,
    raw_score: int,
    flag: int,
    depth: int,
    ply: int,
    alpha: int,
    beta: int,
) -> tuple[bool, int, int, int]:
    """Given a transposition-table slot that matched the current position's key, decide whether
    it settles this node outright. Returns (hit, score, alpha, beta): hit means the caller can
    return `score` immediately (an exact stored score, or a cutoff from a lower/upper bound);
    otherwise alpha/beta come back possibly tightened by a bound that didn't itself cause a
    cutoff, for the caller to keep searching with. Shared between the two probed slots (see
    negamax) so the mate-distance adjustment and bound logic exist in exactly one place.
    """
    if stored_depth < depth:
        return False, 0, alpha, beta
    if raw_score >= MATE_STORE_THRESHOLD:
        s = raw_score - ply
    elif raw_score <= -MATE_STORE_THRESHOLD:
        s = raw_score + ply
    else:
        s = raw_score
    if flag == TT_EXACT:
        return True, s, alpha, beta
    if flag == TT_LOWER and s > alpha:
        alpha = s
    elif flag == TT_UPPER and s < beta:
        beta = s
    if alpha >= beta:
        return True, s, alpha, beta
    return False, 0, alpha, beta


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
    tt_key: np.ndarray,
    tt_depth: np.ndarray,
    tt_score: np.ndarray,
    tt_flag: np.ndarray,
    tt_from: np.ndarray,
    tt_to: np.ndarray,
    tt_promo: np.ndarray,
    killer_from: np.ndarray,
    killer_to: np.ndarray,
    killer_promo: np.ndarray,
    history_table: np.ndarray,
    allow_null: bool,
    ext_budget: int,
    counter_from: np.ndarray,
    counter_to: np.ndarray,
    counter_promo: np.ndarray,
    parent_from: int,
    parent_to: int,
) -> int:
    counters[0] += 1
    if _time_up(deadline, counters):
        return 0

    h = position_hash(bb, meta)

    if ply % 2 == 0:
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

    orig_alpha = alpha
    bucket = int(h & TT_MASK)
    slot_a = bucket * 2
    slot_b = slot_a + 1
    hint_from, hint_to, hint_promo = -1, -1, -1
    if tt_key[slot_a] == h:
        hint_from, hint_to, hint_promo = int(tt_from[slot_a]), int(tt_to[slot_a]), int(
            tt_promo[slot_a]
        )
        hit, s, alpha, beta = _tt_resolve(
            int(tt_depth[slot_a]), int(tt_score[slot_a]), int(tt_flag[slot_a]), depth, ply,
            alpha, beta,
        )
        if hit:
            return s
    elif tt_key[slot_b] == h:
        hint_from, hint_to, hint_promo = int(tt_from[slot_b]), int(tt_to[slot_b]), int(
            tt_promo[slot_b]
        )
        hit, s, alpha, beta = _tt_resolve(
            int(tt_depth[slot_b]), int(tt_score[slot_b]), int(tt_flag[slot_b]), depth, ply,
            alpha, beta,
        )
        if hit:
            return s

    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    if count == 0:
        return -MATE if is_check(bb, meta) else 0

    if depth <= 0:
        return quiescence(bb, meta, alpha, beta, deadline, counters, QUIESCENCE_MAX_PLIES)

    in_check = is_check(bb, meta)

    futile = False
    if (
        depth <= FUTILITY_MAX_DEPTH
        and not in_check
        and -MATE_STORE_THRESHOLD < alpha < MATE_STORE_THRESHOLD
    ):
        futile = evaluate(bb, meta) + FUTILITY_MARGIN[depth] <= alpha

    if (
        allow_null
        and depth >= NULL_MOVE_MIN_DEPTH
        and -MATE_STORE_THRESHOLD < beta < MATE_STORE_THRESHOLD
        and not in_check
        and has_non_pawn_material(bb, meta[0])
    ):
        null_meta = make_null_move(meta)
        null_score = -negamax(
            bb, null_meta, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, deadline, counters,
            ply + 1, history, child_hist_len,
            tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
            killer_from, killer_to, killer_promo, history_table, False, ext_budget,
            counter_from, counter_to, counter_promo, -1, -1,
        )
        if counters[1]:
            return 0
        if null_score >= beta:
            return beta

    if hint_from < 0 and depth >= IID_MIN_DEPTH and not in_check:
        # No hash move to order with, and deep enough that ordering is worth paying for: search
        # this same node at a reduced depth purely to populate one. The recursive call's own TT
        # store (same h, keyed off the exact same position) is the return channel -- re-probing
        # right after picks up whatever move it found, without threading a second return value
        # through negamax just for this.
        negamax(
            bb, meta, depth - IID_REDUCTION, alpha, beta, deadline, counters, ply, history,
            hist_len, tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
            killer_from, killer_to, killer_promo, history_table, allow_null, ext_budget,
            counter_from, counter_to, counter_promo, parent_from, parent_to,
        )
        if counters[1]:
            return 0
        if tt_key[slot_a] == h:
            hint_from, hint_to, hint_promo = int(tt_from[slot_a]), int(tt_to[slot_a]), int(
                tt_promo[slot_a]
            )
        elif tt_key[slot_b] == h:
            hint_from, hint_to, hint_promo = int(tt_from[slot_b]), int(tt_to[slot_b]), int(
                tt_promo[slot_b]
            )

    k1_from = int(killer_from[ply, 0])
    k1_to, k1_promo = int(killer_to[ply, 0]), int(killer_promo[ply, 0])
    k2_from = int(killer_from[ply, 1])
    k2_to, k2_promo = int(killer_to[ply, 1]), int(killer_promo[ply, 1])
    if parent_from >= 0:
        cm_idx = parent_from * 64 + parent_to
        cm_from, cm_to, cm_promo = (
            int(counter_from[cm_idx]), int(counter_to[cm_idx]), int(counter_promo[cm_idx]),
        )
    else:
        cm_from, cm_to, cm_promo = -1, -1, -1
    scores = _score_moves2(
        bb, meta, from_arr, to_arr, promo_arr, count,
        hint_from, hint_to, hint_promo,
        k1_from, k1_to, k1_promo, k2_from, k2_to, k2_promo,
        cm_from, cm_to, cm_promo,
        history_table,
    )
    order = np.argsort(-scores)

    best = -INF
    best_idx = -1
    for oi in range(count):
        idx = order[oi]
        f, t, p = from_arr[idx], to_arr[idx], promo_arr[idx]
        new_bb, new_meta = make_move(bb, meta, f, t, p)
        gives_check = is_check(new_bb, new_meta)
        if ext_budget > 0 and gives_check:
            child_depth = depth
            child_ext_budget = ext_budget - 1
        else:
            child_depth = depth - 1
            child_ext_budget = ext_budget

        if oi == 0:
            score = -negamax(
                new_bb, new_meta, child_depth, -beta, -alpha, deadline, counters,
                ply + 1, history, child_hist_len,
                tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                killer_from, killer_to, killer_promo, history_table, True, child_ext_budget,
                counter_from, counter_to, counter_promo, f, t,
            )
        else:
            if futile and p < 0 and not gives_check and not is_capture(bb, meta, f, t):
                # A quiet move at a shallow, non-check node when even the best case (static eval
                # plus a depth-scaled margin) can't reach alpha -- skip it outright rather than
                # spend a search on it. oi == 0 (the hash/best-ordered move) is never skipped, so
                # this node always has at least one fully-searched move to report a score from.
                continue

            reduction = 0
            if (
                oi >= LMR_MIN_MOVE_INDEX
                and depth >= LMR_MIN_DEPTH
                and p < 0
                and not in_check
                and not is_capture(bb, meta, f, t)
                and not gives_check
            ):
                reduction = LMR_REDUCTION

            score = -negamax(
                new_bb, new_meta, child_depth - reduction, -alpha - 1, -alpha, deadline, counters,
                ply + 1, history, child_hist_len,
                tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                killer_from, killer_to, killer_promo, history_table, True, child_ext_budget,
                counter_from, counter_to, counter_promo, f, t,
            )
            if not counters[1] and reduction > 0 and score > alpha:
                # the reduced-depth probe suggested this late, quiet move might actually be
                # good -- confirm at full depth (still null window) before trusting it enough
                # to maybe trigger the full-window PVS re-search below.
                score = -negamax(
                    new_bb, new_meta, child_depth, -alpha - 1, -alpha, deadline, counters,
                    ply + 1, history, child_hist_len,
                    tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                    killer_from, killer_to, killer_promo, history_table, True, child_ext_budget,
                    counter_from, counter_to, counter_promo, f, t,
                )
            if not counters[1] and alpha < score < beta:
                score = -negamax(
                    new_bb, new_meta, child_depth, -beta, -alpha, deadline, counters,
                    ply + 1, history, child_hist_len,
                    tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                    killer_from, killer_to, killer_promo, history_table, True, child_ext_budget,
                    counter_from, counter_to, counter_promo, f, t,
                )
        if counters[1]:
            return 0
        if score > best:
            best = score
            best_idx = idx
        if best > alpha:
            alpha = best
        if alpha >= beta:
            if p < 0 and not is_capture(bb, meta, f, t):
                if not (f == k1_from and t == k1_to and p == k1_promo):
                    killer_from[ply, 1], killer_to[ply, 1], killer_promo[ply, 1] = (
                        killer_from[ply, 0], killer_to[ply, 0], killer_promo[ply, 0],
                    )
                    killer_from[ply, 0], killer_to[ply, 0], killer_promo[ply, 0] = f, t, p
                history_table[f * 64 + t] += depth * depth
                if parent_from >= 0:
                    cm_idx = parent_from * 64 + parent_to
                    counter_from[cm_idx] = np.int8(f)
                    counter_to[cm_idx] = np.int8(t)
                    counter_promo[cm_idx] = np.int8(p)
            break

    # Two-tier replacement: prefer the depth-preferred slot on a same-key refresh, an empty slot
    # (depth == -1, the new_tt() sentinel), or a search that went at least as deep as what is
    # already there; otherwise fall back to the always-replace slot so this node's result is
    # still cached even though it didn't earn the depth-preferred one.
    if tt_key[slot_a] == h or tt_depth[slot_a] < 0 or depth >= tt_depth[slot_a]:
        write_idx = slot_a
    else:
        write_idx = slot_b

    tt_key[write_idx] = h
    tt_depth[write_idx] = depth
    if best >= MATE_STORE_THRESHOLD:
        tt_score[write_idx] = best + ply
    elif best <= -MATE_STORE_THRESHOLD:
        tt_score[write_idx] = best - ply
    else:
        tt_score[write_idx] = best
    if best <= orig_alpha:
        tt_flag[write_idx] = TT_UPPER
    elif best >= beta:
        tt_flag[write_idx] = TT_LOWER
    else:
        tt_flag[write_idx] = TT_EXACT
    if best_idx != -1:
        tt_from[write_idx] = np.int8(from_arr[best_idx])
        tt_to[write_idx] = np.int8(to_arr[best_idx])
        tt_promo[write_idx] = np.int8(promo_arr[best_idx])

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
    move decision but not throughout the tree. Never consults the transposition table -- always
    recomputed from the real history arrays, so a cached search score can never mask a claim.
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
def _search_root_pass(
    bb: np.ndarray,
    meta: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    deadline: float,
    counters: np.ndarray,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
    history: np.ndarray,
    hist_len: int,
    opponent_history: np.ndarray,
    opponent_hist_len: int,
    tt_key: np.ndarray,
    tt_depth: np.ndarray,
    tt_score: np.ndarray,
    tt_flag: np.ndarray,
    tt_from: np.ndarray,
    tt_to: np.ndarray,
    tt_promo: np.ndarray,
    killer_from: np.ndarray,
    killer_to: np.ndarray,
    killer_promo: np.ndarray,
    history_table: np.ndarray,
    counter_from: np.ndarray,
    counter_to: np.ndarray,
    counter_promo: np.ndarray,
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    promo_arr: np.ndarray,
    order: np.ndarray,
    count: int,
) -> tuple[int, int, int, int, bool]:
    """One root pass over already-ordered moves within a fixed [alpha, beta] window -- factored
    out of search_root so aspiration windows (see there) can re-run this at the same depth with a
    wider window on a fail-high/fail-low without re-ordering or re-generating moves.
    """
    best_score = -INF
    best_from, best_to, best_promo = from_arr[order[0]], to_arr[order[0]], promo_arr[order[0]]

    for oi in range(count):
        idx = order[oi]
        f, t, p = from_arr[idx], to_arr[idx], promo_arr[idx]
        new_bb, new_meta = make_move(bb, meta, f, t, p)
        child_depth = depth if is_check(new_bb, new_meta) else depth - 1
        child_ext_budget = MAX_CHECK_EXTENSIONS - (1 if child_depth == depth else 0)
        if oi == 0:
            score = -negamax(
                new_bb, new_meta, child_depth, -beta, -alpha, deadline, counters, 1, history,
                hist_len, tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                killer_from, killer_to, killer_promo, history_table, True, child_ext_budget,
                counter_from, counter_to, counter_promo, f, t,
            )
        else:
            score = -negamax(
                new_bb, new_meta, child_depth, -alpha - 1, -alpha, deadline, counters, 1, history,
                hist_len, tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                killer_from, killer_to, killer_promo, history_table, True, child_ext_budget,
                counter_from, counter_to, counter_promo, f, t,
            )
            if not counters[1] and alpha < score < beta:
                score = -negamax(
                    new_bb, new_meta, child_depth, -beta, -alpha, deadline, counters, 1, history,
                    hist_len, tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                    killer_from, killer_to, killer_promo, history_table, True, child_ext_budget,
                    counter_from, counter_to, counter_promo, f, t,
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
    tt_key: np.ndarray,
    tt_depth: np.ndarray,
    tt_score: np.ndarray,
    tt_flag: np.ndarray,
    tt_from: np.ndarray,
    tt_to: np.ndarray,
    tt_promo: np.ndarray,
    killer_from: np.ndarray,
    killer_to: np.ndarray,
    killer_promo: np.ndarray,
    history_table: np.ndarray,
    counter_from: np.ndarray,
    counter_to: np.ndarray,
    counter_promo: np.ndarray,
    prev_score: int,
) -> tuple[int, int, int, int, bool]:
    """hist_len counts our-turn positions already recorded, including the root's own -- the
    caller (agent.py) writes history[hist_len - 1] = hash(bb, meta) before calling this, since it
    already needs that hash for the persistent cross-move history anyway. opponent_hist_len
    counts the real game's own past opponent-turn positions, one recorded per move we have
    actually played -- see claim_eligible_for_opponent. tt_* / killer_* / history_table /
    counter_* are threaded straight into negamax; see this module's docstring. prev_score is the
    previous iterative-deepening depth's score, or NO_PREV_SCORE if there isn't one (depth 1) --
    used to center the aspiration window (see module-level constants).
    """
    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    if count == 0:
        return -1, -1, -1, 0, False

    scores = _score_moves(bb, meta, from_arr, to_arr, promo_arr, count, pv_from, pv_to, pv_promo)
    order = np.argsort(-scores)

    if depth <= 2 or prev_score == NO_PREV_SCORE or abs(prev_score) >= MATE_STORE_THRESHOLD:
        alpha, beta = -INF, INF
    else:
        alpha = max(-INF, prev_score - ASPIRATION_WINDOW)
        beta = min(INF, prev_score + ASPIRATION_WINDOW)

    while True:
        best_from, best_to, best_promo, best_score, completed = _search_root_pass(
            bb, meta, depth, alpha, beta, deadline, counters, pv_from, pv_to, pv_promo,
            history, hist_len, opponent_history, opponent_hist_len,
            tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
            killer_from, killer_to, killer_promo, history_table,
            counter_from, counter_to, counter_promo,
            from_arr, to_arr, promo_arr, order, count,
        )
        if not completed:
            return best_from, best_to, best_promo, best_score, False
        if (best_score <= alpha and alpha > -INF) or (best_score >= beta and beta < INF):
            alpha, beta = -INF, INF
            continue
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


def new_tt() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """A fresh, empty transposition table -- parallel arrays, two raw slots per bucket (see this
    module's docstring). depth=-1 marks an empty slot implicitly (any real search depth is >= 0,
    so a real entry always compares >= any depth request the first time it is written; key is
    separately checked for a match anyway).
    """
    key = np.zeros(TT_SIZE, dtype=np.uint64)
    depth = np.full(TT_SIZE, -1, dtype=np.int32)
    score = np.zeros(TT_SIZE, dtype=np.int32)
    flag = np.zeros(TT_SIZE, dtype=np.int8)
    move_from = np.full(TT_SIZE, -1, dtype=np.int8)
    move_to = np.full(TT_SIZE, -1, dtype=np.int8)
    move_promo = np.full(TT_SIZE, -1, dtype=np.int8)
    return key, depth, score, flag, move_from, move_to, move_promo


def new_killers() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    killer_from = np.full((MAX_KILLER_PLY, 2), -1, dtype=np.int8)
    killer_to = np.full((MAX_KILLER_PLY, 2), -1, dtype=np.int8)
    killer_promo = np.full((MAX_KILLER_PLY, 2), -1, dtype=np.int8)
    return killer_from, killer_to, killer_promo


def new_history_table() -> np.ndarray:
    return np.zeros(64 * 64, dtype=np.int32)


def new_counter_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counter-move table: indexed by (parent_from * 64 + parent_to), the move that led to a
    node, this stores whatever quiet move most recently caused a beta cutoff in reply to it --
    see negamax's parent_from/parent_to parameters and _score_moves2's cm_from/cm_to/cm_promo.
    Rebuilt fresh per real move decision, same as killers and history_table: it encodes this
    search's own cutoff history, not the position itself.
    """
    counter_from = np.full(64 * 64, -1, dtype=np.int8)
    counter_to = np.full(64 * 64, -1, dtype=np.int8)
    counter_promo = np.full(64 * 64, -1, dtype=np.int8)
    return counter_from, counter_to, counter_promo
