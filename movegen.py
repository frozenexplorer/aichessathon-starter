"""Pseudo-legal and legal move generation over the bitboard representation in bitboard.py.

Legality is checked the simple way: generate pseudo-legal, make the move, see if the mover's
own king is attacked. Not the fastest approach, but every check reuses the same `attacked_by`
and sliding-attack code that everything else does, so there is only one place attack logic can
be wrong instead of two.

Move lists are fixed-size int8 arrays (from, to, promo) plus a count, not Python lists, so this
stays jit-friendly. MAX_MOVES is well above the theoretical maximum legal moves in any reachable
chess position (218).
"""

from typing import Any

import llvmlite.ir as _ir  # type: ignore[import-untyped]
import numpy as np
from numba import njit
from numba.core import types as _nbtypes
from numba.extending import intrinsic

from attacks import KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, bishop_attacks, rook_attacks
from bitboard import BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE, i64

MAX_MOVES = 256
ONE = np.uint64(1)
NO_SQUARE = np.int8(-1)

# Castling-square constants for generate_pseudo_legal's attacked_by() checks below, wrapped once
# at module level via bitboard.i64 rather than at each call site -- i64() itself is plain,
# un-jitted Python, so calling it directly from inside an njit function body (as opposed to here,
# at ordinary module-import time) fails to compile at all (numba cannot call an untyped Python
# function in nopython mode). See docs/plan.md Phase 1.1(a): a bare int literal crossing an njit
# call boundary is typed as a distinct Literal[int] per value, which is what made attacked_by
# (and, transitively, bishop_attacks/rook_attacks) compile a dozen-odd specialisations before this.
_SQ_C2 = i64(2)
_SQ_C3 = i64(3)
_SQ_C4 = i64(4)
_SQ_C5 = i64(5)
_SQ_C6 = i64(6)
_SQ_C58 = i64(58)
_SQ_C59 = i64(59)
_SQ_C60 = i64(60)
_SQ_C61 = i64(61)
_SQ_C62 = i64(62)


@njit(cache=False)
def _i64(v: int) -> int:
    """Runtime counterpart to bitboard.i64: casts a value read from an int8-dtyped array (or a
    bare `meta[0]`) to int64 *from inside an njit function body*, where bitboard.i64 itself can't
    be called (it is plain, un-jitted Python -- calling it in nopython mode fails to compile
    outright). Python's own `int(...)` looks like it should do the same thing but does not: numba
    treats it as a no-op identity conversion when the input is already some integer type, so
    `int(meta[0])` stays int8, not int64 -- confirmed the hard way (docs/plan.md Phase 1.1(b)/(c))
    when using it caused every from/to/color-consuming function to regress back to multiple
    specialisations. Routing the cast through this dedicated njit function's own explicit
    `np.int64(...)` return forces the widening every caller needs, at the cost of compiling this
    one-line function once per distinct *input* type it is fed (cheap: the body is trivial, and
    NUMBA_OPT=1 inlines it in practice).
    """
    return np.int64(v)  # type: ignore[return-value]


def _build_square_color_mask() -> np.uint64:
    """One bit per square whose (file + rank) is even -- an arbitrary but fixed partition of the
    board into two colour classes, used only to tell whether a set of bishops all sit on the same
    square colour (see is_insufficient_material below); which class is called "light" or "dark"
    never matters, only that the partition is consistent.
    """
    bits = np.uint64(0)
    for square in range(64):
        if (square % 8 + square // 8) % 2 == 0:
            bits |= ONE << np.uint64(square)
    return bits


SQUARE_COLOR_MASK = _build_square_color_mask()


@njit(cache=False)
def occ_color(bb: np.ndarray, color: int) -> np.uint64:
    occ = np.uint64(0)
    for pt in range(6):
        occ |= bb[color * 6 + pt]
    return occ


@njit(cache=False)
def occ_all(bb: np.ndarray) -> np.uint64:
    return occ_color(bb, WHITE) | occ_color(bb, BLACK)


@njit(cache=False)
def king_square(bb: np.ndarray, color: int) -> int:
    king_bb = bb[color * 6 + KING]
    for square in range(64):
        if king_bb & (ONE << np.uint64(square)):
            return square
    return -1


@njit(cache=False)
def attacked_by(bb: np.ndarray, all_occ: np.uint64, color: int, square: int) -> bool:
    """Is `square` attacked by any piece of `color`?"""
    if PAWN_ATTACKS[1 - color, square] & bb[color * 6 + PAWN]:
        return True
    if KNIGHT_ATTACKS[square] & bb[color * 6 + KNIGHT]:
        return True
    if KING_ATTACKS[square] & bb[color * 6 + KING]:
        return True
    if bishop_attacks(square, all_occ) & (bb[color * 6 + BISHOP] | bb[color * 6 + QUEEN]):
        return True
    return bool(rook_attacks(square, all_occ) & (bb[color * 6 + ROOK] | bb[color * 6 + QUEEN]))


@njit(cache=False)
def is_check(bb: np.ndarray, meta: np.ndarray) -> bool:
    color = _i64(meta[0])
    return attacked_by(bb, occ_all(bb), 1 - color, king_square(bb, color))


@njit(cache=False)
def piece_type_at(bb: np.ndarray, color: int, square: int) -> int:
    bit = ONE << np.uint64(square)
    for pt in range(6):
        if bb[color * 6 + pt] & bit:
            return pt
    return -1


@intrinsic
def _llvm_ctpop64(typingctx: Any, val: Any) -> Any:
    """`llvm.ctpop.i64` -- population count in one instruction instead of a Kernighan loop.
    Typed as returning `int64` (not `uint64`) directly -- unifying a `uint64`-typed result with the
    plain `int64` literal `-1` in `_bit_scan`'s other return branch made numba widen the whole
    function's return type to `float64` (verified: a bare `int(uint64_var)` cast unifies with `-1`
    the same way, so this is a general numba quirk, not specific to `intrinsic`); returning `int64`
    from the codegen sidesteps the unification entirely.
    """
    sig = _nbtypes.int64(_nbtypes.uint64)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        return builder.ctpop(args[0])

    return sig, codegen


@intrinsic
def _llvm_cttz64(typingctx: Any, val: Any) -> Any:
    """`llvm.cttz.i64` -- index of the lowest set bit in one instruction instead of a linear scan.
    Result is undefined (per the LLVM intrinsic contract) when `val` is zero -- callers must check
    for zero themselves, which `_bit_scan` below does. See `_llvm_ctpop64` above for why this
    returns `int64` rather than `uint64`.
    """
    sig = _nbtypes.int64(_nbtypes.uint64)

    def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
        is_zero_undef = _ir.Constant(_ir.IntType(1), 0)
        return builder.cttz(args[0], is_zero_undef)

    return sig, codegen


@njit(cache=False)
def _popcount64(bits: np.uint64) -> int:
    return _llvm_ctpop64(np.uint64(bits))  # type: ignore[call-arg]


@njit(cache=False)
def _bit_scan(bits: np.uint64) -> int:
    """Index of the lowest set bit, or -1 if `bits` is zero. Shared home for what used to be two
    separate copies (search.py and evaluate.py both had their own) -- see docs/plan.md Phase 1.7.
    Phase 2.2 swapped the bodies of both this and `_popcount64` for single-instruction LLVM
    intrinsics (`llvm.cttz`/`llvm.ctpop`) in place of a 64-iteration scan / Kernighan loop --
    see `tests/test_bit_ops.py` for the differential check against the old implementations.
    """
    if bits == 0:
        return -1
    return _llvm_cttz64(np.uint64(bits))  # type: ignore[call-arg]


@njit(cache=False)
def is_insufficient_material(bb: np.ndarray) -> bool:
    """Conservative dead-draw recognition, deliberately narrower than the full FIDE rule (or
    python-chess's own Board.is_insufficient_material(), which the harness's Board.outcome()
    already applies to the real game state as an automatic, non-claim terminal condition -- so
    this exists purely to keep the search's own eval of a hypothetical such position accurate
    while descending the tree, not to protect the real game result, which the harness already
    guarantees regardless). True only for: a bare king against a bare king; a king plus exactly
    one minor piece (knight or bishop) against a bare king; or any number of bishops -- with no
    pawns, knights, rooks, or queens anywhere -- all confined to squares of one colour. Every
    other combination (including ones the full rule also treats as insufficient, like a knight on
    each side) returns False, which is always the safe direction: eval simply runs as normal,
    exactly as if this check did not exist.
    """
    if (
        bb[WHITE * 6 + PAWN] or bb[BLACK * 6 + PAWN]
        or bb[WHITE * 6 + ROOK] or bb[BLACK * 6 + ROOK]
        or bb[WHITE * 6 + QUEEN] or bb[BLACK * 6 + QUEEN]
    ):
        return False
    knights = bb[WHITE * 6 + KNIGHT] | bb[BLACK * 6 + KNIGHT]
    bishops = bb[WHITE * 6 + BISHOP] | bb[BLACK * 6 + BISHOP]
    if knights:
        return bishops == 0 and _popcount64(knights) <= 1
    if bishops == 0:
        return True
    same_light = (bishops & SQUARE_COLOR_MASK) == bishops
    same_dark = (bishops & ~SQUARE_COLOR_MASK) == bishops
    return bool(same_light or same_dark)


@njit(cache=False)
def has_non_pawn_material(bb: np.ndarray, color: int) -> bool:
    """False in a king+pawn(s)-only position for `color` -- the null-move zugzwang guard in
    search.py (see its module docstring) skips null-move pruning here, since giving up a move for
    free is exactly the wrong test when every move actually available is zugzwang.
    """
    base = color * 6
    return bool(bb[base + KNIGHT] | bb[base + BISHOP] | bb[base + ROOK] | bb[base + QUEEN])


@njit(cache=False)
def make_null_move(meta: np.ndarray) -> np.ndarray:
    """The position with the side to move passing -- same board, turn flipped, en passant rights
    forfeited (a null move cannot itself be captured en passant, and the right does not survive a
    move that isn't the double pawn push that created it). Used only for null-move pruning, never
    as a real move.
    """
    new_meta = meta.copy()
    new_meta[0] = 1 - meta[0]
    new_meta[5] = NO_SQUARE
    return new_meta


# Phase 4.2 of docs/plan.md: an explicit eager signature -- not a strength change, insurance. See
# search.py's own eager-signature block (negamax/quiescence/search_root) for the full rationale;
# make_move is the one such function outside search.py, called from both search.py and agent.py's
# _search_restricted with args already uniformly int64 today (Phase 1.1(b) fixed the int8/int64
# drift that used to give this two specialisations), so this just makes that structural rather
# than merely observed.
_MAKE_MOVE_SIG = _nbtypes.Tuple(  # type: ignore[no-untyped-call]
    (_nbtypes.uint64[::1], _nbtypes.int8[::1])
)(_nbtypes.uint64[::1], _nbtypes.int8[::1], _nbtypes.int64, _nbtypes.int64, _nbtypes.int64)


@njit(_MAKE_MOVE_SIG, cache=False)
def make_move(
    bb: np.ndarray, meta: np.ndarray, from_sq: int, to_sq: int, promo: int
) -> tuple[np.ndarray, np.ndarray]:
    new_bb = bb.copy()
    new_meta = meta.copy()
    color = _i64(meta[0])
    opponent = 1 - color
    from_bit = ONE << np.uint64(from_sq)
    to_bit = ONE << np.uint64(to_sq)

    moving_pt = piece_type_at(bb, color, from_sq)

    for pt in range(6):
        new_bb[opponent * 6 + pt] &= ~to_bit

    ep_square = meta[5]
    if moving_pt == PAWN and to_sq == ep_square and (from_sq % 8) != (to_sq % 8):
        captured_sq = to_sq - 8 if color == WHITE else to_sq + 8
        new_bb[opponent * 6 + PAWN] &= ~(ONE << np.uint64(captured_sq))

    new_bb[color * 6 + moving_pt] &= ~from_bit
    if moving_pt == PAWN and promo >= 0:
        new_bb[color * 6 + promo] |= to_bit
    else:
        new_bb[color * 6 + moving_pt] |= to_bit

    if moving_pt == KING and abs(to_sq - from_sq) == 2:
        if to_sq > from_sq:
            rook_from, rook_to = from_sq + 3, from_sq + 1
        else:
            rook_from, rook_to = from_sq - 4, from_sq - 1
        new_bb[color * 6 + ROOK] &= ~(ONE << np.uint64(rook_from))
        new_bb[color * 6 + ROOK] |= ONE << np.uint64(rook_to)

    if moving_pt == KING:
        if color == WHITE:
            new_meta[1] = 0
            new_meta[2] = 0
        else:
            new_meta[3] = 0
            new_meta[4] = 0
    if from_sq == 0 or to_sq == 0:
        new_meta[2] = 0
    if from_sq == 7 or to_sq == 7:
        new_meta[1] = 0
    if from_sq == 56 or to_sq == 56:
        new_meta[4] = 0
    if from_sq == 63 or to_sq == 63:
        new_meta[3] = 0

    if moving_pt == PAWN and abs(to_sq - from_sq) == 16:
        new_meta[5] = np.int8((from_sq + to_sq) // 2)
    else:
        new_meta[5] = np.int8(-1)

    new_meta[0] = opponent
    return new_bb, new_meta


@njit(cache=False)
def generate_pseudo_legal(
    bb: np.ndarray, meta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    color = _i64(meta[0])
    opponent = 1 - color
    own = occ_color(bb, color)
    opp = occ_color(bb, opponent)
    all_occ = own | opp
    ep_square = meta[5]

    from_arr = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    to_arr = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    promo_arr = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    count = 0

    forward = 8 if color == WHITE else -8
    start_rank = 1 if color == WHITE else 6
    promo_rank = 7 if color == WHITE else 0

    pawn_bb = bb[color * 6 + PAWN]
    for sq in range(64):
        if not (pawn_bb & (ONE << np.uint64(sq))):
            continue
        rank = sq // 8
        target = sq + forward
        if 0 <= target < 64 and not (all_occ & (ONE << np.uint64(target))):
            if target // 8 == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    from_arr[count], to_arr[count], promo_arr[count] = sq, target, promo
                    count += 1
            else:
                from_arr[count], to_arr[count] = sq, target
                count += 1
            if rank == start_rank:
                target2 = sq + 2 * forward
                if not (all_occ & (ONE << np.uint64(target2))):
                    from_arr[count], to_arr[count] = sq, target2
                    count += 1
        attack_bits = PAWN_ATTACKS[color, sq]
        for t in range(64):
            if not (attack_bits & (ONE << np.uint64(t))):
                continue
            if opp & (ONE << np.uint64(t)):
                if t // 8 == promo_rank:
                    for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                        from_arr[count], to_arr[count], promo_arr[count] = sq, t, promo
                        count += 1
                else:
                    from_arr[count], to_arr[count] = sq, t
                    count += 1
            elif t == ep_square:
                from_arr[count], to_arr[count] = sq, t
                count += 1

    knight_bb = bb[color * 6 + KNIGHT]
    for sq in range(64):
        if not (knight_bb & (ONE << np.uint64(sq))):
            continue
        targets = KNIGHT_ATTACKS[sq] & ~own
        for t in range(64):
            if targets & (ONE << np.uint64(t)):
                from_arr[count], to_arr[count] = sq, t
                count += 1

    bishop_bb = bb[color * 6 + BISHOP]
    for sq in range(64):
        if not (bishop_bb & (ONE << np.uint64(sq))):
            continue
        targets = bishop_attacks(sq, all_occ) & ~own
        for t in range(64):
            if targets & (ONE << np.uint64(t)):
                from_arr[count], to_arr[count] = sq, t
                count += 1

    rook_bb = bb[color * 6 + ROOK]
    for sq in range(64):
        if not (rook_bb & (ONE << np.uint64(sq))):
            continue
        targets = rook_attacks(sq, all_occ) & ~own
        for t in range(64):
            if targets & (ONE << np.uint64(t)):
                from_arr[count], to_arr[count] = sq, t
                count += 1

    queen_bb = bb[color * 6 + QUEEN]
    for sq in range(64):
        if not (queen_bb & (ONE << np.uint64(sq))):
            continue
        targets = (rook_attacks(sq, all_occ) | bishop_attacks(sq, all_occ)) & ~own
        for t in range(64):
            if targets & (ONE << np.uint64(t)):
                from_arr[count], to_arr[count] = sq, t
                count += 1

    king_bb = bb[color * 6 + KING]
    for sq in range(64):
        if not (king_bb & (ONE << np.uint64(sq))):
            continue
        targets = KING_ATTACKS[sq] & ~own
        for t in range(64):
            if targets & (ONE << np.uint64(t)):
                from_arr[count], to_arr[count] = sq, t
                count += 1

        if color == WHITE and sq == 4:
            if (
                meta[1]
                and not (all_occ & ((ONE << np.uint64(5)) | (ONE << np.uint64(6))))
                and not attacked_by(bb, all_occ, BLACK, _SQ_C4)
                and not attacked_by(bb, all_occ, BLACK, _SQ_C5)
                and not attacked_by(bb, all_occ, BLACK, _SQ_C6)
            ):
                from_arr[count], to_arr[count] = 4, 6
                count += 1
            if (
                meta[2]
                and not (
                    all_occ
                    & ((ONE << np.uint64(1)) | (ONE << np.uint64(2)) | (ONE << np.uint64(3)))
                )
                and not attacked_by(bb, all_occ, BLACK, _SQ_C4)
                and not attacked_by(bb, all_occ, BLACK, _SQ_C3)
                and not attacked_by(bb, all_occ, BLACK, _SQ_C2)
            ):
                from_arr[count], to_arr[count] = 4, 2
                count += 1
        elif color == BLACK and sq == 60:
            if (
                meta[3]
                and not (all_occ & ((ONE << np.uint64(61)) | (ONE << np.uint64(62))))
                and not attacked_by(bb, all_occ, WHITE, _SQ_C60)
                and not attacked_by(bb, all_occ, WHITE, _SQ_C61)
                and not attacked_by(bb, all_occ, WHITE, _SQ_C62)
            ):
                from_arr[count], to_arr[count] = 60, 62
                count += 1
            if (
                meta[4]
                and not (
                    all_occ
                    & ((ONE << np.uint64(57)) | (ONE << np.uint64(58)) | (ONE << np.uint64(59)))
                )
                and not attacked_by(bb, all_occ, WHITE, _SQ_C60)
                and not attacked_by(bb, all_occ, WHITE, _SQ_C59)
                and not attacked_by(bb, all_occ, WHITE, _SQ_C58)
            ):
                from_arr[count], to_arr[count] = 60, 58
                count += 1

    return from_arr, to_arr, promo_arr, count


@njit(cache=False)
def generate_legal(
    bb: np.ndarray, meta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from_arr, to_arr, promo_arr, count = generate_pseudo_legal(bb, meta)
    color = _i64(meta[0])

    out_from = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    out_to = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    out_promo = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    out_count = 0

    for i in range(count):
        f, t, p = _i64(from_arr[i]), _i64(to_arr[i]), _i64(promo_arr[i])
        new_bb, _new_meta = make_move(bb, meta, f, t, p)
        ksq = king_square(new_bb, color)
        if not attacked_by(new_bb, occ_all(new_bb), 1 - color, ksq):
            out_from[out_count], out_to[out_count], out_promo[out_count] = f, t, p
            out_count += 1

    return out_from, out_to, out_promo, out_count
