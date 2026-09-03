"""Pseudo-legal and legal move generation over the bitboard representation in bitboard.py.

Legality is checked the simple way: generate pseudo-legal, make the move, see if the mover's
own king is attacked. Not the fastest approach, but every check reuses the same `attacked_by`
and sliding-attack code that everything else does, so there is only one place attack logic can
be wrong instead of two.

Move lists are fixed-size int8 arrays (from, to, promo) plus a count, not Python lists, so this
stays jit-friendly. MAX_MOVES is well above the theoretical maximum legal moves in any reachable
chess position (218).
"""

import numpy as np
from numba import njit

from attacks import KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, bishop_attacks, rook_attacks
from bitboard import BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE

MAX_MOVES = 256
ONE = np.uint64(1)
NO_SQUARE = np.int8(-1)


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
    color = meta[0]
    return attacked_by(bb, occ_all(bb), 1 - color, king_square(bb, color))


@njit(cache=False)
def piece_type_at(bb: np.ndarray, color: int, square: int) -> int:
    bit = ONE << np.uint64(square)
    for pt in range(6):
        if bb[color * 6 + pt] & bit:
            return pt
    return -1


@njit(cache=False)
def make_move(
    bb: np.ndarray, meta: np.ndarray, from_sq: int, to_sq: int, promo: int
) -> tuple[np.ndarray, np.ndarray]:
    new_bb = bb.copy()
    new_meta = meta.copy()
    color = meta[0]
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
    color = meta[0]
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
                and not attacked_by(bb, all_occ, BLACK, 4)
                and not attacked_by(bb, all_occ, BLACK, 5)
                and not attacked_by(bb, all_occ, BLACK, 6)
            ):
                from_arr[count], to_arr[count] = 4, 6
                count += 1
            if (
                meta[2]
                and not (
                    all_occ
                    & ((ONE << np.uint64(1)) | (ONE << np.uint64(2)) | (ONE << np.uint64(3)))
                )
                and not attacked_by(bb, all_occ, BLACK, 4)
                and not attacked_by(bb, all_occ, BLACK, 3)
                and not attacked_by(bb, all_occ, BLACK, 2)
            ):
                from_arr[count], to_arr[count] = 4, 2
                count += 1
        elif color == BLACK and sq == 60:
            if (
                meta[3]
                and not (all_occ & ((ONE << np.uint64(61)) | (ONE << np.uint64(62))))
                and not attacked_by(bb, all_occ, WHITE, 60)
                and not attacked_by(bb, all_occ, WHITE, 61)
                and not attacked_by(bb, all_occ, WHITE, 62)
            ):
                from_arr[count], to_arr[count] = 60, 62
                count += 1
            if (
                meta[4]
                and not (
                    all_occ
                    & ((ONE << np.uint64(57)) | (ONE << np.uint64(58)) | (ONE << np.uint64(59)))
                )
                and not attacked_by(bb, all_occ, WHITE, 60)
                and not attacked_by(bb, all_occ, WHITE, 59)
                and not attacked_by(bb, all_occ, WHITE, 58)
            ):
                from_arr[count], to_arr[count] = 60, 58
                count += 1

    return from_arr, to_arr, promo_arr, count


@njit(cache=False)
def generate_legal(
    bb: np.ndarray, meta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from_arr, to_arr, promo_arr, count = generate_pseudo_legal(bb, meta)
    color = meta[0]

    out_from = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    out_to = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    out_promo = np.full(MAX_MOVES, NO_SQUARE, dtype=np.int8)
    out_count = 0

    for i in range(count):
        f, t, p = from_arr[i], to_arr[i], promo_arr[i]
        new_bb, _new_meta = make_move(bb, meta, f, t, p)
        ksq = king_square(new_bb, color)
        if not attacked_by(new_bb, occ_all(new_bb), 1 - color, ksq):
            out_from[out_count], out_to[out_count], out_promo[out_count] = f, t, p
            out_count += 1

    return out_from, out_to, out_promo, out_count
