"""Material + piece-square evaluation, from the perspective of the side to move (negamax sign
convention: positive is good for whoever moves next).

Tables are the common "simplified evaluation" piece-square values, white's orientation, rank 1
at index 0. Black's score uses the same tables mirrored vertically (square ^ 56 flips the rank,
keeps the file), so there is one table per piece, not two. The king additionally has a second,
endgame-only table (centralizing instead of hiding in a castled corner), blended in by game phase
-- a king that wants safety mid-game is instead an active piece needed for the win once material
has thinned out.

Beyond material + PST, four cheap positional terms are added, all in the same white-minus-black,
own-file/precomputed-mask style as the rest of this module so they stay jit-friendly without a
second pass over the board: pawn structure (doubled, isolated, passed), bishop pair, rooks on
open/semi-open files, and a king pawn-shield bonus that -- like the king PST -- fades out via the
same phase blend as material comes off the board.
"""

import numpy as np
from numba import njit

from bitboard import BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE
from movegen import generate_legal, king_square

PIECE_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)

_PAWN_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_ROOK_PST = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_QUEEN_PST = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_KING_PST = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
_KING_PST_ENDGAME = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]

PST = np.array(
    [_PAWN_PST, _KNIGHT_PST, _BISHOP_PST, _ROOK_PST, _QUEEN_PST, _KING_PST], dtype=np.int32
)
PST_KING_END = np.array(_KING_PST_ENDGAME, dtype=np.int32)

# Standard phase weighting: knight/bishop 1, rook 2, queen 4, pawn/king 0 -- 4*1 + 4*1 + 4*2 + 2*4
# = 24 at the start of the game, tapering to 0 as non-pawn material comes off.
PHASE_WEIGHT = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
PHASE_MAX = 24

ONE = np.uint64(1)
MOBILITY_WEIGHT = np.int32(2)
BISHOP_PAIR_BONUS = np.int32(30)
DOUBLED_PAWN_PENALTY = np.int32(10)
ISOLATED_PAWN_PENALTY = np.int32(15)
PASSED_PAWN_BONUS = np.array([0, 5, 10, 20, 35, 60, 100, 0], dtype=np.int32)
ROOK_SEMI_OPEN_BONUS = np.int32(10)
ROOK_OPEN_BONUS = np.int32(20)
KING_SHIELD_BONUS = np.int32(8)


def _build_file_masks() -> np.ndarray:
    masks = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        bits = np.uint64(0)
        for r in range(8):
            bits |= ONE << np.uint64(r * 8 + f)
        masks[f] = bits
    return masks


def _build_adjacent_file_masks(file_masks: np.ndarray) -> np.ndarray:
    masks = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        bits = np.uint64(0)
        if f > 0:
            bits |= file_masks[f - 1]
        if f < 7:
            bits |= file_masks[f + 1]
        masks[f] = bits
    return masks


def _build_passed_masks(white: bool) -> np.ndarray:
    """Per square, the enemy-pawn squares (own file + adjacent files, every rank strictly ahead
    in the pawn's direction of travel) that would contest or block a pawn there from queening.
    """
    masks = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file, rank = square % 8, square // 8
        files = [f for f in (file - 1, file, file + 1) if 0 <= f < 8]
        ranks = range(rank + 1, 8) if white else range(0, rank)
        bits = np.uint64(0)
        for f in files:
            for r in ranks:
                bits |= ONE << np.uint64(r * 8 + f)
        masks[square] = bits
    return masks


def _build_king_shield_masks(white: bool) -> np.ndarray:
    """Per king square, the two ranks directly ahead (in the king's own forward direction) across
    the king's file and its neighbours -- where a pawn shield would sit.
    """
    masks = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file, rank = square % 8, square // 8
        bits = np.uint64(0)
        for dr in ((1, 2) if white else (-1, -2)):
            r = rank + dr
            if 0 <= r < 8:
                for df in (-1, 0, 1):
                    f = file + df
                    if 0 <= f < 8:
                        bits |= ONE << np.uint64(r * 8 + f)
        masks[square] = bits
    return masks


FILE_MASKS = _build_file_masks()
ADJACENT_FILE_MASKS = _build_adjacent_file_masks(FILE_MASKS)
PASSED_MASK_WHITE = _build_passed_masks(True)
PASSED_MASK_BLACK = _build_passed_masks(False)
KING_SHIELD_WHITE = _build_king_shield_masks(True)
KING_SHIELD_BLACK = _build_king_shield_masks(False)


@njit(cache=False)
def _popcount64(bits: np.uint64) -> int:
    count = 0
    while bits:
        bits &= bits - ONE
        count += 1
    return count


@njit(cache=False)
def game_phase(bb: np.ndarray) -> int:
    phase = 0
    for color in range(2):
        phase += _popcount64(bb[color * 6 + KNIGHT]) * PHASE_WEIGHT[KNIGHT]
        phase += _popcount64(bb[color * 6 + BISHOP]) * PHASE_WEIGHT[BISHOP]
        phase += _popcount64(bb[color * 6 + ROOK]) * PHASE_WEIGHT[ROOK]
        phase += _popcount64(bb[color * 6 + QUEEN]) * PHASE_WEIGHT[QUEEN]
    return phase if phase < PHASE_MAX else PHASE_MAX


@njit(cache=False)
def material_and_pst(bb: np.ndarray, phase: int) -> int:
    score = np.int32(0)
    for pt in range(6):
        white_bb = bb[pt]
        black_bb = bb[6 + pt]
        for square in range(64):
            bit = ONE << np.uint64(square)
            if white_bb & bit:
                if pt == KING:
                    mg, eg = PST[pt, square], PST_KING_END[square]
                    score += PIECE_VALUE[pt] + (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
                else:
                    score += PIECE_VALUE[pt] + PST[pt, square]
            if black_bb & bit:
                mirror = square ^ 56
                if pt == KING:
                    mg, eg = PST[pt, mirror], PST_KING_END[mirror]
                    score -= PIECE_VALUE[pt] + (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
                else:
                    score -= PIECE_VALUE[pt] + PST[pt, mirror]
    return int(score)


@njit(cache=False)
def pawn_structure(bb: np.ndarray) -> int:
    score = np.int32(0)
    white_pawns = bb[WHITE * 6 + PAWN]
    black_pawns = bb[BLACK * 6 + PAWN]

    for f in range(8):
        wc = _popcount64(white_pawns & FILE_MASKS[f])
        bc = _popcount64(black_pawns & FILE_MASKS[f])
        if wc > 1:
            score -= DOUBLED_PAWN_PENALTY * (wc - 1)
        if bc > 1:
            score += DOUBLED_PAWN_PENALTY * (bc - 1)

    for square in range(64):
        bit = ONE << np.uint64(square)
        f = square % 8
        if white_pawns & bit:
            if white_pawns & ADJACENT_FILE_MASKS[f] == 0:
                score -= ISOLATED_PAWN_PENALTY
            if black_pawns & PASSED_MASK_WHITE[square] == 0:
                score += PASSED_PAWN_BONUS[square // 8]
        if black_pawns & bit:
            if black_pawns & ADJACENT_FILE_MASKS[f] == 0:
                score += ISOLATED_PAWN_PENALTY
            if white_pawns & PASSED_MASK_BLACK[square] == 0:
                score -= PASSED_PAWN_BONUS[7 - square // 8]
    return int(score)


@njit(cache=False)
def piece_features(bb: np.ndarray, phase: int) -> int:
    score = np.int32(0)
    white_pawns = bb[WHITE * 6 + PAWN]
    black_pawns = bb[BLACK * 6 + PAWN]

    if _popcount64(bb[WHITE * 6 + BISHOP]) >= 2:
        score += BISHOP_PAIR_BONUS
    if _popcount64(bb[BLACK * 6 + BISHOP]) >= 2:
        score -= BISHOP_PAIR_BONUS

    for square in range(64):
        bit = ONE << np.uint64(square)
        f = square % 8
        if bb[WHITE * 6 + ROOK] & bit and white_pawns & FILE_MASKS[f] == 0:
            open_file = black_pawns & FILE_MASKS[f] == 0
            score += ROOK_OPEN_BONUS if open_file else ROOK_SEMI_OPEN_BONUS
        if bb[BLACK * 6 + ROOK] & bit and black_pawns & FILE_MASKS[f] == 0:
            open_file = white_pawns & FILE_MASKS[f] == 0
            score -= ROOK_OPEN_BONUS if open_file else ROOK_SEMI_OPEN_BONUS

    if phase > 0:
        wk = king_square(bb, WHITE)
        shield = _popcount64(white_pawns & KING_SHIELD_WHITE[wk])
        score += KING_SHIELD_BONUS * shield * phase // PHASE_MAX
        bk = king_square(bb, BLACK)
        shield_b = _popcount64(black_pawns & KING_SHIELD_BLACK[bk])
        score -= KING_SHIELD_BONUS * shield_b * phase // PHASE_MAX

    return int(score)


@njit(cache=False)
def evaluate(bb: np.ndarray, meta: np.ndarray, mobility: int) -> int:
    phase = game_phase(bb)
    score = material_and_pst(bb, phase) + pawn_structure(bb) + piece_features(bb, phase)
    signed = score if meta[0] == WHITE else -score
    return int(signed + MOBILITY_WEIGHT * mobility)


def warm_up() -> None:
    from bitboard import from_fen

    bb, meta = from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    _, _, _, count = generate_legal(bb, meta)
    evaluate(bb, meta, count)
