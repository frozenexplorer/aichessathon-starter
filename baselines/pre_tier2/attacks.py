"""Attack tables and sliding-piece attack generation.

Knight, king and pawn attacks are fixed per square, so they are precomputed once at import.
Bishop, rook and queen attacks depend on occupancy, so they are ray-cast on demand -- one step
at a time in each direction until the board edge or a blocker, rather than magic bitboards.
Slower per query, far simpler to get right, and jitted so it is still fast.
"""

import numpy as np
from numba import njit

from bitboard import BLACK, WHITE

ONE = np.uint64(1)


def _in_bounds(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def _build_knight_attacks() -> np.ndarray:
    table = np.zeros(64, dtype=np.uint64)
    deltas = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    for square in range(64):
        file, rank = square % 8, square // 8
        bits = np.uint64(0)
        for df, dr in deltas:
            if _in_bounds(file + df, rank + dr):
                target = (rank + dr) * 8 + (file + df)
                bits |= ONE << np.uint64(target)
        table[square] = bits
    return table


def _build_king_attacks() -> np.ndarray:
    table = np.zeros(64, dtype=np.uint64)
    deltas = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for square in range(64):
        file, rank = square % 8, square // 8
        bits = np.uint64(0)
        for df, dr in deltas:
            if _in_bounds(file + df, rank + dr):
                target = (rank + dr) * 8 + (file + df)
                bits |= ONE << np.uint64(target)
        table[square] = bits
    return table


def _build_pawn_attacks() -> np.ndarray:
    table = np.zeros((2, 64), dtype=np.uint64)
    for square in range(64):
        file, rank = square % 8, square // 8
        for color, dr in ((WHITE, 1), (BLACK, -1)):
            bits = np.uint64(0)
            for df in (-1, 1):
                if _in_bounds(file + df, rank + dr):
                    target = (rank + dr) * 8 + (file + df)
                    bits |= ONE << np.uint64(target)
            table[color, square] = bits
    return table


KNIGHT_ATTACKS = _build_knight_attacks()
KING_ATTACKS = _build_king_attacks()
PAWN_ATTACKS = _build_pawn_attacks()


@njit(cache=False)
def rook_attacks(square: int, occupied: np.uint64) -> np.uint64:
    file, rank = square % 8, square // 8
    bits = np.uint64(0)
    for df, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        f, r = file + df, rank + dr
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            bits |= ONE << np.uint64(target)
            if occupied & (ONE << np.uint64(target)):
                break
            f += df
            r += dr
    return bits


@njit(cache=False)
def bishop_attacks(square: int, occupied: np.uint64) -> np.uint64:
    file, rank = square % 8, square // 8
    bits = np.uint64(0)
    for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        f, r = file + df, rank + dr
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            bits |= ONE << np.uint64(target)
            if occupied & (ONE << np.uint64(target)):
                break
            f += df
            r += dr
    return bits


@njit(cache=False)
def queen_attacks(square: int, occupied: np.uint64) -> np.uint64:
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)
