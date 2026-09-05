"""Zobrist hashing, for repetition detection only -- never compared across processes or matched
against any external table, so a fixed seed is fine; it only needs to be internally consistent
within one run.
"""

import numpy as np
from numba import njit

from movegen import _bit_scan

_RNG = np.random.default_rng(0xC5E55A7)


def _random_u64(size: int | tuple[int, ...]) -> np.ndarray:
    hi = _RNG.integers(0, 2**32, size=size, dtype=np.uint64)
    lo = _RNG.integers(0, 2**32, size=size, dtype=np.uint64)
    return (hi << np.uint64(32)) | lo


PIECE_KEYS = _random_u64((12, 64))
TURN_KEY = _random_u64(1)[0]
CASTLING_KEYS = _random_u64(4)
EP_KEYS = _random_u64(64)

ONE = np.uint64(1)


@njit(cache=False)
def position_hash(bb: np.ndarray, meta: np.ndarray) -> np.uint64:
    """Phase 2.3 of docs/plan.md: iterate only the set bits of each piece bitboard (via the
    `_llvm_cttz64`-backed `_bit_scan`, ~32 iterations for a full board) instead of scanning all 64
    squares regardless of occupancy -- called on every negamax node plus twice per candidate move
    in claim_eligible_for_opponent, so this is one of the hottest loops in the engine.
    """
    h = np.uint64(0)
    for i in range(12):
        remaining = bb[i]
        while remaining:
            square = _bit_scan(remaining)
            h ^= PIECE_KEYS[i, square]
            remaining &= remaining - ONE
    if meta[0] == 1:
        h ^= TURN_KEY
    for i in range(4):
        if meta[1 + i]:
            h ^= CASTLING_KEYS[i]
    if meta[5] >= 0:
        h ^= EP_KEYS[meta[5]]
    return h
