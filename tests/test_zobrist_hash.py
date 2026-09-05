"""Differential test for zobrist.position_hash (Phase 2.3 of docs/plan.md): the optimized version
iterates only set bits (via _bit_scan) instead of scanning all 64 squares of all 12 bitboards
regardless of occupancy. Checked here against a plain-Python full-scan reference -- the same
algorithm the njit version used before this change -- over a mix of real FENs (opening, midgame,
endgame, en-passant- and castling-eligible positions) so every meta field gets exercised.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bitboard as bbm
import zobrist as zb

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 6",
    "r2qr2k/pp5p/8/3nPbp1/3B1P1b/1BPp4/PQ4P1/R5KR b - - 3 22",
    "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
    "8/8/8/8/8/4k3/4p3/4K3 b - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "k7/8/2K5/8/8/8/8/8 b - - 0 1",
]


def _reference_hash(bb: np.ndarray, meta: np.ndarray) -> int:
    h = np.uint64(0)
    one = np.uint64(1)
    for i in range(12):
        piece_bb = bb[i]
        for square in range(64):
            if piece_bb & (one << np.uint64(square)):
                h ^= zb.PIECE_KEYS[i, square]
    if meta[0] == 1:
        h ^= zb.TURN_KEY
    for i in range(4):
        if meta[1 + i]:
            h ^= zb.CASTLING_KEYS[i]
    if meta[5] >= 0:
        h ^= zb.EP_KEYS[meta[5]]
    return int(h)


def main() -> None:
    mismatches = 0
    for fen in FENS:
        bb, meta = bbm.from_fen(fen)
        got = int(zb.position_hash(bb, meta))
        want = _reference_hash(bb, meta)
        if got != want:
            mismatches += 1
            print(f"MISMATCH fen={fen!r}: got={got} != want={want}")

    print(f"{len(FENS) - mismatches}/{len(FENS)} position hashes agree")
    if mismatches:
        print(f"\n{mismatches} MISMATCH(ES)")
        raise SystemExit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
