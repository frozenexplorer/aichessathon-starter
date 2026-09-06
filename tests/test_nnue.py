"""Differential tests for nnue.py's Phase 3 (docs/plan.md) Stage A scaffolding.

Not a strength test -- there is no trained network yet. This only checks that the feature
encoding and the integer forward pass are *structurally correct*: they agree with a plain-Python
reference implementation, the same bar tests/test_zobrist_hash.py and tests/test_bit_ops.py hold
their njit counterparts to. Nothing here is wired into evaluate.py or agent.py yet.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bitboard import BLACK, WHITE, from_fen
from nnue import (
    HIDDEN1,
    HIDDEN2,
    INPUT_FEATURES,
    MAX_ACTIVE_FEATURES,
    OUTPUT_SCALE,
    W1_SCALE,
    W2_SCALE,
    extract_features,
    feature_index,
    nnue_forward,
)

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r2qr2k/pp5p/8/3nPbp1/3B1P1b/1BPp4/PQ4P1/R5KR b - - 3 22",
    "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
]


def _reference_feature_index(color: int, piece_type: int, square: int, side_to_move: int) -> int:
    if side_to_move == WHITE:
        rel_color, rel_square = color, square
    else:
        rel_color, rel_square = 1 - color, square ^ 56
    return (rel_color * 6 + piece_type) * 64 + rel_square


def _reference_extract_features(bb: np.ndarray, meta: np.ndarray) -> list[int]:
    side_to_move = int(meta[0])
    out = []
    for color in range(2):
        for piece_type in range(6):
            bits = int(bb[color * 6 + piece_type])
            for square in range(64):
                if bits & (1 << square):
                    out.append(_reference_feature_index(color, piece_type, square, side_to_move))
    return sorted(out)


def _reference_forward(
    active: list[int],
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
) -> int:
    a1 = b1.astype(np.int64).copy()
    for f in active:
        a1 += w1[f, :].astype(np.int64)
    a1 = np.maximum(a1, 0)

    a2 = b2.astype(np.int64) * W1_SCALE + a1 @ w2.astype(np.int64)
    a2 = np.maximum(a2, 0)

    total3 = int(b3[0]) * W1_SCALE * W2_SCALE + int(a2 @ w3.astype(np.int64))
    return total3 // OUTPUT_SCALE


def test_feature_index() -> None:
    random.seed(0)
    mismatches = 0
    total = 0
    for _ in range(2000):
        color = random.randint(0, 1)
        piece_type = random.randint(0, 5)
        square = random.randint(0, 63)
        stm = random.choice([WHITE, BLACK])
        got = feature_index(color, piece_type, square, stm)
        want = _reference_feature_index(color, piece_type, square, stm)
        total += 1
        if got != want:
            mismatches += 1
    print(f"{total - mismatches}/{total} feature_index queries agree")
    assert mismatches == 0
    assert 0 <= feature_index(1, 5, 63, BLACK) < INPUT_FEATURES


def test_extract_features() -> None:
    mismatches = 0
    for fen in FENS:
        bb, meta = from_fen(fen)
        got = sorted(int(x) for x in extract_features(bb, meta) if x >= 0)
        want = _reference_extract_features(bb, meta)
        if got != want:
            mismatches += 1
            print(f"MISMATCH on {fen}: got={got} want={want}")
    print(f"{len(FENS) - mismatches}/{len(FENS)} positions' feature sets agree")
    assert mismatches == 0


def test_forward_pass() -> None:
    rng = np.random.default_rng(42)
    mismatches = 0
    trials = 25
    for _ in range(trials):
        w1 = rng.integers(-200, 200, size=(INPUT_FEATURES, HIDDEN1), dtype=np.int16)
        b1 = rng.integers(-500, 500, size=HIDDEN1, dtype=np.int32)
        w2 = rng.integers(-200, 200, size=(HIDDEN1, HIDDEN2), dtype=np.int16)
        b2 = rng.integers(-500, 500, size=HIDDEN2, dtype=np.int32)
        w3 = rng.integers(-200, 200, size=HIDDEN2, dtype=np.int16)
        b3 = rng.integers(-500, 500, size=1, dtype=np.int64)

        n_active = rng.integers(0, MAX_ACTIVE_FEATURES)
        active = rng.choice(INPUT_FEATURES, size=n_active, replace=False)
        active_padded = np.full(MAX_ACTIVE_FEATURES, -1, dtype=np.int64)
        active_padded[:n_active] = active

        got = nnue_forward(active_padded, w1, b1, w2, b2, w3, b3)
        want = _reference_forward(list(active), w1, b1, w2, b2, w3, b3)
        if got != want:
            mismatches += 1
            print(f"MISMATCH: got={got} want={want} n_active={n_active}")
    print(f"{trials - mismatches}/{trials} forward-pass trials agree")
    assert mismatches == 0


def main() -> None:
    test_feature_index()
    test_extract_features()
    test_forward_pass()
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
