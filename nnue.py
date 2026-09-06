"""Phase 3 (docs/plan.md) -- Stage A NNUE evaluation, `768 -> 512 -> 32 -> 1`.

**Not wired in yet.** Nothing in `agent.py` or `evaluate.py` imports this module, so it costs
zero init time and has zero effect on play until a trained `weights/nnue.npz` exists and an
arena run shows it beats the handcrafted eval (docs/plan.md 3.4). This file only defines the
feature encoding and the forward pass, both differential-tested in `tests/test_nnue.py` against
plain-Python reference implementations, the same pattern `tests/test_zobrist_hash.py` and
`tests/test_bit_ops.py` use for their njit counterparts.

Feature encoding: side-to-move perspective, one feature per (colour, piece type, square) tuple,
768 = 12 * 64. From Black's perspective the board is rank-flipped (`square ^ 56`) and the colour
labels are swapped, so the same weights see "my piece on my back rank" the same way regardless of
which side is actually moving -- the standard trick that lets one set of weights serve both
colours instead of training two.

Quantisation: every weight is stored as `int16`, the fixed-point encoding of a real-valued weight
times that layer's `_SCALE`. Because `ReLU(x / s) == ReLU(x) / s` for any `s > 0`, applying ReLU
to the *scaled* integer accumulator and only dividing out the cumulative scale at the very end is
exactly equivalent to running the same network in real arithmetic -- not an approximation, just a
reordering. `int64` accumulators throughout make overflow a non-issue at these layer sizes.
"""

import numpy as np
from numba import njit

from bitboard import WHITE

NUM_SQUARES = 64
NUM_PIECE_TYPES = 6
NUM_COLORS = 2
INPUT_FEATURES = NUM_COLORS * NUM_PIECE_TYPES * NUM_SQUARES  # 768
HIDDEN1 = 512
HIDDEN2 = 32
MAX_ACTIVE_FEATURES = 32  # at most 32 pieces on the board

W1_SCALE = 64
W2_SCALE = 64
W3_SCALE = 64
OUTPUT_SCALE = W1_SCALE * W2_SCALE * W3_SCALE


@njit(cache=False)
def feature_index(color: int, piece_type: int, square: int, side_to_move: int) -> int:
    """Maps a (colour, piece type, square) tuple to its side-to-move-relative feature index."""
    if side_to_move == WHITE:
        rel_color = color
        rel_square = square
    else:
        rel_color = 1 - color
        rel_square = square ^ 56
    return (rel_color * NUM_PIECE_TYPES + piece_type) * NUM_SQUARES + rel_square


@njit(cache=False)
def extract_features(bb: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Returns the (at most 32) active feature indices for this position, -1-padded."""
    side_to_move = meta[0]
    out = np.full(MAX_ACTIVE_FEATURES, -1, dtype=np.int64)
    n = 0
    for color in range(NUM_COLORS):
        for piece_type in range(NUM_PIECE_TYPES):
            bits = bb[color * NUM_PIECE_TYPES + piece_type]
            while bits != np.uint64(0):
                lsb = bits & (~bits + np.uint64(1))
                square = 0
                probe = lsb
                while probe != np.uint64(1):
                    probe >>= np.uint64(1)
                    square += 1
                out[n] = feature_index(color, piece_type, square, side_to_move)
                n += 1
                bits &= bits - np.uint64(1)
    return out


@njit(cache=False)
def nnue_forward(
    active_features: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
) -> int:
    """Integer forward pass. `w1` is `[768, 512]`, `w2` is `[512, 32]`, `w3` is `[32]`, all int16;
    `b1`/`b2` int32, `b3` int64. Returns a centipawn score from the side-to-move's perspective.
    """
    acc1 = np.zeros(HIDDEN1, dtype=np.int64)
    for j in range(HIDDEN1):
        acc1[j] = b1[j]
    for i in range(active_features.shape[0]):
        f = active_features[i]
        if f < 0:
            continue
        for j in range(HIDDEN1):
            acc1[j] += w1[f, j]
    for j in range(HIDDEN1):
        if acc1[j] < 0:
            acc1[j] = 0

    acc2 = np.zeros(HIDDEN2, dtype=np.int64)
    for k in range(HIDDEN2):
        total = np.int64(b2[k]) * W1_SCALE
        for j in range(HIDDEN1):
            total += acc1[j] * w2[j, k]
        acc2[k] = total if total > 0 else 0

    total3 = np.int64(b3[0]) * W1_SCALE * W2_SCALE
    for k in range(HIDDEN2):
        total3 += acc2[k] * w3[k]

    return int(total3 // OUTPUT_SCALE)


def zero_weights() -> tuple[np.ndarray, ...]:
    """Structurally-valid, all-zero weight set -- for shape/plumbing tests only, never for play."""
    w1 = np.zeros((INPUT_FEATURES, HIDDEN1), dtype=np.int16)
    b1 = np.zeros(HIDDEN1, dtype=np.int32)
    w2 = np.zeros((HIDDEN1, HIDDEN2), dtype=np.int16)
    b2 = np.zeros(HIDDEN2, dtype=np.int32)
    w3 = np.zeros(HIDDEN2, dtype=np.int16)
    b3 = np.zeros(1, dtype=np.int64)
    return w1, b1, w2, b2, w3, b3


def load_weights(path: str) -> tuple[np.ndarray, ...]:
    data = np.load(path)
    return (
        data["w1"].astype(np.int16),
        data["b1"].astype(np.int32),
        data["w2"].astype(np.int16),
        data["b2"].astype(np.int32),
        data["w3"].astype(np.int16),
        data["b3"].astype(np.int64),
    )


def save_weights(
    path: str,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        w1=w1.astype(np.int16),
        b1=b1.astype(np.int32),
        w2=w2.astype(np.int16),
        b2=b2.astype(np.int32),
        w3=w3.astype(np.int16),
        b3=b3.astype(np.int64),
    )
