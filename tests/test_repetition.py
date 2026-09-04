"""Direct test of the repetition-draw mechanism in search.py: a position with a known material
edge should score as a real advantage with no repetition history, as exactly 0 (draw) once its
hash already appears twice in history (this would be the 3rd occurrence), and NOT as a draw with
only one prior occurrence, or when the match is at an odd ply (the opponent's hypothetical turn,
which is never tracked -- see search.py's module docstring for why that is sufficient).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bitboard as bbm
import search as sr
import zobrist as zb

# White to move, up a rook -- unambiguous material edge so a draw score (0) is clearly
# distinguishable from a normal eval-based score.
UP_A_ROOK_FEN = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"


def fresh_history() -> np.ndarray:
    return np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)


def call_negamax(hist_len: int, ply: int = 0) -> int:
    bb, meta = bbm.from_fen(UP_A_ROOK_FEN)
    h = zb.position_hash(bb, meta)
    history = fresh_history()
    for i in range(hist_len):
        history[i] = h
    counters = sr.new_counters()
    deadline = 1e18
    return sr.negamax(bb, meta, 0, -sr.INF, sr.INF, deadline, counters, ply, history, hist_len)


def main() -> None:
    failures = 0

    no_repeat = call_negamax(hist_len=0, ply=0)
    print(f"no prior occurrences, our turn: score={no_repeat} (expect a real material edge)")
    if no_repeat <= 100:
        print("  FAIL: expected a clear positive score, material edge was not reflected")
        failures += 1

    one_prior = call_negamax(hist_len=1, ply=0)
    print(f"one prior occurrence (this would be the 2nd), our turn: score={one_prior}")
    if one_prior <= 100:
        print("  FAIL: one prior occurrence must not be scored as a draw")
        failures += 1

    two_prior = call_negamax(hist_len=2, ply=0)
    print(f"two prior occurrences (this would be the 3rd), our turn: score={two_prior}")
    if two_prior != 0:
        print("  FAIL: a 3rd occurrence must score as an immediate draw (0)")
        failures += 1

    odd_ply_with_matches = call_negamax(hist_len=2, ply=1)
    print(f"two prior occurrences, but opponent's (odd) ply: score={odd_ply_with_matches}")
    if odd_ply_with_matches == 0:
        print("  FAIL: odd-ply nodes must never be checked against history")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
