"""Direct tests of search.py's fifty-move-rule draw detection: negamax must score a position as
an immediate draw (0) once halfmove_clock reaches HALFMOVE_DRAW_LIMIT, the same way it already
does for a genuine threefold repetition -- and must NOT draw early, one ply short of the limit,
even with a real material edge on the board (so a false-positive draw claim would be obvious).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bitboard as bbm
import search as sr

# White to move, up a rook -- unambiguous material edge, same shape as test_repetition.py's own
# fixture, so a draw score (0) is clearly distinguishable from a normal eval-based score.
UP_A_ROOK_FEN = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"


def call_negamax(halfmove_clock: int) -> int:
    bb, meta = bbm.from_fen(UP_A_ROOK_FEN)
    history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
    counters = sr.new_counters()
    deadline = 1e18
    tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo = sr.new_tt()
    killer_from, killer_to, killer_promo = sr.new_killers()
    history_table = sr.new_history_table()
    counter_from, counter_to, counter_promo = sr.new_counter_table()
    return sr.negamax(
        bb, meta, 0, -sr.INF, sr.INF, deadline, counters, 0, history, 0,
        tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
        killer_from, killer_to, killer_promo, history_table, True, sr.MAX_CHECK_EXTENSIONS,
        counter_from, counter_to, counter_promo, -1, -1, halfmove_clock, -1, -1,
    )


def main() -> None:
    failures = 0

    under_limit = call_negamax(sr.HALFMOVE_DRAW_LIMIT - 1)
    print(f"halfmove_clock=limit-1: score={under_limit} (expect a real material edge, not 0)")
    if under_limit <= 100:
        print("  FAIL: one ply short of the limit must not be scored as a draw")
        failures += 1

    at_limit = call_negamax(sr.HALFMOVE_DRAW_LIMIT)
    print(f"halfmove_clock=limit: score={at_limit} (expect an immediate draw, 0)")
    if at_limit != 0:
        print("  FAIL: reaching HALFMOVE_DRAW_LIMIT must score as an immediate draw (0)")
        failures += 1

    over_limit = call_negamax(sr.HALFMOVE_DRAW_LIMIT + 5)
    print(f"halfmove_clock=limit+5: score={over_limit} (expect an immediate draw, 0)")
    if over_limit != 0:
        print("  FAIL: past the limit must still score as an immediate draw (0)")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
