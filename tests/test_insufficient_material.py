"""Direct tests of movegen.is_insufficient_material and its use in search.py's negamax/quiescence
(both must score such a position as an immediate draw, 0, the same way they already do for
threefold repetition and the fifty-move rule). Deliberately conservative (see its own docstring):
covers bare king vs bare king, king plus one minor vs bare king, and same-coloured-square bishops
only -- and stays False (the safe direction) for combinations the full FIDE rule also excludes,
like a knight on each side, so those are checked here too, expecting False.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bitboard as bbm
import movegen as mg
import search as sr


def call_negamax(fen: str) -> int:
    bb, meta = bbm.from_fen(fen)
    history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
    counters = sr.new_counters()
    deadline = 1e18
    tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo = sr.new_tt()
    killer_from, killer_to, killer_promo = sr.new_killers()
    history_table = sr.new_history_table()
    counter_from, counter_to, counter_promo = sr.new_counter_table()
    return sr.negamax(
        bb, meta, 3, -sr.INF, sr.INF, deadline, counters, 0, history, 0,
        tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
        killer_from, killer_to, killer_promo, history_table, True, sr.MAX_CHECK_EXTENSIONS,
        counter_from, counter_to, counter_promo, -1, -1, 0,
    )


def main() -> None:
    failures = 0

    insufficient_cases = {
        "bare kings": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "king+knight vs bare king": "4k3/8/8/8/8/8/8/4KN2 w - - 0 1",
        "king+bishop vs bare king": "4k3/8/8/8/8/8/8/4KB2 w - - 0 1",
        "same-colour bishops both sides": "3bk3/8/8/8/8/8/8/2B1K3 w - - 0 1",
    }
    for label, fen in insufficient_cases.items():
        bb, _meta = bbm.from_fen(fen)
        flag = mg.is_insufficient_material(bb)
        print(f"{label}: is_insufficient_material={flag}")
        if not flag:
            print(f"  FAIL: expected {label!r} to be recognised as insufficient material")
            failures += 1
        score = call_negamax(fen)
        print(f"{label}: negamax score={score} (expect an immediate draw, 0)")
        if score != 0:
            print(f"  FAIL: expected negamax to score {label!r} as a draw (0)")
            failures += 1

    sufficient_cases = {
        "opposite-colour bishops both sides": "4k3/3b4/8/8/8/8/8/2B1K3 w - - 0 1",
        "knight on each side": "4k1n1/8/8/8/8/8/8/4KN2 w - - 0 1",
        "up a rook (real material)": "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
    }
    for label, fen in sufficient_cases.items():
        bb, _meta = bbm.from_fen(fen)
        flag = mg.is_insufficient_material(bb)
        print(f"{label}: is_insufficient_material={flag} (expect False)")
        if flag:
            print(f"  FAIL: expected {label!r} to NOT be flagged (conservative, safe direction)")
            failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
