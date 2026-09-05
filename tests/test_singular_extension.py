"""Direct tests of negamax's excluded-move mechanism (search.py's module docstring, "Singular
extensions"): the machinery that lets one recursive call search a node's moves *other than* one
specific move, used to verify whether the hash move stands out. Exercises the exclusion itself
(not the singularity heuristic's tuning, which is not asserted on) since that is the part where a
mistake would be a real correctness bug (an excluded node resolving via a stale, non-excluded TT
entry, or polluting the table with a partial result) rather than a search-quality tradeoff.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bitboard as bbm
import search as sr
import zobrist as zb

# White to move, up a rook -- same fixture as test_repetition.py/test_fifty_move.py. Only one
# rook move is possible from a1 in most directions; using the actual legal moves generated below
# rather than hand-picking one keeps this robust to any future movegen change.
UP_A_ROOK_FEN = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"


def run_negamax(excluded_from: int, excluded_to: int, depth: int = 4) -> tuple[int, np.ndarray]:
    bb, meta = bbm.from_fen(UP_A_ROOK_FEN)
    history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
    counters = sr.new_counters()
    deadline = 1e18
    tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo = sr.new_tt()
    killer_from, killer_to, killer_promo = sr.new_killers()
    history_table = sr.new_history_table()
    counter_from, counter_to, counter_promo = sr.new_counter_table()
    score = sr.negamax(
        bb, meta, depth, -sr.INF, sr.INF, deadline, counters, 0, history, 0,
        tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
        killer_from, killer_to, killer_promo, history_table, True, sr.MAX_CHECK_EXTENSIONS,
        counter_from, counter_to, counter_promo, -1, -1, 0, excluded_from, excluded_to,
    )
    return score, tt_key


def main() -> None:
    failures = 0

    # Baseline: no exclusion, normal search. Establishes the real best score and populates (and
    # is entitled to write) this call's own TT.
    baseline_score, baseline_tt_key = run_negamax(-1, -1)
    print(f"no exclusion: score={baseline_score}")
    if baseline_score <= 100:
        print("  FAIL: expected a clear positive score (up a rook) with no exclusion")
        failures += 1
    if not np.any(baseline_tt_key != 0):
        print("  FAIL: expected the unexcluded search to have written at least one TT entry")
        failures += 1

    # Exclude a real legal move (the king stepping to d1, always legal here) and confirm the
    # search still returns a sane result by searching everything else -- not a crash, not the
    # same call short-circuiting via some stale mechanism, and not obviously broken (still a real
    # material edge, since excluding one king move leaves every other move, including the rook,
    # available).
    excluded_score, _excluded_tt_key = run_negamax(4, 3)  # e1 -> d1, always legal here
    print(f"king e1->d1 excluded: score={excluded_score}")
    if excluded_score <= 100:
        print("  FAIL: expected excluding one non-essential move to still show the material edge")
        failures += 1

    # The excluded-move search must never write to the TT under *its own* position's key -- its
    # children (real moves played from here, each a different position) legitimately write under
    # theirs, so the only correct check is the root position's own bucket specifically, not "no
    # entry anywhere".
    bb, meta = bbm.from_fen(UP_A_ROOK_FEN)
    h = zb.position_hash(bb, meta)
    bucket = int(h & sr.TT_MASK)
    slot_a, slot_b = bucket * 2, bucket * 2 + 1
    history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
    counters = sr.new_counters()
    tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo = sr.new_tt()
    killer_from, killer_to, killer_promo = sr.new_killers()
    history_table = sr.new_history_table()
    counter_from, counter_to, counter_promo = sr.new_counter_table()
    sr.negamax(
        bb, meta, 4, -sr.INF, sr.INF, 1e18, counters, 0, history, 0,
        tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
        killer_from, killer_to, killer_promo, history_table, True, sr.MAX_CHECK_EXTENSIONS,
        counter_from, counter_to, counter_promo, -1, -1, 0, 4, 3,
    )
    root_slot_written = tt_key[slot_a] == h or tt_key[slot_b] == h
    print(f"excluded-move search wrote its OWN position's TT slot: {root_slot_written}")
    if root_slot_written:
        print("  FAIL: an excluded node must never store to the TT under its own position's key")
        failures += 1
    if not np.any(tt_depth > 0):
        print("  FAIL: expected the excluded search's own real-move children to still write "
              "their own (different) positions to the TT")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
