"""Direct test of quiescence()'s in-check handling (search.py's module docstring, "Quiescence in
check"): a side in check has no "decline to respond" option, and a legal evasion that is not
itself a capture (a king step, a block) must still be searched -- previously quiescence always
stood pat and only ever looked at captures, silently mishandling any check reached mid-quiescence
whose only replies are non-captures.

This is checked by calling quiescence() on the same in-check position twice, with check_budget=0
(the old behaviour: falls straight to the stand-pat/captures-only path even in check, since the
budget that would trigger the new evasion search is already exhausted) and with the real
QSEARCH_CHECK_BUDGET. The position has zero legal captures (ncap == 0), so the old path returns
the parent node's own stand-pat unconditionally without ever searching a reply; the fixed path
must instead search the forced king move and return whatever that child position is actually
worth, so the two calls should disagree.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bitboard as bbm
import search as sr

# Black to move, in check from the rook on e1 down the open e-file: no black piece can capture or
# block, so the only legal replies are king steps off the file (d7/d8/f7/f8) -- zero captures.
IN_CHECK_NO_CAPTURES_FEN = "4k3/8/8/8/8/8/8/4R2K b - - 0 1"


def call_quiescence(check_budget: int) -> int:
    bb, meta = bbm.from_fen(IN_CHECK_NO_CAPTURES_FEN)
    counters = np.zeros(2, dtype=np.int64)
    deadline = 1e18
    return sr.quiescence(
        bb, meta, -sr.INF, sr.INF, deadline, counters, sr.QUIESCENCE_MAX_PLIES, check_budget
    )


def main() -> None:
    failures = 0

    old_behaviour = call_quiescence(0)
    fixed_behaviour = call_quiescence(sr.QSEARCH_CHECK_BUDGET)
    print(f"check_budget=0 (old): {old_behaviour}, check_budget=default: {fixed_behaviour}")
    if old_behaviour == fixed_behaviour:
        print(
            "  FAIL: expected the two to disagree -- the fixed path must actually search the "
            "forced king move (a different square, different PST value) rather than reusing the "
            "parent's own stand-pat the way the old, capture-only path did"
        )
        failures += 1

    # A genuine checkmate (in check, zero legal moves at all) must still score as -MATE regardless
    # of check_budget -- unaffected by this fix, both paths share the same top-of-function
    # count == 0 check that ran before either one. Fool's mate: white king h1, black queen h4,
    # black to move is irrelevant here -- it's white to move, checkmated by ...Qxf2 already played.
    checkmated_bb, checkmated_meta = bbm.from_fen(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    )
    mate_counters = np.zeros(2, dtype=np.int64)
    mate_score = sr.quiescence(
        checkmated_bb, checkmated_meta, -sr.INF, sr.INF, 1e18, mate_counters,
        sr.QUIESCENCE_MAX_PLIES, sr.QSEARCH_CHECK_BUDGET,
    )
    print(f"fool's mate, white to move with no legal replies: quiescence score={mate_score}")
    if mate_score != -sr.MATE:
        print("  FAIL: expected an in-check, no-legal-moves position to score as -MATE")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
