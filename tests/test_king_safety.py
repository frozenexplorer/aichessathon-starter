"""Direct tests of evaluate.py's king_safety_score: attacker-weighted pressure on a king's own
ring (KING_ATTACKS[king_sq]), phase-blended like the king PST so it matters in the middlegame and
fades to exactly zero once phase hits 0 (an exposed king in a bare endgame is not a liability).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bitboard as bbm
import evaluate as ev

# Black king on g8; white queen h6 and rook g6 both reach into its ring (KING_ATTACKS[g8] =
# {f7, f8, g7, h7, h8}): queen hits h7/h8 (up the h-file) and g7/f8 (the h6-g7-f8 diagonal), rook
# hits g7 (up the g-file). White king tucked away on a1, irrelevant to either ring.
PRESSURE_FEN = "6k1/8/6RQ/8/8/8/8/K7 w - - 0 1"

# Same board, mirrored (rook/queen now bearing down on the white king instead) to check the sign
# convention the opposite way.
PRESSURE_FEN_MIRRORED = "k7/8/8/8/8/6rq/8/6K1 w - - 0 1"

# Bare kings, far apart: no attacker in range of either ring at any phase.
QUIET_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


def main() -> None:
    failures = 0

    bb, _meta = bbm.from_fen(PRESSURE_FEN)
    full_phase = ev.king_safety_score(bb, ev.PHASE_MAX)
    faded = ev.king_safety_score(bb, 0)
    print(f"queen+rook pressuring black's king ring: full_phase={full_phase} faded={faded}")
    if full_phase <= 0:
        print("  FAIL: expected a positive score (white pressuring black's king)")
        failures += 1
    if faded != 0:
        print("  FAIL: expected the term to fade to exactly zero at phase == 0")
        failures += 1

    mirrored_bb, _meta = bbm.from_fen(PRESSURE_FEN_MIRRORED)
    mirrored_full = ev.king_safety_score(mirrored_bb, ev.PHASE_MAX)
    print(f"same pressure, mirrored onto white's king: full_phase={mirrored_full}")
    if mirrored_full != -full_phase:
        print(f"  FAIL: expected exactly -{full_phase} by symmetry, got {mirrored_full}")
        failures += 1

    quiet_bb, _meta = bbm.from_fen(QUIET_FEN)
    quiet_score = ev.king_safety_score(quiet_bb, ev.PHASE_MAX)
    print(f"bare kings, nothing in range: score={quiet_score}")
    if quiet_score != 0:
        print("  FAIL: expected exactly zero with no attacking pieces on the board")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
