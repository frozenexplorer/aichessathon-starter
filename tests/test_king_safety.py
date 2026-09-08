"""Direct tests of evaluate.py's king_safety_score: attacker-weighted pressure on a king's own
ring (KING_ATTACKS[king_sq]), phase-blended like the king PST so it matters in the middlegame and
fades to exactly zero once phase hits 0 (an exposed king in a bare endgame is not a liability).

docs/fix.md F2: also covers the unsupported-attacker discount -- a non-pawn attacker not defended
by its own side has its contribution halved and is excluded from attacker_count, since it is
disposable pressure the defender can often just remove rather than the sustained, coordinated kind
KING_ATTACK_COUNT_PERCENT's superlinear scaling is meant to reward. Note PRESSURE_FEN above is
*not* an unsupported-attacker case even though neither piece has an outside defender: the rook and
queen sit adjacent on the same rank and mutually defend each other, so both fully count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bitboard as bbm
import evaluate as ev

# Black king on g8; white queen h6 and rook g6 both reach into its ring (KING_ATTACKS[g8] =
# {f7, f8, g7, h7, h8}): queen hits h7/h8 (up the h-file) and g7/f8 (the h6-g7-f8 diagonal), rook
# hits g7 (up the g-file). White king tucked away on a1, irrelevant to either ring. The rook and
# queen mutually defend each other (adjacent on rank 6), so this is a fully-supported case.
PRESSURE_FEN = "6k1/8/6RQ/8/8/8/8/K7 w - - 0 1"

# Same board, mirrored (rook/queen now bearing down on the white king instead) to check the sign
# convention the opposite way.
PRESSURE_FEN_MIRRORED = "k7/8/8/8/8/6rq/8/6K1 w - - 0 1"

# Bare kings, far apart: no attacker in range of either ring at any phase.
QUIET_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

# White queen d3, nothing else of white's on the board to defend it -- a lone, unsupported
# attacker reaching into black's king zone (f5/f6/f7/f8/g5/g6/g7/g8/h5/h6/h7/h8 around Kg8).
UNDEFENDED_QUEEN_FEN = "6k1/8/8/8/8/3Q4/8/K7 w - - 0 1"

# Identical attack pattern into the same zone, but the queen is now backed by a rook on the same
# file -- must fully count (defended attackers are unaffected by the F2 discount).
DEFENDED_QUEEN_FEN = "6k1/8/8/8/8/3Q4/8/K2R4 w - - 0 1"

# White queen g7, one king-move from Black's king on g8, with nothing defending it -- Black could
# simply play ...Kxg7 next move for free. The pre-fix heuristic priced this at a full 147cp; this
# is the concrete "freely capturable attacker" case F2 targets.
CAPTURABLE_QUEEN_FEN = "6k1/6Q1/8/8/8/8/8/K7 w - - 0 1"

# White rook h1 aimed up the open h-file at Black's king zone, blocked by a black pawn on h4
# *before* the zone starts (zone begins at h5) -- the rook's ray never reaches the zone at all, so
# this must be identical with and without the F2 discount (there's nothing to discount: 0 hits).
BLOCKED_ROOK_FEN = "6k1/8/8/8/7p/8/8/K6R w - - 0 1"
UNBLOCKED_ROOK_FEN = "6k1/8/8/8/8/8/8/K6R w - - 0 1"

# Mirror of UNDEFENDED_QUEEN_FEN (rank-flipped and colour-swapped, same convention as
# PRESSURE_FEN_MIRRORED above, so each king keeps its own forward shield direction) -- checks the
# discount applies with the correct sign regardless of which side is attacking.
UNDEFENDED_QUEEN_FEN_MIRRORED = "k7/8/3q4/8/8/8/8/6K1 w - - 0 1"

# Real R54 endgame position (docs/fix.md F2 forensic case): Black's queen f4 and pawn e4 are
# defended, but the bishop on h3 has no defender at all -- it alone pushes attacker_count from 1
# to 2, crossing KING_ATTACK_COUNT_PERCENT's 20%->55% cliff on top of its own halved contribution.
R54_FEN = "2r3k1/1p2r2p/p2p3n/P2P2pB/1P2pqP1/4N2b/R4P1P/2Q3K1 w - - 0 36"


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

    # --- F2: unsupported-attacker discount ---

    undefended_bb, _meta = bbm.from_fen(UNDEFENDED_QUEEN_FEN)
    undefended_score = ev.king_safety_score(undefended_bb, ev.PHASE_MAX)
    print(f"lone undefended queen d3: score={undefended_score}")
    if undefended_score != 75:
        print(f"  FAIL: expected 75 (halved contribution, excluded from attacker_count), "
              f"got {undefended_score}")
        failures += 1

    defended_bb, _meta = bbm.from_fen(DEFENDED_QUEEN_FEN)
    defended_score = ev.king_safety_score(defended_bb, ev.PHASE_MAX)
    print(f"same queen, backed by a rook on the file: score={defended_score}")
    if defended_score != 93:
        print(f"  FAIL: expected 93 (full contribution, defended attackers are unaffected), "
              f"got {defended_score}")
        failures += 1
    if defended_score <= undefended_score:
        print("  FAIL: a defended attacker must score strictly higher than the same "
              "attacker undefended")
        failures += 1

    capturable_bb, _meta = bbm.from_fen(CAPTURABLE_QUEEN_FEN)
    capturable_score = ev.king_safety_score(capturable_bb, ev.PHASE_MAX)
    print(f"undefended queen one king-move from the enemy king (freely capturable): "
          f"score={capturable_score}")
    if capturable_score != 75:
        print(f"  FAIL: expected 75 (was 147 pre-fix), got {capturable_score}")
        failures += 1

    blocked_bb, _meta = bbm.from_fen(BLOCKED_ROOK_FEN)
    unblocked_bb, _meta = bbm.from_fen(UNBLOCKED_ROOK_FEN)
    blocked_score = ev.king_safety_score(blocked_bb, ev.PHASE_MAX)
    unblocked_score = ev.king_safety_score(unblocked_bb, ev.PHASE_MAX)
    print(f"rook blocked before the zone: score={blocked_score}  "
          f"(unblocked, for reference: {unblocked_score})")
    if blocked_score != 50:
        print(f"  FAIL: expected 50 (zero hits -> nothing for the discount to touch, "
              f"only the open-file bonus remains), got {blocked_score}")
        failures += 1

    mirrored_undefended_bb, _meta = bbm.from_fen(UNDEFENDED_QUEEN_FEN_MIRRORED)
    mirrored_undefended_score = ev.king_safety_score(mirrored_undefended_bb, ev.PHASE_MAX)
    print(f"same undefended-queen case, mirrored onto the other king: "
          f"score={mirrored_undefended_score}")
    if mirrored_undefended_score != -undefended_score:
        print(f"  FAIL: expected exactly -{undefended_score} by symmetry, "
              f"got {mirrored_undefended_score}")
        failures += 1

    r54_bb, _meta = bbm.from_fen(R54_FEN)
    r54_phase = ev.game_phase(r54_bb)
    r54_score = ev.king_safety_score(r54_bb, r54_phase)
    print(f"R54 real-game position (phase={r54_phase}): score={r54_score}")
    if r54_score != -15:
        print(f"  FAIL: expected -15 (was -76 pre-fix), got {r54_score}")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
