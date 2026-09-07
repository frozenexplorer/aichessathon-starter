"""Direct tests of evaluate.py's unstoppable_passed_pawn_score against the classical rule of the
square: a passed pawn queens against bare-king defense whenever the defending king's Chebyshev
distance to the promotion square exceeds the pawn's own distance there, with a one-square discount
for the defender when it is their own move (docs/fix.md Part 3c -- "given how new this term is and
how large it is, verify it against a test suite of rule-of-the-square positions before trusting
it, or reduce it to ~250 pending that").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bitboard as bbm
import evaluate as ev

# White pawn e5 (3 squares from queening on e8), black king a8 (Chebyshev distance 4 to e8), white
# king e1 out of the way, no black rook/queen -- the textbook square-rule setup where whether the
# king is inside or outside the square depends entirely on who moves first.
TEMPO_FEN = "k7/8/8/4P3/8/8/8/4K3 w - - 0 1"

# Same pawn, black king right next to the promotion square -- catches it regardless of the move.
CAUGHT_FEN = "4k3/8/8/4P3/8/8/8/4K3 w - - 0 1"

# Tempo setup again, but black keeps a rook -- unstoppable_passed_pawn_score should not fire at
# all once the defender has a rook or queen (see its own docstring).
DEFENDED_FEN = "k6r/8/8/4P3/8/8/8/4K3 w - - 0 1"

# Mirror of TEMPO_FEN: black pawn e4 racing to e1, white king a1 outside the square, no white
# rook/queen -- checks the sign convention and the mirrored tempo/distance arithmetic.
MIRROR_TEMPO_FEN = "4k3/8/8/8/4p3/8/8/K7 b - - 0 1"


def main() -> None:
    failures = 0

    bb, _meta = bbm.from_fen(TEMPO_FEN)

    # Black (the defender) is NOT to move: no tempo discount, king_dist(4) > pawn_dist(3) --
    # outside the square, pawn should be priced as unstoppable.
    not_defenders_move = ev.unstoppable_passed_pawn_score(bb, 0, bbm.WHITE)
    print(f"outside the square, defender not to move: score={not_defenders_move}")
    if not_defenders_move <= 0:
        print("  FAIL: expected a positive (white-favouring) unstoppable-passer bonus")
        failures += 1

    # Black IS to move: tempo discount drops king_dist to 3, tying pawn_dist(3) -- inside the
    # square (a tie is NOT "greater than", so this must NOT be priced as unstoppable).
    defenders_move = ev.unstoppable_passed_pawn_score(bb, 0, bbm.BLACK)
    print(f"same position, defender's own move (tempo): score={defenders_move}")
    if defenders_move != 0:
        print("  FAIL: expected the one-tempo discount to bring the king inside the square (0)")
        failures += 1

    # Fully faded at phase == PHASE_MAX (a middlegame position) regardless of the above -- this
    # term is endgame-only.
    faded = ev.unstoppable_passed_pawn_score(bb, ev.PHASE_MAX, bbm.WHITE)
    print(f"same outside-the-square position, full middlegame phase: score={faded}")
    if faded != 0:
        print("  FAIL: expected the term to fade to exactly zero at phase == PHASE_MAX")
        failures += 1

    caught_bb, _meta = bbm.from_fen(CAUGHT_FEN)
    caught = ev.unstoppable_passed_pawn_score(caught_bb, 0, bbm.WHITE)
    print(f"king adjacent to the promotion square, well inside the square: score={caught}")
    if caught != 0:
        print("  FAIL: expected 0 -- the king plainly catches this pawn")
        failures += 1

    defended_bb, _meta = bbm.from_fen(DEFENDED_FEN)
    defended = ev.unstoppable_passed_pawn_score(defended_bb, 0, bbm.WHITE)
    print(f"outside the square, but defender still has a rook: score={defended}")
    if defended != 0:
        print("  FAIL: expected 0 -- the term must not fire once the defender has a rook/queen")
        failures += 1

    mirror_bb, _meta = bbm.from_fen(MIRROR_TEMPO_FEN)
    mirror_not_defenders_move = ev.unstoppable_passed_pawn_score(mirror_bb, 0, bbm.BLACK)
    print(f"mirrored (black pawn, white defender), defender not to move: "
          f"score={mirror_not_defenders_move}")
    if mirror_not_defenders_move >= 0:
        print("  FAIL: expected a negative (black-favouring) unstoppable-passer bonus")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
