"""Direct tests of evaluate.py's two newest, near-zero-compile-cost additions: a flat tempo bonus
for the side to move, and opposite_bishops_scale, which scales the whole eval down in a pure
opposite-coloured-bishop endgame (famously drawish even a material edge up).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bitboard as bbm
import evaluate as ev

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_FEN_BLACK = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"

# Material-imbalanced position (white up a rook), same placement with only side-to-move flipped.
IMBALANCED_FEN_WHITE = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
IMBALANCED_FEN_BLACK = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"

# White bishop c1 (dark square), black bishop c8 (light square) -- opposite colours, otherwise
# bare kings, no other minor/major piece: the scale should kick in.
OCB_FEN = "2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1"

# Same-coloured bishops instead (white c1 dark, black f8 dark): no scaling.
SAME_COLOUR_FEN = "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1"

# Opposite-coloured bishops as above, but a white rook also on the board: rooks alongside OCB
# don't carry the same drawish tendency, so this must stay unscaled.
OCB_WITH_ROOK_FEN = "2b1k3/8/8/8/8/8/8/R1B1K3 w - - 0 1"

# Bare kings, no bishops at all.
NO_BISHOPS_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


def main() -> None:
    failures = 0

    bb, meta_w = bbm.from_fen(START_FEN)
    _, meta_b = bbm.from_fen(START_FEN_BLACK)
    symmetric_white = ev.evaluate(bb, meta_w)
    symmetric_black = ev.evaluate(bb, meta_b)
    print(f"tempo, symmetric start position: white-to-move={symmetric_white} "
          f"black-to-move={symmetric_black} (expect both == TEMPO_BONUS={int(ev.TEMPO_BONUS)})")
    if symmetric_white != int(ev.TEMPO_BONUS) or symmetric_black != int(ev.TEMPO_BONUS):
        print("  FAIL: a fully symmetric position should score exactly the tempo bonus either way")
        failures += 1

    bb2, meta2_w = bbm.from_fen(IMBALANCED_FEN_WHITE)
    _, meta2_b = bbm.from_fen(IMBALANCED_FEN_BLACK)
    total = ev.evaluate(bb2, meta2_w) + ev.evaluate(bb2, meta2_b)
    print(f"tempo invariant, material-imbalanced position: sum={total} "
          f"(expect exactly 2*TEMPO_BONUS={2 * int(ev.TEMPO_BONUS)} regardless of the imbalance)")
    if total != 2 * int(ev.TEMPO_BONUS):
        print("  FAIL: the material term should cancel out of the sum, leaving only 2x tempo")
        failures += 1

    ocb_bb, _ = bbm.from_fen(OCB_FEN)
    ocb_scale = ev.opposite_bishops_scale(ocb_bb)
    print(f"opposite-coloured bishops, kings+bishops only: scale={ocb_scale} "
          f"(expect OCB_SCALE_PERCENT={int(ev.OCB_SCALE_PERCENT)})")
    if ocb_scale != int(ev.OCB_SCALE_PERCENT):
        failures += 1
        print("  FAIL: expected the drawish OCB scale to apply")

    same_bb, _ = bbm.from_fen(SAME_COLOUR_FEN)
    same_scale = ev.opposite_bishops_scale(same_bb)
    print(f"same-coloured bishops: scale={same_scale} (expect 100)")
    if same_scale != 100:
        failures += 1
        print("  FAIL: same-coloured bishops must not be scaled")

    rook_bb, _ = bbm.from_fen(OCB_WITH_ROOK_FEN)
    rook_scale = ev.opposite_bishops_scale(rook_bb)
    print(f"opposite-coloured bishops plus a rook: scale={rook_scale} (expect 100)")
    if rook_scale != 100:
        failures += 1
        print("  FAIL: OCB alongside a rook should not be scaled")

    none_bb, _ = bbm.from_fen(NO_BISHOPS_FEN)
    none_scale = ev.opposite_bishops_scale(none_bb)
    print(f"no bishops at all: scale={none_scale} (expect 100)")
    if none_scale != 100:
        failures += 1
        print("  FAIL: no bishops on the board should never trigger scaling")

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
