"""Differential test for magic bitboards: attacks.rook_attacks / bishop_attacks (the magic-lookup
versions used everywhere at runtime) must agree with attacks._rook_ray_attacks /
_bishop_ray_attacks (the plain ray-cast reference, kept only for this check and for building the
magic tables at import time) over random occupancies for every square. This is the rigor
docs/FUTURE.md item 7 asks for before magic bitboards can be trusted -- perft alone would only
catch a bug that changes which moves are legal, not one that merely computes the wrong attack set
in a way that happens not to flip legality in the perft battery's specific positions.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import attacks as at

random.seed(2024)
TRIALS_PER_SQUARE = 200


def main() -> None:
    mismatches = 0
    for square in range(64):
        for _ in range(TRIALS_PER_SQUARE):
            occ = np.uint64(random.getrandbits(64))
            rook_magic = int(at.rook_attacks(square, occ))
            rook_ray = int(at._rook_ray_attacks(square, occ))
            if rook_magic != rook_ray:
                mismatches += 1
                print(
                    f"MISMATCH rook sq={square} occ={int(occ)}: "
                    f"magic={rook_magic} != ray={rook_ray}"
                )

            bishop_magic = int(at.bishop_attacks(square, occ))
            bishop_ray = int(at._bishop_ray_attacks(square, occ))
            if bishop_magic != bishop_ray:
                mismatches += 1
                print(
                    f"MISMATCH bishop sq={square} occ={int(occ)}: "
                    f"magic={bishop_magic} != ray={bishop_ray}"
                )

    total = 64 * TRIALS_PER_SQUARE * 2
    print(f"{total - mismatches}/{total} attack queries agree (rook + bishop, all 64 squares)")
    if mismatches:
        print(f"\n{mismatches} MISMATCH(ES)")
        raise SystemExit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
