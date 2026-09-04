"""Direct tests of static exchange evaluation (search.see) against known winning, losing, equal,
and x-ray exchanges -- see search.py's module docstring for why this needs its own tests: a wrong
SEE implementation silently misorders moves or mis-prunes quiescence rather than crashing, so
correctness has to be checked directly against hand-computed expected values, not inferred from
overall game results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

import bitboard as bbm
import search as sr


def see_of(fen: str, from_name: str, to_name: str, promo: int = -1) -> int:
    bb, meta = bbm.from_fen(fen)
    from_sq = chess.parse_square(from_name)
    to_sq = chess.parse_square(to_name)
    return int(sr.see(bb, meta, from_sq, to_sq, promo))


def main() -> None:
    failures = 0

    # White pawn e4 takes an undefended black pawn on d5: a clean, uncontested win of a pawn.
    clean_win = see_of("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4", "d5")
    print(f"undefended pawn capture: see={clean_win} (expect +100)")
    if clean_win != 100:
        print("  FAIL: expected exactly +100")
        failures += 1

    # White pawn e4 takes black pawn d5, which is defended by a black pawn on c6 -- an even
    # pawn-for-pawn trade nets zero.
    equal_trade = see_of("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4", "d5")
    print(f"pawn-for-pawn defended capture: see={equal_trade} (expect 0)")
    if equal_trade != 0:
        print("  FAIL: expected exactly 0")
        failures += 1

    # White knight takes a black pawn on d5 defended by a black pawn on c6: winning the pawn but
    # then losing the knight nets a clear loss (-220 = knight value 320 minus the pawn won, 100).
    losing_capture = see_of("4k3/8/2p5/3p4/5N2/8/8/4K3 w - - 0 1", "f4", "d5")
    print(f"knight-for-pawn, recaptured by a pawn: see={losing_capture} (expect -220)")
    if losing_capture != -220:
        print("  FAIL: expected exactly -220")
        failures += 1

    # A doubled white rook on a1/a2 takes a black rook on a5, defended only by a black pawn on
    # b6. The pawn recaptures (rook-for-rook, net 0 so far), but the second white rook -- only
    # revealed once the first vacates a2, a genuine x-ray -- recaptures the pawn for a net +100.
    # This is the case that would silently break if sliding attacks were not recomputed against
    # the live occupancy at each step of the exchange (see _see_least_valuable_attacker).
    xray_win = see_of("4k3/8/1p6/r7/8/8/R7/R3K3 w - - 0 1", "a2", "a5")
    print(f"rook trade with an x-rayed rook behind it: see={xray_win} (expect +100)")
    if xray_win != 100:
        print("  FAIL: expected exactly +100")
        failures += 1

    # A non-capturing promotion (no defender on the target square): the full queen-for-pawn gain.
    promo_gain = see_of("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7", "e8", bbm.QUEEN)
    print(f"undefended promotion: see={promo_gain} (expect +800)")
    if promo_gain != 800:
        print("  FAIL: expected exactly +800 (queen 900 - pawn 100)")
        failures += 1

    if failures:
        print(f"\n{failures} FAILURE(S)")
        raise SystemExit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
