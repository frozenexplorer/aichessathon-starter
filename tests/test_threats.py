"""Direct tests of evaluate.py's tactical-motif terms: threats_score (pawn threats, hanging
pieces, forks) and pin_and_xray_score (absolute pins, skewers/x-rays). Each case is built so only
the mechanism under test can produce a nonzero score -- e.g. the hanging-piece case gives the
attacker's own equivalent piece a defender, so a naive symmetric read of the position nets to zero
and only an asymmetric implementation passes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bitboard as bbm
import evaluate as ev


def score(fen: str) -> tuple[int, int]:
    bb, _meta = bbm.from_fen(fen)
    return ev.threats_score(bb), ev.pin_and_xray_score(bb)


def main() -> None:
    failures = 0

    # White knight on e5 attacks both the black king (g6) and the undefended black queen (c6): a
    # fork (2 simultaneous targets) plus a hanging-piece bonus for the queen specifically.
    fork_threats, fork_pins = score("8/8/2q3k1/4N3/8/8/4K3/8 w - - 0 1")
    print(f"knight forks king+queen, queen undefended: threats={fork_threats} pins={fork_pins}")
    if fork_threats <= 100:
        print("  FAIL: expected a large threats bonus (fork + hanging queen)")
        failures += 1

    # White rook a1 attacks the undefended black rook a8; white's own rook a1 is defended by the
    # bishop on b2, so only white's attack should register -- a symmetric read would net to zero.
    hanging_threats, hanging_pins = score("r3k3/8/8/8/8/8/1B6/R3K3 w - - 0 1")
    print(f"defended attacker vs undefended target: threats={hanging_threats} pins={hanging_pins}")
    if hanging_threats <= 0:
        print("  FAIL: expected a positive hanging-piece bonus for white only")
        failures += 1

    # White bishop a1 - black knight d4 - black king g7, all on the a1-h8 diagonal: the knight is
    # absolutely pinned (moving it would expose its own king to the bishop).
    pin_threats, pin_pins = score("8/6k1/8/8/3n4/8/8/B3K3 w - - 0 1")
    print(f"bishop pins knight to king: threats={pin_threats} pins={pin_pins}")
    if pin_pins <= 0:
        print("  FAIL: expected a positive absolute-pin bonus")
        failures += 1

    # White rook a1 - black bishop a4 - black queen a8, all on the a-file: an x-ray/skewer, since
    # the queen (more valuable) sits directly behind the bishop from the rook's point of view.
    xray_threats, xray_pins = score("q3k3/8/8/8/b7/8/8/R3K3 w - - 0 1")
    print(f"rook x-rays bishop into queen: threats={xray_threats} pins={xray_pins}")
    if xray_pins <= 0:
        print("  FAIL: expected a positive x-ray/skewer bonus")
        failures += 1

    # Bare kings, far apart: no attacker can reach anything, so both terms must be exactly zero.
    quiet_threats, quiet_pins = score("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    print(f"bare kings, nothing in range: threats={quiet_threats} pins={quiet_pins}")
    if quiet_threats != 0 or quiet_pins != 0:
        print("  FAIL: expected both terms to be exactly zero with no pieces in range")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
