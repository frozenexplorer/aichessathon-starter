"""Direct tests of the two repetition mechanisms in search.py.

negamax: a position with a known material edge should score as a real advantage with no
repetition history, as exactly 0 (draw) once its hash already appears twice in history (this
would be the 3rd occurrence), and NOT as a draw with only one prior occurrence, or when the
match is at an odd ply (the opponent's hypothetical turn, which is never tracked directly).

claim_eligible_for_opponent: the harness's referee auto-draws before either side is asked to move
if the position about to be handed over either (a) has itself already occurred twice before, or
(b) offers the side to move a reply that would create a third occurrence -- in both cases
regardless of what actually gets played. See search.py's module docstring for the two real games
that exposed each condition in turn. This checks both directly: true under either condition,
false when neither holds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bitboard as bbm
import movegen as mg
import search as sr
import zobrist as zb

# White to move, up a rook -- unambiguous material edge so a draw score (0) is clearly
# distinguishable from a normal eval-based score.
UP_A_ROOK_FEN = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"


def fresh_history() -> np.ndarray:
    return np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)


def call_negamax(hist_len: int, ply: int = 0) -> int:
    bb, meta = bbm.from_fen(UP_A_ROOK_FEN)
    h = zb.position_hash(bb, meta)
    history = fresh_history()
    for i in range(hist_len):
        history[i] = h
    counters = sr.new_counters()
    deadline = 1e18
    return sr.negamax(bb, meta, 0, -sr.INF, sr.INF, deadline, counters, ply, history, hist_len)


def main() -> None:
    failures = 0

    no_repeat = call_negamax(hist_len=0, ply=0)
    print(f"no prior occurrences, our turn: score={no_repeat} (expect a real material edge)")
    if no_repeat <= 100:
        print("  FAIL: expected a clear positive score, material edge was not reflected")
        failures += 1

    one_prior = call_negamax(hist_len=1, ply=0)
    print(f"one prior occurrence (this would be the 2nd), our turn: score={one_prior}")
    if one_prior <= 100:
        print("  FAIL: one prior occurrence must not be scored as a draw")
        failures += 1

    two_prior = call_negamax(hist_len=2, ply=0)
    print(f"two prior occurrences (this would be the 3rd), our turn: score={two_prior}")
    if two_prior != 0:
        print("  FAIL: a 3rd occurrence must score as an immediate draw (0)")
        failures += 1

    odd_ply_with_matches = call_negamax(hist_len=2, ply=1)
    print(f"two prior occurrences, but opponent's (odd) ply: score={odd_ply_with_matches}")
    if odd_ply_with_matches == 0:
        print("  FAIL: odd-ply nodes must never be checked against history")
        failures += 1

    # Black to move, king on a8; Kb8 and Ka7 are its only legal moves. Pre-populate history with
    # two occurrences of the position reached by Kb8 (white to move), so black playing Kb8 would
    # be the 3rd occurrence and is available to them right now, whether or not they'd choose it.
    bb, meta = bbm.from_fen("k7/8/2K5/8/8/8/8/8 b - - 0 1")
    after_kb8, after_kb8_meta = mg.make_move(bb, meta, 56, 57, -1)
    reply_hash = zb.position_hash(after_kb8, after_kb8_meta)

    reply_history = fresh_history()
    reply_history[0] = reply_hash
    reply_history[1] = reply_hash
    eligible_b = sr.claim_eligible_for_opponent(bb, meta, reply_history, 2, fresh_history(), 0, 1)
    print(f"condition (b), opponent has a repeat-creating reply available: eligible={eligible_b}")
    if not eligible_b:
        print("  FAIL: expected condition (b) to detect the available Kb8 reply")
        failures += 1

    # Same position (black to move), but this time it is the position ITSELF -- the one we would
    # be handing over -- that has already occurred twice, independent of what black could reply.
    own_hash = zb.position_hash(bb, meta)
    opponent_hist = fresh_history()
    opponent_hist[0] = own_hash
    opponent_hist[1] = own_hash
    eligible_a = sr.claim_eligible_for_opponent(bb, meta, fresh_history(), 0, opponent_hist, 2, 1)
    print(f"condition (a), handed-over position itself recurs a 3rd time: eligible={eligible_a}")
    if not eligible_a:
        print("  FAIL: expected condition (a) to detect the position's own two prior occurrences")
        failures += 1

    not_eligible = sr.claim_eligible_for_opponent(
        bb, meta, fresh_history(), 0, fresh_history(), 0, 1
    )
    print(f"empty histories, neither condition can hold: eligible={not_eligible}")
    if not_eligible:
        print("  FAIL: expected claim_eligible_for_opponent to be false with no prior history")
        failures += 1

    # Recursive case (lookahead): black to move at a8, walled into a single legal reply (Ka7) by
    # its own king on c7. From there (white to move), white has a reply (Kc6) landing on a
    # position that already occurred twice -- but neither bb itself nor black's one reply (W)
    # matches anything directly, so only the lookahead recursion (checking W's own eligibility
    # with the two history arrays swapped) can catch this.
    bb2, meta2 = bbm.from_fen("k7/2K5/8/8/8/8/8/8 b - - 0 1")
    w_bb, w_meta = mg.make_move(bb2, meta2, 56, 48, -1)  # a8 -> a7, black's only legal move
    x_bb, x_meta = mg.make_move(w_bb, w_meta, 50, 42, -1)  # c7 -> c6, white's reply
    x_hash = zb.position_hash(x_bb, x_meta)

    # X is black's turn (after white's Kc6), matching bb2's own parity -- so its hash belongs in
    # opponent_history (bb2's own-parity history), not history (which is checked against black's
    # reply W, white's turn, and must stay empty here so only the recursion catches this).
    x_history = fresh_history()
    x_history[0] = x_hash
    x_history[1] = x_hash
    eligible_lookahead = sr.claim_eligible_for_opponent(
        bb2, meta2, fresh_history(), 0, x_history, 2, 1
    )
    print(f"recursive case, danger two plies deep: eligible={eligible_lookahead}")
    if not eligible_lookahead:
        print("  FAIL: expected lookahead=1 to catch the danger via black's forced Ka7 reply")
        failures += 1

    eligible_no_lookahead = sr.claim_eligible_for_opponent(
        bb2, meta2, fresh_history(), 0, x_history, 2, 0
    )
    print(f"same case with lookahead=0 (should miss it): eligible={eligible_no_lookahead}")
    if eligible_no_lookahead:
        print("  FAIL: lookahead=0 should not see two plies deep -- test construction is wrong")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
