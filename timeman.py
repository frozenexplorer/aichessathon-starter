"""Per-move time budgeting. The referee measures wall time and a flag is a loss, so every
number here leans conservative rather than clever.
"""

SAFETY_MARGIN_MS = 300
MIN_BUDGET_MS = 5
ASSUMED_MOVES_LEFT = 30
INCREMENT_MS = 500

# Below this much clock, skip the tree search entirely (see search.quick_best_move). A depth-1
# search can cascade through thousands of quiescence nodes on a tactical position before the
# first in-search time check fires, and below ~100ms that alone can exceed what is left.
PANIC_MS = 150

# Adaptive time management: a volatile position (score swinging between iterations, in check, a
# capture just played, or few pieces left -- see agent._is_volatile) can spend up to this many
# times the normal budget, so a sharp position gets the depth it needs instead of being cut off
# at the same budget as a quiet one. Bounded by MAX_EXTENSION_FRACTION regardless, so a string of
# volatile moves in a row can never eat the clock down to a flag.
EXTENSION_FACTOR = 2.5
MAX_EXTENSION_FRACTION = 0.35

# A score swing at least this large (centipawns) between the last two completed iterative-
# deepening depths marks the position as volatile -- the same magnitude as a minor-piece-for-
# pawn-ish trade, not noise from a stable evaluation.
SCORE_SWING_CP = 60

# Total pieces (both colours, kings included) at or below this marks a position as a simplified
# endgame worth searching deeper, mirroring the FUTURE.md rationale directly.
LOW_PIECE_COUNT = 12


def budget_ms(time_left_ms: int) -> int:
    """How long this move gets, out of the clock we were handed, for a quiet/stable position."""
    safe_left = max(0, time_left_ms - SAFETY_MARGIN_MS)
    planned = safe_left / ASSUMED_MOVES_LEFT + INCREMENT_MS * 0.8
    return int(max(MIN_BUDGET_MS, min(planned, safe_left)))


def extended_budget_ms(time_left_ms: int) -> int:
    """The most this move may ever spend, if it looks volatile -- still capped well short of the
    whole remaining clock so a run of sharp positions in a row cannot flag us.
    """
    safe_left = max(0, time_left_ms - SAFETY_MARGIN_MS)
    base = budget_ms(time_left_ms)
    cap = min(base * EXTENSION_FACTOR, safe_left * MAX_EXTENSION_FRACTION)
    return int(max(base, min(cap, safe_left)))
