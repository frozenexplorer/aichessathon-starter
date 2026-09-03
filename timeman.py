"""Per-move time budgeting. The referee measures wall time and a flag is a loss, so every
number here leans conservative rather than clever.
"""

SAFETY_MARGIN_MS = 300
MIN_BUDGET_MS = 30
ASSUMED_MOVES_LEFT = 30
INCREMENT_MS = 500

# Below this much clock, skip the tree search entirely (see search.quick_best_move). A depth-1
# search can cascade through thousands of quiescence nodes on a tactical position before the
# first in-search time check fires, and below ~100ms that alone can exceed what is left.
PANIC_MS = 150


def budget_ms(time_left_ms: int) -> int:
    """How long this move gets, out of the clock we were handed."""
    safe_left = max(0, time_left_ms - SAFETY_MARGIN_MS)
    planned = safe_left / ASSUMED_MOVES_LEFT + INCREMENT_MS * 0.8
    return int(max(MIN_BUDGET_MS, min(planned, safe_left)))
