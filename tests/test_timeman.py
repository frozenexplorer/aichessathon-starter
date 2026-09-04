"""Direct tests of timeman.py's two safety reservations (see its module docstring):
SAFETY_MARGIN_MS (clock-proportional, applied before any planning formula) and
POST_SEARCH_BUFFER_MS (fixed, applied to the planned budget itself, covering the real work that
happens after the search loop stops checking its deadline but before get_move actually returns).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import timeman as tm


def main() -> None:
    failures = 0

    # A generous clock, well above every threshold here, so the buffer's effect is visible without
    # any of the clamps (MIN_BUDGET_MS, safe_left, etc.) also being in play.
    time_left = 120_000
    budget = tm.budget_ms(time_left)
    safe_left = time_left - tm.SAFETY_MARGIN_MS
    planned = safe_left / tm.ASSUMED_MOVES_LEFT + tm.INCREMENT_MS * 0.8
    print(f"budget_ms({time_left})={budget}, unbuffered planned={planned:.1f}")
    if budget > planned - tm.POST_SEARCH_BUFFER_MS + 1:
        print("  FAIL: expected budget_ms to have POST_SEARCH_BUFFER_MS subtracted from planned")
        failures += 1

    extended = tm.extended_budget_ms(time_left)
    print(f"extended_budget_ms({time_left})={extended} (expect >= budget_ms={budget})")
    if extended < budget:
        print("  FAIL: the volatile-position budget must never be smaller than the base budget")
        failures += 1
    if extended > safe_left - tm.POST_SEARCH_BUFFER_MS:
        print("  FAIL: extended_budget_ms must also respect POST_SEARCH_BUFFER_MS")
        failures += 1

    # A very low clock: both functions must still return a sane, non-negative, non-crashing value
    # (agent.py never calls these below timeman.PANIC_MS in practice, but the functions should not
    # rely on that caller-side guarantee to stay safe).
    low_thresholds = (
        0, 1, tm.PANIC_MS, tm.SAFETY_MARGIN_MS, tm.SAFETY_MARGIN_MS + tm.POST_SEARCH_BUFFER_MS,
    )
    for low in low_thresholds:
        low_budget = tm.budget_ms(low)
        low_extended = tm.extended_budget_ms(low)
        print(f"time_left={low}: budget_ms={low_budget}, extended_budget_ms={low_extended}")
        if low_budget < tm.MIN_BUDGET_MS or low_extended < tm.MIN_BUDGET_MS:
            print(f"  FAIL: expected both to floor at MIN_BUDGET_MS ({tm.MIN_BUDGET_MS}) at worst")
            failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
