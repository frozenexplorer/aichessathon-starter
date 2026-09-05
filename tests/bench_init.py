"""Benchmark and regression-guard the import-time numba compile cost.

Imports agent.py once (in this process -- there is nothing to warm twice) and reports total wall
time plus a per-function numba compile breakdown, by patching Dispatcher.compile to time every
call before delegating to the real implementation. It then dumps the distinct argument-type
signatures every jitted function ended up with: the whole point of this script is to catch a
function compiling more than once for the same logical call site, which is the mechanism behind
most of the historical init-time regressions (see docs/plan.md's Phase 1.1) and is invisible from
wall time alone until it has already doubled some function's cost.

Not a pytest file, like every other script in tests/ -- run directly:

    python tests/bench_init.py [--budget-s SECONDS]

Exits non-zero if any dispatcher has more than one specialisation, or if total import time exceeds
--budget-s (default: unset, i.e. report only). The real gate is the platform's 90s cap; this script
runs on a dev machine that historically measures ~2-2.5x slower (docs/plan.md's "Measured current
state"), so pass --budget-s deliberately, don't assume the platform number transfers directly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-s", type=float, default=None,
        help="fail if total import time (seconds) exceeds this",
    )
    parser.add_argument(
        "--top", type=int, default=25, help="how many functions to show in the breakdown",
    )
    args = parser.parse_args()

    from numba.core import dispatcher

    times: list[tuple[float, str, str]] = []
    orig_compile = dispatcher.Dispatcher.compile

    def timed_compile(self: object, sig: object) -> object:
        start = time.perf_counter()
        try:
            return orig_compile(self, sig)  # type: ignore[misc]
        finally:
            elapsed = time.perf_counter() - start
            module = getattr(self.py_func, "__module__", "?")  # type: ignore[attr-defined]
            name = self.py_func.__name__  # type: ignore[attr-defined]
            times.append((elapsed, module, name))

    dispatcher.Dispatcher.compile = timed_compile  # type: ignore[method-assign]
    t0 = time.perf_counter()
    import agent  # noqa: F401  (import is the thing being measured)
    total = time.perf_counter() - t0
    dispatcher.Dispatcher.compile = orig_compile  # type: ignore[method-assign]

    print(f"TOTAL_IMPORT_SEC {total:.2f}")
    print(f"SUM_COMPILE_SEC {sum(t for t, _, _ in times):.2f}  (n_compile_calls={len(times)})")

    by_func: dict[tuple[str, str], list[float]] = {}
    for elapsed, module, name in times:
        by_func.setdefault((module, name), []).append(elapsed)

    print(f"\n{'module':<10} {'function':<30} {'seconds':>8} {'sigs':>5}")
    ranked = sorted(by_func.items(), key=lambda kv: -sum(kv[1]))
    for (module, name), elapsed_list in ranked[: args.top]:
        print(f"{module:<10} {name:<30} {sum(elapsed_list):>8.2f} {len(elapsed_list):>5}")

    # Signature audit: import agent's own modules and inspect every dispatcher's .signatures.
    # This is what actually explains a function appearing more than once above -- distinct
    # argument types compiled as distinct specialisations of the same source function.
    import evaluate as ev
    import movegen as mg
    import search as sr
    import zobrist as zb

    print(f"\n{'module':<10} {'function':<30} {'specialisations':>16}")
    multi_spec = 0
    for mod in (sr, mg, ev, zb):
        for attr_name in sorted(dir(mod)):
            obj = getattr(mod, attr_name)
            sigs = getattr(obj, "signatures", None)
            if sigs is None:
                continue
            n = len(sigs)
            marker = "  <-- MULTIPLE" if n > 1 else ""
            print(f"{mod.__name__:<10} {attr_name:<30} {n:>16}{marker}")
            if n > 1:
                multi_spec += 1
                for sig in sigs:
                    print(f"    {sig}")

    failures = []
    if multi_spec:
        failures.append(
            f"{multi_spec} function(s) compiled more than one specialisation "
            "(see docs/plan.md Phase 1.1)"
        )
    if args.budget_s is not None and total > args.budget_s:
        failures.append(f"total import time {total:.2f}s exceeds --budget-s {args.budget_s:.2f}s")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
