"""Differential test for movegen._popcount64 / _bit_scan (Phase 2.2 of docs/plan.md): both now
call single-instruction LLVM intrinsics (llvm.ctpop / llvm.cttz) instead of a Kernighan loop and a
64-iteration linear scan. Checked here against a plain-Python bit-twiddling reference over random
64-bit values, including the edge cases (0, all-ones, single bit) an intrinsic swap is most likely
to get wrong.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import movegen as mg

random.seed(2024)
TRIALS = 5000


def _ref_popcount(bits: int) -> int:
    return bin(bits).count("1")


def _ref_bit_scan(bits: int) -> int:
    if bits == 0:
        return -1
    return (bits & -bits).bit_length() - 1


def main() -> None:
    mismatches = 0
    values = [0, 1, (1 << 64) - 1] + [1 << i for i in range(64)]
    values += [random.getrandbits(64) for _ in range(TRIALS)]

    for v in values:
        bits = np.uint64(v)

        got_pop = int(mg._popcount64(bits))
        want_pop = _ref_popcount(v)
        if got_pop != want_pop:
            mismatches += 1
            print(f"MISMATCH popcount v={v}: got={got_pop} != want={want_pop}")

        got_scan = int(mg._bit_scan(bits))
        want_scan = _ref_bit_scan(v)
        if got_scan != want_scan:
            mismatches += 1
            print(f"MISMATCH bit_scan v={v}: got={got_scan} != want={want_scan}")

    total = len(values) * 2
    print(f"{total - mismatches}/{total} bit-op queries agree (popcount + bit_scan)")
    if mismatches:
        print(f"\n{mismatches} MISMATCH(ES)")
        raise SystemExit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
