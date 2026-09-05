"""Benchmark search node rate on a fixed position set, at a fixed time budget per position.

This is the acceptance test for every Phase 2 (node-rate) change in docs/plan.md: a change is kept
only if this script shows more nodes (or the same nodes, faster) on these positions, not on
"it looks right." Positions:

  blunder   the Tier-1 tactical FEN every tier in docs/STATUS.md was benched on -- sharp
            middlegame, heavy branching, the position that first exposed shallow-search blunders.
  quiet     a quiet middlegame with no immediate tactics, to catch a change that only helps sharp
            positions (e.g. better pruning) at quiet positions' expense (e.g. worse ordering).
  endgame   a simple king-and-pawn endgame, few pieces, to catch a change that regresses the
            low-piece-count code paths (extended time budget, tablebase-adjacent play).

Not pytest -- run directly:

    python tests/bench_nodes.py [--seconds 8]

Pays the full numba compile cost on first run (there is no cache -- see docs/plan.md); that cost is
excluded from the reported node rate by warming with a depth-1 search on each position first.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bitboard as bbm
import movegen as mg
import search as sr

POSITIONS = {
    "blunder": "r2qr2k/pp5p/8/3nPbp1/3B1P1b/1BPp4/PQ4P1/R5KR b - - 3 22",
    "quiet": "r1bq1rk1/ppp2ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 7",
    "endgame": "8/5k2/8/3K4/3P4/8/8/8 w - - 0 1",
}


def run_fixed_time(fen: str, seconds: float) -> tuple[int, int, float]:
    """Iterative deepening at a fixed wall-clock budget; returns (nodes, depth_reached, elapsed)."""
    bb, meta = bbm.from_fen(fen)
    tt = sr.new_tt()
    history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
    opponent_history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)
    killer_from, killer_to, killer_promo = sr.new_killers()
    history_table = sr.new_history_table()
    counter_from, counter_to, counter_promo = sr.new_counter_table()

    deadline = time.perf_counter() + seconds
    depth = 1
    total_nodes = 0
    depth_reached = 0
    prev_score = sr.NO_PREV_SCORE
    start = time.perf_counter()
    while True:
        counters = sr.new_counters()
        f, t, p, score, completed = sr.search_root(
            bb, meta, depth, deadline, counters, -1, -1, -1,
            history, 1, opponent_history, 0,
            *tt,
            killer_from, killer_to, killer_promo, history_table,
            counter_from, counter_to, counter_promo, prev_score, 0,
        )
        total_nodes += int(counters[0])
        if not completed:
            break
        depth_reached = depth
        prev_score = score
        if time.perf_counter() >= deadline or abs(score) >= sr.MATE_THRESHOLD:
            break
        depth += 1
    elapsed = time.perf_counter() - start
    return total_nodes, depth_reached, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0, help="fixed budget per position")
    args = parser.parse_args()

    print("warming numba compile (excluded from timing below)...")
    warm_fen = next(iter(POSITIONS.values()))
    run_fixed_time(warm_fen, seconds=2.0)

    print(f"\n{'position':<10} {'nodes':>10} {'depth':>6} {'elapsed_s':>10} {'nodes/s':>12}")
    for name, fen in POSITIONS.items():
        nodes, depth, elapsed = run_fixed_time(fen, args.seconds)
        rate = nodes / elapsed if elapsed > 0 else 0.0
        print(f"{name:<10} {nodes:>10} {depth:>6} {elapsed:>10.2f} {rate:>12,.0f}")


if __name__ == "__main__":
    main()
