"""Gate-equivalent head-to-head runner: current build vs an older build checked out into
a separate directory (a git worktree). Never edits harness/ -- raises the locally-observed
INIT_BUDGET_S in this process's memory only, exactly like the Tier 17 real-contract check did,
so a slower dev-machine compile on either side doesn't disqualify a build that would clear the
real 90s platform budget. Lives in tools/, not the repo root, so it never ships in the zip.

Usage: .venv/Scripts/python.exe tools/head_to_head.py <opponent_dir> [games] [base_ms] [incr_ms]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.referee as referee

referee.INIT_BUDGET_S = 180.0  # type: ignore[attr-defined]

from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.rules import PLY_CAP  # noqa: E402
from harness.sandbox import local  # noqa: E402


def main() -> None:
    opponent_dir = Path(sys.argv[1]).resolve()
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    base_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 120_000
    increment_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    current = Path(".").resolve()
    wins = draws = losses = 0
    terminations: dict[str, int] = {}

    for game in range(games):
        plays_white = game % 2 == 0
        white, black = (current, opponent_dir) if plays_white else (opponent_dir, current)
        outcome = play_match(local(white), local(black), base_ms, increment_ms, ply_cap=PLY_CAP)
        terminations[outcome.termination] = terminations.get(outcome.termination, 0) + 1
        if outcome.result in ("draw", "void"):
            draws += 1
        elif (outcome.result == "white") == plays_white:
            wins += 1
        else:
            losses += 1
        print(f"game {game + 1}/{games}: {outcome.result} by {outcome.termination}", flush=True)

    score = (wins + draws / 2) / games
    print(f"\ncurrent ({current}) vs {opponent_dir} over {games} games")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print("terminations: " + ", ".join(f"{n} {c}" for n, c in terminations.items()))
    broken = {n: c for n, c in terminations.items() if n in FAILED_TERMINATIONS}
    if broken:
        print("FAILURES: " + ", ".join(f"{n} {c}" for n, c in broken.items()))


if __name__ == "__main__":
    main()
