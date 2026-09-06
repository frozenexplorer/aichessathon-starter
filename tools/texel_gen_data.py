"""Phase 4.1 (docs/plan.md) -- self-play data generation for Texel tuning: reuses
`tools/nnue_selfplay_fast.py`'s in-process bulk self-play (one `agent` import paying its compile
cost exactly once, then many synthetic games driven directly via `chess.Board` + `agent.get_move`)
but records `(FEN, game result)` pairs instead of per-position search-depth labels -- Texel tuning
fits eval constants against real game *outcomes*, not against another search's opinion of the
position, so no per-position labeling search is needed here at all (which is also why this run is
faster per game than nnue_selfplay_fast.py's was).

Result is stored once per game, from White's perspective (1.0 white win, 0.5 draw, 0.0 black win),
against every FEN visited in that game -- `tools/texel_tune.py` converts to the side-to-move's own
perspective at load time, since `evaluate()` itself is side-to-move-relative.

Checkpointing follows the exact same atomic-write/`--resume` pattern nnue_selfplay_fast.py already
proved out (write to a `.tmp.npz` then `Path.replace()`, so a crash mid-write never corrupts the
previous good file) -- see that module's docstring for why this exists at all (a prior run lost 63
minutes of compute to an unrelated machine restart before checkpointing was added).

Never imported by agent.py -- offline tooling, lives in tools/ so harness/package.py never ships
it; output (tools/data/*.npz) is gitignored.

Usage: .venv/Scripts/python.exe tools/texel_gen_data.py <num_games> [base_ms] [incr_ms]
       [--random-plies-max N] [--out tools/data/texel_bulk.npz]
       [--checkpoint-every N] [--resume]
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import numpy as np

import agent

OUT_DIR = Path(__file__).resolve().parent / "data"
PLY_CAP = 300


def reset_agent_state() -> None:
    agent._history_len = 0
    agent._opponent_history_len = 0
    agent._prev_piece_count = None


def play_one_game(
    base_ms: int, increment_ms: int, random_plies_max: int
) -> tuple[list[str], float, str]:
    board = chess.Board()
    for _ in range(random.randint(0, random_plies_max)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))

    reset_agent_state()
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    fens: list[str] = []

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None or len(board.move_stack) >= PLY_CAP:
            if outcome is None or outcome.winner is None:
                result = 0.5
            elif outcome.winner == chess.WHITE:
                result = 1.0
            else:
                result = 0.0
            return fens, result, "complete"

        mover = board.turn
        fen = board.fen()
        fens.append(fen)

        started = time.perf_counter()
        uci = agent.get_move(fen, int(clock[mover]))
        clock[mover] -= (time.perf_counter() - started) * 1000.0
        if clock[mover] < 0:
            result = 1.0 if mover == chess.BLACK else 0.0
            return fens, result, "flag"

        clock[mover] += increment_ms

        try:
            move = chess.Move.from_uci(uci)
        except chess.InvalidMoveError:
            result = 1.0 if mover == chess.BLACK else 0.0
            return fens, result, "illegal"
        if move not in board.legal_moves:
            result = 1.0 if mover == chess.BLACK else 0.0
            return fens, result, "illegal"
        board.push(move)


def save(out_path: Path, all_fens: list[str], all_results: list[float]) -> int:
    n = len(all_fens)
    fens_arr = np.array(all_fens, dtype=object)
    results_arr = np.array(all_results, dtype=np.float32)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, fens=fens_arr, results=results_arr)
    tmp_path.replace(out_path)
    return n


def load_existing(out_path: Path) -> tuple[list[str], list[float]]:
    if not out_path.exists():
        return [], []
    data = np.load(out_path, allow_pickle=True)
    fens = [str(x) for x in data["fens"]]
    results = [float(x) for x in data["results"]]
    return fens, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("num_games", type=int, help="additional games to play this run")
    parser.add_argument("base_ms", type=int, nargs="?", default=3000)
    parser.add_argument("increment_ms", type=int, nargs="?", default=50)
    parser.add_argument("--random-plies-max", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_DIR / "texel_bulk.npz"))
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="write --out to disk every N games, so a crash/restart loses at most N games",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="if --out already exists, load it first and append this run's new games to it",
    )
    args = parser.parse_args()
    out_path = Path(args.out)

    if args.resume:
        all_fens, all_results = load_existing(out_path)
        print(f"resuming from {out_path}: {len(all_fens)} existing samples", flush=True)
    else:
        all_fens, all_results = [], []
    run_start = time.perf_counter()

    for game_idx in range(args.num_games):
        fens, result, termination = play_one_game(
            args.base_ms, args.increment_ms, args.random_plies_max
        )
        for fen in fens:
            all_fens.append(fen)
            all_results.append(result)
        elapsed = time.perf_counter() - run_start
        print(
            f"game {game_idx + 1}/{args.num_games}: {termination}, result={result}, "
            f"{len(fens)} positions ({len(all_fens)} total samples, {elapsed:.0f}s elapsed)",
            flush=True,
        )
        if (game_idx + 1) % args.checkpoint_every == 0:
            save(out_path, all_fens, all_results)
            print(f"  checkpointed {len(all_fens)} samples to {out_path}", flush=True)

    n = save(out_path, all_fens, all_results)
    print(f"\nwrote {n} total samples ({args.num_games} games this run) to {out_path}")


if __name__ == "__main__":
    main()
