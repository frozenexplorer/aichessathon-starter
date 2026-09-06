"""Phase 3 (docs/plan.md) 3.3 -- bulk self-play data generation, in-process.

`tools/nnue_gen_data.py` spawns a fresh subprocess pair per game (matching the real match
protocol exactly, one process per side), which is the right choice for a small fidelity-checking
run but pays each side's ~85-150s numba compile cost on *every single game* -- utterly
prohibitive at the "thousands of games" scale real training data needs (that would be
150s * 2 * thousands, i.e. days, before a single move is even played).

This script instead imports `agent` **once** (paying that compile cost exactly one time for the
whole run, via the same `_warm_up()` call the real process makes at import) and drives self-play
directly in-process: a single `chess.Board`, `agent.get_move` called alternately for whichever
colour is to move, `board.outcome(claim_draw=True)` as the sole authority on when a game ends --
exactly like `harness.referee._play`, just without the two extra processes. Reuses
`nnue_gen_data.SearchLabeler` for depth-fixed `negamax` labels, and a *shared* `SearchLabeler`
(one TT, reused across the whole run) for the same reason `nnue_gen_data.py`'s own labeler does.

Each game opens with a few random legal plies (`--random-plies-max`) before `agent.get_move`
takes over -- without this, replaying the same start position through a fully deterministic
engine with a persistent TT would produce the *same game* every time, which is useless as
training data. This is the standard "random opening, engine finishes it" self-play recipe.

**Deliberately accepted trade-off:** `agent.py`'s own `_history`/`_opponent_history` arrays
(negamax's own repetition-avoidance heuristic, not the game-ending logic) assume one continuous
same-colour sequence, as in a real single-sided match process. Feeding both colours' queries
through the same arrays here interleaves them, which can occasionally blunt that specific
heuristic mid-game. It never affects game-ending correctness -- `python-chess`'s `Board` is the
sole authority on that here, same as `harness.referee` -- so the worst case is very occasionally
slightly weaker self-play, not a wrong result or a crash. Worth it for the throughput gain.

Never imported by agent.py -- offline tooling, lives in tools/ so `harness/package.py` never
ships it; output (`tools/data/*.npz`) is gitignored.

Checkpointing: `--out` is rewritten (atomically -- written to a `.tmp.npz` then renamed over the
real path, so a crash mid-write never corrupts the previous good file) every `--checkpoint-every`
games, not only at the very end. A run that dies partway (power loss, a killed process, a machine
restart) loses at most that many games' worth of compute, not the whole run -- learned the hard
way the first time this script ran for an hour and the process died with nothing on disk, because
the original version only wrote its output after the full loop finished. `--resume` loads
whatever `--out` already holds and appends this run's new games to it, so a genuinely interrupted
run can just be restarted with the same `--out`.

Usage: .venv/Scripts/python.exe tools/nnue_selfplay_fast.py <num_games> [base_ms] [incr_ms]
       [--depth N] [--random-plies-max N] [--out tools/data/self_play_bulk.npz]
       [--checkpoint-every N] [--resume]
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess
import numpy as np
from nnue_gen_data import SearchLabeler

import agent
import bitboard as bbm
from nnue import MAX_ACTIVE_FEATURES, extract_features

OUT_DIR = Path(__file__).resolve().parent / "data"
PLY_CAP = 300


def reset_agent_state() -> None:
    agent._history_len = 0
    agent._opponent_history_len = 0
    agent._prev_piece_count = None


def play_one_game(
    labeler: SearchLabeler, base_ms: int, increment_ms: int, random_plies_max: int
) -> tuple[list[tuple[np.ndarray, int]], str]:
    board = chess.Board()
    for _ in range(random.randint(0, random_plies_max)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))

    reset_agent_state()
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    samples: list[tuple[np.ndarray, int]] = []

    while True:
        if board.outcome(claim_draw=True) is not None or len(board.move_stack) >= PLY_CAP:
            return samples, "complete"

        mover = board.turn
        fen = board.fen()
        bb, meta = bbm.from_fen(fen)
        label = labeler.score(bb, meta, bbm.halfmove_clock(fen))
        features = np.asarray(extract_features(bb, meta), dtype=np.int64)
        samples.append((features, label))

        started = time.perf_counter()
        uci = agent.get_move(fen, int(clock[mover]))
        clock[mover] -= (time.perf_counter() - started) * 1000.0
        if clock[mover] < 0:
            return samples, "flag"
        clock[mover] += increment_ms

        try:
            move = chess.Move.from_uci(uci)
        except chess.InvalidMoveError:
            return samples, "illegal"
        if move not in board.legal_moves:
            return samples, "illegal"
        board.push(move)


def save(out_path: Path, all_features: list[np.ndarray], all_labels: list[int]) -> int:
    n = len(all_labels)
    feature_matrix = np.full((n, MAX_ACTIVE_FEATURES), -1, dtype=np.int64)
    for i, features in enumerate(all_features):
        feature_matrix[i, : len(features)] = features
    labels = np.array(all_labels, dtype=np.int32)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, features=feature_matrix, labels=labels)
    tmp_path.replace(out_path)
    return n


def load_existing(out_path: Path) -> tuple[list[np.ndarray], list[int]]:
    if not out_path.exists():
        return [], []
    data = np.load(out_path)
    features = [row for row in data["features"]]
    labels = [int(x) for x in data["labels"]]
    return features, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("num_games", type=int, help="additional games to play this run")
    parser.add_argument("base_ms", type=int, nargs="?", default=3000)
    parser.add_argument("increment_ms", type=int, nargs="?", default=50)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--random-plies-max", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_DIR / "self_play_bulk.npz"))
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

    labeler = SearchLabeler(args.depth)
    if args.resume:
        all_features, all_labels = load_existing(out_path)
        print(f"resuming from {out_path}: {len(all_labels)} existing samples", flush=True)
    else:
        all_features, all_labels = [], []
    run_start = time.perf_counter()

    for game_idx in range(args.num_games):
        samples, termination = play_one_game(
            labeler, args.base_ms, args.increment_ms, args.random_plies_max
        )
        for features, label in samples:
            all_features.append(features)
            all_labels.append(label)
        elapsed = time.perf_counter() - run_start
        print(
            f"game {game_idx + 1}/{args.num_games}: {termination}, {len(samples)} positions "
            f"({len(all_features)} total samples, {elapsed:.0f}s elapsed)",
            flush=True,
        )
        if (game_idx + 1) % args.checkpoint_every == 0:
            save(out_path, all_features, all_labels)
            print(f"  checkpointed {len(all_labels)} samples to {out_path}", flush=True)

    n = save(out_path, all_features, all_labels)
    print(
        f"\nwrote {n} total samples ({args.num_games} games this run, label depth={args.depth}) "
        f"to {out_path}"
    )


if __name__ == "__main__":
    main()
