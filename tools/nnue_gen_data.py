"""Phase 3 (docs/plan.md) 3.3 -- self-bootstrapped training data generation.

Plays self-play games through the real `harness.sandbox`/`harness.referee` protocol (one process
per side, both pointed at the same agent directory, so game state isolation matches an actual
match instead of reusing `agent.py`'s module-level history/TT globals across synthetic games),
then replays each finished game's PGN through `python-chess` to recover every position visited.

Label: a fixed-depth `search.negamax` score (default depth 4), called the same standalone way
`tests/test_singular_extension.py` calls it -- a huge deadline (no time pressure, a real fixed
depth rather than a time-boxed one) and fresh `killer`/`history_table`/`counter_table` per
position (matching `agent.py`'s own "rebuilt fresh per real move decision" usage), but one shared
`new_tt()` and node-`counters` array reused across the whole run: unrelated positions sharing a TT
is exactly how a real game's TT already gets reused move over move, and reallocating TT's ~300MB
of parallel arrays per sample would dominate this script's own runtime for no benefit. This bakes
in resolved tactics a raw static eval misses. Pass `--depth 0` to fall back to the cheaper V0
label (`evaluate.evaluate`, no search at all) for a quick pipeline smoke test.

`AGENTS.md:61-64` permits training on labels our own engine produced -- this route needs no
external engine. Never imported by agent.py -- this is offline tooling, lives in tools/ (not the
repo root) per docs/plan.md 3.0 so `harness/package.py` never ships it, and its output
(tools/data/*.npz) is gitignored: it is regenerable multi-megabyte data, not source.

Usage: .venv/Scripts/python.exe tools/nnue_gen_data.py <agent_dir> <num_games> [base_ms] [incr_ms]
       [--depth N]
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.pgn
import numpy as np

import harness.referee as referee

referee.INIT_BUDGET_S = 180.0  # type: ignore[attr-defined]

import bitboard as bbm  # noqa: E402
import evaluate as ev  # noqa: E402
import search as sr  # noqa: E402
from harness.referee import play_match  # noqa: E402
from harness.sandbox import local  # noqa: E402
from nnue import MAX_ACTIVE_FEATURES, extract_features  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data"
NO_DEADLINE = 1e18


def positions_from_pgn(pgn_text: str) -> list[str]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []
    board = game.board()
    fens = []
    for move in game.mainline_moves():
        board.push(move)
        fens.append(board.fen())
    return fens


class SearchLabeler:
    """Fixed-depth `negamax` scorer with a TT shared across every position it labels."""

    def __init__(self, depth: int):
        self.depth = depth
        self.tt = sr.new_tt()
        self.counters = sr.new_counters()
        self.history = np.zeros(sr.HISTORY_CAPACITY, dtype=np.uint64)

    def score(self, bb: np.ndarray, meta: np.ndarray, halfmove_clock: int) -> int:
        killer_from, killer_to, killer_promo = sr.new_killers()
        history_table = sr.new_history_table()
        cont_hist = sr.new_continuation_history()
        counter_from, counter_to, counter_promo = sr.new_counter_table()
        return int(
            sr.negamax(
                bb, meta, self.depth, -sr.INF, sr.INF, NO_DEADLINE, self.counters, 0,
                self.history, 0, *self.tt, killer_from, killer_to, killer_promo, history_table,
                cont_hist, True, sr.MAX_CHECK_EXTENSIONS, counter_from, counter_to, counter_promo,
                -1, -1, halfmove_clock, -1, -1,
            )
        )


def main() -> None:
    agent_dir = Path(sys.argv[1]).resolve()
    num_games = int(sys.argv[2])
    positional_args = [a for a in sys.argv[3:] if not a.startswith("--")]
    base_ms = int(positional_args[0]) if len(positional_args) > 0 else 10_000
    increment_ms = int(positional_args[1]) if len(positional_args) > 1 else 100
    depth = 4
    if "--depth" in sys.argv:
        depth = int(sys.argv[sys.argv.index("--depth") + 1])
    labeler = SearchLabeler(depth) if depth > 0 else None

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []

    for game_idx in range(num_games):
        outcome = play_match(local(agent_dir), local(agent_dir), base_ms, increment_ms)
        fens = positions_from_pgn(outcome.pgn)
        for fen in fens:
            bb, meta = bbm.from_fen(fen)
            if labeler is not None:
                label = labeler.score(bb, meta, bbm.halfmove_clock(fen))
            else:
                label = int(ev.evaluate(bb, meta))
            features = np.asarray(extract_features(bb, meta), dtype=np.int64)
            all_features.append(features)
            all_labels.append(label)
        print(
            f"game {game_idx + 1}/{num_games}: {outcome.result} by {outcome.termination}, "
            f"{len(fens)} positions ({len(all_features)} total samples so far)",
            flush=True,
        )

    n = len(all_labels)
    feature_matrix = np.full((n, MAX_ACTIVE_FEATURES), -1, dtype=np.int64)
    for i, features in enumerate(all_features):
        feature_matrix[i, : len(features)] = features
    labels = np.array(all_labels, dtype=np.int32)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "self_play_v0.npz"
    np.savez_compressed(out_path, features=feature_matrix, labels=labels)
    print(f"\nwrote {n} samples from {num_games} games (label depth={depth}) to {out_path}")


if __name__ == "__main__":
    main()
