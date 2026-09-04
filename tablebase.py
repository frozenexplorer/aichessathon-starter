"""Syzygy WDL endgame tablebase, root-only.

Filters the root move list down to WDL-optimal moves (never conceding a win or a draw) using
the files shipped in weights/syzygy/, then lets the normal search choose among those with its
own eval and repetition-avoidance. WDL alone says whether a position is objectively won, drawn
or lost, not how to make progress toward it -- filtering rather than picking a move outright is
what actually converts a won endgame instead of just not blundering it away.

Only probed at the root, not inside the search tree: that would mean building a chess.Board
from our bitboard state at every node just to call into python-chess's tablebase reader, a much
larger and riskier change for a benefit that mostly matters once, at the point a game actually
reaches one of these endgames.

WDL-only (no DTZ): the complete 3-4-man set is ~1.3MB, downloaded whole so there is no coverage
gap within that scope. The complete 5-man set is ~380MB -- worth a future look, but well past
what is worth spending out of the 50MB cap for a first version.

Every failure mode here (missing directory, a position outside the shipped material range, a
python-chess exception) degrades to returning None, meaning "use the normal, already-proven
search" -- this module must never be able to turn a bug into a lost game.
"""

from pathlib import Path

import chess
import chess.syzygy

MAX_PIECES = 4

_DIR = Path(__file__).resolve().parent / "weights" / "syzygy"
_PROMOTION_INDEX = {chess.QUEEN: 4, chess.ROOK: 3, chess.BISHOP: 2, chess.KNIGHT: 1}

try:
    _tablebase: chess.syzygy.Tablebase | None = chess.syzygy.open_tablebase(str(_DIR))
except Exception:
    _tablebase = None


def piece_count(bb: object) -> int:
    total = 0
    for value in bb:  # type: ignore[attr-defined]
        total += bin(int(value)).count("1")
    return total


def best_moves(fen: str) -> list[tuple[int, int, int]] | None:
    """WDL-optimal legal moves as (from_square, to_square, promotion_index) triples, promotion
    index using bitboard.py's convention (-1 for none). None if unavailable for this position.
    """
    if _tablebase is None:
        return None
    try:
        board = chess.Board(fen)
        best: int | None = None
        candidates: list[tuple[int, int, int]] = []
        for move in board.legal_moves:
            board.push(move)
            try:
                wdl = -_tablebase.probe_wdl(board)
            finally:
                board.pop()
            promo = _PROMOTION_INDEX.get(move.promotion, -1) if move.promotion else -1
            triple = (move.from_square, move.to_square, promo)
            if best is None or wdl > best:
                best = wdl
                candidates = [triple]
            elif wdl == best:
                candidates.append(triple)
        return candidates if candidates else None
    except Exception:
        return None
