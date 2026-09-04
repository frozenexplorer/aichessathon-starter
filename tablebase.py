"""Syzygy WDL + DTZ endgame tablebase, root-only.

Filters the root move list down to WDL-optimal moves (never conceding a win or a draw) using the
files shipped in weights/syzygy/, then narrows further by DTZ (distance-to-zero) among those
tied on WDL, then lets the normal search choose among what is left with its own eval and
repetition-avoidance. WDL alone says whether a position is objectively won, drawn or lost, not
how to make progress toward it: a search that only avoids WDL-losing moves can still drift
between several different won-but-not-shrinking lines forever, which is exactly what caused the
KBN vs K conversion gap documented in docs/STATUS.md's "Future" section before this. DTZ fixes
that: among WDL-tied moves, prefer the smallest DTZ magnitude while winning (fastest forced
zeroing move, i.e. real progress) and the largest while lost (the best practical resistance,
since no move actually saves a provably lost position). A draw needs no DTZ tie-break -- every
move preserving a draw has DTZ 0, and any one of them stays a draw.

Only probed at the root, not inside the search tree: that would mean building a chess.Board from
our bitboard state at every node just to call into python-chess's tablebase reader, a much larger
and riskier change for a benefit that mostly matters once, at the point a game actually reaches
one of these endgames.

Same material scope as before (complete 3-4-man), now with both file kinds: WDL ~1.3MB, DTZ
~3.1MB, ~4.4MB total, from the same source. The complete 5-man set is ~380MB -- worth a future
look, but well past what is worth spending out of the 50MB cap for this pass.

Every failure mode here (missing directory, a position outside the shipped material range, a
python-chess exception, DTZ files present for WDL but missing for a specific combination) has a
narrower fallback beneath it rather than dropping straight to "use the normal, unrestricted
search": a DTZ failure falls back to the WDL-tied move list, and a WDL failure (or no tablebase
at all) falls back to None -- this module must never be able to turn a bug into a lost game.
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
    """WDL-optimal legal moves, narrowed by DTZ where available, as (from_square, to_square,
    promotion_index) triples, promotion index using bitboard.py's convention (-1 for none). None
    if unavailable for this position.
    """
    if _tablebase is None:
        return None
    try:
        board = chess.Board(fen)
        best_wdl: int | None = None
        candidates: list[tuple[chess.Move, tuple[int, int, int]]] = []
        for move in board.legal_moves:
            board.push(move)
            try:
                wdl = -_tablebase.probe_wdl(board)
            finally:
                board.pop()
            promo = _PROMOTION_INDEX.get(move.promotion, -1) if move.promotion else -1
            triple = (move.from_square, move.to_square, promo)
            if best_wdl is None or wdl > best_wdl:
                best_wdl = wdl
                candidates = [(move, triple)]
            elif wdl == best_wdl:
                candidates.append((move, triple))
        if not candidates or best_wdl is None:
            return None
        wdl_tied = [triple for _, triple in candidates]
        if best_wdl == 0 or len(candidates) == 1:
            return wdl_tied
        try:
            narrowed = _narrow_by_dtz(board, candidates, best_wdl)
        except Exception:
            narrowed = None
        return narrowed if narrowed else wdl_tied
    except Exception:
        return None


def _narrow_by_dtz(
    board: chess.Board,
    candidates: list[tuple[chess.Move, tuple[int, int, int]]],
    best_wdl: int,
) -> list[tuple[int, int, int]] | None:
    assert _tablebase is not None
    scored: list[tuple[int, tuple[int, int, int]]] = []
    for move, triple in candidates:
        board.push(move)
        try:
            dtz = -_tablebase.probe_dtz(board)
        finally:
            board.pop()
        scored.append((abs(dtz), triple))
    target = min(m for m, _ in scored) if best_wdl > 0 else max(m for m, _ in scored)
    return [triple for magnitude, triple in scored if magnitude == target]
