"""Bitboard position representation.

A position is two small numpy arrays, not an object, so every search node can be produced by
copying and mutating arrays inside numba-jitted code.

bb   uint64[12]   piece bitboards: bb[color * 6 + piece_type], one bit per occupied square
meta int8[6]      [turn, castle_wk, castle_wq, castle_bk, castle_bq, ep_square]
                   turn: 0 = white to move, 1 = black. castle_*: 1 if that right still exists.
                   ep_square: the square a pawn just double-pushed past, or -1.

Squares follow python-chess: a1 = 0, b1 = 1, ... h1 = 7, a2 = 8, ... h8 = 63 (file + 8 * rank).
This lets FEN parsing and cross-checks reuse python-chess directly instead of a hand-rolled
parser, which is exactly the kind of code we do not want two implementations of.
"""

import chess
import numpy as np


def i64(value: int) -> int:
    """Returns `value` as a *runtime* np.int64 while type-checking as plain `int`.

    Numba types a bare Python int crossing an njit call boundary as a distinct `Literal[int]`
    per value -- passing the same logical constant as both `Literal[int](0)` (a literal call
    argument) and plain `int64` (an arithmetic result) compiles two separate specialisations of
    every function it reaches (see docs/plan.md Phase 1.1(a)). Wrapping as `np.int64(...)` fixes
    that, but a numpy scalar doesn't satisfy the `int`-annotated parameters mypy sees on njit
    dispatchers (which mirror the wrapped function's own type hints) -- hence the return-type lie
    here. Safe either way: every njit function these constants feed treats the value as a 64-bit
    int regardless of which concrete Python type carries it.
    """
    return np.int64(value)  # type: ignore[return-value]


WHITE = i64(0)
BLACK = i64(1)

PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5

PIECE_TYPES = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

_PYCHESS_PIECE_TYPE = {
    chess.PAWN: PAWN,
    chess.KNIGHT: KNIGHT,
    chess.BISHOP: BISHOP,
    chess.ROOK: ROOK,
    chess.QUEEN: QUEEN,
    chess.KING: KING,
}


def empty_bb() -> np.ndarray:
    return np.zeros(12, dtype=np.uint64)


def from_fen(fen: str) -> tuple[np.ndarray, np.ndarray]:
    """Build (bb, meta) from a FEN string, via python-chess so parsing is battle-tested."""
    board = chess.Board(fen)
    bb = empty_bb()
    for square, piece in board.piece_map().items():
        color = WHITE if piece.color == chess.WHITE else BLACK
        piece_type = _PYCHESS_PIECE_TYPE[piece.piece_type]
        bb[color * 6 + piece_type] |= np.uint64(1) << np.uint64(square)

    meta = np.zeros(6, dtype=np.int8)
    meta[0] = WHITE if board.turn == chess.WHITE else BLACK
    meta[1] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    meta[2] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    meta[3] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    meta[4] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    meta[5] = board.ep_square if board.ep_square is not None else -1
    return bb, meta


def halfmove_clock(fen: str) -> int:
    """The FEN's own halfmove clock (plies since the last pawn move or capture) -- read directly
    from the FEN each call rather than tracked incrementally in agent.py, so it is always correct
    even if a game's starting position (see docs/IDEAS.md: rated games start from curated
    positions, not necessarily the standard start) does not itself begin at 0. Used for the
    fifty-move-rule draw detection in search.py -- see its module docstring.
    """
    return chess.Board(fen).halfmove_clock


_FILES = "abcdefgh"


def square_name(square: int) -> str:
    return _FILES[square % 8] + str(square // 8 + 1)


def move_uci(from_square: int, to_square: int, promotion: int = -1) -> str:
    promo = {PAWN: "", KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q", KING: ""}
    suffix = promo[promotion] if promotion >= 0 else ""
    return square_name(from_square) + square_name(to_square) + suffix
