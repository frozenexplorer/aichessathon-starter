"""Correctness gate for movegen.py: compare our legal moves against python-chess's, move by
move, over random games from a battery of positions. This is the thing that has to pass clean
before movegen is trusted for anything else -- a bug here is a silent, occasional loss on the
platform, not a crash we'd notice locally.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

import bitboard as bbm
import movegen as mg

POSITIONS = {
    "start": chess.STARTING_FEN,
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "tricky": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "promotion": "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
    "en_passant": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
}


def legal_uci_ours(bb: object, meta: object) -> set[str]:
    from_arr, to_arr, promo_arr, count = mg.generate_legal(bb, meta)
    return {bbm.move_uci(from_arr[i], to_arr[i], promo_arr[i]) for i in range(count)}


def legal_uci_reference(board: chess.Board) -> set[str]:
    return {move.uci() for move in board.legal_moves}


def check_position(name: str, fen: str, max_plies: int, games: int, rng: random.Random) -> int:
    mismatches = 0
    for game in range(games):
        board = chess.Board(fen)
        for ply in range(max_plies):
            if board.is_game_over():
                break
            bb, meta = bbm.from_fen(board.fen())
            ours = legal_uci_ours(bb, meta)
            reference = legal_uci_reference(board)
            if ours != reference:
                mismatches += 1
                print(f"MISMATCH {name} game={game} ply={ply} fen={board.fen()}")
                print(f"  ours only:      {sorted(ours - reference)}")
                print(f"  reference only: {sorted(reference - ours)}")
                return mismatches
            move = rng.choice(list(board.legal_moves))
            board.push(move)
    return mismatches


def perft(bb: object, meta: object, depth: int) -> int:
    if depth == 0:
        return 1
    from_arr, to_arr, promo_arr, count = mg.generate_legal(bb, meta)
    if depth == 1:
        return count
    nodes = 0
    for i in range(count):
        new_bb, new_meta = mg.make_move(bb, meta, from_arr[i], to_arr[i], promo_arr[i])
        nodes += perft(new_bb, new_meta, depth - 1)
    return nodes


PERFT_EXPECTED = {
    ("start", 1): 20,
    ("start", 2): 400,
    ("start", 3): 8_902,
    ("start", 4): 197_281,
    ("kiwipete", 1): 48,
    ("kiwipete", 2): 2_039,
    ("kiwipete", 3): 97_862,
}


def check_perft() -> int:
    failures = 0
    for (name, depth), expected in PERFT_EXPECTED.items():
        bb, meta = bbm.from_fen(POSITIONS[name])
        got = perft(bb, meta, depth)
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"perft {name} depth={depth}: {got} (expected {expected}) [{status}]")
    return failures


def main() -> None:
    rng = random.Random(1)
    total_mismatches = 0
    for name, fen in POSITIONS.items():
        mismatches = check_position(name, fen, max_plies=60, games=20, rng=rng)
        total_mismatches += mismatches
        print(f"{name}: {mismatches} mismatches")

    perft_failures = check_perft()

    if total_mismatches or perft_failures:
        print(f"\nFAILED: {total_mismatches} differential mismatches, {perft_failures} perft")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
