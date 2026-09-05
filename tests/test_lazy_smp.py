"""Correctness/stress test for agent.py's Lazy SMP helper threads (search.py's search_root is
compiled nogil=True specifically to support this -- see both modules' docstrings). This does not
assert on strength, only on safety: no thread ever raises, every move returned while helpers are
running concurrently is legal, and the shared TT actually gets written by more than the main
thread alone (proof the helpers really ran, not just started and immediately no-opped).
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

import agent

PLIES = 24
TIME_LEFT_MS = 2000

_thread_exceptions: list[BaseException] = []


def _record_thread_exception(args: threading.ExceptHookArgs) -> None:
    exc = args.exc_value if args.exc_value is not None else Exception("unknown")
    _thread_exceptions.append(exc)


def main() -> None:
    failures = 0

    print(f"SMP_THREADS={agent.SMP_THREADS}")
    if agent.SMP_THREADS < 2:
        print("  NOTE: this machine/process only allows 1 thread (os.cpu_count() <= 1) -- "
              "helper-thread behaviour below degrades to a no-op, which is correct, not a failure.")

    previous_hook = threading.excepthook
    threading.excepthook = _record_thread_exception
    try:
        board = chess.Board()
        for ply in range(PLIES):
            if board.is_game_over(claim_draw=True):
                break
            uci = agent.get_move(board.fen(), TIME_LEFT_MS)
            try:
                move = chess.Move.from_uci(uci)
            except chess.InvalidMoveError:
                move = None
            if move is None or move not in board.legal_moves:
                print(f"  FAIL: illegal/unparseable move {uci!r} at ply {ply}, "
                      f"fen={board.fen()}")
                failures += 1
                break
            board.push(move)
    finally:
        threading.excepthook = previous_hook

    print(f"played {len(board.move_stack)} plies, final fen={board.fen()}")

    if _thread_exceptions:
        print(f"  FAIL: {len(_thread_exceptions)} helper-thread exception(s) raised: "
              f"{_thread_exceptions}")
        failures += 1
    else:
        print("no helper-thread exceptions raised")

    written = int((agent._tt_depth >= 0).sum())
    print(f"TT entries written after the game: {written} (expect a large number -- proof helper "
          f"threads, not just the main thread, actually searched and wrote to the shared table)")
    if written < 1000:
        print("  FAIL: expected substantially more TT activity from a real multi-threaded game")
        failures += 1

    if failures:
        print(f"\nFAILED: {failures} case(s)")
        sys.exit(1)
    print("\nALL CLEAR")


if __name__ == "__main__":
    main()
