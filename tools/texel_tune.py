"""Phase 4.1 (docs/plan.md) -- Texel tuning: fits evaluate.py's material values and
piece-square tables against real self-play game outcomes via the classic Texel tuning method
(coordinate/local search minimizing sigmoid-fitted squared error -- see chessprogramming.org
"Texel's Tuning Method").

IMPORTANT, learned the hard way: PIECE_VALUE/PST/PST_KING_END being plain numpy-array module
globals in evaluate.py does NOT make them safe to mutate in place for this purpose. Verified
directly (see the differential check `_verify_parity` runs before tuning starts): numba freezes a
global array's VALUES into the compiled machine code at first call -- mutating the same array
object's contents afterward has zero effect on already-compiled njit functions, unlike an array
passed as an ARGUMENT, which numba always reads live from memory on every call (this is exactly
why history_table/cont_hist/the TT arrays elsewhere in this codebase work at all: they are always
threaded through as parameters, never read as globals). So tuning here works by threading
`piece_value`/`pst`/`pst_king_end` as explicit parameters into a small parameterized clone of
evaluate.material_and_pst (`_material_and_pst_tunable`, below) and a matching orchestration clone
of evaluate.evaluate (`_eval_tunable`) that calls every OTHER term via evaluate.py's real,
unmodified functions -- `_verify_parity` asserts this clone produces byte-identical scores to the
real evaluate() when handed evaluate.py's own current constants, across a batch of real
self-play positions, before any tuning is trusted.

Scope, deliberately: material values (excluding the king's fixed-0 sentinel, which cancels
identically in the white-minus-black sum regardless of value -- both sides always have exactly
one king) and every piece-square table, including the endgame king table. These are the
parameters classic Texel tuning targets most, and the only ones this module builds a parameterized
clone for. Everything else evaluate() computes (pawn structure, piece features, mobility, threats,
pins, king safety, outposts, space/storm, tempo, OCB scale) is left at its current hand-picked
value and computed via evaluate.py's own real functions unchanged -- extending the parameterized
clone to those too would mean copying most of evaluate.py a second time, for comparatively small
expected payoff next to material+PST, so it was judged not worth the added correctness risk and
implementation time this close to the deadline (see docs/plan.md Phase 4.1).

Standard win-probability sigmoid, K=400 (the conventional centipawn scaling most engines already
assume -- not separately fit here): p = 1 / (1 + 10^(-eval/400)).

Data: tools/texel_gen_data.py's (FEN, game result) self-play output, result stored white-relative
(1.0/0.5/0.0). The fit target is the SIDE TO MOVE's own win probability at that position --
evaluate() is side-to-move-relative, so flipped to `1 - result` when black is to move.

batch_sse is its own tiny njit function so the per-position loop runs as compiled code calling
_eval_tunable directly, rather than paying Python/numba call-dispatch overhead once per position
per trial -- the difference between a tuning run finishing in minutes versus not finishing before
the deadline at this position count times parameter count.

Never imported by agent.py -- offline tooling, lives in tools/ so harness/package.py never ships
it. Output is new literal values for evaluate.py's arrays, printed for manual review and copy-in
(docs/plan.md's own budget arithmetic calls for "same constants, better values" -- no shipped
file, no init-time or zip-space cost from this at all), not written back to any file automatically.

Usage: .venv/Scripts/python.exe tools/texel_tune.py <dataset.npz> [<dataset2.npz> ...]
       [--sample N] [--passes N] [--steps 4,2,1] [--seed N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from numba import njit

import bitboard as bbm
from bitboard import KING, WHITE
from evaluate import (
    PHASE_MAX,
    PIECE_VALUE,
    PST,
    PST_KING_END,
    TEMPO_BONUS,
    evaluate,
    game_phase,
    king_safety_score,
    mobility_score,
    opposite_bishops_scale,
    outpost_score,
    passed_pawn_king_distance,
    piece_features,
    pin_and_xray_score,
    space_and_storm_score,
    threats_score,
)
from evaluate import pawn_structure as ev_pawn_structure

K = 400.0
ONE = np.uint64(1)


@njit(cache=False)
def _material_and_pst_tunable(
    bb: np.ndarray,
    phase: int,
    piece_value: np.ndarray,
    pst: np.ndarray,
    pst_king_end: np.ndarray,
) -> int:
    """Exact parameterized copy of evaluate.material_and_pst -- see this module's docstring for
    why a copy, not the real function with mutated globals.
    """
    score = np.int32(0)
    for pt in range(6):
        white_bb = bb[pt]
        black_bb = bb[6 + pt]
        for square in range(64):
            bit = ONE << np.uint64(square)
            if white_bb & bit:
                if pt == KING:
                    mg, eg = pst[pt, square], pst_king_end[square]
                    score += (
                        piece_value[pt] + (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
                    )
                else:
                    score += piece_value[pt] + pst[pt, square]
            if black_bb & bit:
                mirror = square ^ 56
                if pt == KING:
                    mg, eg = pst[pt, mirror], pst_king_end[mirror]
                    score -= (
                        piece_value[pt] + (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
                    )
                else:
                    score -= piece_value[pt] + pst[pt, mirror]
    return int(score)


@njit(cache=False)
def _eval_tunable(
    bb: np.ndarray,
    meta: np.ndarray,
    piece_value: np.ndarray,
    pst: np.ndarray,
    pst_king_end: np.ndarray,
) -> int:
    """Mirrors evaluate.evaluate's orchestration exactly, swapping only material_and_pst for the
    parameterized clone above -- every other term is evaluate.py's real, unmodified function.
    """
    phase = game_phase(bb)
    score = (
        _material_and_pst_tunable(bb, phase, piece_value, pst, pst_king_end)
        + ev_pawn_structure(bb, phase)
        + piece_features(bb, phase)
        + mobility_score(bb)
        + passed_pawn_king_distance(bb, phase)
        + threats_score(bb)
        + pin_and_xray_score(bb)
        + king_safety_score(bb, phase)
        + outpost_score(bb)
        + space_and_storm_score(bb, phase)
    )
    score = score * opposite_bishops_scale(bb) // 100
    stm_score = int(score) if meta[0] == WHITE else int(-score)
    return stm_score + int(TEMPO_BONUS)


@njit(cache=False)
def batch_sse(
    bbs: np.ndarray,
    metas: np.ndarray,
    targets: np.ndarray,
    n: int,
    piece_value: np.ndarray,
    pst: np.ndarray,
    pst_king_end: np.ndarray,
) -> float:
    total = 0.0
    for i in range(n):
        s = _eval_tunable(bbs[i], metas[i], piece_value, pst, pst_king_end)
        p = 1.0 / (1.0 + 10.0 ** (-s / K))
        d = p - targets[i]
        total += d * d
    return total


def verify_parity(bbs: np.ndarray, metas: np.ndarray, n: int) -> None:
    """The one thing that must be true before any tuning result is trustworthy: with evaluate.py's
    own current constants, _eval_tunable must reproduce evaluate() exactly, position for position.
    """
    piece_value = PIECE_VALUE.copy()
    pst = PST.copy()
    pst_king_end = PST_KING_END.copy()
    mismatches = 0
    for i in range(n):
        real = int(evaluate(bbs[i], metas[i]))
        cloned = int(_eval_tunable(bbs[i], metas[i], piece_value, pst, pst_king_end))
        if real != cloned:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH at position {i}: real={real} cloned={cloned}", flush=True)
    if mismatches:
        raise SystemExit(
            f"parity check FAILED: {mismatches}/{n} positions disagree -- tuning would be "
            f"fitting a different function than the real evaluate(). Aborting."
        )
    print(f"parity check passed: {n}/{n} positions agree with real evaluate()", flush=True)


def load_dataset(paths: list[str]) -> tuple[list[str], list[float]]:
    fens: list[str] = []
    results: list[float] = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        fens.extend(str(x) for x in data["fens"])
        results.extend(float(x) for x in data["results"])
    return fens, results


def build_arrays(
    fens: list[str], results: list[float], sample: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n_total = len(fens)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(sample, n_total), replace=False)
    n = len(idx)
    bbs = np.zeros((n, 12), dtype=np.uint64)
    metas = np.zeros((n, 6), dtype=np.int8)
    targets = np.zeros(n, dtype=np.float64)
    for out_i, in_i in enumerate(idx):
        bb, meta = bbm.from_fen(fens[in_i])
        bbs[out_i] = bb
        metas[out_i] = meta
        result = results[in_i]
        targets[out_i] = result if meta[0] == WHITE else (1.0 - result)
    return bbs, metas, targets, n


def build_param_list(
    piece_value: np.ndarray, pst: np.ndarray, pst_king_end: np.ndarray
) -> list[tuple[np.ndarray, tuple[int, ...]]]:
    params: list[tuple[np.ndarray, tuple[int, ...]]] = []
    for pt in range(5):  # PAWN..QUEEN -- KING (index 5) excluded, see module docstring
        params.append((piece_value, (pt,)))
    for pt in range(6):
        for sq in range(64):
            params.append((pst, (pt, sq)))
    for sq in range(64):
        params.append((pst_king_end, (sq,)))
    return params


def tune(
    bbs: np.ndarray,
    metas: np.ndarray,
    targets: np.ndarray,
    n: int,
    piece_value: np.ndarray,
    pst: np.ndarray,
    pst_king_end: np.ndarray,
    passes: int,
    steps: list[int],
) -> float:
    params = build_param_list(piece_value, pst, pst_king_end)
    best_error = batch_sse(bbs, metas, targets, n, piece_value, pst, pst_king_end)
    print(f"positions={n} parameters={len(params)} initial_sse={best_error:.4f}", flush=True)

    for step in steps:
        for pass_idx in range(passes):
            improved = 0
            for arr, idx in params:
                old = int(arr[idx])
                arr[idx] = old + step
                e_plus = batch_sse(bbs, metas, targets, n, piece_value, pst, pst_king_end)
                if e_plus < best_error:
                    best_error = e_plus
                    improved += 1
                    continue
                arr[idx] = old - step
                e_minus = batch_sse(bbs, metas, targets, n, piece_value, pst, pst_king_end)
                if e_minus < best_error:
                    best_error = e_minus
                    improved += 1
                    continue
                arr[idx] = old
            print(
                f"step={step} pass {pass_idx + 1}/{passes}: sse={best_error:.4f}, "
                f"improved {improved}/{len(params)} params",
                flush=True,
            )
            if improved == 0:
                print(f"  converged at step={step}", flush=True)
                break
    return best_error


def dump_python_literals(
    piece_value: np.ndarray, pst: np.ndarray, pst_king_end: np.ndarray
) -> None:
    print("\n# --- tuned constants: paste over the matching lines in evaluate.py ---\n")

    def pst_name(pt: int) -> str:
        return ["_PAWN_PST", "_KNIGHT_PST", "_BISHOP_PST", "_ROOK_PST", "_QUEEN_PST", "_KING_PST"][
            pt
        ]

    print(f"PIECE_VALUE = np.array({piece_value.tolist()}, dtype=np.int32)\n")
    for pt in range(6):
        row = pst[pt].tolist()
        print(f"{pst_name(pt)} = [")
        for r in range(8):
            print("    " + ", ".join(str(v) for v in row[r * 8:(r + 1) * 8]) + ",")
        print("]")
    king_end = pst_king_end.tolist()
    print("_KING_PST_ENDGAME = [")
    for r in range(8):
        print("    " + ", ".join(str(v) for v in king_end[r * 8:(r + 1) * 8]) + ",")
    print("]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--steps", type=str, default="8,4,2,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parity-check-n", type=int, default=500)
    args = parser.parse_args()
    steps = [int(s) for s in args.steps.split(",")]

    fens, results = load_dataset(args.datasets)
    print(f"loaded {len(fens)} positions from {len(args.datasets)} file(s)", flush=True)
    bbs, metas, targets, n = build_arrays(fens, results, args.sample, args.seed)

    verify_parity(bbs, metas, min(args.parity_check_n, n))

    piece_value = PIECE_VALUE.copy()
    pst = PST.copy()
    pst_king_end = PST_KING_END.copy()

    tune(bbs, metas, targets, n, piece_value, pst, pst_king_end, args.passes, steps)
    dump_python_literals(piece_value, pst, pst_king_end)


if __name__ == "__main__":
    main()
