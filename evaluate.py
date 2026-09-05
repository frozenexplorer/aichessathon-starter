"""Material + piece-square evaluation, from the perspective of the side to move (negamax sign
convention: positive is good for whoever moves next).

Tables are the common "simplified evaluation" piece-square values, white's orientation, rank 1
at index 0. Black's score uses the same tables mirrored vertically (square ^ 56 flips the rank,
keeps the file), so there is one table per piece, not two. The king additionally has a second,
endgame-only table (centralizing instead of hiding in a castled corner), blended in by game phase
-- a king that wants safety mid-game is instead an active piece needed for the win once material
has thinned out.

Beyond material + PST, six cheap positional terms are added, all in the same white-minus-black,
own-file/precomputed-mask style as the rest of this module so they stay jit-friendly without a
second pass over the board: pawn structure (doubled, isolated, passed -- the passed-pawn bonus
itself phase-blended like the king PST, since a passed pawn is worth far more with fewer pieces
left to stop it), bishop pair, rooks on open/semi-open files, a king pawn-shield bonus that --
like the king PST -- fades out via the same phase blend as material comes off the board, piece
mobility (knight/bishop/rook/queen squares attacked, excluding squares held by an own piece) --
affordable per node now that sliding attacks are magic-bitboard lookups (see attacks.py) rather
than ray-casts -- and, phased in the same way as the passed-pawn bonus, king proximity to each
passed pawn's promotion square (the "rule of the square" -- whichever king is closer tends to
decide whether the pawn queens or gets caught).

Two more terms cover tactical motifs a depth-limited search can miss until it searches far enough
to reach the actual capture, but which are visible one ply after the threatening move is made if
eval itself notices: `threats_score` (a pawn attacking a more valuable piece, an attacked piece
with no defender, and a fork bonus when one piece attacks two or more enemy pieces -- including
the enemy king -- at once, since the opponent can only save one) and `pin_and_xray_score`
(absolute pins to the king, plus skewers/x-rays -- a second enemy piece directly behind the first
along the same ray from one of our sliders). Both recompute the ray with a candidate blocker
removed rather than tracking rays incrementally, the same "discovered attacker falls out for free"
trick `search.see` already relies on for the same reason.

One more, `king_safety_score`: attacker-weighted pressure on each king's own ring (the up-to-eight
squares immediately around it) -- the gap between "material is fine" and "about to get mated" that
material/PST/mobility/threats scoring alone under-values until a search reads all the way to the
mating net itself, previously covered only by the pawn-shield bonus above (which values a king's
own cover, not the enemy's actual reach toward it). Phase-blended the opposite way from the
passed-pawn/king-distance terms: full strength with pieces still on the board to attack with,
fading to zero in the endgame, where an exposed king stops being a liability and becomes an asset
(the king PST's own phase blend already covers that side of it).

Two small additions on top of all the above, both zero/near-zero compile cost since they add no
new control flow to speak of: a flat tempo bonus (`TEMPO_BONUS`) for the side to move, added after
every other term and after the perspective flip so it always rewards whoever's turn it is; and
`opposite_bishops_scale`, which scales the whole eval down to `OCB_SCALE_PERCENT` percent in a pure
opposite-coloured-bishop endgame (exactly one bishop each, opposite-coloured squares, no other
minor or major piece for either side) -- these are famously drawish even a material edge up, since
the bishops can never contest the same squares, so an unscaled eval overstates real winning
chances. Deliberately scoped to bishops-and-pawns-only endgames: OCB alongside rooks or queens does
not carry the same drawish tendency, so those are left at full weight.
"""

import numpy as np
from numba import njit

from attacks import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)
from bitboard import BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE
from movegen import SQUARE_COLOR_MASK, attacked_by, king_square, occ_color, piece_type_at

PIECE_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)

_PAWN_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_ROOK_PST = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_QUEEN_PST = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_KING_PST = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
_KING_PST_ENDGAME = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]

PST = np.array(
    [_PAWN_PST, _KNIGHT_PST, _BISHOP_PST, _ROOK_PST, _QUEEN_PST, _KING_PST], dtype=np.int32
)
PST_KING_END = np.array(_KING_PST_ENDGAME, dtype=np.int32)

# Standard phase weighting: knight/bishop 1, rook 2, queen 4, pawn/king 0 -- 4*1 + 4*1 + 4*2 + 2*4
# = 24 at the start of the game, tapering to 0 as non-pawn material comes off.
PHASE_WEIGHT = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
PHASE_MAX = 24

ONE = np.uint64(1)
MOBILITY_WEIGHT = np.int32(2)
BISHOP_PAIR_BONUS = np.int32(30)
DOUBLED_PAWN_PENALTY = np.int32(10)
ISOLATED_PAWN_PENALTY = np.int32(15)
PASSED_PAWN_BONUS_MID = np.array([0, 5, 10, 20, 35, 60, 100, 0], dtype=np.int32)
PASSED_PAWN_BONUS_END = np.array([0, 10, 20, 40, 70, 120, 200, 0], dtype=np.int32)
ROOK_SEMI_OPEN_BONUS = np.int32(10)
ROOK_OPEN_BONUS = np.int32(20)
KING_SHIELD_BONUS = np.int32(8)
KING_DISTANCE_WEIGHT = np.int32(4)
PAWN_THREAT_BONUS = np.int32(35)
HANGING_PIECE_DIVISOR = np.int32(8)
FORK_BONUS = np.int32(30)
PIN_KING_DIVISOR = np.int32(6)
XRAY_FLAT_BONUS = np.int32(6)
XRAY_HEAVY_DIVISOR = np.int32(10)
KING_ZONE_ATTACK_WEIGHT = np.array([2, 20, 20, 30, 45, 0], dtype=np.int32)
TEMPO_BONUS = np.int32(10)
OCB_SCALE_PERCENT = np.int32(60)


def _build_file_masks() -> np.ndarray:
    masks = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        bits = np.uint64(0)
        for r in range(8):
            bits |= ONE << np.uint64(r * 8 + f)
        masks[f] = bits
    return masks


def _build_adjacent_file_masks(file_masks: np.ndarray) -> np.ndarray:
    masks = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        bits = np.uint64(0)
        if f > 0:
            bits |= file_masks[f - 1]
        if f < 7:
            bits |= file_masks[f + 1]
        masks[f] = bits
    return masks


def _build_passed_masks(white: bool) -> np.ndarray:
    """Per square, the enemy-pawn squares (own file + adjacent files, every rank strictly ahead
    in the pawn's direction of travel) that would contest or block a pawn there from queening.
    """
    masks = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file, rank = square % 8, square // 8
        files = [f for f in (file - 1, file, file + 1) if 0 <= f < 8]
        ranks = range(rank + 1, 8) if white else range(0, rank)
        bits = np.uint64(0)
        for f in files:
            for r in ranks:
                bits |= ONE << np.uint64(r * 8 + f)
        masks[square] = bits
    return masks


def _build_king_shield_masks(white: bool) -> np.ndarray:
    """Per king square, the two ranks directly ahead (in the king's own forward direction) across
    the king's file and its neighbours -- where a pawn shield would sit.
    """
    masks = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file, rank = square % 8, square // 8
        bits = np.uint64(0)
        for dr in ((1, 2) if white else (-1, -2)):
            r = rank + dr
            if 0 <= r < 8:
                for df in (-1, 0, 1):
                    f = file + df
                    if 0 <= f < 8:
                        bits |= ONE << np.uint64(r * 8 + f)
        masks[square] = bits
    return masks


FILE_MASKS = _build_file_masks()
ADJACENT_FILE_MASKS = _build_adjacent_file_masks(FILE_MASKS)
PASSED_MASK_WHITE = _build_passed_masks(True)
PASSED_MASK_BLACK = _build_passed_masks(False)
KING_SHIELD_WHITE = _build_king_shield_masks(True)
KING_SHIELD_BLACK = _build_king_shield_masks(False)


@njit(cache=False)
def _popcount64(bits: np.uint64) -> int:
    count = 0
    while bits:
        bits &= bits - ONE
        count += 1
    return count


@njit(cache=False)
def game_phase(bb: np.ndarray) -> int:
    phase = 0
    for color in range(2):
        phase += _popcount64(bb[color * 6 + KNIGHT]) * PHASE_WEIGHT[KNIGHT]
        phase += _popcount64(bb[color * 6 + BISHOP]) * PHASE_WEIGHT[BISHOP]
        phase += _popcount64(bb[color * 6 + ROOK]) * PHASE_WEIGHT[ROOK]
        phase += _popcount64(bb[color * 6 + QUEEN]) * PHASE_WEIGHT[QUEEN]
    return phase if phase < PHASE_MAX else PHASE_MAX


@njit(cache=False)
def material_and_pst(bb: np.ndarray, phase: int) -> int:
    score = np.int32(0)
    for pt in range(6):
        white_bb = bb[pt]
        black_bb = bb[6 + pt]
        for square in range(64):
            bit = ONE << np.uint64(square)
            if white_bb & bit:
                if pt == KING:
                    mg, eg = PST[pt, square], PST_KING_END[square]
                    score += PIECE_VALUE[pt] + (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
                else:
                    score += PIECE_VALUE[pt] + PST[pt, square]
            if black_bb & bit:
                mirror = square ^ 56
                if pt == KING:
                    mg, eg = PST[pt, mirror], PST_KING_END[mirror]
                    score -= PIECE_VALUE[pt] + (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
                else:
                    score -= PIECE_VALUE[pt] + PST[pt, mirror]
    return int(score)


@njit(cache=False)
def pawn_structure(bb: np.ndarray, phase: int) -> int:
    score = np.int32(0)
    white_pawns = bb[WHITE * 6 + PAWN]
    black_pawns = bb[BLACK * 6 + PAWN]

    for f in range(8):
        wc = _popcount64(white_pawns & FILE_MASKS[f])
        bc = _popcount64(black_pawns & FILE_MASKS[f])
        if wc > 1:
            score -= DOUBLED_PAWN_PENALTY * (wc - 1)
        if bc > 1:
            score += DOUBLED_PAWN_PENALTY * (bc - 1)

    for square in range(64):
        bit = ONE << np.uint64(square)
        f = square % 8
        rank = square // 8
        if white_pawns & bit:
            if white_pawns & ADJACENT_FILE_MASKS[f] == 0:
                score -= ISOLATED_PAWN_PENALTY
            if black_pawns & PASSED_MASK_WHITE[square] == 0:
                mid, end = PASSED_PAWN_BONUS_MID[rank], PASSED_PAWN_BONUS_END[rank]
                score += (mid * phase + end * (PHASE_MAX - phase)) // PHASE_MAX
        if black_pawns & bit:
            if black_pawns & ADJACENT_FILE_MASKS[f] == 0:
                score += ISOLATED_PAWN_PENALTY
            if white_pawns & PASSED_MASK_BLACK[square] == 0:
                mid, end = PASSED_PAWN_BONUS_MID[7 - rank], PASSED_PAWN_BONUS_END[7 - rank]
                score -= (mid * phase + end * (PHASE_MAX - phase)) // PHASE_MAX
    return int(score)


@njit(cache=False)
def _chebyshev(sq1: int, sq2: int) -> int:
    df = abs(sq1 % 8 - sq2 % 8)
    dr = abs(sq1 // 8 - sq2 // 8)
    return df if df > dr else dr


@njit(cache=False)
def passed_pawn_king_distance(bb: np.ndarray, phase: int) -> int:
    """Rule-of-the-square proxy: for each passed pawn, reward the friendly king being closer to
    its promotion square than the enemy king (and penalise the reverse), phase-blended like the
    passed-pawn bonus above -- king proximity to a passed pawn's race is an endgame concern, not a
    middlegame one where the king is still tucked away for safety.
    """
    endgame_weight = PHASE_MAX - phase
    if endgame_weight == 0:
        return 0
    score = np.int32(0)
    white_pawns = bb[WHITE * 6 + PAWN]
    black_pawns = bb[BLACK * 6 + PAWN]
    wk = king_square(bb, WHITE)
    bk = king_square(bb, BLACK)

    for square in range(64):
        bit = ONE << np.uint64(square)
        f = square % 8
        if white_pawns & bit and black_pawns & PASSED_MASK_WHITE[square] == 0:
            promo = 56 + f
            diff = _chebyshev(bk, promo) - _chebyshev(wk, promo)
            score += KING_DISTANCE_WEIGHT * diff * endgame_weight // PHASE_MAX
        if black_pawns & bit and white_pawns & PASSED_MASK_BLACK[square] == 0:
            promo = f
            diff = _chebyshev(wk, promo) - _chebyshev(bk, promo)
            score -= KING_DISTANCE_WEIGHT * diff * endgame_weight // PHASE_MAX
    return int(score)


@njit(cache=False)
def piece_features(bb: np.ndarray, phase: int) -> int:
    score = np.int32(0)
    white_pawns = bb[WHITE * 6 + PAWN]
    black_pawns = bb[BLACK * 6 + PAWN]

    if _popcount64(bb[WHITE * 6 + BISHOP]) >= 2:
        score += BISHOP_PAIR_BONUS
    if _popcount64(bb[BLACK * 6 + BISHOP]) >= 2:
        score -= BISHOP_PAIR_BONUS

    for square in range(64):
        bit = ONE << np.uint64(square)
        f = square % 8
        if bb[WHITE * 6 + ROOK] & bit and white_pawns & FILE_MASKS[f] == 0:
            open_file = black_pawns & FILE_MASKS[f] == 0
            score += ROOK_OPEN_BONUS if open_file else ROOK_SEMI_OPEN_BONUS
        if bb[BLACK * 6 + ROOK] & bit and black_pawns & FILE_MASKS[f] == 0:
            open_file = white_pawns & FILE_MASKS[f] == 0
            score -= ROOK_OPEN_BONUS if open_file else ROOK_SEMI_OPEN_BONUS

    if phase > 0:
        wk = king_square(bb, WHITE)
        shield = _popcount64(white_pawns & KING_SHIELD_WHITE[wk])
        score += KING_SHIELD_BONUS * shield * phase // PHASE_MAX
        bk = king_square(bb, BLACK)
        shield_b = _popcount64(black_pawns & KING_SHIELD_BLACK[bk])
        score -= KING_SHIELD_BONUS * shield_b * phase // PHASE_MAX

    return int(score)


@njit(cache=False)
def mobility_score(bb: np.ndarray) -> int:
    """Differential piece mobility (white minus black): for each knight/bishop/rook/queen, the
    squares it attacks that aren't occupied by a piece of its own colour, one flat weight per
    square regardless of piece type -- a real per-piece activity signal (independent of whose
    turn it is), unlike counting the side to move's total legal moves. Pawns and kings are
    excluded: pawn mobility barely varies and rewarding king mobility encourages an unsafe king
    walk, especially mid-game.
    """
    score = np.int32(0)
    white_occ = occ_color(bb, WHITE)
    black_occ = occ_color(bb, BLACK)
    all_occ = white_occ | black_occ

    for square in range(64):
        bit = ONE << np.uint64(square)
        if bb[WHITE * 6 + KNIGHT] & bit:
            score += _popcount64(KNIGHT_ATTACKS[square] & ~white_occ)
        if bb[BLACK * 6 + KNIGHT] & bit:
            score -= _popcount64(KNIGHT_ATTACKS[square] & ~black_occ)
        if bb[WHITE * 6 + BISHOP] & bit:
            score += _popcount64(bishop_attacks(square, all_occ) & ~white_occ)
        if bb[BLACK * 6 + BISHOP] & bit:
            score -= _popcount64(bishop_attacks(square, all_occ) & ~black_occ)
        if bb[WHITE * 6 + ROOK] & bit:
            score += _popcount64(rook_attacks(square, all_occ) & ~white_occ)
        if bb[BLACK * 6 + ROOK] & bit:
            score -= _popcount64(rook_attacks(square, all_occ) & ~black_occ)
        if bb[WHITE * 6 + QUEEN] & bit:
            score += _popcount64(queen_attacks(square, all_occ) & ~white_occ)
        if bb[BLACK * 6 + QUEEN] & bit:
            score -= _popcount64(queen_attacks(square, all_occ) & ~black_occ)

    return int(score) * int(MOBILITY_WEIGHT)


@njit(cache=False)
def _bit_scan(bits: np.uint64) -> int:
    for square in range(64):
        if bits & (ONE << np.uint64(square)):
            return square
    return -1


@njit(cache=False)
def threats_score(bb: np.ndarray) -> int:
    """A pawn attacking a more valuable piece, any piece attacking an undefended enemy piece, and
    a fork bonus when one piece attacks two or more enemy pieces (the enemy king included, since
    that is still only one of the two-plus pieces the opponent can save) at once. Attacker-centric
    (loop over our pieces, look at what each attacks) rather than victim-centric, since a fork is
    naturally a property of the attacking piece, not any one of its targets.
    """
    score = np.int32(0)
    white_occ = occ_color(bb, WHITE)
    black_occ = occ_color(bb, BLACK)
    all_occ = white_occ | black_occ

    for color in range(2):
        sign = np.int32(1) if color == WHITE else np.int32(-1)
        enemy = 1 - color
        enemy_occ = black_occ if color == WHITE else white_occ
        enemy_king_bb = bb[enemy * 6 + KING]

        remaining = bb[color * 6 + PAWN]
        while remaining:
            sq = _bit_scan(remaining)
            remaining &= remaining - ONE
            hit = PAWN_ATTACKS[color, sq] & enemy_occ & ~bb[enemy * 6 + PAWN]
            if hit:
                score += sign * PAWN_THREAT_BONUS * np.int32(_popcount64(hit))

        for pt in (KNIGHT, BISHOP, ROOK, QUEEN, KING):
            remaining = bb[color * 6 + pt]
            while remaining:
                sq = _bit_scan(remaining)
                remaining &= remaining - ONE
                if pt == KNIGHT:
                    atk = KNIGHT_ATTACKS[sq]
                elif pt == BISHOP:
                    atk = bishop_attacks(sq, all_occ)
                elif pt == ROOK:
                    atk = rook_attacks(sq, all_occ)
                elif pt == QUEEN:
                    atk = queen_attacks(sq, all_occ)
                else:
                    atk = KING_ATTACKS[sq]

                all_hits = atk & enemy_occ
                if all_hits == 0:
                    continue
                hit_count = _popcount64(all_hits)
                if hit_count >= 2:
                    score += sign * FORK_BONUS * np.int32(hit_count - 1)

                sub = all_hits & ~enemy_king_bb
                while sub:
                    tsq = _bit_scan(sub)
                    sub &= sub - ONE
                    if not attacked_by(bb, all_occ, enemy, tsq):
                        target_pt = piece_type_at(bb, enemy, tsq)
                        score += sign * (PIECE_VALUE[target_pt] // HANGING_PIECE_DIVISOR)

    return int(score)


@njit(cache=False)
def _ray_pin_score(
    bb: np.ndarray,
    sq: int,
    all_occ: np.uint64,
    own_occ: np.uint64,
    enemy_occ: np.uint64,
    enemy: int,
    enemy_king_bb: np.uint64,
    use_bishop: bool,
) -> int:
    full = bishop_attacks(sq, all_occ) if use_bishop else rook_attacks(sq, all_occ)
    candidates = full & enemy_occ & ~enemy_king_bb
    total = 0
    remaining = candidates
    while remaining:
        c = _bit_scan(remaining)
        remaining &= remaining - ONE
        occ_without = all_occ & ~(ONE << np.uint64(c))
        extended = (
            bishop_attacks(sq, occ_without) if use_bishop else rook_attacks(sq, occ_without)
        )
        # Isolate to squares newly reachable now that c is gone -- extended still includes every
        # other ray's original blocker unchanged (e.g. an own piece on a different rank/diagonal),
        # and intersecting the whole thing against occ_without would wrongly pick those up too.
        revealed = (extended & ~full) & occ_without
        if revealed == 0 or revealed & own_occ:
            continue
        candidate_pt = piece_type_at(bb, enemy, c)
        if revealed & enemy_king_bb:
            total += int(PIECE_VALUE[candidate_pt]) // int(PIN_KING_DIVISOR)
        else:
            revealed_sq = _bit_scan(revealed)
            revealed_pt = piece_type_at(bb, enemy, revealed_sq)
            bonus = int(XRAY_FLAT_BONUS)
            if PIECE_VALUE[revealed_pt] > PIECE_VALUE[candidate_pt]:
                diff = int(PIECE_VALUE[revealed_pt]) - int(PIECE_VALUE[candidate_pt])
                bonus += diff // int(XRAY_HEAVY_DIVISOR)
            total += bonus
    return total


@njit(cache=False)
def pin_and_xray_score(bb: np.ndarray) -> int:
    """Absolute pins (a piece that cannot move without exposing its own king to one of our
    sliders) and skewers/x-rays (a second enemy piece directly behind the first along the same
    ray) -- both invisible to a search that has not yet looked past the first blocker on that ray,
    but real pressure a static eval can flag as soon as the pinning/skewering piece is placed.
    """
    score = np.int32(0)
    white_occ = occ_color(bb, WHITE)
    black_occ = occ_color(bb, BLACK)
    all_occ = white_occ | black_occ

    for color in range(2):
        sign = np.int32(1) if color == WHITE else np.int32(-1)
        enemy = 1 - color
        own_occ = white_occ if color == WHITE else black_occ
        enemy_occ = black_occ if color == WHITE else white_occ
        enemy_king_bb = bb[enemy * 6 + KING]

        remaining = bb[color * 6 + BISHOP]
        while remaining:
            sq = _bit_scan(remaining)
            remaining &= remaining - ONE
            score += sign * _ray_pin_score(
                bb, sq, all_occ, own_occ, enemy_occ, enemy, enemy_king_bb, True
            )

        remaining = bb[color * 6 + ROOK]
        while remaining:
            sq = _bit_scan(remaining)
            remaining &= remaining - ONE
            score += sign * _ray_pin_score(
                bb, sq, all_occ, own_occ, enemy_occ, enemy, enemy_king_bb, False
            )

        remaining = bb[color * 6 + QUEEN]
        while remaining:
            sq = _bit_scan(remaining)
            remaining &= remaining - ONE
            score += sign * _ray_pin_score(
                bb, sq, all_occ, own_occ, enemy_occ, enemy, enemy_king_bb, True
            )
            score += sign * _ray_pin_score(
                bb, sq, all_occ, own_occ, enemy_occ, enemy, enemy_king_bb, False
            )

    return int(score)


@njit(cache=False)
def king_safety_score(bb: np.ndarray, phase: int) -> int:
    """Attacker-weighted pressure on each king's own ring (KING_ATTACKS[king_sq], its up-to-eight
    adjacent squares): for each enemy piece, how many of those squares it currently attacks,
    weighted per piece type (a queen's reach into the ring is far more dangerous than a knight's).
    Phase-blended like the king PST -- full strength with material still on the board to attack
    with, zero once phase hits 0. King itself excluded as an attacker (kings do not approach the
    enemy king in the phases this term is active for) and as a target weight (KING_ZONE_ATTACK_
    WEIGHT[KING] == 0, since only the opponent's non-king pieces threatening the ring matter here).
    """
    if phase == 0:
        return 0
    score = np.int32(0)
    white_occ = occ_color(bb, WHITE)
    black_occ = occ_color(bb, BLACK)
    all_occ = white_occ | black_occ

    for color in range(2):
        sign = np.int32(1) if color == WHITE else np.int32(-1)
        enemy = 1 - color
        zone = KING_ATTACKS[king_square(bb, enemy)]
        danger = np.int32(0)

        remaining = bb[color * 6 + PAWN]
        while remaining:
            sq = _bit_scan(remaining)
            remaining &= remaining - ONE
            hits = _popcount64(PAWN_ATTACKS[color, sq] & zone)
            danger += KING_ZONE_ATTACK_WEIGHT[PAWN] * np.int32(hits)

        for pt in (KNIGHT, BISHOP, ROOK, QUEEN):
            remaining = bb[color * 6 + pt]
            while remaining:
                sq = _bit_scan(remaining)
                remaining &= remaining - ONE
                if pt == KNIGHT:
                    atk = KNIGHT_ATTACKS[sq]
                elif pt == BISHOP:
                    atk = bishop_attacks(sq, all_occ)
                elif pt == ROOK:
                    atk = rook_attacks(sq, all_occ)
                else:
                    atk = queen_attacks(sq, all_occ)
                hits = _popcount64(atk & zone)
                danger += KING_ZONE_ATTACK_WEIGHT[pt] * np.int32(hits)

        score += sign * (danger * np.int32(phase) // np.int32(PHASE_MAX))

    return int(score)


@njit(cache=False)
def opposite_bishops_scale(bb: np.ndarray) -> int:
    """A percentage (0-100) to scale the rest of the eval by. A pure opposite-coloured-bishop
    endgame (exactly one bishop each, on opposite-coloured squares, with no other minor or major
    piece for either side -- rooks/queens alongside OCB do not carry the same drawish tendency) is
    famously drawish even a pawn or two up: the bishops can never contest the same squares, so the
    stronger side often cannot force real progress. 100 (no scaling) otherwise.
    """
    if (
        bb[WHITE * 6 + KNIGHT] or bb[BLACK * 6 + KNIGHT]
        or bb[WHITE * 6 + ROOK] or bb[BLACK * 6 + ROOK]
        or bb[WHITE * 6 + QUEEN] or bb[BLACK * 6 + QUEEN]
    ):
        return 100
    white_bishops = bb[WHITE * 6 + BISHOP]
    black_bishops = bb[BLACK * 6 + BISHOP]
    if _popcount64(white_bishops) != 1 or _popcount64(black_bishops) != 1:
        return 100
    white_light = (white_bishops & SQUARE_COLOR_MASK) != 0
    black_light = (black_bishops & SQUARE_COLOR_MASK) != 0
    if white_light == black_light:
        return 100
    return int(OCB_SCALE_PERCENT)


@njit(cache=False)
def evaluate(bb: np.ndarray, meta: np.ndarray) -> int:
    phase = game_phase(bb)
    score = (
        material_and_pst(bb, phase)
        + pawn_structure(bb, phase)
        + piece_features(bb, phase)
        + mobility_score(bb)
        + passed_pawn_king_distance(bb, phase)
        + threats_score(bb)
        + pin_and_xray_score(bb)
        + king_safety_score(bb, phase)
    )
    score = score * opposite_bishops_scale(bb) // 100
    stm_score = int(score) if meta[0] == WHITE else int(-score)
    return stm_score + int(TEMPO_BONUS)


def warm_up() -> None:
    from bitboard import from_fen

    bb, meta = from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    evaluate(bb, meta)
