"""Attack tables and sliding-piece attack generation.

Knight, king and pawn attacks are fixed per square, so they are precomputed once at import.
Bishop, rook and queen attacks depend on occupancy, so they use magic bitboards: a per-square
lookup table sized to that square's relevant occupancy (the squares that could actually block a
ray from it, excluding the board edge in each direction -- what happens past the edge never
depends on what's on the edge square itself), indexed by ((occupied & mask) * magic) >> shift.
The magic numbers themselves are precomputed offline (find_magics-style random search, verified
against a plain ray-cast reference for every occupancy subset of the relevant mask) and hardcoded
below -- nothing at runtime searches for a magic.

ROOK_ATTACK_TABLE / BISHOP_ATTACK_TABLE ship precomputed in weights/attacks.npz (~2.3 MB) rather
than being rebuilt at import: building them from scratch means one ray-cast per occupancy subset
per square (107,648 calls total across both tables), and doing that through njit-compiled
_rook_ray_attacks / _bishop_ray_attacks cost two extra compiles for no runtime benefit (the
build loop itself only ever runs once, at import, never in the search hot path -- see
docs/plan.md Phase 1.5). _rook_ray_attacks / _bishop_ray_attacks are therefore plain, un-jitted
Python here: they exist only for _build_magic_table (used to regenerate weights/attacks.npz when
the magic numbers below ever change) and as the differential-test oracle in
tests/test_magic_attacks.py, neither of which is on the init-time or search-time path.
_CHECKSUM guards against a corrupt or stale weights/attacks.npz failing loudly at import instead
of silently producing wrong attacks.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numba import njit

from bitboard import BLACK, WHITE

ONE = np.uint64(1)


def _in_bounds(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def _build_knight_attacks() -> np.ndarray:
    table = np.zeros(64, dtype=np.uint64)
    deltas = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
    for square in range(64):
        file, rank = square % 8, square // 8
        bits = np.uint64(0)
        for df, dr in deltas:
            if _in_bounds(file + df, rank + dr):
                target = (rank + dr) * 8 + (file + df)
                bits |= ONE << np.uint64(target)
        table[square] = bits
    return table


def _build_king_attacks() -> np.ndarray:
    table = np.zeros(64, dtype=np.uint64)
    deltas = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for square in range(64):
        file, rank = square % 8, square // 8
        bits = np.uint64(0)
        for df, dr in deltas:
            if _in_bounds(file + df, rank + dr):
                target = (rank + dr) * 8 + (file + df)
                bits |= ONE << np.uint64(target)
        table[square] = bits
    return table


def _build_pawn_attacks() -> np.ndarray:
    table = np.zeros((2, 64), dtype=np.uint64)
    for square in range(64):
        file, rank = square % 8, square // 8
        for color, dr in ((WHITE, 1), (BLACK, -1)):
            bits = np.uint64(0)
            for df in (-1, 1):
                if _in_bounds(file + df, rank + dr):
                    target = (rank + dr) * 8 + (file + df)
                    bits |= ONE << np.uint64(target)
            table[color, square] = bits
    return table


KNIGHT_ATTACKS = _build_knight_attacks()
KING_ATTACKS = _build_king_attacks()
PAWN_ATTACKS = _build_pawn_attacks()


def _rook_ray_attacks(square: int, occupied: np.uint64) -> np.uint64:
    file, rank = square % 8, square // 8
    bits = np.uint64(0)
    for df, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        f, r = file + df, rank + dr
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            bits |= ONE << np.uint64(target)
            if occupied & (ONE << np.uint64(target)):
                break
            f += df
            r += dr
    return bits


def _bishop_ray_attacks(square: int, occupied: np.uint64) -> np.uint64:
    file, rank = square % 8, square // 8
    bits = np.uint64(0)
    for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        f, r = file + df, rank + dr
        while 0 <= f < 8 and 0 <= r < 8:
            target = r * 8 + f
            bits |= ONE << np.uint64(target)
            if occupied & (ONE << np.uint64(target)):
                break
            f += df
            r += dr
    return bits


# Precomputed offline (random search over candidate magics, each verified to map every subset of
# its square's relevant-occupancy mask to a collision-free table index against _rook_ray_attacks /
# _bishop_ray_attacks) -- see this module's docstring. Regenerating these is a dev-time exercise,
# never something the shipped engine does.
_ROOK_MASKS_RAW = [
    282578800148862, 565157600297596, 1130315200595066, 2260630401190006,
    4521260802379886, 9042521604759646, 18085043209519166, 36170086419038334,
    282578800180736, 565157600328704, 1130315200625152, 2260630401218048,
    4521260802403840, 9042521604775424, 18085043209518592, 36170086419037696,
    282578808340736, 565157608292864, 1130315208328192, 2260630408398848,
    4521260808540160, 9042521608822784, 18085043209388032, 36170086418907136,
    282580897300736, 565159647117824, 1130317180306432, 2260632246683648,
    4521262379438080, 9042522644946944, 18085043175964672, 36170086385483776,
    283115671060736, 565681586307584, 1130822006735872, 2261102847592448,
    4521664529305600, 9042787892731904, 18085034619584512, 36170077829103616,
    420017753620736, 699298018886144, 1260057572672512, 2381576680245248,
    4624614895390720, 9110691325681664, 18082844186263552, 36167887395782656,
    35466950888980736, 34905104758997504, 34344362452452352, 33222877839362048,
    30979908613181440, 26493970160820224, 17522093256097792, 35607136465616896,
    9079539427579068672, 8935706818303361536, 8792156787827803136, 8505056726876686336,
    7930856604974452736, 6782456361169985536, 4485655873561051136, 9115426935197958144,
]
_ROOK_MAGICS_RAW = [
    612489824210784898, 8376712900168916992, 144133055677301248, 2341889398519890048,
    2413936031708544000, 13907116748968168448, 11709361230203322496, 1333065770535485824,
    2351019745528266752, 70506451582976, 4644474555728000, 4035788259031322658,
    144537417729312768, 140746086687744, 1126467111486025, 288371185597546624,
    433050900940980256, 9007474134765576, 144133880310595648, 635148834971879428,
    864973703015827456, 9228018023305054720, 4399674429968, 178120893155588,
    9219447950705152, 4659044185408536704, 873698879614222376, 216199189721251968,
    290275366031616, 15433906326121481232, 4822509540081928, 4910190250556539136,
    1315061263830614048, 9223407227677720577, 325526497769033728, 5767000094737960960,
    4644423040239616, 144255942752469504, 17626612892227, 1188970646919446596,
    35735201939456, 9291697931485216, 4616401284009164832, 18085909716140064,
    2882906294023389312, 4755803405593641088, 5145720059592706, 229684132429561881,
    4719772964076126336, 140874931503232, 166668370853527680, 148917855539233024,
    4918493811914573312, 630644693927198848, 10397703257769681920, 9223512787228573824,
    4760517084897608289, 35412081065985, 1153044688565961539, 193514183074049,
    1873778937176784913, 18295890666194977, 6769137451268, 2307532140471976962,
]
_ROOK_BITS = [
    12, 11, 11, 11, 11, 11, 11, 12, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 12, 11, 11, 11, 11, 11, 11, 12,
]
_BISHOP_MASKS_RAW = [
    18049651735527936, 70506452091904, 275415828992, 1075975168,
    38021120, 8657588224, 2216338399232, 567382630219776,
    9024825867763712, 18049651735527424, 70506452221952, 275449643008,
    9733406720, 2216342585344, 567382630203392, 1134765260406784,
    4512412933816832, 9024825867633664, 18049651768822272, 70515108615168,
    2491752130560, 567383701868544, 1134765256220672, 2269530512441344,
    2256206450263040, 4512412900526080, 9024834391117824, 18051867805491712,
    637888545440768, 1135039602493440, 2269529440784384, 4539058881568768,
    1128098963916800, 2256197927833600, 4514594912477184, 9592139778506752,
    19184279556981248, 2339762086609920, 4538784537380864, 9077569074761728,
    562958610993152, 1125917221986304, 2814792987328512, 5629586008178688,
    11259172008099840, 22518341868716544, 9007336962655232, 18014673925310464,
    2216338399232, 4432676798464, 11064376819712, 22137335185408,
    44272556441600, 87995357200384, 35253226045952, 70506452091904,
    567382630219776, 1134765260406784, 2832480465846272, 5667157807464448,
    11333774449049600, 22526811443298304, 9024825867763712, 18049651735527936,
]
_BISHOP_MAGICS_RAW = [
    1443405913989054481, 1408551771767328, 9802157158115180608, 1155209039353774086,
    1190081718443311104, 4630272524901613570, 10740034748548224, 107786567485472,
    2892612791331136512, 1130302382674304, 4521221911830656, 1170945815917691139,
    4760447751517389120, 288380752085649408, 8070750840794791936, 18614183243456512,
    10238721534329856, 290482313975038016, 36354321184587840, 145241096607367744,
    1407409780686912, 1162192389152832, 9223937203055119376, 9223512782967308800,
    185791162787825664, 1738710518409462272, 2287604003308096, 5782626336841465888,
    72202729581191185, 19422874013008516, 648711865106948, 19141535987728648,
    1153379144111753216, 669740054611972, 2305878812061599777, 1731926528964952193,
    198167458870333696, 144423059921625608, 19162303749441536, 11602402939687176340,
    1174600816970965056, 282093788073986, 289532336163268609, 105836593022976,
    8800476079104, 1369659989932646916, 45044797739894528, 20355310313017888,
    8106770219417995008, 74769341416513, 4620840578060665877, 288231773123515392,
    17660939206656, 633331919159369, 1231770406168363018, 149749120117076993,
    282714143983616, 3458764668456667136, 216788517232331012, 72339618779104322,
    13835621018192856068, 576496074192126464, 36064153460441986, 9385574217044525104,
]
_BISHOP_BITS = [
    6, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6,
]
MAX_ROOK_BITS = 12
MAX_BISHOP_BITS = 9

ROOK_MASKS = np.array(_ROOK_MASKS_RAW, dtype=np.uint64)
ROOK_MAGICS = np.array(_ROOK_MAGICS_RAW, dtype=np.uint64)
ROOK_SHIFTS = np.array([64 - b for b in _ROOK_BITS], dtype=np.int64)
BISHOP_MASKS = np.array(_BISHOP_MASKS_RAW, dtype=np.uint64)
BISHOP_MAGICS = np.array(_BISHOP_MAGICS_RAW, dtype=np.uint64)
BISHOP_SHIFTS = np.array([64 - b for b in _BISHOP_BITS], dtype=np.int64)


def _subsets_of(mask: int) -> list[int]:
    """Carry-Rippler enumeration of every subset of the bits set in `mask`."""
    subsets = [0]
    subset = 0
    while subset != mask:
        subset = (subset - mask) & mask
        subsets.append(subset)
    return subsets


def _build_magic_table(
    masks_raw: list[int],
    magics_raw: list[int],
    bits: list[int],
    ray_fn: Callable[[int, np.uint64], np.uint64],
    table_size: int,
) -> np.ndarray:
    table = np.zeros((64, table_size), dtype=np.uint64)
    for square in range(64):
        mask = masks_raw[square]
        magic = magics_raw[square]
        shift = 64 - bits[square]
        for subset in _subsets_of(mask):
            attack = int(ray_fn(square, np.uint64(subset)))
            idx = ((subset * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
            table[square, idx] = attack
    return table


def _checksum(rook: np.ndarray, bishop: np.ndarray) -> int:
    """Order-independent sanity check over both tables -- a load-time corruption/staleness guard,
    not a cryptographic one. XOR-reduce rather than sum so it doesn't rely on uint64 wraparound
    behaving any particular way.
    """
    return int(np.bitwise_xor.reduce(rook.ravel()) ^ np.bitwise_xor.reduce(bishop.ravel()))


# Hardcoded once, when weights/attacks.npz was generated from the magic numbers above (see
# _build_magic_table) -- regenerate both together if _ROOK_MAGICS_RAW/_BISHOP_MAGICS_RAW (or the
# masks/bits) ever change; a stale pair fails the assert below instead of silently shipping wrong
# attack data.
_EXPECTED_CHECKSUM = 4792111478498951490

_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "attacks.npz"

if _WEIGHTS_PATH.exists():
    with np.load(_WEIGHTS_PATH) as _npz:
        ROOK_ATTACK_TABLE = _npz["rook"]
        BISHOP_ATTACK_TABLE = _npz["bishop"]
    assert _checksum(ROOK_ATTACK_TABLE, BISHOP_ATTACK_TABLE) == _EXPECTED_CHECKSUM, (
        f"weights/attacks.npz is corrupt or stale (checksum mismatch) -- regenerate it from "
        f"the magic numbers in {__file__} via _build_magic_table"
    )
else:
    # Dev-time / regeneration fallback only -- the shipped zip always includes weights/attacks.npz
    # (see docs/plan.md Phase 1.5), so this path is not expected to run on the real platform.
    ROOK_ATTACK_TABLE = _build_magic_table(
        _ROOK_MASKS_RAW, _ROOK_MAGICS_RAW, _ROOK_BITS, _rook_ray_attacks, 1 << MAX_ROOK_BITS
    )
    BISHOP_ATTACK_TABLE = _build_magic_table(
        _BISHOP_MASKS_RAW, _BISHOP_MAGICS_RAW, _BISHOP_BITS, _bishop_ray_attacks,
        1 << MAX_BISHOP_BITS,
    )


@njit(cache=False)
def rook_attacks(square: int, occupied: np.uint64) -> np.uint64:
    idx = int(
        ((occupied & ROOK_MASKS[square]) * ROOK_MAGICS[square]) >> np.uint64(ROOK_SHIFTS[square])
    )
    return np.uint64(ROOK_ATTACK_TABLE[square, idx])


@njit(cache=False)
def bishop_attacks(square: int, occupied: np.uint64) -> np.uint64:
    idx = int(
        ((occupied & BISHOP_MASKS[square]) * BISHOP_MAGICS[square])
        >> np.uint64(BISHOP_SHIFTS[square])
    )
    return np.uint64(BISHOP_ATTACK_TABLE[square, idx])


@njit(cache=False)
def queen_attacks(square: int, occupied: np.uint64) -> np.uint64:
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)
