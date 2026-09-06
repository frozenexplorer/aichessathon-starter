"""Negamax with alpha-beta, iterative deepening, a transposition table, killer/history move
ordering, principal variation search (PVS), and quiescence search.

Time safety: numba cannot call time.perf_counter() directly in nopython mode, so the search
drops back to Python via `objmode` every 128 nodes to check a wall-clock deadline. On abort it
propagates a sentinel up immediately rather than unwinding cleanly -- cheap because the check is
frequent enough that the overrun past the deadline is small and bounded by remaining call depth.
A root search that aborts before finishing is discarded entirely by the caller (timeman-driven
iterative deepening in agent.py), never returned as a partial, possibly-misordered result.

Below timeman.PANIC_MS remaining, agent.py skips the tree search altogether and plays
quick_best_move instead: even a depth-1 search can cascade through thousands of quiescence
nodes on a tactical position before the first check point, and at a few tens of milliseconds of
clock left that alone can be enough to flag. quick_best_move never recurses, so it cannot.

Transposition table: a fixed-size, two-tier hash table (parallel numpy arrays, not a dict -- numba
nopython mode has no fast Python dict), keyed on the low bits of the Zobrist hash with a full
64-bit key comparison to detect index collisions. Each bucket (TT_BUCKETS of them) has two raw
slots: a depth-preferred one, replaced only on a same-key refresh, an empty slot, or a search that
went at least as deep as what's already there, and an always-replace one that a fresh position
falls back to otherwise -- so a deep, expensive result isn't evicted by shallow, plentiful ones,
while a node from the current search is still never simply dropped for lack of a slot (see
_tt_resolve and the store logic at the end of negamax). Stores a depth, a score (bound-adjusted
for mate distance so a cached "mate in N from here" is still correct when reused at a different
ply), a bound type (exact/lower/upper), and the best move found, so a re-visited node can either
return immediately (sufficient stored depth, exact bound, or a bound that already causes a
cutoff) or at minimum reuse the stored move as the first one tried. Persists across the whole
game (agent.py owns the arrays, created once), since positions can transpose across our own
move choices even when they don't recur outright. It is deliberately not consulted for the
repetition-forced draw score (see below) -- that check runs first and returns before the table
is ever read or written for that node, so a cached score can never paper over an actual repeat.

Killer moves (two per ply), a from/to history table, and a counter-move table give cheap move
ordering for quiet moves that caused a beta cutoff elsewhere in the tree, on top of the
hash-move-first, then MVV-LVA capture ordering already in place. The counter-move table is indexed
by the move that led to a node (parent_from/parent_to, threaded through negamax) rather than by
ply or by the moving side's own from/to squares: it answers "what quiet reply most recently beat
this specific opponent move," a more targeted signal than the from/to history table's coarser
one. All three are rebuilt fresh per real move decision (agent.py allocates them per get_move
call), since they encode this search's own cutoff history, not anything about the position itself.

History malus: at a beta cutoff, every other quiet move already tried at that same node (earlier
in the ordering, so it had the same chance to cut off) is penalised in history_table by the same
magnitude the actual cutoff move is rewarded by -- not merely withheld a bonus, but pushed below
an untried (history == 0) move the next time ordering consults it. Doubles the resolution of the
history heuristic (a move's score now reflects both how often it cuts off and how often it was
tried and didn't) at the cost of one extra pass over the moves already tried before the cutoff --
bounded by the same move list this node already generated, not a new one.

Principal variation search: the first move at each node (hash move or best-scoring by ordering)
is searched with the full alpha-beta window; every other move first gets a cheap null-window
probe (-alpha-1, -alpha) and is only re-searched with the full window if that probe suggests it
might actually beat alpha. Cuts the cost of nodes that ordering already got right, which the
hash-move-first and killer/history ordering above make the common case.

Aspiration windows: search_root's first pass at a given depth (depth > 2, and only once the
previous depth's score is a real, non-mate value) searches a narrow window centered on the
previous iterative-deepening depth's score rather than -INF..INF, on the premise that the score
rarely swings far in one more ply. A fail-high or fail-low re-searches the same depth with the
full window -- one extra pass in the rare case, never a correctness risk, since the final result
is only ever accepted once it comes from a pass whose window did not clip it.

Null-move pruning: at a non-leaf node deep enough (NULL_MOVE_MIN_DEPTH), negamax additionally
tries giving the side to move a free pass (make_null_move: same board, turn flipped, en passant
forfeited) and searching the rest at depth - 1 - NULL_MOVE_REDUCTION with a null window just above
beta. If even a free move for the opponent is not enough to drop the score below beta, the real
position is assumed to cut off too and the whole subtree is pruned. Two zugzwang guards, since a
free move can look safe in a position where every real move only makes things worse: skipped
while in check (the null move would leave an illegal, check-blind position), and skipped when the
side to move has no non-pawn material left (has_non_pawn_material) -- exactly the king+pawn
endgames zugzwang is common in, which is also the class of position DTZ was added to get right
(see tablebase.py). `allow_null` additionally forbids two null moves in a row (a null move whose
own child is itself another null-move probe proves nothing and just burns depth), threaded through
every real-move recursive call as True and through the null-move probe's own call as False.

Static exchange evaluation (see()): a real exchange calculation (least-valuable-attacker-first,
swap-list algorithm) replaces MVV-LVA's rough capture-ordering heuristic in _move_score, and
quiescence uses it to skip searching a capture outright once its SEE is negative -- a capture
that loses material cannot help quiescing the position, and exploring it anyway was the direct
cause of the 500K-2M+ node counts observed at depth 7-9 in sharp middlegames (see docs/STATUS.md).
Ignores pins, same simplification essentially every engine's SEE makes -- see() is a heuristic for
ordering and pruning, never a legality check, so a wrong answer there misorders or mis-prunes but
cannot make the search return an illegal move.

Delta pruning: quiescence additionally skips a capture once stand-pat plus its SEE, plus a safety
margin (DELTA_MARGIN), still can't reach alpha -- catching the case SEE-pruning alone misses, a
capture that genuinely wins material but not enough to close a large existing gap. Same
SEE-descending move order as SEE-pruning makes both a single monotonic break condition (see
quiescence's move loop): once either trips, every remaining, lower-scored capture trips it too.

Quiescence in check: none of the above (stand-pat, capture-only move generation) is valid while
the side to move is in check -- there is no "decline to respond" option, and a legal evasion that
is not itself a capture (a king step, a block) is otherwise never even generated. quiescence
special-cases this: skip the stand-pat/beta cutoff and search every legal move (generate_legal
already restricts to legal evasions when in check), the same posture as a normal negamax node.
Bounded by its own check_budget (mirroring negamax's ext_budget/MAX_CHECK_EXTENSIONS) rather than
qdepth, since quiescence carries no history array and so cannot detect a perpetual-check line via
repetition the way negamax can -- once check_budget runs out, a still-in-check node falls back to
the ordinary stand-pat/captures-only path so recursion still terminates. This only extends,
matching how the same forcing-line reasoning already works for search extensions elsewhere.

Late move reductions: inside negamax's move loop (never at the root -- search_root explores every
root move at full depth), a quiet move late enough in the ordering (see LMR_MIN_MOVE_INDEX) gets
searched shallower first, on the premise that TT/killer/history ordering has already almost
certainly put the moves worth full depth ahead of it. Only promotes to a full-depth re-search if
the reduced probe still beats alpha, and from there falls into the same full-window PVS re-search
as any other move that beats alpha -- so a reduction can only cost extra nodes re-confirming a
move, never silently accept a wrong score, since the final accepted score for any move that raises
alpha always comes from a full-depth search. Skipped entirely when the current node or the
resulting position is in check, or the move is a capture or promotion -- exactly the tactical moves
a shallower search is least equipped to judge. The reduction itself comes from LMR_TABLE, indexed
by (depth, move index): the standard log(depth) * log(move index) shape real engines start from,
rather than the previous flat one-ply reduction, so a move that is both very late and very deep
gets reduced further than one just past the LMR_MIN_* thresholds, capped at LMR_MAX_REDUCTION. Pure
lookup-table data built once at import (like FUTILITY_MARGIN/LMP_THRESHOLD above), so this changes
nothing about negamax's own compiled control flow or init cost.

Search extensions: a move that gives check gets its child searched at the same depth rather than
depth - 1 -- a full ply deeper than normal, since a forced reply to check is exactly the kind of
forcing line a depth-limited search most needs the extra ply for. ext_budget bounds how many of
these one line can stack (MAX_CHECK_EXTENSIONS), so a long forcing check sequence cannot inflate
one line's effective depth arbitrarily at every other line's expense within the same time budget.
Depth extends, never ply: the TT still stores against the exact depth value passed to that node,
so a cached entry is always compared like-for-like regardless of how much extension went into
reaching it, and the mate-distance adjustment (keyed on ply, not depth) is unaffected either way.

Singular extensions: at a deep enough node (SE_MIN_DEPTH) with a hash move backed by a deep,
trustworthy TT entry (SE_TT_DEPTH_MARGIN, and not merely an upper bound), a verification search of
this exact node's *other* moves -- at (depth - 1) // 2, in a narrow window just below the hash
move's own stored score (SE_MARGIN_PER_DEPTH * depth) -- checks whether any of them can even come
close. If none can, the hash move is "singular" (clearly better than every alternative this search
can see) and its own child gets the same one-ply extension search extensions above already give a
checking move, so the search commits real depth to confirming a move the ordering already trusts
rather than assuming the trust is warranted. Implemented via excluded_from/excluded_to: a per-node
exclusion, not inherited by any recursive call the excluded search itself makes (its own null-move,
IID, and move-loop children all pass -1, -1, same as every ordinary node) -- so nesting is
impossible by construction (the trigger itself also requires excluded_from < 0, belt and braces).
An excluded node must never resolve via its own TT entry (that entry was written for the *whole*
move set, not the one this search is deliberately missing) or write one either: both the
early-return in _tt_resolve's caller and the end-of-function store are skipped whenever
excluded_from >= 0, so the real TT entry for this position is exactly as if the verification search
had never run.

Multi-cut: rides on the exact verification search singular extensions above already pays for,
reading its result the other way rather than running a second, separate reduced-depth search of
its own. That search already excludes the hash move and asks whether anything else can reach
singular_beta (a narrow window just below the hash move's own stored score); if the hash move
turns out not to be singular (some other move reached singular_beta) AND singular_beta is itself
>= the real beta, then two independent moves at this node -- the hash move (whose stored score
backed a lower-bound/exact entry above singular_beta + margin) and the one the verification search
just found -- both clear the real beta on their own. That is normally treated as strong enough
evidence to cut the whole node without searching the rest of it, the same way a null-move cutoff
does, and for the same reason: enough independent evidence of a fail-high that spending the full
move loop to confirm it is very unlikely to pay for itself. Returns beta itself, not the possibly-
higher singular_beta, matching null-move pruning's own conservative cutoff value below.

Futility pruning: at a shallow node (depth <= FUTILITY_MAX_DEPTH) not in check, a quiet, non-check
move is skipped outright once the static eval plus a depth-scaled margin still can't reach alpha
-- the position already looks bad enough that a quiet move is very unlikely to save it, so the
search cost is spent on captures/promotions/checks instead. oi == 0 is never skipped, so a node
this applies to always has at least one fully-searched move to report a real score from.

Reverse futility / static null-move pruning: at a shallow node (depth <= RFP_MAX_DEPTH) not in
check, if the static eval already clears beta by more than a depth-scaled margin (RFP_MARGIN),
the whole node returns that eval outright without generating or searching a single move -- the
opponent's best defense is assumed unable to drag a position this good back down to beta. Shares
its static-eval computation with futility pruning above (STATIC_EVAL_MAX_DEPTH gates a single
evaluate() call reused by both, computed at most once per node) since both fire in the same
shallow-and-not-in-check regime. Distinct target from futility pruning: this prunes the entire
node before move generation, futility prunes individual moves within a node already being
searched.

Late move pruning (LMP): inside the move loop, a quiet, non-check move at a shallow node
(depth <= LMP_MAX_DEPTH) whose position in the ordering (oi) is at or past LMP_THRESHOLD[depth] is
skipped outright, on the same "ordering has already put anything worth searching ahead of this"
premise LMR relies on, but pruning entirely rather than searching at reduced depth first. The
threshold grows with depth, so a more expensive (and more trustworthy) node tolerates more late
moves before pruning kicks in. Checked after futility pruning's own skip in the same branch, so
oi == 0 is never reachable here either -- a node always has at least one fully-searched move.

Internal iterative deepening: a node deep enough (IID_MIN_DEPTH) with no hash move to order by --
a genuine TT miss, not merely an entry too shallow for a cutoff, since that still yields a hint --
searches itself at depth - IID_REDUCTION purely to populate one before the real move ordering.
Implemented by re-invoking negamax on the exact same (bb, meta, ply, hist_len): the sub-search's
own TT store, keyed by the same position hash, is the return channel -- re-probing right after
picks up whatever move it found, avoiding a second return value threaded through negamax just for
this. Self-terminating (depth strictly decreases each nesting) and skipped in check, matching the
other depth-side heuristics' posture toward tactical nodes.

Repetition, mechanism 1 -- our own turn recurring: negamax and search_root take a shared
`history` array of zobrist hashes plus `hist_len`, the count of entries that precede the current
node. Only our-turn positions (even ply, root = ply 0) are ever recorded or checked here -- the
array is indexed by "how many our-turn positions have occurred so far on the current path",
which a plain depth-first walk keeps correct with no explicit push/pop: a node writes its own
slot once, and a later sibling at the same ply simply overwrites it when the search backtracks.
agent.py owns the persistent real-game prefix and writes the root's own slot before calling
search_root; everything at ply >= 1 is written by negamax itself as it descends. If a line would
make a position recur a third time, it scores as an immediate draw (0) instead of running eval
or search on it further -- and this check runs before the transposition table is touched, so a
position that is genuinely a repeat on this path is never short-circuited by a stale cached score
from a path where it wasn't.

That alone is not sufficient, for two separate reasons, both found by a won game that kept
drawing (see agent.py's module docstring for the two real games). First: the harness's referee
checks Board.outcome(claim_draw=True) *before* asking either side for a move, and python-chess's
can_claim_threefold_repetition() fires the instant the side to move *has some legal reply* that
would create a third occurrence -- not only once one is actually played. Second: it also fires
when the *current* position (the one about to be handed over, the opponent's turn) has itself
already occurred twice before -- a repeat of an opponent-turn position counts on its own, not
only when it happens to line up with one of ours.

claim_eligible_for_opponent checks both: condition two is a direct lookup against
`opponent_history`, the real game's own past opponent-turn positions (one appended per move we
actually play, in agent.py -- not extended during search, since only the real, played sequence
matters for what has actually occurred); condition one is one extra generate_legal on the
position we would hand over, matching each of the opponent's replies against `history`. Either
way, search_root / agent._search_restricted cap that candidate move's score at 0, so a move that
merely offers the position is treated the same as one that plays it. Root-only, not inside
negamax's own recursion: condition one is a full extra movegen per candidate move, affordable
once per real move decision but not throughout the tree. Untouched by the transposition table --
it always recomputes from the real history arrays, never from a cached score.

Fifty-move rule: `halfmove_clock`, threaded through negamax/_search_root_pass/search_root the same
way `hist_len` is, counts plies since the last pawn move or capture on the current path -- the
real game's own value (bitboard.halfmove_clock(fen), read straight from the FEN each get_move
call rather than tracked incrementally, so it is correct even if a game does not start at 0) plus
whatever the search itself has added while descending. Reset to 0 for a child whose move is a pawn
move or capture, otherwise incremented by 1 -- computed once per move in negamax's own loop and
reused for both the recursive call and (previously) the redundant is_capture calls in the
futility/LMP/LMR checks just above it. At HALFMOVE_DRAW_LIMIT (100 plies), a node returns an
immediate draw (0), checked in the same early position as the repetition check above (before the
transposition table) and for the same reason -- a cached score must never mask a real forced draw.
Main-search-only correctness gap, deliberately accepted: quiescence does not thread or check the
running count itself (see is_insufficient_material below for the one draw check it does share),
since its own additional plies either reset the count immediately (a capture) or are bounded to a
handful (the in-check evasion search, QSEARCH_CHECK_BUDGET) -- missing the exact ply the count
crosses 100 inside that narrow window is a negligible risk against the complexity of threading a
seventh parameter through quiescence's own recursion for it.

Insufficient material: movegen.is_insufficient_material(bb) is a pure, path-independent property
of the current board alone (bare king vs bare king, king plus exactly one minor against a bare
king, or same-coloured-square bishops only -- see its own docstring for the deliberately
conservative subset and why under-detecting is always the safe direction), so unlike the fifty-move
count it needs no parameter threading at all: negamax and quiescence both check it directly,
immediately after the halfmove/time-up checks, and both return an immediate draw (0) when it
holds. This does not protect the real game result -- the harness's own Board.outcome() already
treats insufficient material as an automatic, non-claim terminal condition on the actual position
played -- it exists purely so the search's own evaluation of a hypothetical such position reached
while descending the tree is accurate rather than running ordinary material/PST/threat scoring on
a position that is provably a dead draw.

"""

import math
import time

import numpy as np
from numba import njit, objmode
from numba import types as nbtypes

from attacks import KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS, bishop_attacks, rook_attacks
from bitboard import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE, i64
from evaluate import PIECE_VALUE, evaluate
from movegen import (
    _bit_scan,
    _i64,
    generate_legal,
    has_non_pawn_material,
    is_check,
    is_insufficient_material,
    king_square,
    make_move,
    make_null_move,
    occ_all,
    piece_type_at,
)
from zobrist import position_hash

MATE = 1_000_000
INF = 2_000_000
CHECK_INTERVAL = 127
# Phase 2.7 of docs/plan.md: a non-capturing promotion used to score 500 + PIECE_VALUE[promo]
# (<=1400 for a queen), which sits BELOW killers (5001/5002), the counter-move slot (5000), and any
# quiet move with history >= 1400 -- and _score_moves2's "base != 0 skips killer/history" short
# circuit meant a promotion could never claw its way back up from there either. Raised above the
# killer/counter-move/history band (all capped at 4999-5002) and kept below the capture floor
# (10_000 + the worst plausible SEE loss) so "promotes, does not also capture" now sorts where it
# belongs: after captures, before quiet moves.
PROMOTION_BASE = 9_000
QUIESCENCE_MAX_PLIES = i64(24)
QSEARCH_CHECK_BUDGET = i64(6)
ONE = np.uint64(1)

# i64() itself is plain, un-jitted Python (see bitboard.py), so it can only be called from
# ordinary module-level/Python code, never from inside an njit function body -- these two are for
# the njit-to-njit call sites below that would otherwise pass a bare int literal (claim_eligible_
# for_opponent's lookahead=1, quick_best_move's pv sentinel -1) and mint a fresh Literal[int]
# specialisation (see docs/plan.md Phase 1.1(a)).
_LOOKAHEAD_ONE = i64(1)
_NO_PV = i64(-1)

# Delta pruning in quiescence: a capture whose SEE, added to the stand-pat eval, still can't
# reach alpha within this safety margin is skipped -- see quiescence's move loop. A generous
# margin (bigger than a minor piece) since this is a hard skip, not a reduction.
DELTA_MARGIN = 200

# Aspiration windows: search_root centers the first pass's alpha-beta window on the previous
# iterative-deepening depth's score instead of always searching -INF..INF. Phase 2.7 of
# docs/plan.md: a fail-high/fail-low used to jump straight from a +-50 window to the full
# -INF..INF, throwing away a whole re-search's worth of the previous pass's work every time the
# score moved by even a little more than 50 -- widen through +-200 first and only fall back to
# the full window if that also fails. NO_PREV_SCORE (a value no real score or mate score can ever
# equal, since MATE < INF strictly) tells search_root there is no previous depth to center on --
# used for depth 1 and whenever the prior score was itself outside the mate threshold.
ASPIRATION_WINDOWS = np.array([50, 200], dtype=np.int64)
NO_PREV_SCORE = INF

# Null-move pruning: give the side to move a free pass and search the rest at a reduced depth; if
# that is still enough to cause a beta cutoff, the real move would only do better, so the whole
# subtree is pruned. NULL_MOVE_MIN_DEPTH keeps depth - 1 - NULL_MOVE_REDUCTION non-negative (any
# smaller and it degrades to a quiescence call anyway, not a useful probe) and NULL_MOVE_REDUCTION
# is the standard R=2. Guarded against zugzwang two ways -- see negamax.
NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2

# Late move reductions: a quiet move searched late in the ordering (oi >= LMR_MIN_MOVE_INDEX,
# after the hash/killer/history-backed moves have already had their full-depth say) gets less
# depth first; only a reduced-depth score that still beats alpha earns a full-depth re-search
# before PVS's own full-window re-search gets a chance -- see docs/FUTURE.md item 5.
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE_INDEX = 3

# LMR_TABLE[depth][move_index] replaces a flat one-ply reduction with the standard depth/move-index
# log-log formula (real engines' usual starting point): a move both late in the ordering and deep
# in the tree earns a bigger reduction than one just barely past the LMR_MIN_* thresholds, since
# the ordering (TT/killers/history/counter-move) is more likely to have already found the real
# move by then. Capped at LMR_MAX_REDUCTION so a pathological depth/move-index combination can
# never reduce so far that a real tactic falls out of the search; capped table dimensions (the
# largest depth/move-index this search realistically reaches) with lookups clamped to them below,
# not because a bigger index is wrong, just to keep the table itself small. Built once at import in
# plain Python (like FUTILITY_MARGIN/LMP_THRESHOLD above) -- zero compile cost, this is data, not
# control flow.
LMR_TABLE_MAX_DEPTH = 64
LMR_TABLE_MAX_MOVE_INDEX = 128
LMR_MAX_REDUCTION = 4
LMR_TABLE = np.zeros((LMR_TABLE_MAX_DEPTH + 1, LMR_TABLE_MAX_MOVE_INDEX + 1), dtype=np.int64)
for _d in range(1, LMR_TABLE_MAX_DEPTH + 1):
    for _m in range(1, LMR_TABLE_MAX_MOVE_INDEX + 1):
        _r = int(0.5 + math.log(_d) * math.log(_m) / 2.25)
        LMR_TABLE[_d, _m] = min(_r, LMR_MAX_REDUCTION)
del _d, _m, _r

# Search extensions: a move that gives check gets its child searched a full ply deep (depth is
# not decremented) instead of the usual depth - 1, since a forced reply to check is exactly the
# kind of forcing, tactically-critical line a depth-limited search is otherwise most likely to
# misjudge. ext_budget, threaded through negamax and decremented once per extension granted,
# caps how many of these a single line can stack -- unbounded stacking (e.g. a long forcing check
# sequence) would let one line's effective depth grow arbitrarily, at the expense of every
# sibling line sharing the same time budget. MAX_CHECK_EXTENSIONS is deliberately small: enough
# for a real forcing sequence, far short of enough to meaningfully skew the time budget.
MAX_CHECK_EXTENSIONS = i64(8)

# Futility pruning: at a shallow, non-check node, a quiet move whose best case (the static eval
# plus a depth-scaled margin) still can't reach alpha is skipped outright rather than searched --
# a quiet move rarely swings the score by more than a pawn or so, so if the position already
# looks this bad before the move, searching it is very unlikely to change the outcome. Margins
# widen with depth since more plies of search behind the static eval means more room for a quiet
# move to matter after all. Index 0 is never read (depth <= 0 returns via quiescence earlier).
FUTILITY_MAX_DEPTH = 3
FUTILITY_MARGIN = np.array([0, 150, 300, 450], dtype=np.int64)
RFP_MAX_DEPTH = 6
RFP_MARGIN = 90
STATIC_EVAL_MAX_DEPTH = 6  # max(RFP_MAX_DEPTH, FUTILITY_MAX_DEPTH, RAZOR_MAX_DEPTH) -- shared gate
LMP_MAX_DEPTH = 4
LMP_THRESHOLD = np.array([0, 6, 10, 15, 21], dtype=np.int64)

# Razoring: at a very shallow, non-check node, a static eval already trailing alpha by more than a
# depth-scaled margin almost certainly cannot be rescued by a quiet move -- confirm with a
# quiescence call (not a blind return, so a tactical shot the static eval missed still survives)
# and return its score outright if that also fails to reach alpha. Index 0 is never read, same as
# FUTILITY_MARGIN above.
RAZOR_MAX_DEPTH = 2
RAZOR_MARGIN = np.array([0, 300, 500], dtype=np.int64)

# SEE pruning in the main search (quiescence already does this for its own captures -- see
# DELTA_MARGIN above): at a shallow node not in check, a capture whose SEE is clearly losing by
# more than a depth-scaled margin is skipped outright rather than searched, same rationale as
# futility pruning for quiet moves. Never applied to the first (hash/best-ordered) move.
SEE_PRUNE_MAX_DEPTH = 5
SEE_PRUNE_MARGIN = 100

# History-based LMR: a late, quiet move's reduction (from LMR_TABLE below) is nudged by how well
# history/continuation-history have rated it in the past -- a move both quiets and history-hot
# almost certainly deserves the shallower search, while one that has actively cut a lot of moves
# below it (a strongly negative malus score) deserves an extra ply off. +-1 only, so this can never
# push a reduction far from what the depth/move-index table alone would already give.
LMR_HISTORY_HIGH = 2000
LMR_HISTORY_LOW = -2000

# Singular extensions: only at a node deep enough (SE_MIN_DEPTH) that the extra verification
# search below is worth its own cost, with a hash move backed by a TT entry both deep enough
# (within SE_TT_DEPTH_MARGIN of the current depth) and trustworthy (not merely an upper bound).
# SE_MARGIN_PER_DEPTH sets how far below the hash move's own score every alternative must fail to
# count as "singular". See this module's docstring for the excluded-move mechanism this needs.
SE_MIN_DEPTH = 7
SE_TT_DEPTH_MARGIN = 3
SE_MARGIN_PER_DEPTH = 2

# Fifty-move rule: 100 plies (50 full moves by each side) with no pawn move and no capture is an
# automatic draw. See this module's docstring for where the running count comes from and why.
HALFMOVE_DRAW_LIMIT = 100

# Internal iterative deepening: a node deep enough to matter but with no hash move to order by
# (a TT miss, or a stored entry too shallow to have one -- see hint_from in negamax) gets a
# reduced-depth search of itself first, purely to seed move ordering before the real search of
# it. IID_MIN_DEPTH keeps this to nodes where good ordering is worth the extra sub-search.
IID_MIN_DEPTH = 4
IID_REDUCTION = 2

# Our-turn positions per game are bounded by rules.PLY_CAP / 2 (~150); in-search growth is
# bounded by MAX_DEPTH / 2 (agent.py caps depth at 64, so ~32 more). 512 leaves ample headroom.
HISTORY_CAPACITY = 512

# Negamax's own recursion only deepens through ply while depth > 0, and depth only ever
# decreases from the root's initial value (agent.py caps that at 64), so 128 is ample headroom
# for indexing killer moves by ply, matching HISTORY_CAPACITY's own margin above its real need.
MAX_KILLER_PLY = 128

# Two-tier: each bucket has two slots, a depth-preferred one (kept unless a same-key refresh or a
# search that went at least as deep wants to replace it) and an always-replace one (a pure
# recency slot, so a fresh position from the current search is never simply dropped because the
# depth-preferred slot happens to be occupied by something deeper). TT_BUCKETS is the addressable
# bucket count (indexed by the low bits of the Zobrist hash); the arrays below are twice that many
# raw slots. 16M buckets * 2 slots * 20 bytes/slot (parallel arrays below) is a fixed ~640MB,
# independent of game length -- no growth, no eviction bookkeeping beyond the two-slot policy.
# Doubled a third time from 8M buckets (~320MB, itself already doubled twice before -- Tier 14,
# Tier 16): Phase 4.2 of docs/plan.md -- real-platform init now has real margin (~40-50s of 90s),
# so there is room to spend pure RAM on fewer collisions over a full game's worth of nodes, at no
# compile or init-time cost whatsoever (a pure array-size constant, same code either way). The
# agent contract documents a 2 GB memory ceiling, so 640MB is a known-safe ~32% of it, not a guess.
TT_BUCKETS = 1 << 24
TT_SIZE = TT_BUCKETS * 2
TT_MASK = np.uint64(TT_BUCKETS - 1)
TT_EXACT = np.int8(0)
TT_LOWER = np.int8(1)
TT_UPPER = np.int8(2)
MATE_STORE_THRESHOLD = MATE - 1000

# Phase 4.2 of docs/plan.md: explicit eager signatures on negamax/quiescence/search_root (and
# make_move, in movegen.py) -- not a strength change, insurance. Phase 1.1 already normalised
# every call site to a single consistent type per parameter (Literal[int] constants wrapped in
# np.int64(...), int8/int64 array-index drift fixed), which is *why* bench_init.py already shows
# exactly one specialisation per function today -- but that is an empirically observed fact about
# today's call sites, not a structural guarantee. A signature declared here makes a second
# specialisation impossible for numba to ever produce, so a future call site that regresses back
# to a bare Python int or a mismatched dtype fails loudly and immediately at import time (a
# TypingError) instead of silently compiling a second specialisation of the most expensive
# function in the engine mid-game, on the clock -- exactly the Phase 1.1(c) hazard this guards
# against structurally rather than by convention.
_BB_T = nbtypes.uint64[::1]
_META_T = nbtypes.int8[::1]
_I64 = nbtypes.int64
_F64 = nbtypes.float64
_B1 = nbtypes.boolean
_I64_1D = nbtypes.int64[::1]
_U64_1D = nbtypes.uint64[::1]
_I32_1D = nbtypes.int32[::1]
_I8_1D = nbtypes.int8[::1]
_I8_2D = nbtypes.int8[:, ::1]
_I32_2D = nbtypes.int32[:, ::1]

_QUIESCENCE_SIG = _I64(_BB_T, _META_T, _I64, _I64, _F64, _I64_1D, _I64, _I64)

_NEGAMAX_SIG = _I64(
    _BB_T, _META_T, _I64, _I64, _I64, _F64, _I64_1D, _I64,
    _U64_1D, _I64,
    _U64_1D, _I32_1D, _I32_1D, _I8_1D, _I8_1D, _I8_1D, _I8_1D,
    _I8_2D, _I8_2D, _I8_2D,
    _I32_1D, _I32_2D,
    _B1, _I64,
    _I8_1D, _I8_1D, _I8_1D,
    _I64, _I64, _I64, _I64, _I64,
)

_SEARCH_ROOT_RET = nbtypes.Tuple((_I64, _I64, _I64, _I64, _B1))  # type: ignore[no-untyped-call]
_SEARCH_ROOT_SIG = _SEARCH_ROOT_RET(
    _BB_T, _META_T, _I64, _F64, _I64_1D, _I64, _I64, _I64,
    _U64_1D, _I64, _U64_1D, _I64,
    _U64_1D, _I32_1D, _I32_1D, _I8_1D, _I8_1D, _I8_1D, _I8_1D,
    _I8_2D, _I8_2D, _I8_2D,
    _I32_1D, _I32_2D,
    _I8_1D, _I8_1D, _I8_1D,
    _I64, _I64,
)


@njit(cache=False)
def _time_up(deadline: float, counters: np.ndarray) -> bool:
    if counters[0] & CHECK_INTERVAL == 0:
        with objmode(now="float64"):
            now = time.perf_counter()
        if now >= deadline:
            counters[1] = 1
    return bool(counters[1] != 0)


@njit(cache=False)
def is_capture(bb: np.ndarray, meta: np.ndarray, from_sq: int, to_sq: int) -> bool:
    color = _i64(meta[0])
    opponent = 1 - color
    to_bit = ONE << np.uint64(to_sq)
    for pt in range(6):
        if bb[opponent * 6 + pt] & to_bit:
            return True
    if to_sq == meta[5] and piece_type_at(bb, color, from_sq) == PAWN:
        return (from_sq % 8) != (to_sq % 8)
    return False


@njit(cache=False)
def _gives_check_direct(
    bb: np.ndarray, meta: np.ndarray, from_sq: int, to_sq: int, promo: int, moving_pt: int
) -> bool:
    """Cheap, DIRECT-check-only approximation of "does playing (from_sq, to_sq, promo) give
    check?", without calling make_move + is_check (a full board copy). Computes only the moved
    piece's own attack pattern from `to_sq` against the enemy king, using the post-move occupancy
    updated as a single scalar rather than a new board -- misses discovered checks (moving a piece
    off a line that unveils an attack from a different piece), which a full is_check would catch.

    Phase 2.6 of docs/plan.md: negamax uses this ONLY to gate the futility/LMP prune checks, so a
    pruned move never pays for a board copy at all; every move that survives pruning still gets the
    real, accurate `is_check(new_bb, new_meta)` off its real `make_move` result (needed anyway for
    the recursive search and for check-extension eligibility, which -- unlike pruning -- must not
    silently miss discovered checks).
    """
    color = _i64(meta[0])
    opponent = 1 - color
    king_sq = king_square(bb, opponent)
    if king_sq < 0:
        return False
    king_bit = ONE << np.uint64(king_sq)

    occ_after = occ_all(bb) & ~(ONE << np.uint64(from_sq))
    occ_after |= ONE << np.uint64(to_sq)

    on_square_pt = promo if (moving_pt == PAWN and promo >= 0) else moving_pt
    if on_square_pt == PAWN:
        return bool(PAWN_ATTACKS[color, to_sq] & king_bit)
    if on_square_pt == KNIGHT:
        return bool(KNIGHT_ATTACKS[to_sq] & king_bit)
    if on_square_pt == BISHOP:
        return bool(bishop_attacks(to_sq, occ_after) & king_bit)
    if on_square_pt == ROOK:
        return bool(rook_attacks(to_sq, occ_after) & king_bit)
    if on_square_pt == QUEEN:
        return bool((bishop_attacks(to_sq, occ_after) | rook_attacks(to_sq, occ_after)) & king_bit)
    return False


# Exchange chains are bounded by total pieces on the board (32), so 32 gain-array slots is
# ample headroom -- see() never appends past one entry per capture in the chain.
SEE_MAX_DEPTH = 32


@njit(cache=False)
def _see_least_valuable_attacker(
    bb: np.ndarray, occ: np.uint64, color: int, square: int
) -> tuple[int, int]:
    """The lowest-value piece of `color` attacking `square` given `occ`, as (from_square,
    piece_type), or (-1, -1) if none. `bb` is the ORIGINAL, unmutated board -- see() below tracks
    the exchange purely via `occ` (one uint64, bits cleared as pieces are used), so every type
    lookup here must intersect the original per-type bitboard with `occ` to exclude pieces already
    spent earlier in the chain. Sliding attacks are recomputed against the live `occ` on every call,
    which is what makes discovered ("x-ray") attackers fall out for free as pieces are removed
    during the exchange -- once a blocker is cleared from `occ`, bishop_attacks/rook_attacks simply
    see past it on the next call. Same ray-cast-over-magic-bitboards tradeoff as the rest of
    attacks.py.
    """
    attackers = PAWN_ATTACKS[1 - color, square] & bb[color * 6 + PAWN] & occ
    if attackers:
        return _bit_scan(attackers), PAWN
    attackers = KNIGHT_ATTACKS[square] & bb[color * 6 + KNIGHT] & occ
    if attackers:
        return _bit_scan(attackers), KNIGHT
    attackers = bishop_attacks(square, occ) & bb[color * 6 + BISHOP] & occ
    if attackers:
        return _bit_scan(attackers), BISHOP
    attackers = rook_attacks(square, occ) & bb[color * 6 + ROOK] & occ
    if attackers:
        return _bit_scan(attackers), ROOK
    attackers = (
        (bishop_attacks(square, occ) | rook_attacks(square, occ)) & bb[color * 6 + QUEEN] & occ
    )
    if attackers:
        return _bit_scan(attackers), QUEEN
    attackers = KING_ATTACKS[square] & bb[color * 6 + KING] & occ
    if attackers:
        return _bit_scan(attackers), KING
    return -1, -1


@njit(cache=False)
def see(bb: np.ndarray, meta: np.ndarray, from_sq: int, to_sq: int, promo: int) -> int:
    """Static exchange evaluation of playing (from_sq, to_sq, promo): the net material change on
    `to_sq` after both sides recapture there optimally (least-valuable-attacker first), not just
    the value of what this one move immediately wins. Standard swap-list algorithm (see
    chessprogramming.org "SEE - The Swap Algorithm"): gain[d] is what the side capturing at ply d
    nets (what they take, minus the previous ply's gain from their opponent's perspective), and
    the final backward min-max pass lets either side stop the exchange early if continuing it
    would lose more than declining to.

    Ignores pins and check-legality of intermediate recaptures (a standard SEE simplification --
    it can misjudge a recapture that is actually illegal because the recapturing piece is pinned),
    which is why this is a move-ordering and quiescence-pruning heuristic, never a legality check.

    Phase 2.4 of docs/plan.md: tracks the exchange with a single `occ` bitboard mutated in place
    (bits cleared as pieces are spent) instead of a full `bb.copy()` per call -- `bb` itself is
    never written to, only read through `occ`'s mask in `_see_least_valuable_attacker` above.
    """
    color = _i64(meta[0])
    opponent = 1 - color
    moving_pt = piece_type_at(bb, color, from_sq)
    ep_square = meta[5]
    is_ep = moving_pt == PAWN and to_sq == ep_square and (from_sq % 8) != (to_sq % 8)
    victim_pt = PAWN if is_ep else piece_type_at(bb, opponent, to_sq)

    gain = np.zeros(SEE_MAX_DEPTH, dtype=np.int64)
    gain[0] = PIECE_VALUE[victim_pt] if victim_pt >= 0 else 0
    if promo >= 0:
        gain[0] += PIECE_VALUE[promo] - PIECE_VALUE[PAWN]

    occ = occ_all(bb)
    occ &= ~(ONE << np.uint64(from_sq))
    if is_ep:
        captured_sq = to_sq - 8 if color == WHITE else to_sq + 8
        occ &= ~(ONE << np.uint64(captured_sq))
    # The mover always ends up on to_sq -- for en passant, to_sq itself was empty beforehand, so
    # this bit needs setting explicitly rather than relying on the victim having occupied it.
    occ |= ONE << np.uint64(to_sq)

    on_square_pt = promo if (moving_pt == PAWN and promo >= 0) else moving_pt
    on_square_value = PIECE_VALUE[on_square_pt]

    side = opponent
    depth = 0
    while depth < SEE_MAX_DEPTH - 1:
        atk_sq, atk_pt = _see_least_valuable_attacker(bb, occ, side, to_sq)
        if atk_sq < 0:
            break
        depth += 1
        gain[depth] = on_square_value - gain[depth - 1]

        occ &= ~(ONE << np.uint64(atk_sq))
        promotes = atk_pt == PAWN and (
            (side == WHITE and to_sq >= 56) or (side != WHITE and to_sq <= 7)
        )
        on_square_pt = QUEEN if promotes else atk_pt
        on_square_value = PIECE_VALUE[on_square_pt]
        side = 1 - side

    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return int(gain[0])


@njit(cache=False)
def _move_score(
    bb: np.ndarray,
    meta: np.ndarray,
    from_sq: int,
    to_sq: int,
    promo: int,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
) -> int:
    if from_sq == pv_from and to_sq == pv_to and promo == pv_promo:
        return 1_000_000

    score = 0
    if promo >= 0:
        score += PROMOTION_BASE + PIECE_VALUE[promo]

    color = _i64(meta[0])
    opponent = 1 - color
    victim_pt = piece_type_at(bb, opponent, to_sq)
    is_ep = to_sq == meta[5] and piece_type_at(bb, color, from_sq) == PAWN
    if victim_pt >= 0 or (is_ep and (from_sq % 8) != (to_sq % 8)):
        score += 10_000 + see(bb, meta, from_sq, to_sq, promo)
    return score


@njit(cache=False)
def _score_moves(
    bb: np.ndarray,
    meta: np.ndarray,
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    promo_arr: np.ndarray,
    count: int,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
) -> np.ndarray:
    scores = np.empty(count, dtype=np.int64)
    for i in range(count):
        scores[i] = _move_score(
            bb, meta, _i64(from_arr[i]), _i64(to_arr[i]), _i64(promo_arr[i]),
            pv_from, pv_to, pv_promo,
        )
    return scores


@njit(cache=False)
def _score_moves2(
    bb: np.ndarray,
    meta: np.ndarray,
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    promo_arr: np.ndarray,
    count: int,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
    k1_from: int,
    k1_to: int,
    k1_promo: int,
    k2_from: int,
    k2_to: int,
    k2_promo: int,
    cm_from: int,
    cm_to: int,
    cm_promo: int,
    history_table: np.ndarray,
    cont_hist: np.ndarray,
    prev_idx: int,
) -> np.ndarray:
    """Same per-move scoring as _score_moves, but a plain quiet move (base score exactly 0: not
    the hash/PV move, not a promotion, not a capture) additionally gets a killer-move bonus, or
    failing that a counter-move bonus (cm_from/cm_to/cm_promo: whatever quiet move most recently
    caused a cutoff in reply to the move that led to this node -- see negamax's counter_from/to/
    promo table), or failing that its from/to history score plus a continuation-history bonus
    keyed by (prev_idx, this move's [piece type, to-square]) when prev_idx >= 0 (see
    new_continuation_history) -- ranked killers, then counter-move, then history+continuation, all
    below every capture and promotion, above zero. Inlined into one function (rather than calling a
    per-move helper in the loop, as an earlier version did) purely to keep numba's compile graph
    smaller -- this and negamax are the two most expensive functions to JIT in the whole engine.
    """
    scores = np.empty(count, dtype=np.int64)
    for i in range(count):
        from_sq = _i64(from_arr[i])
        to_sq = _i64(to_arr[i])
        promo = _i64(promo_arr[i])
        base = _move_score(bb, meta, from_sq, to_sq, promo, pv_from, pv_to, pv_promo)
        if base != 0:
            scores[i] = base
        elif from_sq == k1_from and to_sq == k1_to and promo == k1_promo:
            scores[i] = 5002
        elif from_sq == k2_from and to_sq == k2_to and promo == k2_promo:
            scores[i] = 5001
        elif from_sq == cm_from and to_sq == cm_to and promo == cm_promo:
            scores[i] = 5000
        else:
            h = history_table[from_sq * 64 + to_sq]
            if prev_idx >= 0:
                moving_pt = piece_type_at(bb, _i64(meta[0]), from_sq)
                h += cont_hist[prev_idx, moving_pt * 64 + to_sq]
            scores[i] = h if h < 4999 else 4999
    return scores


# Phase 2.5 of docs/plan.md: sentinel a picked move's score is overwritten with so _argmax never
# selects it again. Below every real move-ordering score (the highest is the PV/hash move's flat
# 1_000_000, still well under INF) and below quiescence's own -INF "not a capture" sentinel, so it
# can never be mistaken for either.
_SELECTED = -INF - 1


@njit(cache=False)
def _argmax(scores: np.ndarray, count: int) -> int:
    """Index of the highest-scoring not-yet-picked move. Called once per move actually examined
    (negamax's and quiescence's move loops both break out on a cutoff/prune well before visiting
    every move), replacing a single `np.argsort(-scores)` that always fully sorts the whole list up
    front -- most nodes fail high on the first few moves and never need the rest ordered at all.
    """
    idx = 0
    best = scores[0]
    for i in range(1, count):
        if scores[i] > best:
            best = scores[i]
            idx = i
    return idx


@njit(_QUIESCENCE_SIG, cache=False)
def quiescence(
    bb: np.ndarray,
    meta: np.ndarray,
    alpha: int,
    beta: int,
    deadline: float,
    counters: np.ndarray,
    qdepth: int,
    check_budget: int,
) -> int:
    counters[0] += 1
    if _time_up(deadline, counters):
        return 0

    if is_insufficient_material(bb):
        return 0

    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    if count == 0:
        return -MATE if is_check(bb, meta) else 0

    if check_budget > 0 and is_check(bb, meta):
        for i in range(count):
            f = _i64(from_arr[i])
            t = _i64(to_arr[i])
            p = _i64(promo_arr[i])
            new_bb, new_meta = make_move(bb, meta, f, t, p)
            score = -quiescence(
                new_bb, new_meta, -beta, -alpha, deadline, counters, qdepth, check_budget - 1
            )
            if counters[1]:
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    stand_pat = evaluate(bb, meta)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth <= 0:
        return alpha

    scores = np.full(count, -INF, dtype=np.int64)
    ncap = 0
    for i in range(count):
        f = _i64(from_arr[i])
        t = _i64(to_arr[i])
        p = _i64(promo_arr[i])
        if is_capture(bb, meta, f, t) or p == QUEEN:
            scores[i] = see(bb, meta, f, t, p)
            ncap += 1
    if ncap == 0:
        return alpha

    for _ in range(ncap):
        idx = _argmax(scores, count)
        see_score = scores[idx]
        scores[idx] = _SELECTED
        if see_score < 0 or stand_pat + see_score + DELTA_MARGIN <= alpha:
            # SEE-descending order: everything from here on scores at least as low, so once
            # either condition trips it holds for the rest of the loop too. The first
            # (see_score < 0) is SEE-pruning -- a capture that loses material outright cannot
            # help (see docs/FUTURE.md item 4). The second is delta pruning: even a *winning*
            # capture skipped here still can't close the gap to alpha by more than a safety
            # margin, catching the case SEE-pruning alone misses (a real but too-small gain
            # against a large existing deficit).
            break
        f = _i64(from_arr[idx])
        t = _i64(to_arr[idx])
        p = _i64(promo_arr[idx])
        new_bb, new_meta = make_move(bb, meta, f, t, p)
        score = -quiescence(
            new_bb, new_meta, -beta, -alpha, deadline, counters, qdepth - 1, check_budget
        )
        if counters[1]:
            return 0
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def _tt_resolve(
    stored_depth: int,
    raw_score: int,
    flag: int,
    depth: int,
    ply: int,
    alpha: int,
    beta: int,
) -> tuple[bool, int, int, int]:
    """Given a transposition-table slot that matched the current position's key, decide whether
    it settles this node outright. Returns (hit, score, alpha, beta): hit means the caller can
    return `score` immediately (an exact stored score, or a cutoff from a lower/upper bound);
    otherwise alpha/beta come back possibly tightened by a bound that didn't itself cause a
    cutoff, for the caller to keep searching with. Shared between the two probed slots (see
    negamax) so the mate-distance adjustment and bound logic exist in exactly one place.
    """
    if stored_depth < depth:
        return False, 0, alpha, beta
    if raw_score >= MATE_STORE_THRESHOLD:
        s = raw_score - ply
    elif raw_score <= -MATE_STORE_THRESHOLD:
        s = raw_score + ply
    else:
        s = raw_score
    if flag == TT_EXACT:
        return True, s, alpha, beta
    if flag == TT_LOWER and s > alpha:
        alpha = s
    elif flag == TT_UPPER and s < beta:
        beta = s
    if alpha >= beta:
        return True, s, alpha, beta
    return False, 0, alpha, beta


@njit(_NEGAMAX_SIG, cache=False)
def negamax(
    bb: np.ndarray,
    meta: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    deadline: float,
    counters: np.ndarray,
    ply: int,
    history: np.ndarray,
    hist_len: int,
    tt_key: np.ndarray,
    tt_depth: np.ndarray,
    tt_score: np.ndarray,
    tt_flag: np.ndarray,
    tt_from: np.ndarray,
    tt_to: np.ndarray,
    tt_promo: np.ndarray,
    killer_from: np.ndarray,
    killer_to: np.ndarray,
    killer_promo: np.ndarray,
    history_table: np.ndarray,
    cont_hist: np.ndarray,
    allow_null: bool,
    ext_budget: int,
    counter_from: np.ndarray,
    counter_to: np.ndarray,
    counter_promo: np.ndarray,
    parent_from: int,
    parent_to: int,
    halfmove_clock: int,
    excluded_from: int,
    excluded_to: int,
) -> int:
    counters[0] += 1
    if _time_up(deadline, counters):
        return 0

    if halfmove_clock >= HALFMOVE_DRAW_LIMIT:
        return 0

    if is_insufficient_material(bb):
        return 0

    h = position_hash(bb, meta)

    if ply % 2 == 0:
        matches = 0
        for i in range(hist_len):
            if history[i] == h:
                matches += 1
        if matches >= 2:
            return 0
        history[hist_len] = h
        child_hist_len = hist_len + 1
    else:
        child_hist_len = hist_len

    orig_alpha = alpha
    bucket = int(h & TT_MASK)
    slot_a = bucket * 2
    slot_b = slot_a + 1
    hint_from, hint_to, hint_promo = -1, -1, -1
    hint_tt_depth, hint_tt_score, hint_tt_flag = -1, 0, int(TT_EXACT)
    if tt_key[slot_a] == h:
        hint_from, hint_to, hint_promo = int(tt_from[slot_a]), int(tt_to[slot_a]), int(
            tt_promo[slot_a]
        )
        hint_tt_depth, hint_tt_score, hint_tt_flag = (
            int(tt_depth[slot_a]) - 1, int(tt_score[slot_a]), int(tt_flag[slot_a]),
        )
        if excluded_from < 0:
            hit, s, alpha, beta = _tt_resolve(
                hint_tt_depth, hint_tt_score, hint_tt_flag, depth, ply, alpha, beta,
            )
            if hit:
                return s
    elif tt_key[slot_b] == h:
        hint_from, hint_to, hint_promo = int(tt_from[slot_b]), int(tt_to[slot_b]), int(
            tt_promo[slot_b]
        )
        hint_tt_depth, hint_tt_score, hint_tt_flag = (
            int(tt_depth[slot_b]) - 1, int(tt_score[slot_b]), int(tt_flag[slot_b]),
        )
        if excluded_from < 0:
            hit, s, alpha, beta = _tt_resolve(
                hint_tt_depth, hint_tt_score, hint_tt_flag, depth, ply, alpha, beta,
            )
            if hit:
                return s

    if depth <= 0:
        return quiescence(
            bb, meta, alpha, beta, deadline, counters, QUIESCENCE_MAX_PLIES, QSEARCH_CHECK_BUDGET
        )

    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    if count == 0:
        return -MATE if is_check(bb, meta) else 0

    in_check = is_check(bb, meta)

    static_eval = 0
    have_static_eval = False
    if not in_check and depth <= STATIC_EVAL_MAX_DEPTH:
        static_eval = evaluate(bb, meta)
        have_static_eval = True

    if (
        have_static_eval
        and depth <= RFP_MAX_DEPTH
        and -MATE_STORE_THRESHOLD < beta < MATE_STORE_THRESHOLD
        and static_eval - RFP_MARGIN * depth >= beta
    ):
        # Reverse futility / static null-move pruning: the static eval already clears beta by more
        # than a depth-scaled margin, so even a defense that outplays this crude estimate is
        # assumed unable to drag the score back down to beta -- return outright rather than search
        # the node at all. Distinct from futile below, which skips individual moves, not the node.
        return static_eval

    futile = False
    if (
        have_static_eval
        and depth <= FUTILITY_MAX_DEPTH
        and -MATE_STORE_THRESHOLD < alpha < MATE_STORE_THRESHOLD
    ):
        futile = static_eval + FUTILITY_MARGIN[depth] <= alpha

    if (
        have_static_eval
        and depth <= RAZOR_MAX_DEPTH
        and -MATE_STORE_THRESHOLD < alpha < MATE_STORE_THRESHOLD
        and static_eval + RAZOR_MARGIN[depth] <= alpha
    ):
        razor_score = quiescence(
            bb, meta, alpha, beta, deadline, counters, QUIESCENCE_MAX_PLIES, QSEARCH_CHECK_BUDGET
        )
        if counters[1]:
            return 0
        if razor_score <= alpha:
            return razor_score

    if (
        allow_null
        and depth >= NULL_MOVE_MIN_DEPTH
        and -MATE_STORE_THRESHOLD < beta < MATE_STORE_THRESHOLD
        and not in_check
        and has_non_pawn_material(bb, _i64(meta[0]))
    ):
        null_meta = make_null_move(meta)
        null_score = -negamax(
            bb, null_meta, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, deadline, counters,
            ply + 1, history, child_hist_len,
            tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
            killer_from, killer_to, killer_promo, history_table, cont_hist, False, ext_budget,
            counter_from, counter_to, counter_promo, -1, -1, halfmove_clock + 1, -1, -1,
        )
        if counters[1]:
            return 0
        if null_score >= beta:
            return beta

    if hint_from < 0 and depth >= IID_MIN_DEPTH and not in_check and excluded_from < 0:
        # No hash move to order with, and deep enough that ordering is worth paying for: search
        # this same node at a reduced depth purely to populate one. The recursive call's own TT
        # store (same h, keyed off the exact same position) is the return channel -- re-probing
        # right after picks up whatever move it found, without threading a second return value
        # through negamax just for this.
        negamax(
            bb, meta, depth - IID_REDUCTION, alpha, beta, deadline, counters, ply, history,
            hist_len, tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
            killer_from, killer_to, killer_promo, history_table, cont_hist, allow_null, ext_budget,
            counter_from, counter_to, counter_promo, parent_from, parent_to, halfmove_clock,
            -1, -1,
        )
        if counters[1]:
            return 0
        if tt_key[slot_a] == h:
            hint_from, hint_to, hint_promo = int(tt_from[slot_a]), int(tt_to[slot_a]), int(
                tt_promo[slot_a]
            )
            hint_tt_depth, hint_tt_score, hint_tt_flag = (
                int(tt_depth[slot_a]) - 1, int(tt_score[slot_a]), int(tt_flag[slot_a]),
            )
        elif tt_key[slot_b] == h:
            hint_from, hint_to, hint_promo = int(tt_from[slot_b]), int(tt_to[slot_b]), int(
                tt_promo[slot_b]
            )
            hint_tt_depth, hint_tt_score, hint_tt_flag = (
                int(tt_depth[slot_b]) - 1, int(tt_score[slot_b]), int(tt_flag[slot_b]),
            )

    singular_extension = False
    if (
        excluded_from < 0
        and hint_from >= 0
        and depth >= SE_MIN_DEPTH
        and hint_tt_depth >= depth - SE_TT_DEPTH_MARGIN
        and hint_tt_flag != TT_UPPER
        and -MATE_STORE_THRESHOLD < hint_tt_score < MATE_STORE_THRESHOLD
    ):
        singular_beta = hint_tt_score - SE_MARGIN_PER_DEPTH * depth
        verify_score = negamax(
            bb, meta, (depth - 1) // 2, singular_beta - 1, singular_beta, deadline, counters,
            ply, history, hist_len,
            tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
            killer_from, killer_to, killer_promo, history_table, cont_hist, False, ext_budget,
            counter_from, counter_to, counter_promo, parent_from, parent_to, halfmove_clock,
            hint_from, hint_to,
        )
        if counters[1]:
            return 0
        if verify_score < singular_beta:
            singular_extension = True
        elif singular_beta >= beta:
            # Multi-cut: the verification search above already excludes the hash move and still
            # found some OTHER move reaching singular_beta, which here is itself >= the real beta
            # -- so at least two independent moves at this node can each reach beta on their own
            # (the hash move's own stored score backs a lower-bound/exact entry that cleared
            # hint_tt_score >= singular_beta + margin, and this other move clears singular_beta
            # too). That is normally strong enough evidence to cut the whole node without
            # searching the rest of it -- the same reduced-verification-search result the
            # singular-extension check above already paid for, just read the other way, so this
            # costs no extra search of its own. Returns beta itself (not the possibly-higher
            # singular_beta), the same conservative bound null-move pruning below returns on its
            # own cutoff.
            return beta

    k1_from = int(killer_from[ply, 0])
    k1_to, k1_promo = int(killer_to[ply, 0]), int(killer_promo[ply, 0])
    k2_from = int(killer_from[ply, 1])
    k2_to, k2_promo = int(killer_to[ply, 1]), int(killer_promo[ply, 1])
    if parent_from >= 0:
        cm_idx = parent_from * 64 + parent_to
        cm_from, cm_to, cm_promo = (
            int(counter_from[cm_idx]), int(counter_to[cm_idx]), int(counter_promo[cm_idx]),
        )
    else:
        cm_from, cm_to, cm_promo = -1, -1, -1
    prev_idx = -1
    if parent_from >= 0:
        prev_pt = piece_type_at(bb, 1 - _i64(meta[0]), parent_to)
        prev_idx = prev_pt * 64 + parent_to
    scores = _score_moves2(
        bb, meta, from_arr, to_arr, promo_arr, count,
        hint_from, hint_to, hint_promo,
        k1_from, k1_to, k1_promo, k2_from, k2_to, k2_promo,
        cm_from, cm_to, cm_promo,
        history_table, cont_hist, prev_idx,
    )
    best = -INF
    best_idx = -1
    # Phase 2.5 of docs/plan.md: pick the next-best remaining move one at a time (_argmax over the
    # not-yet-picked scores) instead of a single np.argsort(-scores) up front -- most nodes fail
    # high on the first few moves and never need the rest ordered. picked_idx/move_is_cap record
    # what _argmax picked (and whether it was a capture) at each position, since the history-malus
    # backward pass below needs to revisit earlier positions and there is no materialised order
    # array to read them back from anymore.
    picked_idx = np.empty(count, dtype=np.int64)
    move_is_cap = np.empty(count, dtype=np.bool_)
    for oi in range(count):
        idx = _argmax(scores, count)
        scores[idx] = _SELECTED
        picked_idx[oi] = idx
        f = _i64(from_arr[idx])
        t = _i64(to_arr[idx])
        p = _i64(promo_arr[idx])
        # is_capture is a pure function of (bb, meta, f, t) -- computed unconditionally (even for
        # an excluded move, below) so move_is_cap[oi] is always valid for the malus loop to read
        # back later, rather than recomputing it there a second (Phase 2.7 of docs/plan.md: this
        # used to run three times per cutoff move -- here, at the cutoff check, and in the malus
        # loop -- now once).
        is_cap = is_capture(bb, meta, f, t)
        move_is_cap[oi] = is_cap
        if excluded_from >= 0 and f == excluded_from and t == excluded_to:
            continue
        moving_pt = piece_type_at(bb, _i64(meta[0]), f)
        moved_pawn = moving_pt == PAWN
        child_halfmove_clock = 0 if (is_cap or moved_pawn) else halfmove_clock + 1

        if oi == 0:
            new_bb, new_meta = make_move(bb, meta, f, t, p)
            gives_check = is_check(new_bb, new_meta)
            if ext_budget > 0 and gives_check:
                child_depth = depth
                child_ext_budget = ext_budget - 1
            else:
                child_depth = depth - 1
                child_ext_budget = ext_budget
            if singular_extension and child_depth == depth - 1 and f == hint_from and t == hint_to:
                child_depth = depth
            score = -negamax(
                new_bb, new_meta, child_depth, -beta, -alpha, deadline, counters,
                ply + 1, history, child_hist_len,
                tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                killer_from, killer_to, killer_promo, history_table, cont_hist, True,
                child_ext_budget,
                counter_from, counter_to, counter_promo, f, t, child_halfmove_clock, -1, -1,
            )
        else:
            # Phase 2.6 of docs/plan.md: futility and LMP both used to run only after make_move (a
            # full board copy) + is_check had already been paid for on every single move, purely to
            # learn gives_check for a check that then often just skips the move anyway. Try the
            # cheap, direct-check-only _gives_check_direct first (only meaningful when both prunes'
            # other guards already hold); if it says the move survives, or isn't even a candidate
            # for either prune, fall through to the real make_move + is_check below exactly as
            # before -- this only ever changes the cost of a move that ends up skipped, never the
            # accuracy of one that gets searched.
            if (
                is_cap
                and not in_check
                and depth <= SEE_PRUNE_MAX_DEPTH
                and see(bb, meta, f, t, p) < -SEE_PRUNE_MARGIN * depth
            ):
                # SEE pruning: this capture loses material by more than a depth-scaled margin --
                # quiescence already prunes exactly this shape of move at its own leaves (see
                # DELTA_MARGIN), this just applies the same reasoning one ply higher, in the main
                # search, where such a capture would otherwise be fully searched.
                continue

            if p < 0 and not in_check and not is_cap:
                gives_check_direct = _gives_check_direct(bb, meta, f, t, p, moving_pt)
                if not gives_check_direct:
                    if futile:
                        # A quiet move at a shallow, non-check node when even the best case
                        # (static eval plus a depth-scaled margin) can't reach alpha -- skip it
                        # outright rather than spend a board copy and a search on it. oi == 0 (the
                        # hash/best-ordered move) is never skipped, so this node always has at
                        # least one fully-searched move to report a score from.
                        continue
                    if depth <= LMP_MAX_DEPTH and oi >= LMP_THRESHOLD[depth]:
                        # Late move pruning: this far into the ordering at a shallow depth,
                        # TT/killer/history/counter-move ordering has already almost certainly put
                        # every move worth searching ahead of this one -- skip outright rather than
                        # even the reduced-depth probe LMR below would spend on it. LMP_THRESHOLD
                        # grows with depth so a deeper (more expensive, more trustworthy) node
                        # tolerates more late moves before pruning.
                        continue

            new_bb, new_meta = make_move(bb, meta, f, t, p)
            gives_check = is_check(new_bb, new_meta)
            if ext_budget > 0 and gives_check:
                child_depth = depth
                child_ext_budget = ext_budget - 1
            else:
                child_depth = depth - 1
                child_ext_budget = ext_budget

            reduction = 0
            if (
                oi >= LMR_MIN_MOVE_INDEX
                and depth >= LMR_MIN_DEPTH
                and p < 0
                and not in_check
                and not is_cap
                and not gives_check
            ):
                d_idx = depth if depth < LMR_TABLE_MAX_DEPTH else LMR_TABLE_MAX_DEPTH
                m_idx = oi if oi < LMR_TABLE_MAX_MOVE_INDEX else LMR_TABLE_MAX_MOVE_INDEX
                reduction = int(LMR_TABLE[d_idx, m_idx])
                h_score = history_table[f * 64 + t]
                if prev_idx >= 0:
                    h_score += cont_hist[prev_idx, moving_pt * 64 + t]
                if h_score >= LMR_HISTORY_HIGH:
                    reduction = max(0, reduction - 1)
                elif h_score <= LMR_HISTORY_LOW:
                    reduction += 1

            score = -negamax(
                new_bb, new_meta, child_depth - reduction, -alpha - 1, -alpha, deadline, counters,
                ply + 1, history, child_hist_len,
                tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                killer_from, killer_to, killer_promo, history_table, cont_hist, True,
                child_ext_budget,
                counter_from, counter_to, counter_promo, f, t, child_halfmove_clock, -1, -1,
            )
            if not counters[1] and reduction > 0 and score > alpha:
                # the reduced-depth probe suggested this late, quiet move might actually be
                # good -- confirm at full depth (still null window) before trusting it enough
                # to maybe trigger the full-window PVS re-search below.
                score = -negamax(
                    new_bb, new_meta, child_depth, -alpha - 1, -alpha, deadline, counters,
                    ply + 1, history, child_hist_len,
                    tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                    killer_from, killer_to, killer_promo, history_table, cont_hist, True,
                    child_ext_budget,
                    counter_from, counter_to, counter_promo, f, t, child_halfmove_clock, -1, -1,
                )
            if not counters[1] and alpha < score < beta:
                score = -negamax(
                    new_bb, new_meta, child_depth, -beta, -alpha, deadline, counters,
                    ply + 1, history, child_hist_len,
                    tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                    killer_from, killer_to, killer_promo, history_table, cont_hist, True,
                    child_ext_budget,
                    counter_from, counter_to, counter_promo, f, t, child_halfmove_clock, -1, -1,
                )
        if counters[1]:
            return 0
        if score > best:
            best = score
            best_idx = idx
        if best > alpha:
            alpha = best
        if alpha >= beta:
            if p < 0 and not is_cap:
                bonus = depth * depth
                if not (f == k1_from and t == k1_to and p == k1_promo):
                    killer_from[ply, 1], killer_to[ply, 1], killer_promo[ply, 1] = (
                        killer_from[ply, 0], killer_to[ply, 0], killer_promo[ply, 0],
                    )
                    killer_from[ply, 0], killer_to[ply, 0], killer_promo[ply, 0] = f, t, p
                history_table[f * 64 + t] += bonus
                if prev_idx >= 0:
                    cont_hist[prev_idx, moving_pt * 64 + t] += bonus
                if parent_from >= 0:
                    cm_idx = parent_from * 64 + parent_to
                    counter_from[cm_idx] = np.int8(f)
                    counter_to[cm_idx] = np.int8(t)
                    counter_promo[cm_idx] = np.int8(p)
                # History malus: every other quiet move already tried at this node had the same
                # chance to cut off and didn't, so it is penalised by the same magnitude the
                # actual cutoff move is rewarded by -- not just "not rewarded", actively pushed
                # below untried (history == 0) moves next time. Sharpens ordering everywhere,
                # which matters most in exactly the sharp, many-candidate tactical positions this
                # tier exists for. cont_hist gets the same treatment, keyed by each malus move's
                # own [piece type, to-square] rather than the cutoff move's.
                for oi2 in range(oi):
                    idx2 = picked_idx[oi2]
                    f2 = _i64(from_arr[idx2])
                    t2 = _i64(to_arr[idx2])
                    p2 = _i64(promo_arr[idx2])
                    if p2 < 0 and not move_is_cap[oi2]:
                        history_table[f2 * 64 + t2] -= bonus
                        if prev_idx >= 0:
                            pt2 = piece_type_at(bb, _i64(meta[0]), f2)
                            cont_hist[prev_idx, pt2 * 64 + t2] -= bonus
            break

    # Two-tier replacement: prefer the depth-preferred slot on a same-key refresh, an empty slot
    # (tt_depth == 0, the new_tt() sentinel now that a real entry stores depth + 1 -- see
    # docs/plan.md Phase 1.8), or a search that went at least as deep as what is already there;
    # otherwise fall back to the always-replace slot so this node's result is still cached even
    # though it didn't earn the depth-preferred one. Skipped entirely for an excluded-move search:
    # that result reflects only the moves other than excluded_from/to, not this position's real
    # value, so storing it under the position's real key would corrupt every future non-excluded
    # probe of it.
    if excluded_from < 0:
        if tt_key[slot_a] == h or tt_depth[slot_a] == 0 or depth + 1 >= tt_depth[slot_a]:
            write_idx = slot_a
        else:
            write_idx = slot_b

        tt_key[write_idx] = h
        tt_depth[write_idx] = depth + 1
        if best >= MATE_STORE_THRESHOLD:
            tt_score[write_idx] = best + ply
        elif best <= -MATE_STORE_THRESHOLD:
            tt_score[write_idx] = best - ply
        else:
            tt_score[write_idx] = best
        if best <= orig_alpha:
            tt_flag[write_idx] = TT_UPPER
        elif best >= beta:
            tt_flag[write_idx] = TT_LOWER
        else:
            tt_flag[write_idx] = TT_EXACT
        if best_idx != -1:
            tt_from[write_idx] = np.int8(from_arr[best_idx])
            tt_to[write_idx] = np.int8(to_arr[best_idx])
            tt_promo[write_idx] = np.int8(promo_arr[best_idx])

    return best


@njit(cache=False)
def claim_eligible_for_opponent(
    bb: np.ndarray,
    meta: np.ndarray,
    history: np.ndarray,
    hist_len: int,
    opponent_history: np.ndarray,
    opponent_hist_len: int,
    lookahead: int,
) -> bool:
    """True if handing over the position `bb`/`meta` (the opponent's turn, right after one of
    our candidate moves) would let the harness's referee auto-draw before we are ever asked to
    move again -- python-chess's can_claim_threefold_repetition(), which Board.outcome(claim_draw
    =True) calls before asking either side for a move, matches on two conditions:

    (a) this exact position has already occurred twice before (this handoff would be the 3rd) --
        checked directly against opponent_history, the real game's own past opponent-turn
        positions (recorded once per move we actually play, in agent.py).
    (b) the opponent has some legal reply that would make a position recur a third time --
        checked by generating their replies and matching each against `history`, our own
        real-plus-in-search-so-far turn positions (see this module's docstring).

    Neither depends on what the opponent would actually choose to play, so a move of ours
    creating either condition must be scored as if it directly caused the draw.

    That is still not the whole story: the same auto-draw applies to *our* next position too, and
    it applies before we are ever consulted -- so even if handing over `bb`/`meta` looks safe, an
    opponent reply we do not control could land us on a position where the referee ends the game
    on our own next (still unconsulted) turn. `lookahead` recurses one call further per unit,
    with the two history arrays swapped, to check exactly that for each of the opponent's
    replies; two real games were needed to find these two conditions plus this recursive case
    between them (see agent.py's module docstring), so this is deliberately bounded rather than
    assumed complete for arbitrary depth -- lookahead=1, used at the call sites in search_root /
    agent._search_restricted, closes both games actually observed while keeping the cost bounded
    (branching^2 per candidate root move, still small at these endgame piece counts).

    Deliberately root-only (called from search_root / agent._search_restricted, not from inside
    negamax's own recursion): each level is a full extra generate_legal, affordable once per real
    move decision but not throughout the tree. Never consults the transposition table -- always
    recomputed from the real history arrays, so a cached search score can never mask a claim.
    """
    h = position_hash(bb, meta)
    matches = 0
    for j in range(opponent_hist_len):
        if opponent_history[j] == h:
            matches += 1
    if matches >= 2:
        return True

    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    for i in range(count):
        new_bb, new_meta = make_move(
            bb, meta, _i64(from_arr[i]), _i64(to_arr[i]), _i64(promo_arr[i])
        )
        reply_hash = position_hash(new_bb, new_meta)
        reply_matches = 0
        for j in range(hist_len):
            if history[j] == reply_hash:
                reply_matches += 1
        if reply_matches >= 2:
            return True
        if lookahead > 0 and claim_eligible_for_opponent(
            new_bb, new_meta, opponent_history, opponent_hist_len, history, hist_len,
            lookahead - 1,
        ):
            return True
    return False


@njit(cache=False)
def _search_root_pass(
    bb: np.ndarray,
    meta: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    deadline: float,
    counters: np.ndarray,
    history: np.ndarray,
    hist_len: int,
    opponent_history: np.ndarray,
    opponent_hist_len: int,
    tt_key: np.ndarray,
    tt_depth: np.ndarray,
    tt_score: np.ndarray,
    tt_flag: np.ndarray,
    tt_from: np.ndarray,
    tt_to: np.ndarray,
    tt_promo: np.ndarray,
    killer_from: np.ndarray,
    killer_to: np.ndarray,
    killer_promo: np.ndarray,
    history_table: np.ndarray,
    cont_hist: np.ndarray,
    counter_from: np.ndarray,
    counter_to: np.ndarray,
    counter_promo: np.ndarray,
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    promo_arr: np.ndarray,
    order: np.ndarray,
    count: int,
    halfmove_clock: int,
) -> tuple[int, int, int, int, bool]:
    """One root pass over already-ordered moves within a fixed [alpha, beta] window -- factored
    out of search_root so aspiration windows (see there) can re-run this at the same depth with a
    wider window on a fail-high/fail-low without re-ordering or re-generating moves.
    """
    best_score = -INF
    best_from = _i64(from_arr[order[0]])
    best_to = _i64(to_arr[order[0]])
    best_promo = _i64(promo_arr[order[0]])

    for oi in range(count):
        idx = order[oi]
        f = _i64(from_arr[idx])
        t = _i64(to_arr[idx])
        p = _i64(promo_arr[idx])
        child_halfmove_clock = (
            0 if (is_capture(bb, meta, f, t) or piece_type_at(bb, _i64(meta[0]), f) == PAWN)
            else halfmove_clock + 1
        )
        new_bb, new_meta = make_move(bb, meta, f, t, p)
        child_depth = depth if is_check(new_bb, new_meta) else depth - 1
        child_ext_budget = MAX_CHECK_EXTENSIONS - (1 if child_depth == depth else 0)
        if oi == 0:
            score = -negamax(
                new_bb, new_meta, child_depth, -beta, -alpha, deadline, counters, 1, history,
                hist_len, tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                killer_from, killer_to, killer_promo, history_table, cont_hist, True,
                child_ext_budget,
                counter_from, counter_to, counter_promo, f, t, child_halfmove_clock, -1, -1,
            )
        else:
            score = -negamax(
                new_bb, new_meta, child_depth, -alpha - 1, -alpha, deadline, counters, 1, history,
                hist_len, tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                killer_from, killer_to, killer_promo, history_table, cont_hist, True,
                child_ext_budget,
                counter_from, counter_to, counter_promo, f, t, child_halfmove_clock, -1, -1,
            )
            if not counters[1] and alpha < score < beta:
                score = -negamax(
                    new_bb, new_meta, child_depth, -beta, -alpha, deadline, counters, 1, history,
                    hist_len, tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
                    killer_from, killer_to, killer_promo, history_table, cont_hist, True,
                    child_ext_budget,
                    counter_from, counter_to, counter_promo, f, t, child_halfmove_clock, -1, -1,
                )
        if counters[1]:
            return best_from, best_to, best_promo, best_score, False
        if claim_eligible_for_opponent(
            new_bb, new_meta, history, hist_len, opponent_history, opponent_hist_len,
            _LOOKAHEAD_ONE,
        ):
            score = min(score, 0)
        if score > best_score:
            best_score = score
            best_from, best_to, best_promo = f, t, p
        if best_score > alpha:
            alpha = best_score

    return best_from, best_to, best_promo, best_score, True


@njit(_SEARCH_ROOT_SIG, cache=False)
def search_root(
    bb: np.ndarray,
    meta: np.ndarray,
    depth: int,
    deadline: float,
    counters: np.ndarray,
    pv_from: int,
    pv_to: int,
    pv_promo: int,
    history: np.ndarray,
    hist_len: int,
    opponent_history: np.ndarray,
    opponent_hist_len: int,
    tt_key: np.ndarray,
    tt_depth: np.ndarray,
    tt_score: np.ndarray,
    tt_flag: np.ndarray,
    tt_from: np.ndarray,
    tt_to: np.ndarray,
    tt_promo: np.ndarray,
    killer_from: np.ndarray,
    killer_to: np.ndarray,
    killer_promo: np.ndarray,
    history_table: np.ndarray,
    cont_hist: np.ndarray,
    counter_from: np.ndarray,
    counter_to: np.ndarray,
    counter_promo: np.ndarray,
    prev_score: int,
    halfmove_clock: int,
) -> tuple[int, int, int, int, bool]:
    """hist_len counts our-turn positions already recorded, including the root's own -- the
    caller (agent.py) writes history[hist_len - 1] = hash(bb, meta) before calling this, since it
    already needs that hash for the persistent cross-move history anyway. opponent_hist_len
    counts the real game's own past opponent-turn positions, one recorded per move we have
    actually played -- see claim_eligible_for_opponent. tt_* / killer_* / history_table /
    counter_* are threaded straight into negamax; see this module's docstring. prev_score is the
    previous iterative-deepening depth's score, or NO_PREV_SCORE if there isn't one (depth 1) --
    used to center the aspiration window (see module-level constants). halfmove_clock is the real
    game's own fifty-move-rule count (bitboard.halfmove_clock(fen), read by agent.py) -- see this
    module's docstring's "Fifty-move rule" paragraph.
    """
    from_arr, to_arr, promo_arr, count = generate_legal(bb, meta)
    if count == 0:
        return -1, -1, -1, 0, False

    scores = _score_moves(bb, meta, from_arr, to_arr, promo_arr, count, pv_from, pv_to, pv_promo)
    order = np.argsort(-scores)

    use_aspiration = (
        depth > 2 and prev_score != NO_PREV_SCORE and abs(prev_score) < MATE_STORE_THRESHOLD
    )
    window_idx = 0
    if use_aspiration:
        alpha = max(-INF, prev_score - ASPIRATION_WINDOWS[window_idx])
        beta = min(INF, prev_score + ASPIRATION_WINDOWS[window_idx])
    else:
        alpha, beta = -INF, INF

    while True:
        best_from, best_to, best_promo, best_score, completed = _search_root_pass(
            bb, meta, depth, alpha, beta, deadline, counters,
            history, hist_len, opponent_history, opponent_hist_len,
            tt_key, tt_depth, tt_score, tt_flag, tt_from, tt_to, tt_promo,
            killer_from, killer_to, killer_promo, history_table, cont_hist,
            counter_from, counter_to, counter_promo,
            from_arr, to_arr, promo_arr, order, count, halfmove_clock,
        )
        if not completed:
            return best_from, best_to, best_promo, best_score, False
        fail_low = best_score <= alpha and alpha > -INF
        fail_high = best_score >= beta and beta < INF
        if fail_low or fail_high:
            window_idx += 1
            if window_idx < len(ASPIRATION_WINDOWS):
                # Widen only the side that actually failed -- Phase 4.2 of docs/plan.md: the other
                # side already bounded the real score on this pass, so keeping it tight instead of
                # re-centering both bounds on prev_score avoids re-searching a window this pass
                # already proved is wide enough on that side.
                if fail_low:
                    alpha = max(-INF, prev_score - ASPIRATION_WINDOWS[window_idx])
                else:
                    beta = min(INF, prev_score + ASPIRATION_WINDOWS[window_idx])
            else:
                alpha, beta = -INF, INF
            continue
        return best_from, best_to, best_promo, best_score, True


@njit(cache=False)
def quick_best_move(
    bb: np.ndarray,
    meta: np.ndarray,
    from_arr: np.ndarray,
    to_arr: np.ndarray,
    promo_arr: np.ndarray,
    count: int,
) -> tuple[int, int, int]:
    """The best-looking legal move by MVV-LVA/promotion score alone, no search. Never recurses,
    so it has no way to overrun a deadline -- the fallback for when there is no time to search.
    """
    scores = _score_moves(
        bb, meta, from_arr, to_arr, promo_arr, count,
        _NO_PV, _NO_PV, _NO_PV,
    )
    best_idx = 0
    best_score = scores[0]
    for i in range(1, count):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    return from_arr[best_idx], to_arr[best_idx], promo_arr[best_idx]


def new_counters() -> np.ndarray:
    return np.zeros(2, dtype=np.int64)


def new_tt() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """A fresh, empty transposition table -- parallel arrays, two raw slots per bucket (see this
    module's docstring). Stored depth is the real search depth plus one, so 0 (not -1) marks an
    empty slot: a real entry's stored depth is always >= 1 (negamax only ever reaches the store
    site with depth >= 1, the depth <= 0 case having already returned via quiescence), which lets
    every array here be zero-initialised (a lazy calloc) instead of eagerly written by np.full --
    proportionally less RSS at whatever TT_BUCKETS currently is, no behaviour change (key is
    separately checked for a match anyway, and every stored-depth read subtracts 1 back off --
    see negamax).
    """
    key = np.zeros(TT_SIZE, dtype=np.uint64)
    depth = np.zeros(TT_SIZE, dtype=np.int32)
    score = np.zeros(TT_SIZE, dtype=np.int32)
    flag = np.zeros(TT_SIZE, dtype=np.int8)
    move_from = np.zeros(TT_SIZE, dtype=np.int8)
    move_to = np.zeros(TT_SIZE, dtype=np.int8)
    move_promo = np.zeros(TT_SIZE, dtype=np.int8)
    return key, depth, score, flag, move_from, move_to, move_promo


def new_killers() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    killer_from = np.full((MAX_KILLER_PLY, 2), -1, dtype=np.int8)
    killer_to = np.full((MAX_KILLER_PLY, 2), -1, dtype=np.int8)
    killer_promo = np.full((MAX_KILLER_PLY, 2), -1, dtype=np.int8)
    return killer_from, killer_to, killer_promo


def new_history_table() -> np.ndarray:
    return np.zeros(64 * 64, dtype=np.int32)


# Phase 4.2 of docs/plan.md: continuation history. history_table's from/to score and the
# counter-move table (below) both only look at the CURRENT move in isolation (from/to) or at the
# single most recent cutoff reply to a given parent move; continuation history instead accumulates
# a graduated score for (parent move's [piece type, to-square], this move's [piece type,
# to-square]), so a quiet move that has repeatedly worked well as a *reply to this specific kind of
# parent move* outranks one that merely has a good from/to score in general. Indexed by piece type
# (not from-square) for the parent half since what matters is what piece just arrived on
# parent_to, not where it came from. Folded into history_table's own quiet-move fallback score in
# _score_moves2 (added, not compared against), same "one extra signal, not a replacement" posture
# _score_moves2's counter-move/killer bands already take relative to history_table.
CONT_HIST_SIZE = 6 * 64


def new_continuation_history() -> np.ndarray:
    return np.zeros((CONT_HIST_SIZE, CONT_HIST_SIZE), dtype=np.int32)


def new_counter_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Counter-move table: indexed by (parent_from * 64 + parent_to), the move that led to a
    node, this stores whatever quiet move most recently caused a beta cutoff in reply to it --
    see negamax's parent_from/parent_to parameters and _score_moves2's cm_from/cm_to/cm_promo.
    Rebuilt fresh per real move decision, same as killers and history_table: it encodes this
    search's own cutoff history, not the position itself.
    """
    counter_from = np.full(64 * 64, -1, dtype=np.int8)
    counter_to = np.full(64 * 64, -1, dtype=np.int8)
    counter_promo = np.full(64 * 64, -1, dtype=np.int8)
    return counter_from, counter_to, counter_promo
