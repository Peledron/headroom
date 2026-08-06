"""Optimal cache-breakpoint placement over conversation history.

Anthropic allows at most 4 ``cache_control`` breakpoints per request. When a
prefix change busts the cache at message depth ``c``, the provider's longest
matching prefix ends at the deepest breakpoint at or above ``c``, so every
unchanged token between that breakpoint and ``c`` is re-written at the write
premium even though its bytes did not move. The controllable loss of an anchor
set ``A`` is therefore::

    E_c[ c - anchor_below(c, A) ]

with ``c`` drawn from where this session's churn actually lands (the
``PrefixCacheTracker.churn_depth_samples`` ring, fed by structural bust
detection). Minimizing that expectation over up to ``k`` anchor positions is a
one-dimensional k-segmentation problem, solved exactly here by dynamic
programming over a quantized candidate grid. The HR_MID_ANCHOR heuristic (one
fixed anchor near mid-depth) is the k=1, no-data special case.

Positions are quantized to a coarse grid for the same reason mid_anchor
quantizes: an anchor that drifts with every appended message is itself a new
byte at a new position each turn, which busts the very cache it protects. The
grid keeps anchors byte-stable between growth jumps.

Pure functions, no proxy state. The handler owns attaching the markers and
respecting the provider's 4-breakpoint budget.
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from dataclasses import dataclass

MIN_DEPTH = 32
"""Shallowest anchorable message index. Above this the head breakpoints the
client already places cover the loss, and an anchor would waste budget."""

TAIL_GUARD = 16
"""Messages at the tail left unanchored. The tail churns every turn by
construction (it is the active exchange), so an anchor there never survives."""

QUANTUM = 64
"""Minimum separation between two live anchors, and the HR_MID_ANCHOR grid.

Two anchors closer than this split the same churn region twice, spending a
scarce breakpoint on a segment the neighbouring anchor already covers.
"""

CANDIDATE_STEP = 16
"""Grid step for DP candidate positions.

This used to equal :data:`QUANTUM`, which made the DP a no-op on ordinary
sessions: with a 64-step grid, a 32 floor and a 16 tail guard, every
conversation between 80 and 143 messages has exactly one candidate, so the
argmin had nothing to choose and always returned the heuristic's answer.
Measured over 1321 production turns, the DP proposed a position differing from
the heuristic on 5 percent of turns, and the remainder of the observed spread
was the handler's anchorable-message backscan, not the DP.

The coarse grid was there for byte-stability, on the reasoning that an anchor
drifting every turn busts the cache it protects. That job now belongs to
:func:`stabilize_anchor_depths`, which prices every move against the rewrite it
forces and refuses the ones that do not clear it. With the price gate in place
the grid no longer has to enforce stability by being blunt, so it is refined to
where the DP can actually express a preference.
"""


def fallback_anchor_depth(n_messages: int) -> int:
    """The classic HR_MID_ANCHOR position: quantized, tail-guarded mid-depth."""
    anchor = max(MIN_DEPTH, (n_messages - TAIL_GUARD) // QUANTUM * QUANTUM)
    return min(anchor, n_messages - TAIL_GUARD)


def _candidate_positions(n_messages: int) -> list[int]:
    """Quantized grid of anchorable depths, oldest to newest."""
    ceiling = n_messages - TAIL_GUARD
    return [p for p in range(CANDIDATE_STEP, ceiling + 1, CANDIDATE_STEP) if p >= MIN_DEPTH]


class _RewriteScale:
    """Converts a pair of message depths into what re-writing between them costs.

    The loss an anchor set controls is billed in tokens, but the DP's natural
    coordinate is the message index. Those two only agree when messages are the
    same size, and they are not: one tool result can outweigh thirty user turns.
    Scoring in messages therefore optimizes a proxy that can point away from the
    bill, so when per-message token estimates are available they are used and
    the depth coordinate is mapped through their cumulative sum.

    Without weights this degrades exactly to the old message-count distance, so
    callers with no token estimate keep the previous behaviour.
    """

    __slots__ = ("_cum", "_n")

    def __init__(self, n_messages: int, weights: list[float] | None = None) -> None:
        self._n = max(0, int(n_messages))
        if weights and len(weights) >= self._n and self._n > 0:
            cum = [0.0] * (self._n + 1)
            for i in range(self._n):
                cum[i + 1] = cum[i] + max(0.0, float(weights[i]))
            self._cum = cum if cum[self._n] > 0 else None
        else:
            self._cum = None

    def at(self, depth: float) -> float:
        """Cumulative cost of the prefix ending at ``depth``.

        Churn samples arrive as a fraction of the history and so land between
        message boundaries. Truncating to the message below drops that
        remainder, which both understates the depth of a sample and, under
        uniform weights, leaves a floor the unweighted path does not apply.
        The two coordinates then disagree on near-ties for no better reason
        than which branch ran. Charging the partial message pro rata keeps the
        uniform case an exact multiple of the message distance.
        """
        if self._cum is None:
            return float(depth)
        if depth <= 0:
            return 0.0
        if depth >= self._n:
            return self._cum[self._n]
        index = int(depth)
        remainder = depth - index
        base = self._cum[index]
        if remainder <= 0.0:
            return base
        return base + remainder * (self._cum[index + 1] - base)

    def span(self, low: float, high: float) -> float:
        """Cost of re-writing everything between two depths."""
        return max(0.0, self.at(high) - self.at(low))


def optimal_anchor_depths(
    n_messages: int,
    churn_fractions: list[float],
    k_anchors: int,
    message_weights: list[float] | None = None,
) -> list[int]:
    """Place up to ``k_anchors`` breakpoints to minimize expected rewrite depth.

    ``churn_fractions`` are depth fractions (0 = head, 1 = tail) of observed
    structural busts. Each is scaled to this request's message count: churn is
    an artifact of the client's behavior at a relative depth (hook injection,
    reminder rewriting), so the fraction transfers across turns better than an
    absolute index does.

    With no observations the empirical objective is undefined, so the single
    mid-depth heuristic anchor is returned; the caller needs no separate
    fallback path. With observations, an exact segmentation DP runs over the
    quantized grid: ``f[i][j]`` is the minimal loss over samples below grid
    position ``i`` with the ``j``-th anchor placed exactly there. Grid sizes
    are tiny (hundreds of messages / QUANTUM), so the cubic-ish table is
    microseconds, not a hot-path concern.

    Returns sorted ascending depths, possibly fewer than ``k_anchors`` when
    the grid is smaller than the budget. Empty when nothing is anchorable.
    """
    if k_anchors <= 0 or n_messages <= MIN_DEPTH + TAIL_GUARD:
        return []
    candidates = _candidate_positions(n_messages)
    if not candidates:
        return []
    if not churn_fractions:
        return [fallback_anchor_depth(n_messages)][:k_anchors]

    samples = sorted(
        min(max(f, 0.0), 1.0) * n_messages for f in churn_fractions
    )
    k = min(k_anchors, len(candidates))
    scale = _RewriteScale(n_messages, message_weights)

    # Prefix sums over the samples' scaled positions. segment_loss is called
    # once per (anchor, bound) pair in a cubic-ish table, so evaluating it by
    # scanning the sample ring made the DP quadratic in the ring size. The
    # refined candidate grid multiplies that table by sixteen, which the linear
    # scan could not afford, and the sums make each call two bisects.
    scaled = [scale.at(c) for c in samples]
    running = [0.0] * (len(scaled) + 1)
    for i, value in enumerate(scaled):
        running[i + 1] = running[i] + value

    def segment_loss(anchor: float, bound: float) -> float:
        """Loss from samples in [anchor, bound): rewrite cost back to the anchor."""
        low = bisect_left(samples, anchor)
        high = bisect_left(samples, bound)
        if high <= low:
            return 0.0
        return (running[high] - running[low]) - (high - low) * scale.at(anchor)

    # Boundary sentinels: position 0 stands for the head breakpoints (always
    # a valid fallback prefix), the far bound closes the last segment.
    far = float(n_messages) + 1.0
    m = len(candidates)
    inf = float("inf")
    # f[j][i]: minimal loss over samples below candidates[i], using j anchors,
    # the j-th sitting at candidates[i].
    f = [[inf] * m for _ in range(k + 1)]
    for i, pos in enumerate(candidates):
        f[1][i] = segment_loss(0.0, pos)
    for j in range(2, k + 1):
        for i, pos in enumerate(candidates):
            for h in range(i):
                prev = f[j - 1][h]
                if prev >= inf:
                    continue
                cost = prev + segment_loss(candidates[h], pos)
                if cost < f[j][i]:
                    f[j][i] = cost
    # Close the final segment and pick the best anchor count <= k: an unused
    # anchor can never raise the loss, but a sparse sample set can make two
    # anchors tie with three, so scan all j.
    best_loss = inf
    best_choice: tuple[int, int] | None = None
    for j in range(1, k + 1):
        for i, pos in enumerate(candidates):
            if f[j][i] >= inf:
                continue
            total = f[j][i] + segment_loss(pos, far)
            if total < best_loss:
                best_loss = total
                best_choice = (j, i)
    if best_choice is None:
        return [fallback_anchor_depth(n_messages)][:k_anchors]

    # Recover the argmin path by re-walking the table.
    j, i = best_choice
    chosen = [candidates[i]]
    while j > 1:
        pos = candidates[i]
        target = f[j][i]
        for h in range(i):
            if (
                f[j - 1][h] < inf
                and abs(f[j - 1][h] + segment_loss(candidates[h], pos) - target)
                < 1e-9
            ):
                j, i = j - 1, h
                chosen.append(candidates[i])
                break
        else:  # pragma: no cover - table invariant: a predecessor always exists
            break
    return sorted(chosen)


EXPECTED_REMAINING_BUSTS = 4.0
"""How many future busts a re-placed anchor is assumed to still serve.

A move pays its rewrite once and collects its loss improvement once per later
bust, so the two are only comparable across a horizon. Four is the conservative
end of the observed per-session bust count: overstating it would let a marginal
gain buy a certain rewrite, which is the failure this gate exists to stop.
"""


def _anchor_below(position: float, anchors: list[int]) -> float:
    """Deepest anchor at or above ``position``, or the head at 0."""
    best = 0.0
    for anchor in anchors:
        if anchor <= position and anchor > best:
            best = float(anchor)
    return best


def _mean_rewrite_depth(
    anchors: list[int],
    samples: list[float],
    scale: _RewriteScale | None = None,
) -> float:
    """Average cost re-written per bust under this anchor set.

    In messages when no scale is supplied, in estimated tokens when one is.
    """
    if not samples:
        return 0.0
    if scale is None:
        return sum(c - _anchor_below(c, anchors) for c in samples) / len(samples)
    return sum(scale.span(_anchor_below(c, anchors), c) for c in samples) / len(samples)


def _move_write_cost(old: int, new: int, scale: _RewriteScale | None = None) -> float:
    """Cost re-written at the 1h premium by relocating a live anchor.

    Asymmetric on purpose. Moving forward keeps the old entry as a matching
    prefix, so only the gap between the two positions is new. Moving backward
    lands on a boundary no cached entry ends at, so the whole prefix below the
    new position is written again. That asymmetry is why an anchor that walks
    back and forth across a grid boundary is so expensive: the return leg pays
    for everything, not just the distance travelled.
    """
    if scale is None:
        return float(new - old if new > old else new)
    return scale.span(old, new) if new > old else scale.at(new)


def stabilize_anchor_depths(
    previous_depths: list[int],
    proposed_depths: list[int],
    n_messages: int,
    churn_fractions: list[float],
    k_anchors: int,
    message_weights: list[float] | None = None,
) -> list[int]:
    """Hold live anchors in place unless moving them is priced worth it.

    :func:`optimal_anchor_depths` re-solves from scratch every turn against two
    inputs that both move: churn fractions are rescaled by the current message
    count, and the sample ring itself turns over. The argmin therefore walks
    even when nothing about the conversation changed, and the QUANTUM grid only
    defers the walk until the drift crosses a boundary, at which point the
    anchor jumps a full quantum. Measured over 825 steady-state turns of
    production traffic, that produced 75 anchor moves and 1.59M tokens of 1h
    cache write, 37 forward and 38 backward, in oscillations of 62 to 65
    messages. An anchor exists to be a stable prefix, so re-placing it must
    clear the rewrite it forces, not merely improve an expectation.

    Previous anchors that are still anchorable are kept. Spare budget goes to
    proposed positions that are not already covered. A previous anchor is only
    displaced when the loss it would shed, over
    :data:`EXPECTED_REMAINING_BUSTS`, beats what the move re-writes.
    """
    if k_anchors <= 0:
        return []
    ceiling = n_messages - TAIL_GUARD
    live = sorted({d for d in previous_depths if MIN_DEPTH <= d <= ceiling})
    proposed = sorted({d for d in proposed_depths if MIN_DEPTH <= d <= ceiling})
    if not live:
        return proposed[:k_anchors]

    samples = sorted(min(max(f, 0.0), 1.0) * n_messages for f in churn_fractions)
    scale = _RewriteScale(n_messages, message_weights)
    kept = live[:k_anchors]
    # Spend leftover budget first. Adding an anchor never displaces a cached
    # prefix, so it needs no price gate beyond not duplicating one we hold.
    for depth in proposed:
        if len(kept) >= k_anchors:
            break
        if all(abs(depth - held) >= QUANTUM for held in kept):
            kept.append(depth)
    kept.sort()
    if not samples:
        return kept

    # With the budget full, a proposed position can only enter by evicting a
    # live one. Every swap is priced against the same baseline and at most one
    # is applied: two moves in a turn would each pay a rewrite while only the
    # combined placement was ever costed, which is how an oscillation starts.
    #
    # A relocation must also clear QUANTUM, independently of what it prices at.
    # Churn held at a constant fraction of a growing history sits at a
    # forward-marching absolute message, so the argmin advances a little every
    # turn and each small step prices well on its own: shedding sixteen messages
    # of expected depth over four future busts easily covers writing the
    # sixteen-message gap once. Followed every turn that is a treadmill, paying
    # a rewrite per step to chase a target that never stops moving. The coarse
    # candidate grid used to damp this implicitly by making the DP incapable of
    # proposing a near neighbour; now that the grid is refined for placement
    # accuracy, the hysteresis has to be stated where it actually belongs, on
    # moving a live anchor rather than on choosing where to put a new one.
    baseline_loss = _mean_rewrite_depth(kept, samples, scale)
    best: list[int] | None = None
    best_loss = baseline_loss
    for depth in proposed:
        if depth in kept:
            continue
        for held in kept:
            if abs(depth - held) < QUANTUM:
                continue
            candidate = sorted([d for d in kept if d != held] + [depth])
            loss = _mean_rewrite_depth(candidate, samples, scale)
            gain = (baseline_loss - loss) * EXPECTED_REMAINING_BUSTS
            if gain > _move_write_cost(held, depth, scale) and loss < best_loss:
                best, best_loss = candidate, loss
    return best if best is not None else kept


_KEEP_WARM_PRICE_PER_TOKEN_TURN = 0.1
"""Price of keeping one history token warm for one expected future turn,
relative to the write premium below."""

_ABSTRACT_BUST_PRICE = 1.25
"""Price of a summary-triggered suffix bust, per replaced-plus-busted token.
Matches the provider's 5m cache write multiplier, the closest priced analogue
to a forced rewrite of the summarized span and everything after it."""


@dataclass(frozen=True)
class AbstractVsKeepWarmDecision:
    """Priced comparison between two futures for the same history span.

    ``keep_warm_cost`` is what the span costs left untouched: ``history_tokens``
    re-read at the keep-warm rate, once per expected remaining turn.
    ``abstract_cost`` is the one-time price of replacing the span with a
    ``summary_tokens``-token summary, which busts the ``suffix_tokens`` after
    it. Abstraction wins only when it is strictly cheaper over the horizon.
    """

    would_abstract: bool
    keep_warm_cost: float
    abstract_cost: float
    history_tokens: int
    summary_tokens: int
    suffix_tokens: int
    expected_remaining_turns: float


def decide_abstract_vs_keep_warm(
    history_tokens: int,
    summary_tokens: int,
    suffix_tokens: int,
    expected_remaining_turns: float,
) -> AbstractVsKeepWarmDecision:
    """Price abstracting a history span against leaving it warm in cache.

    Pure and log-only: the caller decides whether to act on the result, this
    function only computes it. Negative inputs are clamped to zero so a bad
    upstream estimate cannot flip the sign of either cost.
    """
    history_tokens = max(0, history_tokens)
    summary_tokens = max(0, summary_tokens)
    suffix_tokens = max(0, suffix_tokens)
    expected_remaining_turns = max(0.0, expected_remaining_turns)

    keep_warm_cost = (
        _KEEP_WARM_PRICE_PER_TOKEN_TURN * history_tokens * expected_remaining_turns
    )
    abstract_cost = _ABSTRACT_BUST_PRICE * (summary_tokens + suffix_tokens)
    return AbstractVsKeepWarmDecision(
        would_abstract=keep_warm_cost > abstract_cost,
        keep_warm_cost=keep_warm_cost,
        abstract_cost=abstract_cost,
        history_tokens=history_tokens,
        summary_tokens=summary_tokens,
        suffix_tokens=suffix_tokens,
        expected_remaining_turns=expected_remaining_turns,
    )


class AbstractVsKeepWarmArmStats:
    """Log-only tally of the priced arm's recommendation versus production.

    Nothing here changes a request. It exists so a canary comparison (see
    workstream E) can ask "how often would this arm have disagreed" without
    replaying traffic.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._would_abstract = 0
        self._would_keep_warm = 0
        self._agrees_with_production = 0
        self._disagrees_with_production = 0

    def record(
        self, decision: AbstractVsKeepWarmDecision, *, production_would_abstract: bool
    ) -> None:
        with self._lock:
            if decision.would_abstract:
                self._would_abstract += 1
            else:
                self._would_keep_warm += 1
            if decision.would_abstract == production_would_abstract:
                self._agrees_with_production += 1
            else:
                self._disagrees_with_production += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "would_abstract": self._would_abstract,
                "would_keep_warm": self._would_keep_warm,
                "agrees_with_production": self._agrees_with_production,
                "disagrees_with_production": self._disagrees_with_production,
            }


abstract_vs_keep_warm_stats = AbstractVsKeepWarmArmStats()
"""Process-wide singleton, mirroring the OperationalAudit/TouchRegistry
pattern of one counters object exposed through the module and read by
/stats. Safe under the GIL plus its own lock; no request state lives here."""
