"""Decide when compaction is worth its cost, instead of waiting for a limit.

Claude Code compacts when the context window is nearly full. That is a safety
rule, not an economic one, and it fires at the worst possible moment: deep
inside a task, on a prefix that has been read cheaply for hundreds of turns,
with no say in where the summary boundary lands.

Compaction has a real price and a real return, so it can be priced like any
other prefix mutation. Replacing a live prefix of ``S`` tokens with a retained
``S_new`` costs one write of the new prefix plus the output tokens spent writing
the summary::

    cost = S_new * write_multiplier + T_summary * output_multiplier

and every later turn reads ``dS = S - S_new`` fewer tokens::

    per_turn_gain = dS * read_multiplier

so it repays after::

    break_even_turns = cost / (read_multiplier * dS)

At list prices, S = 150k compacting to 30k with a 6k summary repays in under six
turns. That is the load-bearing number here. Compaction is far cheaper than the
"only when forced" default implies, because the thing it buys back is the read
on every remaining turn, and reads are what a long session is made of.

Three moments change the arithmetic rather than the policy:

Already busting. When the turn is rewriting the prefix anyway, the write of the
new prefix is not an extra cost, it is a discount: the client would have written
all ``S`` tokens and now writes only ``S_new``. Only the summary output is left
to pay. This is the same rule the effort router and the masking gate arrived at
independently, do structural work on turns that are already paying for it.

Task boundary. Mid-task compaction has a cost this model cannot see, the working
context the agent still needs. So the arithmetic alone is not enough off a
boundary, and the margin there is deliberately stricter. On a boundary, where a
handover was going to happen anyway, break-even is the whole test.

Forced soon. Past the context threshold compaction is not a choice, so the only
question left is whether it happens at a boundary chosen for a good reason or at
whatever token happens to cross the line.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

# Same input-token units as the effort pricing model.
from headroom.proxy.effort_pricing import (
    DEFAULT_CACHE_READ_MULTIPLIER,
    DEFAULT_CACHE_WRITE_MULTIPLIER,
    DEFAULT_OUTPUT_MULTIPLIER,
)

# Off a task boundary the model is blind to the cost of losing working context,
# so require the saving to clear the price by this factor before recommending.
OFF_BOUNDARY_MARGIN = 2.0

# Past this fraction of the context limit the client compacts regardless.
FORCED_FRACTION = 0.92

# The same cap the effort model uses: a horizon nobody should trust.
MAX_CREDITED_TURNS = 40

# What fraction of the live prefix survives a compaction, before anything has
# been measured. Claude Code summaries land near a fifth of what they replace.
DEFAULT_RETAINED_FRACTION = 0.2

# And what the summary itself costs to write, as a fraction of what it retains.
DEFAULT_SUMMARY_FRACTION = 0.2

MIN_OBSERVATIONS = 2


@dataclass
class CompactionAdvice:
    """Whether to compact now, and the arithmetic behind the answer."""

    recommend: bool
    reason: str
    cost: float = 0.0
    per_turn_gain: float = 0.0
    expected_gain: float = 0.0
    break_even_turns: float | None = None
    tokens_reclaimed: int = 0
    urgency: str = "none"

    @property
    def net_gain(self) -> float:
        return self.expected_gain - self.cost

    def as_log_fields(self) -> str:
        parts = [
            f"recommend={self.recommend}",
            f"reason={self.reason}",
            f"urgency={self.urgency}",
            f"reclaim={self.tokens_reclaimed}",
            f"cost={self.cost:.0f}",
            f"gain={self.expected_gain:.0f}",
            f"net={self.net_gain:.0f}",
        ]
        if self.break_even_turns is not None:
            parts.append(f"break_even_turns={self.break_even_turns:.1f}")
        return " ".join(parts)


class CompactionShapeModel:
    """Learns what a compaction actually costs and reclaims, per session.

    The defaults are a starting guess. Every compaction the proxy witnesses
    reports its real before/after sizes, so the estimate converges on what this
    client and this workload actually do rather than on a constant.
    """

    def __init__(self) -> None:
        self._retained_total = 0.0
        self._summary_total = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, live_tokens: int, retained_tokens: int, summary_tokens: int) -> None:
        """Record one witnessed compaction, as fractions of the prefix it replaced."""
        if live_tokens <= 0 or retained_tokens <= 0 or retained_tokens >= live_tokens:
            return
        with self._lock:
            self._retained_total += retained_tokens / live_tokens
            self._summary_total += max(0, summary_tokens) / live_tokens
            self._count += 1

    @property
    def retained_fraction(self) -> float:
        with self._lock:
            if self._count < MIN_OBSERVATIONS:
                return DEFAULT_RETAINED_FRACTION
            return self._retained_total / self._count

    @property
    def summary_fraction(self) -> float:
        with self._lock:
            if self._count < MIN_OBSERVATIONS:
                return DEFAULT_SUMMARY_FRACTION
            return self._summary_total / self._count

    @property
    def observation_count(self) -> int:
        with self._lock:
            return self._count

    def export_state(self) -> dict[str, Any]:
        """Serialize for the cross-restart snapshot. Three ratios, no content."""
        with self._lock:
            return {
                "retained_total": self._retained_total,
                "summary_total": self._summary_total,
                "count": self._count,
            }

    def restore_state(self, blob: dict[str, Any] | None) -> int:
        """Merge a previously exported snapshot. Returns the observations added.

        Compactions are rare, a handful per long session, so an estimate that
        reset on every restart would spend its whole life on the default guess.
        """
        if not isinstance(blob, dict):
            return 0
        try:
            count = int(blob.get("count", 0) or 0)
            retained = float(blob.get("retained_total", 0.0) or 0.0)
            summary = float(blob.get("summary_total", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0
        if count <= 0 or retained <= 0:
            return 0
        with self._lock:
            self._retained_total += retained
            self._summary_total += summary
            self._count += count
        return count


_SHARED_SHAPE_MODEL = CompactionShapeModel()


def shared_shape_model() -> CompactionShapeModel:
    """The process-wide compaction shape estimate."""
    return _SHARED_SHAPE_MODEL


def advise_compaction(
    *,
    live_tokens: int,
    expected_remaining_turns: float,
    at_task_boundary: bool = False,
    prefix_already_busting: bool = False,
    context_limit: int = 0,
    retained_tokens: int | None = None,
    summary_output_tokens: int | None = None,
    shape: CompactionShapeModel | None = None,
    cache_read_multiplier: float = DEFAULT_CACHE_READ_MULTIPLIER,
    cache_write_multiplier: float = DEFAULT_CACHE_WRITE_MULTIPLIER,
    output_multiplier: float = DEFAULT_OUTPUT_MULTIPLIER,
) -> CompactionAdvice:
    """Price compacting now against carrying the current prefix forward.

    ``retained_tokens`` and ``summary_output_tokens`` override the learned
    shape, for callers that know the real numbers. Everything is in input-token
    cost units, so the result is directly comparable to the effort router's and
    the masking gate's.
    """
    if live_tokens <= 0:
        return CompactionAdvice(recommend=False, reason="no_context")

    shape = shape or shared_shape_model()
    if retained_tokens is None:
        retained_tokens = int(live_tokens * shape.retained_fraction)
    if summary_output_tokens is None:
        summary_output_tokens = int(live_tokens * shape.summary_fraction)

    retained_tokens = max(0, retained_tokens)
    summary_output_tokens = max(0, summary_output_tokens)
    reclaimed = live_tokens - retained_tokens
    if reclaimed <= 0:
        return CompactionAdvice(recommend=False, reason="no_reduction")

    # On a busting turn the client rewrites all `live_tokens` regardless, so
    # compacting writes `retained_tokens` instead of `live_tokens`. The prefix
    # write is not a cost here, it is already spent.
    if prefix_already_busting:
        cost = summary_output_tokens * output_multiplier
    else:
        cost = (
            retained_tokens * cache_write_multiplier
            + summary_output_tokens * output_multiplier
        )

    per_turn_gain = reclaimed * cache_read_multiplier
    break_even_turns = cost / per_turn_gain if per_turn_gain > 0 else None
    credited = max(0.0, min(expected_remaining_turns, MAX_CREDITED_TURNS))
    expected_gain = per_turn_gain * credited

    # Past the threshold the client compacts on its own. Say so early enough
    # that the boundary can still be chosen rather than stumbled into.
    forced = context_limit > 0 and live_tokens >= context_limit * FORCED_FRACTION
    if forced:
        return CompactionAdvice(
            recommend=True,
            reason="forced_soon",
            cost=cost,
            per_turn_gain=per_turn_gain,
            expected_gain=expected_gain,
            break_even_turns=break_even_turns,
            tokens_reclaimed=reclaimed,
            urgency="forced_soon",
        )

    margin = 1.0 if at_task_boundary else OFF_BOUNDARY_MARGIN
    if expected_gain > cost * margin:
        reason = "pays_back_at_task_boundary" if at_task_boundary else "pays_back_mid_task"
        return CompactionAdvice(
            recommend=True,
            reason=reason,
            cost=cost,
            per_turn_gain=per_turn_gain,
            expected_gain=expected_gain,
            break_even_turns=break_even_turns,
            tokens_reclaimed=reclaimed,
            urgency="advisory",
        )

    return CompactionAdvice(
        recommend=False,
        reason="horizon_too_short" if at_task_boundary else "below_mid_task_margin",
        cost=cost,
        per_turn_gain=per_turn_gain,
        expected_gain=expected_gain,
        break_even_turns=break_even_turns,
        tokens_reclaimed=reclaimed,
    )


# Below this share of the context limit, compaction is far enough away that its
# cost should not be pulling on a masking decision made now.
IMMINENT_FRACTION = 0.70

# A mutation that buys back less than this many turns is churn: the prefix is
# rewritten and the threshold arrives again almost immediately.
MIN_TURNS_BOUGHT = 2.0


@dataclass
class DeferralAdvice:
    """What deferring compaction is worth to a mutation being priced now."""

    credit: float = 0.0
    reason: str = "not_imminent"
    turns_bought: float = 0.0
    turns_until_forced: float = 0.0
    avoided_probability: float = 0.0
    compaction_cost: float = 0.0

    @property
    def recommend(self) -> bool:
        return self.credit > 0.0

    def as_log_fields(self) -> str:
        return (
            f"reason={self.reason} credit={self.credit:.0f} "
            f"turns_bought={self.turns_bought:.1f} "
            f"until_forced={self.turns_until_forced:.1f} "
            f"p_avoided={self.avoided_probability:.3f}"
        )


def deferral_credit(
    *,
    live_tokens: int,
    reclaimed_tokens: int,
    tokens_per_turn: float,
    expected_remaining_turns: float,
    context_limit: int,
    shape: CompactionShapeModel | None = None,
    cache_write_multiplier: float = DEFAULT_CACHE_WRITE_MULTIPLIER,
    output_multiplier: float = DEFAULT_OUTPUT_MULTIPLIER,
) -> DeferralAdvice:
    """Price what a compression bust is worth purely for postponing compaction.

    Every gate in this proxy asks the same question, does this mutation's read
    saving repay its rewrite. Near the context limit that question is missing a
    term. The prefix is not going to be read forever: it is going to be
    compacted, and compaction costs a full retained-prefix write plus the output
    tokens of the summary. A mutation that reclaims tokens pushes that moment
    further out, and if the session ends before it arrives, the whole compaction
    is never paid at all.

    Turns until compaction are modelled as memoryless, matching the horizon
    estimator's own assumption. With mean remaining turns ``E``, a session still
    running after ``U`` more turns has probability ``exp(-U/E)``, so the chance
    this mutation is what dodges compaction is the gap between the two
    survival values::

        P(avoided) = exp(-U_before / E) - exp(-U_after / E)

    The credit is the compaction cost times that probability. It is a credit and
    not a decision: the caller adds it to whatever the mutation is already worth
    and applies its own threshold. Nothing here recommends busting a prefix that
    the read arithmetic alone says to leave alone, it only stops the arithmetic
    from pretending the prefix lives forever.
    """
    if live_tokens <= 0 or context_limit <= 0:
        return DeferralAdvice(reason="no_context")
    if reclaimed_tokens <= 0:
        return DeferralAdvice(reason="nothing_reclaimed")

    forced_at = context_limit * FORCED_FRACTION
    if live_tokens < context_limit * IMMINENT_FRACTION:
        return DeferralAdvice(reason="not_imminent")

    # Growth has to come from somewhere. A session that is not growing never
    # reaches the threshold, so there is nothing to defer.
    per_turn = max(1.0, float(tokens_per_turn))
    turns_until_forced = max(0.0, (forced_at - live_tokens) / per_turn)
    turns_bought = reclaimed_tokens / per_turn
    if turns_bought < MIN_TURNS_BOUGHT:
        return DeferralAdvice(
            reason="too_little_bought",
            turns_bought=turns_bought,
            turns_until_forced=turns_until_forced,
        )

    horizon = max(1.0, min(float(expected_remaining_turns), float(MAX_CREDITED_TURNS)))
    survives_before = math.exp(-turns_until_forced / horizon)
    survives_after = math.exp(-(turns_until_forced + turns_bought) / horizon)
    avoided = max(0.0, survives_before - survives_after)

    shape = shape or shared_shape_model()
    retained = live_tokens * shape.retained_fraction
    summary = live_tokens * shape.summary_fraction
    compaction_cost = retained * cache_write_multiplier + summary * output_multiplier

    return DeferralAdvice(
        credit=compaction_cost * avoided,
        reason="defers_compaction",
        turns_bought=turns_bought,
        turns_until_forced=turns_until_forced,
        avoided_probability=avoided,
        compaction_cost=compaction_cost,
    )
