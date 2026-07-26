"""Price a reasoning-effort switch against the prefix rewrite it costs.

Lowering effort on a mechanical continuation saves output tokens. On Anthropic
it also mutates the request body, and any body mutation ahead of the cached
suffix rewrites the whole prefix. The router used to resolve that tension with a
blanket rule: never switch when the request uses prompt caching. Safe, and wrong
whenever the stretch is long enough to repay the rewrite, or when the prefix is
being rewritten anyway.

The arithmetic. Switching on a warm prefix of S tokens replaces S cache reads
with S cache writes, so the marginal cost in input-token units is

    switch_cost = S * (write_multiplier - read_multiplier)     ~= 1.15 * S

Each turn at the lower effort saves dOut output tokens, worth

    per_turn_gain = dOut * output_multiplier                    ~= 5 * dOut

so the switch repays after

    break_even_turns = 1.15 * S / (5 * dOut) = 0.23 * S / dOut

At S = 150k and dOut = 2,000 that is about 17 turns. A short mechanical stretch
never repays it. A long one does. The decision therefore needs a horizon, not a
constant, and it needs dOut measured for the model in play rather than assumed.

Two cases make the switch free, and both matter more than the marginal case:
a cold prefix has no S to rewrite, and a turn whose prefix is already busting is
paying the rewrite regardless. Effort should move on those turns by preference.

dOut is learned in-session. Providers do not publish how much reasoning a given
effort level buys, it varies by model, and a static table would be stale on the
next model release. Observations come from response usage, keyed by (model,
effort), and the model declines to guess until it has seen both levels.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Anthropic list-price ratios, expressed in base input-token units.
DEFAULT_CACHE_READ_MULTIPLIER = 0.1
DEFAULT_CACHE_WRITE_MULTIPLIER = 1.25
DEFAULT_OUTPUT_MULTIPLIER = 5.0

# Below this many observations at a level the mean is noise, not a measurement.
MIN_OBSERVATIONS = 3

# A horizon nobody should trust. Mechanical stretches that look infinite are
# usually a misclassification, so cap what the forecast is allowed to promise.
MAX_CREDITED_TURNS = 40


@dataclass
class EffortPrices:
    """Per-token price ratios, in input-token units."""

    cache_read_multiplier: float = DEFAULT_CACHE_READ_MULTIPLIER
    cache_write_multiplier: float = DEFAULT_CACHE_WRITE_MULTIPLIER
    output_multiplier: float = DEFAULT_OUTPUT_MULTIPLIER

    @property
    def rewrite_multiplier(self) -> float:
        """Marginal cost of turning a cache read into a cache write."""
        return self.cache_write_multiplier - self.cache_read_multiplier


@dataclass
class EffortDecision:
    """Why the router did or did not move effort this turn."""

    switch: bool
    reason: str
    switch_cost: float = 0.0
    expected_gain: float = 0.0
    break_even_turns: float | None = None
    delta_output_tokens: float | None = None

    @property
    def net_gain(self) -> float:
        return self.expected_gain - self.switch_cost

    def as_log_fields(self) -> str:
        parts = [
            f"switch={self.switch}",
            f"reason={self.reason}",
            f"cost={self.switch_cost:.0f}",
            f"gain={self.expected_gain:.0f}",
            f"net={self.net_gain:.0f}",
        ]
        if self.break_even_turns is not None:
            parts.append(f"break_even_turns={self.break_even_turns:.1f}")
        if self.delta_output_tokens is not None:
            parts.append(f"d_out={self.delta_output_tokens:.0f}")
        return " ".join(parts)


@dataclass
class _Samples:
    count: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class EffortCostModel:
    """Learns output-token cost per (model, effort) from observed responses."""

    def __init__(self) -> None:
        self._samples: dict[tuple[str, str], _Samples] = {}
        self._lock = threading.Lock()

    def observe(self, model: str, effort: str, output_tokens: int) -> None:
        """Record what one turn at ``effort`` actually cost in output tokens."""
        if not model or not effort or output_tokens <= 0:
            return
        with self._lock:
            entry = self._samples.setdefault((model, effort), _Samples())
            entry.count += 1
            entry.total += float(output_tokens)

    def expected_output_tokens(self, model: str, effort: str) -> float | None:
        """Mean output tokens for this pair, or None if not yet measured."""
        with self._lock:
            entry = self._samples.get((model, effort))
            if entry is None or entry.count < MIN_OBSERVATIONS:
                return None
            return entry.mean

    def delta_output_tokens(
        self, model: str, from_effort: str, to_effort: str
    ) -> float | None:
        """Output tokens saved per turn by moving from one effort to another.

        None when either level is unmeasured. Returns a positive number only
        when the target level is actually cheaper, since a switch that costs
        more output than it saves is never worth a prefix rewrite.
        """
        high = self.expected_output_tokens(model, from_effort)
        low = self.expected_output_tokens(model, to_effort)
        if high is None or low is None:
            return None
        return high - low

    def observation_count(self, model: str, effort: str) -> int:
        with self._lock:
            entry = self._samples.get((model, effort))
            return entry.count if entry else 0

    def export_state(self) -> dict[str, Any]:
        """Serialize the observations for the cross-restart snapshot.

        Six turns of traffic have to land before a delta exists, and a restart
        that dropped them would put the router back to refusing every switch
        for the first several turns of every session. The counts are model and
        effort labels with integer sums, no conversation content.
        """
        with self._lock:
            return {
                "samples": [
                    [model, effort, entry.count, entry.total]
                    for (model, effort), entry in self._samples.items()
                ]
            }

    def restore_state(self, blob: dict[str, Any] | None) -> int:
        """Merge a previously exported snapshot. Returns the entries restored."""
        if not isinstance(blob, dict):
            return 0
        rows = blob.get("samples")
        if not isinstance(rows, list):
            return 0
        restored = 0
        with self._lock:
            for row in rows:
                try:
                    model, effort, count, total = row
                    count = int(count)
                    total = float(total)
                except (TypeError, ValueError):
                    continue
                if not model or not effort or count <= 0 or total <= 0:
                    continue
                entry = self._samples.setdefault((str(model), str(effort)), _Samples())
                entry.count += count
                entry.total += total
                restored += 1
        return restored


@dataclass
class EffortPricingContext:
    """Live cache state for one turn, so the router can price a switch.

    The handler owns the numbers, the shaper owns the body mutation, and this
    carries the former to the latter without the shaper reaching into tracker
    internals.
    """

    model: str
    prefix_tokens: int
    cost_model: EffortCostModel
    prefix_already_busting: bool = False
    expected_remaining_turns: float = 0.0
    prices: EffortPrices | None = None

    # What this lineage has already forwarded, if anything. The switch is
    # priced once. After that the value has to stay put, because the cost of
    # effort is not the level, it is the change: output_config sits ahead of
    # every message, so each flip rewrites the entire prefix. Deciding per turn
    # from the turn kind alternates the value as the conversation moves between
    # tool runs and user asks, and pays a full rewrite on each alternation.
    committed_effort: str | None = None
    commit: Callable[[str], None] | None = None

    def record(self, effort: str) -> None:
        """Remember the value this lineage is now committed to forwarding."""
        self.committed_effort = effort
        if self.commit is not None:
            try:
                self.commit(effort)
            except Exception:
                pass

    def decide(self, *, from_effort: str, to_effort: str) -> EffortDecision:
        return price_effort_switch(
            prefix_tokens=self.prefix_tokens,
            delta_output_tokens=self.cost_model.delta_output_tokens(
                self.model, from_effort, to_effort
            ),
            expected_remaining_turns=self.expected_remaining_turns,
            prefix_already_busting=self.prefix_already_busting,
            prices=self.prices,
        )


_SHARED_COST_MODEL = EffortCostModel()


def shared_cost_model() -> EffortCostModel:
    """The process-wide observation store.

    Deltas are only useful once both effort levels have been seen, which takes
    several turns, so the observations have to outlive a single request. One
    store per process, keyed internally by (model, effort), keeps a busy proxy
    from mixing models while still letting every session benefit from what the
    previous ones measured.
    """
    return _SHARED_COST_MODEL


def price_effort_switch(
    *,
    prefix_tokens: int,
    delta_output_tokens: float | None,
    expected_remaining_turns: float,
    prefix_already_busting: bool = False,
    prices: EffortPrices | None = None,
) -> EffortDecision:
    """Decide whether lowering effort repays the prefix rewrite it triggers.

    ``delta_output_tokens`` is the per-turn output saving, as measured by
    :class:`EffortCostModel`. None means unmeasured, and an unmeasured saving is
    not a reason to spend a known rewrite.
    """
    prices = prices or EffortPrices()

    # Free cases first. Neither depends on knowing dOut, so they apply from the
    # first turn of a session, before any measurement exists.
    if prefix_tokens <= 0:
        return EffortDecision(
            switch=True,
            reason="free_cold_prefix",
            delta_output_tokens=delta_output_tokens,
        )
    if prefix_already_busting:
        return EffortDecision(
            switch=True,
            reason="free_prefix_already_busting",
            delta_output_tokens=delta_output_tokens,
        )

    if delta_output_tokens is None:
        return EffortDecision(
            switch=False,
            reason="unmeasured_delta",
        )
    if delta_output_tokens <= 0:
        return EffortDecision(
            switch=False,
            reason="no_output_saving",
            delta_output_tokens=delta_output_tokens,
        )

    switch_cost = prefix_tokens * prices.rewrite_multiplier
    per_turn_gain = delta_output_tokens * prices.output_multiplier
    break_even_turns = switch_cost / per_turn_gain if per_turn_gain > 0 else None

    credited_turns = max(0.0, min(expected_remaining_turns, MAX_CREDITED_TURNS))
    expected_gain = per_turn_gain * credited_turns

    if expected_gain > switch_cost:
        reason = "pays_back_within_horizon"
        switch = True
    else:
        reason = "horizon_too_short"
        switch = False

    return EffortDecision(
        switch=switch,
        reason=reason,
        switch_cost=switch_cost,
        expected_gain=expected_gain,
        break_even_turns=break_even_turns,
        delta_output_tokens=delta_output_tokens,
    )
