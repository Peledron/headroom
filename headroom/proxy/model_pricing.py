"""Price a model switch against the prefix rewrite it costs.

:mod:`headroom.proxy.model_router` decides which model a request is *allowed*
to use. This module decides whether moving is worth it. The two are separate
because the rule engine reads one request and the price reads the session.

Routing an easy task to a cheaper model looks like free money per turn. It is
not, because prompt caches are keyed by model. Switching abandons the warm
prefix and rewrites the whole thing at the new model's price, and coming back
rewrites it again at the old one. In input-token units of the expensive model,
with a prefix of S tokens and a price ratio r for the cheap model,

    switch_cost = S * write_multiplier * r          (rewrite over there)
    return_cost = S * write_multiplier              (rewrite back here)

while each turn spent on the cheap model saves the price difference on
everything that turn touches,

    per_turn_gain = (1 - r) * (S * read_multiplier + new_input + out * out_mult)

At S = 150k, r = 0.2, and a turn that reads the prefix and writes 1,500 output
tokens, the switch costs about 37k and the return about 188k, against a gain
near 18k per turn. Twelve turns of easy work before it repays, and that is
with the return counted. A single easy question routed cheap is a clear loss,
which is exactly the case a per-request rule engine would route.

Three things follow, and they are what this module encodes:

The decision needs a horizon, not a threshold. One easy turn never repays a
rewrite, a long mechanical stretch does.

The decision has to be sticky per lineage. Difficulty alternates turn to turn
as tool runs and user asks interleave, so a per-turn choice pays a full
rewrite on every alternation and loses on all of them. Once a lineage has
committed to a model it stays there, the same way effort does in
:mod:`headroom.proxy.effort_pricing`.

The free cases matter more than the marginal one. A cold prefix has nothing to
rewrite, and a turn whose prefix is busting anyway pays the rewrite either
way. Those are where routing is actually worth taking.

What this module deliberately does not decide is whether the cheap model is
good enough for the work. That is a quality question, it needs a live A/B
against real traffic, and no amount of arithmetic here substitutes for it.
Because that measurement does not exist yet, the difficulty estimate below
admits one shape of turn and no other: a fresh, short, plainly phrased
question. Anything that might be work stays where it is.

The price ratio is not derivable either. Per-model list prices are not
published in a form the proxy can read, so the ratio is asserted by an
operator through ``HEADROOM_MODEL_ROUTE_PRICES`` and the gate stands aside
without it. Between that and the difficulty filter, a switch takes an
operator's explicit setup, and the handler adds the last requirement: a
switch nobody is told about does not happen, so the route is refused on any
path that cannot carry the notice.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Anthropic list-price ratios, expressed in base input-token units. Same
# numbers as ``effort_pricing``, repeated rather than imported so neither
# module quietly changes the other's arithmetic.
DEFAULT_CACHE_READ_MULTIPLIER = 0.1
DEFAULT_CACHE_WRITE_MULTIPLIER = 1.25
DEFAULT_OUTPUT_MULTIPLIER = 5.0

# No horizon past this is credited. A stretch that looks endlessly mechanical
# is usually a misread of the traffic, not a forecast worth spending on.
MAX_CREDITED_TURNS = 40

# Score at or below which a turn counts as easy. Deliberately strict: the cost
# of a wrong route is a wrong answer, the cost of a missed one is some tokens.
DEFAULT_EASY_THRESHOLD = 0.35

# Where a turn starts before any signal is read. Work of unknown shape is not
# easy work. Scoring from zero would make silence look like a quick question,
# and silence is what the middle of a long task looks like.
UNKNOWN_TURN_SCORE = 0.5

# Longest ask that can still be a quick question. Past this the turn is
# carrying constraints, whatever it looks like.
QUICK_QUESTION_CHARS = 200

_CODE_FENCE = re.compile(r"```|^\s*diff --git|^\s*@@ ", re.MULTILINE)

# Every word below is kept because it measured, not because it sounds hard.
# See docs/difficulty-gate-population-2026-07-26.md. An earlier list held
# twelve hand-picked words, of which ten did not predict effort in 2713 real
# turns and four predicted the opposite: "architect", "race", "trade-off" and
# "explain" all fired on turns that finished in a quarter of the median work.
# These two carry a 3.00x lift on follow-on assistant turns.
_HARD_WORDS = re.compile(r"\b(deadlock|prove|proving)\b", re.IGNORECASE)

# Language that sets up an unattended run. 1.38x lift, so a small weight: it
# says the turn is opening a stretch of work, not that the work is subtle.
_RUN_CONTROL = re.compile(
    r"\b(untill|until|acknowledge|breakers?|recovery|reaches|"
    r"exit condition|stopping condition)\b",
    re.IGNORECASE,
)

# Assent that opens work: "yes do that", "good, now merge it". 2.12x lift
# overall and the strongest signal measured, but the reason it matters is the
# tail rather than the median. These are the turns that ran 76 to 158 further
# assistant turns and a fifth of a million output tokens, and they look easy
# by every other signal here: short, no code, phrased as a question.
#
# Deliberately excludes "continue", "proceed" and the other resumption words.
# Those are confounded with context size rather than difficulty. Stratified by
# prefix size they run a median of 1 follow-on turn under 50k, the cheapest
# turn shape in the corpus, and only climb past 50k where ``large_context``
# already accounts for them. Scoring them here would count that twice.
_STEERING_ASSENT = re.compile(
    r"^\W*(yes|yeah|yep|no|nope|ok|okay|well|good|sure|right|correct|"
    r"exactly|indeed)\b",
    re.IGNORECASE,
)

# What a quick question looks like: a short ask that wants a fact back. The
# question mark is the strongest single signal, the openers cover the ones
# phrased as requests ("tell me what X does").
_QUICK_QUESTION = re.compile(
    r"\?\s*$|^\s*(what|which|when|where|who|is|are|does|do|can|list|show|"
    r"tell me|remind me)\b",
    re.IGNORECASE,
)


@dataclass
class ModelPrices:
    """Per-token price ratios, in input-token units of the expensive model."""

    cache_read_multiplier: float = DEFAULT_CACHE_READ_MULTIPLIER
    cache_write_multiplier: float = DEFAULT_CACHE_WRITE_MULTIPLIER
    output_multiplier: float = DEFAULT_OUTPUT_MULTIPLIER


@dataclass
class DifficultyEstimate:
    """How hard this turn looks, from signals the proxy can read locally.

    ``score`` runs 0 (mechanical) to 1 (hard). ``signals`` names what moved
    it, so a routing decision in the log can be argued with rather than
    merely believed.
    """

    score: float
    signals: tuple[str, ...] = ()

    def is_easy(self, threshold: float = DEFAULT_EASY_THRESHOLD) -> bool:
        return self.score <= threshold


@dataclass
class ModelSwitchDecision:
    """Whether to move this lineage to the cheap model, and what it costs."""

    switch: bool
    reason: str
    switch_cost: float = 0.0
    return_cost: float = 0.0
    expected_gain: float = 0.0
    break_even_turns: float | None = None
    difficulty: float | None = None

    @property
    def total_cost(self) -> float:
        return self.switch_cost + self.return_cost

    @property
    def net_gain(self) -> float:
        return self.expected_gain - self.total_cost

    def as_log_fields(self) -> str:
        parts = [
            f"switch={self.switch}",
            f"reason={self.reason}",
            f"cost={self.total_cost:.0f}",
            f"gain={self.expected_gain:.0f}",
            f"net={self.net_gain:.0f}",
        ]
        if self.break_even_turns is not None:
            parts.append(f"break_even_turns={self.break_even_turns:.1f}")
        if self.difficulty is not None:
            parts.append(f"difficulty={self.difficulty:.2f}")
        return " ".join(parts)


def estimate_difficulty(
    *,
    latest_user_text: str = "",
    tool_count: int = 0,
    is_tool_continuation: bool = False,
    input_tokens: int = 0,
) -> DifficultyEstimate:
    """Score a turn from local signals only.

    Every signal here is something the proxy already has in hand, so the
    estimate costs no model call and cannot itself become the expensive part
    of the request. The text signals were rebuilt on 2026-07-26 against 2713
    real turns, ranking candidate vocabulary by the assistant work that
    actually followed it, so each weight below is ordered by a measured lift
    rather than by how hard the word sounds. Two-thirds of the vocabulary
    that cleared the bar on one half of the corpus failed to replicate on the
    other half, which is the reason the surviving lists are short. Prefer
    tuning the threshold over adding words back.

    The scoring is asymmetric on purpose. A turn is hard until something says
    otherwise, and only one thing says otherwise: a fresh, short, plainly
    phrased question. Everything else adds. An earlier version scored from
    zero and treated a tool continuation as evidence of easy work, which had
    it backwards, since a tool continuation is the middle of a task someone is
    depending on. Being wrong here costs a wrong answer on real work, so the
    estimate has to earn "easy" rather than fall into it.
    """
    score = UNKNOWN_TURN_SCORE
    signals: list[str] = []

    # A turn that hands tool output back is work already under way, with no
    # fresh ask to judge. Nothing about it is a quick question.
    if is_tool_continuation:
        score += 0.3
        signals.append("mid_task")

    # ``tool_count`` says what the client offers, not what the turn needs, and
    # a coding client offers everything on every request. It is kept in the
    # signature because callers pass it, and deliberately unscored.
    if tool_count > 0:
        signals.append("tools_available")

    # Either of these alone has to clear the default threshold on its own.
    # Pasted code and a question about why something is the way it is are the
    # two turns where a cheap model's mistake is least likely to be noticed,
    # so neither gets to combine with something else before it disqualifies.
    text = latest_user_text or ""
    if _CODE_FENCE.search(text):
        score += 0.4
        signals.append("code_or_diff")
    if _HARD_WORDS.search(text):
        score += 0.4
        signals.append("hard_words")

    # Assent that opens work. Same weight as pasted code, for the same reason:
    # it has to disqualify on its own, because nothing else in this function
    # sees anything unusual about "yes, do that".
    if _STEERING_ASSENT.match(text.strip()):
        score += 0.4
        signals.append("steering_assent")

    # Opening an unattended stretch. Small weight, so it tips a turn that is
    # already borderline rather than deciding one by itself.
    if _RUN_CONTROL.search(text):
        score += 0.15
        signals.append("run_control")

    # Long asks carry more constraints to satisfy at once. The cut points are
    # coarse on purpose, a smooth curve here would imply a precision the
    # signal does not have.
    if len(text) > 600:
        score += 0.25
        signals.append("long_ask")
    elif len(text) > 200:
        score += 0.1
        signals.append("medium_ask")

    if input_tokens > 100_000:
        score += 0.2
        signals.append("large_context")

    # The one way down. Everything it requires is a way of saying the turn is
    # a question rather than a piece of work: asked now, asked briefly, no
    # code in it, nothing that wants reasoning about a system, and not an
    # instruction wearing a question mark.
    #
    # The last two exclusions are what stop the discount landing on the worst
    # possible turns. Measured over the local corpus they drop 35 asks out of
    # 401, and those 35 have a p99 of 158 follow-on assistant turns against
    # 116 for the population they were removed from. The additive weights
    # above would disqualify them anyway. Restating the exclusions here keeps
    # the reported signals honest, so no routing log claims a steering message
    # was a quick question.
    if (
        not is_tool_continuation
        and text.strip()
        and len(text) <= QUICK_QUESTION_CHARS
        and not _CODE_FENCE.search(text)
        and not _HARD_WORDS.search(text)
        and not _STEERING_ASSENT.match(text.strip())
        and not _RUN_CONTROL.search(text)
        and _QUICK_QUESTION.search(text.strip())
    ):
        score -= 0.35
        signals.append("quick_question")

    return DifficultyEstimate(
        score=min(1.0, max(0.0, score)), signals=tuple(signals)
    )


def price_model_switch(
    *,
    prefix_tokens: int,
    price_ratio: float,
    difficulty: DifficultyEstimate | float | None = None,
    expected_remaining_turns: float = 0.0,
    expected_output_tokens: float = 0.0,
    expected_new_input_tokens: float = 0.0,
    prefix_already_busting: bool = False,
    returns_to_expensive: bool = True,
    easy_threshold: float = DEFAULT_EASY_THRESHOLD,
    prices: ModelPrices | None = None,
) -> ModelSwitchDecision:
    """Decide whether moving to the cheaper model repays its rewrites.

    ``price_ratio`` is the cheap model's base input price over the expensive
    one's. ``returns_to_expensive`` counts the trip back, which is the honest
    default: a session that got hard once will get hard again.
    """
    prices = prices or ModelPrices()

    score = difficulty.score if isinstance(difficulty, DifficultyEstimate) else difficulty

    if score is not None and score > easy_threshold:
        return ModelSwitchDecision(switch=False, reason="not_easy", difficulty=score)

    if price_ratio >= 1.0:
        return ModelSwitchDecision(
            switch=False, reason="not_cheaper", difficulty=score
        )

    # Free cases. Neither needs a horizon, because neither spends anything.
    if prefix_tokens <= 0:
        return ModelSwitchDecision(
            switch=True, reason="free_cold_prefix", difficulty=score
        )
    if prefix_already_busting:
        return ModelSwitchDecision(
            switch=True, reason="free_prefix_already_busting", difficulty=score
        )

    switch_cost = prefix_tokens * prices.cache_write_multiplier * price_ratio
    return_cost = (
        prefix_tokens * prices.cache_write_multiplier if returns_to_expensive else 0.0
    )

    per_turn_tokens = (
        prefix_tokens * prices.cache_read_multiplier
        + expected_new_input_tokens
        + expected_output_tokens * prices.output_multiplier
    )
    per_turn_gain = (1.0 - price_ratio) * per_turn_tokens

    total_cost = switch_cost + return_cost
    break_even_turns = total_cost / per_turn_gain if per_turn_gain > 0 else None

    credited_turns = max(0.0, min(expected_remaining_turns, MAX_CREDITED_TURNS))
    expected_gain = per_turn_gain * credited_turns

    if per_turn_gain <= 0:
        return ModelSwitchDecision(
            switch=False,
            reason="no_per_turn_gain",
            switch_cost=switch_cost,
            return_cost=return_cost,
            difficulty=score,
        )

    decision = ModelSwitchDecision(
        switch=expected_gain > total_cost,
        reason="repays_horizon" if expected_gain > total_cost else "horizon_too_short",
        switch_cost=switch_cost,
        return_cost=return_cost,
        expected_gain=expected_gain,
        break_even_turns=break_even_turns,
        difficulty=score,
    )
    return decision


@dataclass
class ModelPricingContext:
    """Live cache state for one lineage, so the router can price a switch.

    The handler owns the numbers and the router owns the rewrite, this
    carries the former to the latter. ``committed_model`` is what makes the
    decision sticky: once a lineage is on a model it stays there until the
    lineage itself breaks, because re-deciding per turn pays a rewrite on
    every alternation between easy and hard work.
    """

    prefix_tokens: int = 0
    price_ratio: float = 1.0
    expected_remaining_turns: float = 0.0
    expected_output_tokens: float = 0.0
    expected_new_input_tokens: float = 0.0
    prefix_already_busting: bool = False
    easy_threshold: float = DEFAULT_EASY_THRESHOLD
    prices: ModelPrices | None = None
    committed_model: str | None = None
    commit: Callable[[str], None] | None = None
    signals: dict[str, Any] = field(default_factory=dict)

    def record(self, model: str) -> None:
        """Remember the model this lineage is now committed to forwarding."""
        self.committed_model = model
        if self.commit is not None:
            try:
                self.commit(model)
            except Exception:
                pass

    def decide(self, difficulty: DifficultyEstimate | float | None) -> ModelSwitchDecision:
        if self.committed_model is not None:
            return ModelSwitchDecision(
                switch=False,
                reason="lineage_committed",
                difficulty=(
                    difficulty.score
                    if isinstance(difficulty, DifficultyEstimate)
                    else difficulty
                ),
            )
        return price_model_switch(
            prefix_tokens=self.prefix_tokens,
            price_ratio=self.price_ratio,
            difficulty=difficulty,
            expected_remaining_turns=self.expected_remaining_turns,
            expected_output_tokens=self.expected_output_tokens,
            expected_new_input_tokens=self.expected_new_input_tokens,
            prefix_already_busting=self.prefix_already_busting,
            easy_threshold=self.easy_threshold,
            prices=self.prices,
        )
