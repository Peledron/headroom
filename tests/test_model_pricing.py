"""A model switch has to outrun two prefix rewrites, not one.

The quality half of routing (is the cheap model good enough) cannot be
settled here, it needs a live A/B. What can be pinned is the arithmetic and
the stickiness, which is where a per-request rule engine loses money even
when its quality judgement is right.
"""

from __future__ import annotations

import pytest

from headroom.proxy.model_pricing import (
    DEFAULT_EASY_THRESHOLD,
    MAX_CREDITED_TURNS,
    DifficultyEstimate,
    ModelPricingContext,
    ModelPrices,
    estimate_difficulty,
    price_model_switch,
)


def _price(**kwargs):
    """A warm 150k prefix on a model costing a fifth as much."""
    params = {
        "prefix_tokens": 150_000,
        "price_ratio": 0.2,
        "difficulty": 0.0,
        "expected_remaining_turns": 20.0,
        "expected_output_tokens": 1_500.0,
        "expected_new_input_tokens": 500.0,
    }
    params.update(kwargs)
    return price_model_switch(**params)


class TestDifficultyEstimate:
    def test_a_turn_with_no_signals_is_not_easy(self):
        """Silence is what the middle of a task looks like, not a quick ask."""
        assert not estimate_difficulty().is_easy()

    def test_a_short_standalone_question_is_easy(self):
        estimate = estimate_difficulty(latest_user_text="what port does the proxy listen on?")
        assert estimate.is_easy()
        assert "quick_question" in estimate.signals

    def test_a_bare_tool_continuation_is_not_easy(self):
        """Tool output comes back in the middle of work someone is depending on."""
        estimate = estimate_difficulty(is_tool_continuation=True)
        assert not estimate.is_easy()
        assert "mid_task" in estimate.signals

    def test_a_question_arriving_mid_task_is_not_easy(self):
        estimate = estimate_difficulty(
            latest_user_text="what port does the proxy listen on?",
            is_tool_continuation=True,
        )
        assert not estimate.is_easy()
        assert "quick_question" not in estimate.signals

    def test_a_long_question_is_not_a_quick_question(self):
        estimate = estimate_difficulty(latest_user_text="what about " + "x" * 400 + "?")
        assert not estimate.is_easy()

    def test_a_question_with_code_in_it_is_not_easy(self):
        estimate = estimate_difficulty(latest_user_text="what does this do?\n```py\nx=1\n```")
        assert not estimate.is_easy()
        assert "quick_question" not in estimate.signals

    def test_a_design_question_is_not_easy(self):
        estimate = estimate_difficulty(
            latest_user_text="why did you choose this architecture over the other one"
        )
        assert not estimate.is_easy()
        assert "hard_words" in estimate.signals

    def test_pasted_code_is_not_easy(self):
        estimate = estimate_difficulty(latest_user_text="fix this\n```py\nx=1\n```")
        assert not estimate.is_easy()
        assert "code_or_diff" in estimate.signals

    def test_a_diff_reads_as_code_without_a_fence(self):
        estimate = estimate_difficulty(
            latest_user_text="look at\ndiff --git a/x b/x\n@@ -1 +1 @@\n"
        )
        assert "code_or_diff" in estimate.signals

    def test_the_score_never_leaves_the_unit_interval(self):
        hardest = estimate_difficulty(
            latest_user_text="why " + ("refactor this deadlock " * 60) + "```x```",
            tool_count=40,
            input_tokens=180_000,
        )
        assert hardest.score == 1.0

        easiest = estimate_difficulty(latest_user_text="what is the default port?")
        assert 0.0 <= easiest.score <= DEFAULT_EASY_THRESHOLD

    def test_offering_tools_is_recorded_but_does_not_move_the_score(self):
        """Every request from a coding client carries every tool it has."""
        without = estimate_difficulty(latest_user_text="ok")
        with_tools = estimate_difficulty(latest_user_text="ok", tool_count=12)
        assert with_tools.score == without.score
        assert "tools_available" in with_tools.signals

    def test_a_longer_ask_never_scores_easier(self):
        """Monotonic in length, so a bigger ask cannot sneak under the gate."""
        lengths = [10, 100, 300, 700, 4000]
        scores = [estimate_difficulty(latest_user_text="a" * n).score for n in lengths]
        assert scores == sorted(scores)

    def test_context_size_only_raises_the_score(self):
        small = estimate_difficulty(latest_user_text="ok", input_tokens=1_000)
        large = estimate_difficulty(latest_user_text="ok", input_tokens=150_000)
        assert large.score > small.score

    def test_signals_name_what_moved_the_score(self):
        estimate = estimate_difficulty(
            latest_user_text="why is this slow", tool_count=3, input_tokens=150_000
        )
        assert set(estimate.signals) == {
            "tools_available",
            "hard_words",
            "large_context",
        }


class TestPriceModelSwitch:
    def test_a_hard_turn_is_refused_before_any_arithmetic(self):
        decision = _price(difficulty=0.9)
        assert not decision.switch
        assert decision.reason == "not_easy"
        assert decision.switch_cost == 0.0

    def test_a_model_that_is_not_cheaper_is_refused(self):
        decision = _price(price_ratio=1.0)
        assert not decision.switch
        assert decision.reason == "not_cheaper"

    def test_a_dearer_model_is_refused(self):
        decision = _price(price_ratio=3.0)
        assert not decision.switch
        assert decision.reason == "not_cheaper"

    def test_a_cold_prefix_is_free_to_move(self):
        decision = _price(prefix_tokens=0)
        assert decision.switch
        assert decision.reason == "free_cold_prefix"
        assert decision.total_cost == 0.0

    def test_a_prefix_already_busting_is_free_to_move(self):
        decision = _price(prefix_already_busting=True)
        assert decision.switch
        assert decision.reason == "free_prefix_already_busting"

    def test_the_free_cases_still_respect_the_difficulty_gate(self):
        """Free is about the rewrite, not about the work being suitable."""
        assert not _price(prefix_tokens=0, difficulty=0.9).switch
        assert not _price(prefix_already_busting=True, difficulty=0.9).switch

    def test_one_easy_turn_never_repays_the_rewrite(self):
        decision = _price(expected_remaining_turns=1.0)
        assert not decision.switch
        assert decision.reason == "horizon_too_short"
        assert decision.net_gain < 0

    def test_a_long_mechanical_stretch_does_repay(self):
        decision = _price(expected_remaining_turns=MAX_CREDITED_TURNS)
        assert decision.switch
        assert decision.reason == "repays_horizon"
        assert decision.net_gain > 0

    def test_the_return_trip_is_charged(self):
        staying = _price(returns_to_expensive=False)
        returning = _price(returns_to_expensive=True)
        assert returning.total_cost > staying.total_cost
        assert staying.return_cost == 0.0

    def test_the_documented_example(self):
        """150k prefix, ratio 0.2, 1,500 out and 500 new in per turn."""
        decision = _price(expected_remaining_turns=1.0)
        assert decision.switch_cost == pytest.approx(150_000 * 1.25 * 0.2)
        assert decision.return_cost == pytest.approx(150_000 * 1.25)
        per_turn = 150_000 * 0.1 + 500 + 1_500 * 5
        assert decision.break_even_turns == pytest.approx(
            (37_500 + 187_500) / (0.8 * per_turn)
        )
        assert decision.break_even_turns is not None
        assert 12 < decision.break_even_turns < 13

    def test_no_horizon_past_the_credit_cap_is_counted(self):
        at_cap = _price(expected_remaining_turns=MAX_CREDITED_TURNS)
        past_cap = _price(expected_remaining_turns=MAX_CREDITED_TURNS * 100)
        assert past_cap.expected_gain == at_cap.expected_gain

    def test_a_negative_horizon_is_floored_not_credited(self):
        decision = _price(expected_remaining_turns=-5.0)
        assert not decision.switch
        assert decision.expected_gain == 0.0

    def test_a_turn_with_nothing_to_save_is_refused(self):
        decision = _price(
            prefix_tokens=1,
            expected_output_tokens=0.0,
            expected_new_input_tokens=0.0,
            prices=ModelPrices(cache_read_multiplier=0.0),
        )
        assert not decision.switch
        assert decision.reason == "no_per_turn_gain"
        assert decision.break_even_turns is None

    def test_a_longer_horizon_never_makes_the_switch_worse(self):
        gains = [_price(expected_remaining_turns=t).expected_gain for t in (0, 5, 20, 40)]
        assert gains == sorted(gains)

    def test_a_bigger_prefix_needs_a_longer_horizon(self):
        small = _price(prefix_tokens=20_000).break_even_turns
        large = _price(prefix_tokens=400_000).break_even_turns
        assert small is not None and large is not None
        assert large > small

    def test_a_cheaper_model_repays_sooner(self):
        cheap = _price(price_ratio=0.05).break_even_turns
        dear = _price(price_ratio=0.8).break_even_turns
        assert cheap is not None and dear is not None
        assert cheap < dear

    def test_an_estimate_and_a_bare_score_decide_alike(self):
        from_estimate = _price(difficulty=DifficultyEstimate(score=0.9))
        from_float = _price(difficulty=0.9)
        assert from_estimate.reason == from_float.reason == "not_easy"

    def test_an_absent_difficulty_skips_the_gate(self):
        decision = _price(difficulty=None, expected_remaining_turns=MAX_CREDITED_TURNS)
        assert decision.switch
        assert decision.difficulty is None

    def test_the_threshold_is_tunable(self):
        assert not _price(difficulty=0.5).switch
        decision = _price(
            difficulty=0.5,
            easy_threshold=0.6,
            expected_remaining_turns=MAX_CREDITED_TURNS,
        )
        assert decision.switch

    def test_the_log_line_carries_the_arithmetic(self):
        fields = _price(expected_remaining_turns=MAX_CREDITED_TURNS).as_log_fields()
        for key in ("switch=", "reason=", "cost=", "gain=", "net=", "difficulty="):
            assert key in fields


class TestStickiness:
    def _context(self, **kwargs):
        params = {
            "prefix_tokens": 150_000,
            "price_ratio": 0.2,
            "expected_remaining_turns": MAX_CREDITED_TURNS,
            "expected_output_tokens": 1_500.0,
            "expected_new_input_tokens": 500.0,
        }
        params.update(kwargs)
        return ModelPricingContext(**params)

    def test_an_uncommitted_lineage_prices_the_switch(self):
        assert self._context().decide(0.0).switch

    def test_a_committed_lineage_stays_put(self):
        context = self._context()
        context.record("claude-haiku-4-5-20251001")
        decision = context.decide(0.0)
        assert not decision.switch
        assert decision.reason == "lineage_committed"

    def test_the_commitment_holds_when_the_turn_turns_hard(self):
        """The alternation this prevents is what makes per-turn routing lose."""
        context = self._context()
        context.record("claude-haiku-4-5-20251001")
        assert context.decide(0.95).reason == "lineage_committed"

    def test_committing_notifies_the_owner(self):
        seen: list[str] = []
        context = self._context(commit=seen.append)
        context.record("claude-haiku-4-5-20251001")
        assert seen == ["claude-haiku-4-5-20251001"]
        assert context.committed_model == "claude-haiku-4-5-20251001"

    def test_a_failing_commit_hook_does_not_break_the_turn(self):
        def explode(_model: str) -> None:
            raise RuntimeError("store is down")

        context = self._context(commit=explode)
        context.record("claude-haiku-4-5-20251001")
        assert context.committed_model == "claude-haiku-4-5-20251001"

    def test_a_default_context_refuses_by_price(self):
        """Ratio 1.0 by default, so an unconfigured context never routes."""
        assert ModelPricingContext().decide(0.0).reason == "not_cheaper"


def test_only_a_quick_question_clears_the_default_threshold():
    """The gate admits one shape of turn, since quality is unmeasured."""
    admitted = [
        "what is the default port?",
        "list the open ports",
        "is the proxy running?",
    ]
    refused = [
        "",
        "run the test suite and fix what breaks",
        "why is the cache ratio bad?",
        "what does this do?\n```py\nx=1\n```",
    ]
    for text in admitted:
        assert estimate_difficulty(latest_user_text=text).score <= DEFAULT_EASY_THRESHOLD, text
    for text in refused:
        assert estimate_difficulty(latest_user_text=text).score > DEFAULT_EASY_THRESHOLD, text
