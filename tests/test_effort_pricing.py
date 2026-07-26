"""The effort router must price its own prefix rewrite.

Lowering effort saves output tokens and mutates the request body. On a cached
Anthropic request that mutation rewrites the whole prefix, so the saving has to
clear the rewrite before the switch is worth making. These tests pin the
arithmetic, the free cases, and the refusal to act on an unmeasured guess.
"""

from __future__ import annotations

import pytest

from headroom.proxy.effort_pricing import (
    MAX_CREDITED_TURNS,
    MIN_OBSERVATIONS,
    EffortCostModel,
    EffortPrices,
    EffortPricingContext,
    price_effort_switch,
)


class TestCostModel:
    def test_declines_to_guess_before_enough_samples(self):
        model = EffortCostModel()
        for _ in range(MIN_OBSERVATIONS - 1):
            model.observe("claude-opus-5", "high", 4000)
        assert model.expected_output_tokens("claude-opus-5", "high") is None

    def test_reports_the_mean_once_measured(self):
        model = EffortCostModel()
        for tokens in (3000, 4000, 5000):
            model.observe("claude-opus-5", "high", tokens)
        assert model.expected_output_tokens("claude-opus-5", "high") == 4000

    def test_delta_needs_both_levels(self):
        model = EffortCostModel()
        for _ in range(MIN_OBSERVATIONS):
            model.observe("claude-opus-5", "high", 4000)
        assert model.delta_output_tokens("claude-opus-5", "high", "low") is None

        for _ in range(MIN_OBSERVATIONS):
            model.observe("claude-opus-5", "low", 1500)
        assert model.delta_output_tokens("claude-opus-5", "high", "low") == 2500

    def test_models_are_kept_apart(self):
        """A delta learned on one model must not price another."""
        model = EffortCostModel()
        for _ in range(MIN_OBSERVATIONS):
            model.observe("claude-opus-5", "high", 4000)
            model.observe("claude-opus-5", "low", 1000)
            model.observe("claude-haiku-4-5", "high", 900)
        assert model.delta_output_tokens("claude-opus-5", "high", "low") == 3000
        assert model.delta_output_tokens("claude-haiku-4-5", "high", "low") is None

    def test_junk_observations_are_ignored(self):
        model = EffortCostModel()
        model.observe("", "high", 100)
        model.observe("claude-opus-5", "", 100)
        model.observe("claude-opus-5", "high", 0)
        model.observe("claude-opus-5", "high", -5)
        assert model.observation_count("claude-opus-5", "high") == 0


class TestCostModelPersistence:
    """Six turns of traffic buy the first delta, so restarts must not eat it."""

    def _measured(self) -> EffortCostModel:
        model = EffortCostModel()
        for _ in range(MIN_OBSERVATIONS):
            model.observe("claude-opus-5", "high", 4000)
            model.observe("claude-opus-5", "low", 1000)
        return model

    def test_round_trip_preserves_the_delta(self):
        source = self._measured()
        target = EffortCostModel()
        assert target.restore_state(source.export_state()) == 2
        assert target.delta_output_tokens("claude-opus-5", "high", "low") == 3000

    def test_a_restored_model_can_price_immediately(self):
        target = EffortCostModel()
        target.restore_state(self._measured().export_state())
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=target.delta_output_tokens(
                "claude-opus-5", "high", "low"
            ),
            expected_remaining_turns=30,
        )
        assert decision.switch is True
        assert decision.reason == "pays_back_within_horizon"

    def test_restore_merges_rather_than_replaces(self):
        """A live process keeps what it has measured since starting."""
        target = EffortCostModel()
        for _ in range(MIN_OBSERVATIONS):
            target.observe("claude-opus-5", "high", 4000)
        target.restore_state(self._measured().export_state())
        assert target.observation_count("claude-opus-5", "high") == MIN_OBSERVATIONS * 2
        assert target.expected_output_tokens("claude-opus-5", "high") == 4000

    def test_junk_snapshots_are_discarded_not_partially_applied(self):
        target = EffortCostModel()
        assert target.restore_state(None) == 0
        assert target.restore_state({"samples": "nope"}) == 0
        assert target.restore_state({"samples": [["m", "high"], 7, None]}) == 0
        assert target.restore_state({"samples": [["m", "high", -1, 5.0]]}) == 0
        assert target.observation_count("m", "high") == 0

    def test_the_snapshot_carries_no_conversation_content(self):
        """It rides in the lineage file, but it is counters, not text."""
        blob = self._measured().export_state()
        assert set(blob) == {"samples"}
        for model, effort, count, total in blob["samples"]:
            assert isinstance(model, str)
            assert isinstance(effort, str)
            assert isinstance(count, int)
            assert isinstance(total, float)


class TestFreeCases:
    def test_cold_prefix_switches_without_a_measurement(self):
        """Nothing to rewrite, so the switch cannot cost anything."""
        decision = price_effort_switch(
            prefix_tokens=0,
            delta_output_tokens=None,
            expected_remaining_turns=1,
        )
        assert decision.switch is True
        assert decision.reason == "free_cold_prefix"

    def test_already_busting_prefix_switches_without_a_measurement(self):
        """The rewrite is being paid regardless, so the switch rides along."""
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=None,
            expected_remaining_turns=1,
            prefix_already_busting=True,
        )
        assert decision.switch is True
        assert decision.reason == "free_prefix_already_busting"


class TestPricedCases:
    def test_unmeasured_delta_never_spends_a_known_rewrite(self):
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=None,
            expected_remaining_turns=100,
        )
        assert decision.switch is False
        assert decision.reason == "unmeasured_delta"

    def test_a_switch_that_saves_nothing_is_refused(self):
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=0,
            expected_remaining_turns=100,
        )
        assert decision.switch is False
        assert decision.reason == "no_output_saving"

    def test_a_switch_that_costs_more_output_is_refused(self):
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=-500,
            expected_remaining_turns=100,
        )
        assert decision.switch is False

    def test_short_horizon_on_a_warm_prefix_is_refused(self):
        """The measured live case: one mechanical turn under a 150k prefix."""
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=2000,
            expected_remaining_turns=1,
        )
        assert decision.switch is False
        assert decision.reason == "horizon_too_short"
        # 1.15 * 150k = 172,500 against 5 * 2000 = 10,000.
        assert decision.switch_cost == pytest.approx(172_500)
        assert decision.expected_gain == pytest.approx(10_000)

    def test_long_horizon_repays_the_rewrite(self):
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=2000,
            expected_remaining_turns=30,
        )
        assert decision.switch is True
        assert decision.reason == "pays_back_within_horizon"

    def test_break_even_matches_the_closed_form(self):
        """break_even_turns should equal 0.23 * S / dOut at list prices."""
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=2000,
            expected_remaining_turns=1,
        )
        assert decision.break_even_turns == pytest.approx(0.23 * 150_000 / 2000)

    def test_a_small_prefix_repays_quickly(self):
        decision = price_effort_switch(
            prefix_tokens=5_000,
            delta_output_tokens=2000,
            expected_remaining_turns=2,
        )
        assert decision.switch is True

    def test_the_horizon_is_capped(self):
        """An implausible horizon must not justify an arbitrary rewrite."""
        huge = price_effort_switch(
            prefix_tokens=10_000_000,
            delta_output_tokens=1,
            expected_remaining_turns=10_000,
        )
        assert huge.switch is False
        capped = price_effort_switch(
            prefix_tokens=1_000,
            delta_output_tokens=100,
            expected_remaining_turns=10_000,
        )
        assert capped.expected_gain == pytest.approx(
            100 * 5.0 * MAX_CREDITED_TURNS
        )

    def test_negative_horizon_is_treated_as_zero(self):
        decision = price_effort_switch(
            prefix_tokens=150_000,
            delta_output_tokens=2000,
            expected_remaining_turns=-5,
        )
        assert decision.switch is False
        assert decision.expected_gain == 0

    def test_prices_are_overridable(self):
        """A provider whose output is cheap should switch less readily."""
        cheap_output = EffortPrices(output_multiplier=1.0)
        decision = price_effort_switch(
            prefix_tokens=50_000,
            delta_output_tokens=2000,
            expected_remaining_turns=10,
            prices=cheap_output,
        )
        assert decision.switch is False

        rich_output = EffortPrices(output_multiplier=10.0)
        better = price_effort_switch(
            prefix_tokens=50_000,
            delta_output_tokens=2000,
            expected_remaining_turns=10,
            prices=rich_output,
        )
        assert better.switch is True


class TestPricingContext:
    def _model(self) -> EffortCostModel:
        model = EffortCostModel()
        for _ in range(MIN_OBSERVATIONS):
            model.observe("claude-opus-5", "high", 4000)
            model.observe("claude-opus-5", "low", 2000)
        return model

    def test_context_prices_a_warm_prefix_as_refused(self):
        ctx = EffortPricingContext(
            model="claude-opus-5",
            prefix_tokens=150_000,
            cost_model=self._model(),
            expected_remaining_turns=2,
        )
        decision = ctx.decide(from_effort="high", to_effort="low")
        assert decision.switch is False

    def test_context_rides_along_on_a_busting_turn(self):
        ctx = EffortPricingContext(
            model="claude-opus-5",
            prefix_tokens=150_000,
            cost_model=self._model(),
            prefix_already_busting=True,
            expected_remaining_turns=2,
        )
        assert ctx.decide(from_effort="high", to_effort="low").switch is True

    def test_log_fields_are_single_line(self):
        ctx = EffortPricingContext(
            model="claude-opus-5",
            prefix_tokens=150_000,
            cost_model=self._model(),
            expected_remaining_turns=2,
        )
        rendered = ctx.decide(from_effort="high", to_effort="low").as_log_fields()
        assert "\n" not in rendered
        assert "switch=False" in rendered
        assert "break_even_turns=" in rendered
