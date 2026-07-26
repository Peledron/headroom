"""Compaction should be priced, not triggered by a context threshold.

The default rule fires when the window is nearly full, which is the worst
moment: mid-task, on a prefix that has been reading cheaply for hundreds of
turns. These tests pin the arithmetic that replaces it, the three moments that
change the answer, and the refusal to compact mid-task on a thin margin.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.compaction_advisor import (
    DEFAULT_RETAINED_FRACTION,
    MAX_CREDITED_TURNS,
    MIN_OBSERVATIONS,
    OFF_BOUNDARY_MARGIN,
    CompactionShapeModel,
    advise_compaction,
    deferral_credit,
)


class TestArithmetic:
    def test_the_headline_case_repays_in_a_few_turns(self):
        """150k compacting to 30k with a 6k summary, the documented example."""
        advice = advise_compaction(
            live_tokens=150_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            expected_remaining_turns=1,
            at_task_boundary=True,
        )
        # cost = 30_000 * 1.25 + 6_000 * 5 = 67_500
        # per_turn_gain = 120_000 * 0.1 = 12_000
        assert advice.cost == pytest.approx(67_500)
        assert advice.per_turn_gain == pytest.approx(12_000)
        assert advice.break_even_turns == pytest.approx(67_500 / 12_000)
        assert advice.break_even_turns is not None  # None only when the gain is zero
        assert advice.break_even_turns < 6
        assert advice.tokens_reclaimed == 120_000

    def test_a_long_horizon_at_a_boundary_recommends(self):
        advice = advise_compaction(
            live_tokens=150_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            expected_remaining_turns=20,
            at_task_boundary=True,
        )
        assert advice.recommend is True
        assert advice.reason == "pays_back_at_task_boundary"
        assert advice.urgency == "advisory"

    def test_a_short_horizon_at_a_boundary_declines(self):
        advice = advise_compaction(
            live_tokens=150_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            expected_remaining_turns=2,
            at_task_boundary=True,
        )
        assert advice.recommend is False
        assert advice.reason == "horizon_too_short"

    def test_the_horizon_is_capped(self):
        advice = advise_compaction(
            live_tokens=150_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            expected_remaining_turns=10_000,
            at_task_boundary=True,
        )
        assert advice.expected_gain == pytest.approx(12_000 * MAX_CREDITED_TURNS)

    def test_no_context_is_not_a_recommendation(self):
        advice = advise_compaction(live_tokens=0, expected_remaining_turns=50)
        assert advice.recommend is False
        assert advice.reason == "no_context"

    def test_a_summary_that_reclaims_nothing_is_refused(self):
        advice = advise_compaction(
            live_tokens=50_000,
            retained_tokens=50_000,
            summary_output_tokens=1_000,
            expected_remaining_turns=40,
            at_task_boundary=True,
        )
        assert advice.recommend is False
        assert advice.reason == "no_reduction"


class TestTheThreeMoments:
    def test_mid_task_requires_a_wider_margin(self):
        """Same numbers, different moment: the boundary is what flips it."""
        common: dict[str, Any] = {
            "live_tokens": 150_000,
            "retained_tokens": 30_000,
            "summary_output_tokens": 6_000,
            "expected_remaining_turns": 8,
        }
        at_boundary = advise_compaction(at_task_boundary=True, **common)
        mid_task = advise_compaction(at_task_boundary=False, **common)

        assert at_boundary.recommend is True
        assert mid_task.recommend is False
        assert mid_task.reason == "below_mid_task_margin"
        # Both saw identical arithmetic. Only the required margin differed.
        assert at_boundary.expected_gain == mid_task.expected_gain
        assert at_boundary.cost == mid_task.cost

    def test_mid_task_still_recommends_when_the_margin_clears(self):
        advice = advise_compaction(
            live_tokens=150_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            expected_remaining_turns=MAX_CREDITED_TURNS,
            at_task_boundary=False,
        )
        assert advice.recommend is True
        assert advice.reason == "pays_back_mid_task"
        assert advice.expected_gain > advice.cost * OFF_BOUNDARY_MARGIN

    def test_an_already_busting_turn_only_pays_for_the_summary(self):
        """The prefix write is being spent regardless, so it is not a cost."""
        warm = advise_compaction(
            live_tokens=150_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            expected_remaining_turns=4,
            at_task_boundary=True,
        )
        busting = advise_compaction(
            live_tokens=150_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            expected_remaining_turns=4,
            at_task_boundary=True,
            prefix_already_busting=True,
        )
        assert busting.cost == pytest.approx(6_000 * 5.0)
        assert busting.cost < warm.cost
        assert busting.recommend is True
        assert warm.recommend is False

    def test_past_the_threshold_it_is_no_longer_a_choice(self):
        advice = advise_compaction(
            live_tokens=190_000,
            context_limit=200_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            expected_remaining_turns=1,
            at_task_boundary=False,
        )
        assert advice.recommend is True
        assert advice.reason == "forced_soon"
        assert advice.urgency == "forced_soon"

    def test_below_the_threshold_the_limit_does_not_force_anything(self):
        advice = advise_compaction(
            live_tokens=100_000,
            context_limit=200_000,
            retained_tokens=20_000,
            summary_output_tokens=4_000,
            expected_remaining_turns=1,
            at_task_boundary=True,
        )
        assert advice.reason != "forced_soon"


class TestShapeModel:
    def test_defaults_apply_before_anything_is_measured(self):
        shape = CompactionShapeModel()
        assert shape.retained_fraction == DEFAULT_RETAINED_FRACTION
        advice = advise_compaction(
            live_tokens=100_000,
            expected_remaining_turns=40,
            at_task_boundary=True,
            shape=shape,
        )
        assert advice.tokens_reclaimed == 80_000

    def test_it_learns_from_witnessed_compactions(self):
        shape = CompactionShapeModel()
        for _ in range(MIN_OBSERVATIONS):
            shape.observe(live_tokens=100_000, retained_tokens=50_000, summary_tokens=10_000)
        assert shape.retained_fraction == pytest.approx(0.5)
        assert shape.summary_fraction == pytest.approx(0.1)

    def test_nonsense_observations_are_ignored(self):
        shape = CompactionShapeModel()
        shape.observe(live_tokens=0, retained_tokens=10, summary_tokens=1)
        shape.observe(live_tokens=100, retained_tokens=0, summary_tokens=1)
        # A "compaction" that grew the prefix is a misread, not a measurement.
        shape.observe(live_tokens=100, retained_tokens=200, summary_tokens=1)
        assert shape.observation_count == 0

    def test_explicit_tokens_override_the_learned_shape(self):
        shape = CompactionShapeModel()
        for _ in range(MIN_OBSERVATIONS):
            shape.observe(live_tokens=100_000, retained_tokens=90_000, summary_tokens=1_000)
        advice = advise_compaction(
            live_tokens=100_000,
            retained_tokens=10_000,
            summary_output_tokens=2_000,
            expected_remaining_turns=40,
            at_task_boundary=True,
            shape=shape,
        )
        assert advice.tokens_reclaimed == 90_000


class TestShapePersistence:
    def test_round_trip_preserves_the_estimate(self):
        source = CompactionShapeModel()
        for _ in range(MIN_OBSERVATIONS):
            source.observe(live_tokens=100_000, retained_tokens=40_000, summary_tokens=8_000)

        target = CompactionShapeModel()
        assert target.restore_state(source.export_state()) == MIN_OBSERVATIONS
        assert target.retained_fraction == pytest.approx(source.retained_fraction)
        assert target.summary_fraction == pytest.approx(source.summary_fraction)

    def test_a_junk_snapshot_leaves_the_default_intact(self):
        target = CompactionShapeModel()
        assert target.restore_state(None) == 0
        assert target.restore_state({"count": "many"}) == 0
        assert target.restore_state({"count": 3}) == 0
        assert target.retained_fraction == DEFAULT_RETAINED_FRACTION


class TestDeferralCredit:
    """The credit a mutation earns for pushing compaction further away.

    Read arithmetic alone treats a prefix as if it will be read forever. Near
    the context limit that is wrong: the prefix is about to be replaced by a
    summary, at a price. These pin when that term is allowed to count.
    """

    def _credit(self, **overrides: Any):
        kwargs: dict[str, Any] = dict(
            live_tokens=160_000,
            reclaimed_tokens=20_000,
            tokens_per_turn=2_000.0,
            expected_remaining_turns=20.0,
            context_limit=200_000,
        )
        kwargs.update(overrides)
        return deferral_credit(**kwargs)

    def test_far_from_the_limit_there_is_nothing_to_defer(self):
        advice = self._credit(live_tokens=60_000)
        assert advice.credit == 0.0
        assert advice.reason == "not_imminent"

    def test_near_the_limit_a_real_reclaim_earns_credit(self):
        advice = self._credit()
        assert advice.reason == "defers_compaction"
        assert advice.credit > 0.0
        assert advice.turns_bought == pytest.approx(10.0)
        # 200k * 0.92 = 184k forced; 24k of headroom at 2k a turn.
        assert advice.turns_until_forced == pytest.approx(12.0)

    def test_a_reclaim_too_small_to_buy_turns_is_churn(self):
        advice = self._credit(reclaimed_tokens=1_000)
        assert advice.credit == 0.0
        assert advice.reason == "too_little_bought"

    def test_reclaiming_nothing_earns_nothing(self):
        advice = self._credit(reclaimed_tokens=0)
        assert advice.credit == 0.0
        assert advice.reason == "nothing_reclaimed"

    def test_credit_rises_as_the_limit_gets_closer(self):
        near = self._credit(live_tokens=180_000)
        further = self._credit(live_tokens=145_000)
        assert near.credit > further.credit
        assert near.avoided_probability > further.avoided_probability

    def test_credit_never_exceeds_the_compaction_it_dodges(self):
        advice = self._credit(reclaimed_tokens=120_000)
        assert advice.credit <= advice.compaction_cost
        assert 0.0 <= advice.avoided_probability <= 1.0

    def test_a_session_about_to_end_gets_little_credit(self):
        """A short horizon rarely reaches compaction, so dodging it is worth less."""
        ending = self._credit(expected_remaining_turns=1.0)
        continuing = self._credit(expected_remaining_turns=30.0)
        assert ending.credit < continuing.credit

    def test_a_learned_shape_moves_the_credit(self):
        shape = CompactionShapeModel()
        for _ in range(MIN_OBSERVATIONS):
            # A summary that retains almost everything reclaims little, so the
            # compaction it dodges is cheap and the credit falls.
            shape.observe(live_tokens=100_000, retained_tokens=95_000, summary_tokens=500)
        assert self._credit(shape=shape).credit < self._credit().credit

    def test_no_context_limit_means_no_arithmetic(self):
        assert self._credit(context_limit=0).reason == "no_context"
        assert self._credit(live_tokens=0).reason == "no_context"

    def test_log_fields_carry_the_decision(self):
        fields = self._credit().as_log_fields()
        assert "reason=defers_compaction" in fields
        assert "turns_bought=" in fields
        assert "p_avoided=" in fields
