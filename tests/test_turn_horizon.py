"""The horizon every cache gate prices against should come from what happened.

Lindy answers the same for a typo fix and a thirty-file refactor. These tests
pin the self-labelling loop that replaces it, the conservatism that keeps a
wrong horizon from spending rewrites, and the refusal to predict from too few
episodes.
"""

from __future__ import annotations

import pytest

from headroom.proxy.turn_horizon import (
    HORIZON_QUANTILE,
    MAX_EPISODES_PER_BUCKET,
    MAX_HORIZON,
    MIN_EPISODES,
    TurnFeatures,
    TurnHorizonModel,
    expected_remaining_turns,
)


def _hard_turn(session_turn: int) -> TurnFeatures:
    """A turn deep in a long tool run, working hard, growing context fast."""
    return TurnFeatures(
        session_turn=session_turn,
        turns_since_user_ask=20,
        output_tokens_ewma=7000,
        prefix_growth_per_turn=9000,
        error_fraction=0.2,
    )


def _easy_turn(session_turn: int) -> TurnFeatures:
    """A turn right after a user ask, cheap output, barely growing."""
    return TurnFeatures(
        session_turn=session_turn,
        turns_since_user_ask=1,
        output_tokens_ewma=300,
        prefix_growth_per_turn=200,
        error_fraction=0.0,
    )


class TestBucketing:
    def test_difficulty_separates_the_buckets(self):
        assert _hard_turn(50).bucket() != _easy_turn(50).bucket()

    def test_similar_turns_share_a_bucket(self):
        """Bins are wide on purpose: a per-turn bucket never reaches quorum."""
        a = TurnFeatures(session_turn=40, turns_since_user_ask=12, output_tokens_ewma=2500)
        b = TurnFeatures(session_turn=45, turns_since_user_ask=15, output_tokens_ewma=2900)
        assert a.bucket() == b.bucket()

    def test_a_boundary_crossing_changes_the_bucket(self):
        low = TurnFeatures(output_tokens_ewma=1999)
        high = TurnFeatures(output_tokens_ewma=2001)
        assert low.bucket() != high.bucket()


class TestSelfLabelling:
    def test_a_boundary_labels_every_pending_turn(self):
        model = TurnHorizonModel()
        for turn in range(10, 20):
            model.observe_turn(_hard_turn(turn))
        assert model.observe_task_boundary(session_turn=30) == 10

    def test_a_turn_at_the_boundary_itself_is_not_an_episode(self):
        """Zero remaining turns is not a horizon, it is the boundary."""
        model = TurnHorizonModel()
        model.observe_turn(_hard_turn(30))
        assert model.observe_task_boundary(session_turn=30) == 0

    def test_pending_turns_are_cleared_after_labelling(self):
        model = TurnHorizonModel()
        model.observe_turn(_hard_turn(10))
        model.observe_task_boundary(session_turn=20)
        assert model.observe_task_boundary(session_turn=40) == 0

    def test_labels_are_capped_at_the_trusted_horizon(self):
        model = TurnHorizonModel()
        for _ in range(MIN_EPISODES):
            model.observe_turn(_hard_turn(1))
            model.observe_task_boundary(session_turn=5000)
        assert model.predict(_hard_turn(1)) == MAX_HORIZON


class TestPrediction:
    def test_it_declines_below_quorum(self):
        """One episode short is still a refusal, not a guess from eleven."""
        model = TurnHorizonModel()
        for _ in range(MIN_EPISODES - 1):
            model.observe_turn(_hard_turn(0))
            model.observe_task_boundary(session_turn=8)
        assert model.episode_count(_hard_turn(0)) == MIN_EPISODES - 1
        assert model.predict(_hard_turn(0)) is None

    def test_it_predicts_once_a_bucket_has_evidence(self):
        model = TurnHorizonModel()
        for _ in range(MIN_EPISODES):
            model.observe_turn(_hard_turn(0))
            model.observe_task_boundary(session_turn=8)
        assert model.predict(_hard_turn(0)) == 8

    def test_it_predicts_the_pessimistic_end_not_the_mean(self):
        """A wrong-high horizon spends a rewrite. A wrong-low one misses a saving."""
        model = TurnHorizonModel()
        # Twelve episodes: eleven short tasks and one very long one.
        for _ in range(MIN_EPISODES - 1):
            model.observe_turn(_hard_turn(0))
            model.observe_task_boundary(session_turn=4)
        model.observe_turn(_hard_turn(0))
        model.observe_task_boundary(session_turn=40)

        predicted = model.predict(_hard_turn(0))
        assert predicted == 4
        mean = ((MIN_EPISODES - 1) * 4 + 40) / MIN_EPISODES
        assert predicted is not None and predicted < mean

    def test_the_quantile_is_below_the_median(self):
        assert HORIZON_QUANTILE < 0.5

    def test_hard_and_easy_buckets_predict_differently(self):
        """The whole point: difficulty, not elapsed turns, drives the answer."""
        model = TurnHorizonModel()
        for _ in range(MIN_EPISODES):
            model.observe_turn(_hard_turn(0))
            model.observe_task_boundary(session_turn=35)
        for _ in range(MIN_EPISODES):
            model.observe_turn(_easy_turn(0))
            model.observe_task_boundary(session_turn=3)

        hard = model.predict(_hard_turn(0))
        easy = model.predict(_easy_turn(0))
        assert hard is not None and easy is not None
        assert hard > easy

    def test_a_bucket_forgets_its_oldest_episodes(self):
        model = TurnHorizonModel()
        for _ in range(MAX_EPISODES_PER_BUCKET + 20):
            model.observe_turn(_hard_turn(0))
            model.observe_task_boundary(session_turn=6)
        assert model.episode_count(_hard_turn(0)) == MAX_EPISODES_PER_BUCKET


class TestFallbackContract:
    def test_lindy_is_used_until_the_table_can_answer(self):
        model = TurnHorizonModel()
        horizon, source = expected_remaining_turns(_hard_turn(0), fallback=9.0, model=model)
        assert horizon == 9.0
        assert source == "lindy"

    def test_the_learned_value_wins_once_it_exists(self):
        model = TurnHorizonModel()
        for _ in range(MIN_EPISODES):
            model.observe_turn(_hard_turn(0))
            model.observe_task_boundary(session_turn=25)
        horizon, source = expected_remaining_turns(_hard_turn(0), fallback=9.0, model=model)
        assert horizon == 25.0
        assert source == "learned"

    def test_both_paths_respect_the_trusted_cap(self):
        model = TurnHorizonModel()
        horizon, source = expected_remaining_turns(_hard_turn(0), fallback=10_000, model=model)
        assert horizon == MAX_HORIZON
        assert source == "lindy"

    def test_a_negative_fallback_is_floored(self):
        model = TurnHorizonModel()
        horizon, _ = expected_remaining_turns(_hard_turn(0), fallback=-5, model=model)
        assert horizon == 0.0

    def test_a_broken_model_leaves_the_caller_no_worse_off(self):
        class Exploding(TurnHorizonModel):
            def predict(self, _features: TurnFeatures) -> float | None:
                raise RuntimeError("table corrupt")

        horizon, source = expected_remaining_turns(
            _hard_turn(0), fallback=7.0, model=Exploding()
        )
        assert horizon == 7.0
        assert source == "lindy"


class TestPersistence:
    def test_round_trip_preserves_a_learned_bucket(self):
        source = TurnHorizonModel()
        for _ in range(MIN_EPISODES):
            source.observe_turn(_hard_turn(0))
            source.observe_task_boundary(session_turn=18)

        target = TurnHorizonModel()
        assert target.restore_state(source.export_state()) == MIN_EPISODES
        assert target.predict(_hard_turn(0)) == source.predict(_hard_turn(0))

    def test_pending_turns_are_not_exported(self):
        """Their label depends on a boundary the next process will never see."""
        source = TurnHorizonModel()
        source.observe_turn(_hard_turn(0))
        target = TurnHorizonModel()
        assert target.restore_state(source.export_state()) == 0

    def test_a_junk_snapshot_is_discarded_not_partly_applied(self):
        target = TurnHorizonModel()
        assert target.restore_state(None) == 0
        assert target.restore_state({"episodes": "nope"}) == 0
        assert target.restore_state({"episodes": [[["a"], [1]]]}) == 0
        assert target.restore_state({"episodes": [[[1, 2], []]]}) == 0
        assert target.episode_count() == 0

    def test_out_of_range_labels_are_dropped(self):
        target = TurnHorizonModel()
        restored = target.restore_state(
            {"episodes": [[[0, 0, 0, 0, 0], [5, -1, 0, MAX_HORIZON + 100, 7]]]}
        )
        assert restored == 2

    def test_the_snapshot_carries_no_conversation_content(self):
        source = TurnHorizonModel()
        for _ in range(MIN_EPISODES):
            source.observe_turn(_hard_turn(0))
            source.observe_task_boundary(session_turn=18)
        blob = source.export_state()
        assert set(blob) == {"episodes"}
        for bucket, values in blob["episodes"]:
            assert all(isinstance(x, int) for x in bucket)
            assert all(isinstance(v, int) for v in values)


class TestHorizonFeedsTheGates:
    def test_a_learned_horizon_changes_a_compaction_verdict(self):
        """End to end: difficulty evidence flips a gate that Lindy would decline."""
        from headroom.proxy.compaction_advisor import advise_compaction

        model = TurnHorizonModel()
        for _ in range(MIN_EPISODES):
            model.observe_turn(_hard_turn(0))
            model.observe_task_boundary(session_turn=30)

        lindy_horizon, _ = expected_remaining_turns(
            _hard_turn(0), fallback=2.0, model=TurnHorizonModel()
        )
        learned_horizon, source = expected_remaining_turns(
            _hard_turn(0), fallback=2.0, model=model
        )
        assert source == "learned"

        common = dict(
            live_tokens=150_000,
            retained_tokens=30_000,
            summary_output_tokens=6_000,
            at_task_boundary=True,
        )
        assert advise_compaction(expected_remaining_turns=lindy_horizon, **common).recommend is False
        assert advise_compaction(expected_remaining_turns=learned_horizon, **common).recommend is True


@pytest.mark.parametrize("turns_since_ask", [0, 5, 50, 500])
def test_bucketing_never_raises_on_extremes(turns_since_ask: int):
    features = TurnFeatures(
        session_turn=turns_since_ask * 3,
        turns_since_user_ask=turns_since_ask,
        output_tokens_ewma=float(turns_since_ask) * 100,
        prefix_growth_per_turn=float(turns_since_ask) * 50,
        error_fraction=min(1.0, turns_since_ask / 100),
    )
    assert len(features.bucket()) == 5
