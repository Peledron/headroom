"""Structural bust detection and hazard survival on PrefixCacheTracker.

Covers observe_client_churn (the pre-forward divergence scale that lets the
mutation gates treat already-busted history as free to compress) and
survival_p_alive (the empirical-hazard replacement for the linear idle proxy).
"""

from headroom.cache.prefix_tracker import PrefixCacheTracker
from headroom.proxy.handlers.anthropic import (
    _first_diverged_index_from_fraction,
    _structural_bust_requires_fresh_5m,
)


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _tracker_with_history(messages: list[dict]) -> PrefixCacheTracker:
    tracker = PrefixCacheTracker("anthropic")
    tracker.update_from_response(
        cache_read_tokens=1000,
        cache_write_tokens=100,
        messages=messages,
        original_messages=messages,
    )
    return tracker


def test_structural_bust_uses_fresh_5m_only_for_material_warm_prefix_loss() -> None:
    assert _structural_bust_requires_fresh_5m(0.33, 10)
    assert not _structural_bust_requires_fresh_5m(0.50, 10)
    assert not _structural_bust_requires_fresh_5m(0.10, 0)


class TestFirstDivergedIndexFromFraction:
    def test_no_prior_turn_returns_none(self) -> None:
        assert _first_diverged_index_from_fraction(1.0, 0) is None

    def test_fully_alive_returns_prior_length(self) -> None:
        assert _first_diverged_index_from_fraction(1.0, 4) == 4

    def test_mid_divergence_recovers_exact_index(self) -> None:
        # Mirrors TestObserveClientChurn.test_mid_divergence_reports_surviving_fraction:
        # k=2 of n=4, fraction 0.5.
        assert _first_diverged_index_from_fraction(0.5, 4) == 2

    def test_head_rewrite_recovers_zero(self) -> None:
        assert _first_diverged_index_from_fraction(0.0, 2) == 0

    def test_end_to_end_with_real_tracker(self) -> None:
        # The full pipeline: PrefixCacheTracker computes the fraction,
        # get_last_original_messages supplies n, the helper recovers k.
        history = [_msg("user", str(i)) for i in range(4)]
        tracker = _tracker_with_history(history)
        current = history[:2] + [_msg("user", "REWRITTEN")] + history[3:]
        fraction = tracker.observe_client_churn(current)
        prev_n = len(tracker.get_last_original_messages())
        assert _first_diverged_index_from_fraction(fraction, prev_n) == 2


class TestObserveClientChurn:
    def test_no_history_reports_fully_alive(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        assert tracker.observe_client_churn([_msg("user", "hi")]) == 1.0
        assert tracker.churn_depth_samples == []

    def test_stable_prefix_reports_fully_alive(self) -> None:
        history = [_msg("user", "a"), _msg("assistant", "b")]
        tracker = _tracker_with_history(history)
        current = history + [_msg("user", "c")]
        assert tracker.observe_client_churn(current) == 1.0
        assert tracker.churn_depth_samples == []

    def test_mid_divergence_reports_surviving_fraction(self) -> None:
        history = [_msg("user", str(i)) for i in range(4)]
        tracker = _tracker_with_history(history)
        current = history[:2] + [_msg("user", "REWRITTEN")] + history[3:]
        assert tracker.observe_client_churn(current) == 0.5
        assert tracker.churn_depth_samples == [0.5]

    def test_head_rewrite_reports_dead(self) -> None:
        history = [_msg("user", "a"), _msg("assistant", "b")]
        tracker = _tracker_with_history(history)
        current = [_msg("user", "REWRITTEN"), _msg("assistant", "b")]
        assert tracker.observe_client_churn(current) == 0.0
        assert tracker.churn_depth_samples == [0.0]

    def test_truncated_history_counts_as_divergence(self) -> None:
        history = [_msg("user", str(i)) for i in range(4)]
        tracker = _tracker_with_history(history)
        assert tracker.observe_client_churn(history[:2]) == 0.5

    def test_cache_control_movement_is_not_churn(self) -> None:
        # The canonicalizer ignores cache-directive noise, so a client that
        # moves its breakpoint marker must not register as a structural bust.
        history = [_msg("user", "a"), _msg("assistant", "b")]
        tracker = _tracker_with_history(history)
        moved = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "a",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            _msg("assistant", "b"),
        ]
        assert tracker.observe_client_churn(moved) == 1.0

    def test_samples_ring_is_bounded(self) -> None:
        history = [_msg("user", str(i)) for i in range(4)]
        tracker = _tracker_with_history(history)
        for _ in range(40):
            tracker.observe_client_churn(
                [_msg("user", "X")] + history[1:]
            )
        assert len(tracker.churn_depth_samples) == 32


class TestSurvivalPAlive:
    def test_too_few_samples_returns_fallback(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        tracker.record_turn_gap(10.0)
        assert tracker.survival_p_alive(300.0, 0.4) == 0.4

    def test_fast_cadence_lifts_estimate_above_linear(self) -> None:
        # All gaps well inside the TTL: the empirical rate is 1.0, so the
        # blend must land strictly above a pessimistic linear fallback.
        tracker = PrefixCacheTracker("anthropic")
        for _ in range(8):
            tracker.record_turn_gap(5.0)
        result = tracker.survival_p_alive(300.0, 0.2)
        assert 0.2 < result <= 1.0
        # Full ring: weight is 8/12, so blended = 8/12*1.0 + 4/12*0.2.
        assert abs(result - (8 / 12 + (4 / 12) * 0.2)) < 1e-9

    def test_slow_cadence_pulls_estimate_below_linear(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        for _ in range(8):
            tracker.record_turn_gap(900.0)
        result = tracker.survival_p_alive(300.0, 0.9)
        assert result < 0.9

    def test_result_clamped_and_ttl_guard(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        for _ in range(4):
            tracker.record_turn_gap(5.0)
        assert tracker.survival_p_alive(0.0, 5.0) == 1.0
        assert tracker.survival_p_alive(0.0, -5.0) == 0.0


class TestExpectedReadsWithinTtl:
    def test_too_few_samples_returns_fallback(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        tracker.record_turn_gap(10.0)
        assert tracker.expected_reads_within_ttl(300.0, 10.0) == 10.0

    def test_rapid_cadence_forecasts_more_reads_than_default(self) -> None:
        # Every observed gap inside the TTL: p caps at 0.95, forecast 19.
        tracker = PrefixCacheTracker("anthropic")
        for _ in range(8):
            tracker.record_turn_gap(5.0)
        r = tracker.expected_reads_within_ttl(300.0, 10.0)
        expected = (8 / 12) * 19.0 + (4 / 12) * 10.0
        assert abs(r - expected) < 1e-9
        assert r > 10.0

    def test_sporadic_cadence_forecasts_fewer_reads(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        for _ in range(8):
            tracker.record_turn_gap(900.0)
        r = tracker.expected_reads_within_ttl(300.0, 10.0)
        # p = 0 so the forecast term is 0, only the fallback blend remains.
        assert abs(r - (4 / 12) * 10.0) < 1e-9

    def test_ttl_guard_returns_fallback(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        for _ in range(4):
            tracker.record_turn_gap(5.0)
        assert tracker.expected_reads_within_ttl(0.0, 7.0) == 7.0


class TestExpectedSessionReads:
    """A mask outlives the cache run, so it is priced over the session."""

    def _advance(self, tracker: PrefixCacheTracker, turns: int) -> None:
        for _ in range(turns):
            tracker.record_turn_gap(5.0)
            tracker.update_from_response(0, 0, [])

    def test_short_session_defers_to_the_ttl_run_forecast(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        self._advance(tracker, 3)
        assert tracker.expected_session_reads(
            300.0, 10.0
        ) == tracker.expected_reads_within_ttl(300.0, 10.0)

    def test_long_session_outgrows_the_ttl_run_cap(self) -> None:
        # The within-TTL forecast cannot exceed 19 by construction. A session
        # that has already run 60 turns should price a permanent edit well
        # past that, which is what admits a batch the old horizon declined.
        tracker = PrefixCacheTracker("anthropic")
        self._advance(tracker, 60)
        run = tracker.expected_reads_within_ttl(300.0, 10.0)
        session = tracker.expected_session_reads(300.0, 10.0)
        assert run <= 19.0
        assert session == 60.0
        assert session > run

    def test_horizon_is_capped(self) -> None:
        tracker = PrefixCacheTracker("anthropic")
        self._advance(tracker, 260)
        assert tracker.expected_session_reads(300.0, 10.0) == 200.0

    def test_never_below_the_within_ttl_forecast(self) -> None:
        # Sporadic cadence, so the run forecast is small, but the floor still
        # holds: this must never be more conservative than what it replaces.
        tracker = PrefixCacheTracker("anthropic")
        for _ in range(40):
            tracker.record_turn_gap(900.0)
            tracker.update_from_response(0, 0, [])
        assert tracker.expected_session_reads(
            300.0, 10.0
        ) >= tracker.expected_reads_within_ttl(300.0, 10.0)
