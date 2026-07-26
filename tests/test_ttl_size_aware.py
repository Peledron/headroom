"""Size-aware TTL promotion.

The reactive rule only looked at recent cadence, so a session that idled past
the 5m tier re-wrote its whole prefix and then dropped straight back to 5m as
soon as the cadence sped up again. Measured on the live corpus, 32 of 948 turns
took that path and burned 40% of all cache-write tokens. Promotion has to weigh
the size of the prefix at risk, not just the gap pattern.
"""

from headroom.cache.prefix_tracker import PrefixCacheTracker


def _tracker(cached_tokens: int, steady_write: float | None) -> PrefixCacheTracker:
    tracker = PrefixCacheTracker(provider="anthropic")
    tracker._cached_token_count = cached_tokens
    tracker._steady_write_tokens = steady_write
    return tracker


def test_large_prefix_after_a_breach_prefers_1h():
    tracker = _tracker(cached_tokens=200_000, steady_write=7_000.0)
    tracker.record_turn_gap(30.0)
    tracker.record_turn_gap(600.0)  # one demonstrated breach
    tracker.record_turn_gap(20.0)
    tracker.record_turn_gap(15.0)  # cadence is fast again
    assert tracker.prefers_long_ttl() is True
    assert tracker.recommended_ttl() == "1h"


def test_small_prefix_stays_on_5m_even_after_a_breach():
    # Re-writing a small prefix is cheaper than paying the 1h premium.
    tracker = _tracker(cached_tokens=5_000, steady_write=2_000.0)
    tracker.record_turn_gap(30.0)
    tracker.record_turn_gap(600.0)
    tracker.record_turn_gap(20.0)
    assert tracker.prefers_long_ttl() is False


def test_fast_session_without_any_breach_stays_on_5m():
    tracker = _tracker(cached_tokens=300_000, steady_write=7_000.0)
    for _ in range(6):
        tracker.record_turn_gap(20.0)
    assert tracker.prefers_long_ttl() is False
    assert tracker.recommended_ttl() == "5m"


def test_breach_counter_survives_the_gap_window_rolling_over():
    # _turn_gaps keeps only the last 8 gaps; the breach evidence must not age
    # out with it, otherwise a long session forgets it ever idled.
    tracker = _tracker(cached_tokens=200_000, steady_write=7_000.0)
    tracker.record_turn_gap(600.0)
    for _ in range(12):
        tracker.record_turn_gap(10.0)
    assert tracker._ttl_breaches == 1
    assert tracker.prefers_long_ttl() is True
