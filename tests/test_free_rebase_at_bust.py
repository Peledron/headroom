"""Compression is free on the turn the prefix is already gone.

A rebase normally has to earn back the cost of re-writing the cached suffix.
When the prefix has provably outlived its TTL tier that suffix is re-billed
regardless, so the rewrite penalty is zero and any positive saving is worth
taking. On the live corpus these turns were 3.4% of traffic and 40% of all
cache-write tokens, and they re-sent uncompressed history they were paying to
write anyway.
"""

from headroom.proxy.handlers.anthropic import _prefix_certainly_lapsed
from headroom.proxy.hybrid_mode import HybridModeController


def _controller() -> HybridModeController:
    return HybridModeController("anthropic")


def _decide(controller: HybridModeController, **overrides):
    kwargs = dict(
        message_count=40,
        frozen_message_count=30,
        estimated_savings_tokens=8_000,
        cached_suffix_tokens=150_000,
        expected_reads=20.0,
        p_alive=0.9,
        context_pressure=0.4,
        total_tokens=200_000,
    )
    kwargs.update(overrides)
    return controller.decide(**kwargs)


def test_dead_prefix_rebases_even_though_a_warm_prefix_would_not():
    controller = _controller()
    warm = _decide(controller)
    dead = _decide(controller, prefix_known_dead=True)
    assert dead.should_rebase is True
    assert dead.reason == "free_rebase_at_bust"
    # The warm turn faces a 150k rewrite to save 8k, so it must decline.
    assert warm.should_rebase is False


def test_dead_prefix_with_nothing_to_save_does_not_rebase():
    controller = _controller()
    decision = _decide(controller, estimated_savings_tokens=0, prefix_known_dead=True)
    assert decision.should_rebase is False


class _Tracker:
    def __init__(self, idle):
        self._idle = idle

    def peek_idle_seconds(self):
        return self._idle


def test_lapse_detection_requires_a_provable_gap():
    assert _prefix_certainly_lapsed(_Tracker(600.0), 300.0) is True
    assert _prefix_certainly_lapsed(_Tracker(120.0), 300.0) is False
    # An unknown idle time must not be read as a lapse.
    assert _prefix_certainly_lapsed(_Tracker(None), 300.0) is False
    assert _prefix_certainly_lapsed(_Tracker(600.0), None) is False
