"""Tools/system growth ahead of the messages is a total prefix bust.

Deferred tool loading appends schemas to the `tools` array, which the
provider matches ahead of every message. Measured on the replay corpus at
1.63M tokens of forced write, 14% of all billed write, at roughly 90k per
occurrence. Message-only churn detection reported the prefix fully alive on
those turns, so the bust was invisible and the queued compression never got
flushed on the one turn where the rewrite was already paid for.
"""

from headroom.cache.prefix_tracker import PrefixCacheTracker


def _messages(n: int) -> list[dict]:
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def _tracker() -> PrefixCacheTracker:
    return PrefixCacheTracker("anthropic")


def test_identical_messages_with_grown_tools_read_as_total_bust():
    tracker = _tracker()
    msgs = _messages(40)

    # First turn establishes the baseline head.
    assert tracker.observe_client_churn(msgs, head_fingerprint="head-16-tools") == 1.0

    # Same messages, byte for byte. Only the tools array grew, exactly what
    # ToolSearch does when it resolves a deferred schema.
    alive = tracker.observe_client_churn(msgs, head_fingerprint="head-21-tools")

    assert alive == 0.0, "tools growth kills the whole prefix, not part of it"


def test_stable_head_leaves_message_churn_untouched():
    tracker = _tracker()
    msgs = _messages(40)

    tracker.observe_client_churn(msgs, head_fingerprint="stable")
    # Pure append under a stable head is the healthy case and must stay 1.0.
    alive = tracker.observe_client_churn(_messages(48), head_fingerprint="stable")

    assert alive == 1.0


def test_head_fingerprint_is_optional():
    # Callers that do not pass a fingerprint keep the old message-only path.
    tracker = _tracker()
    tracker.observe_client_churn(_messages(10))
    assert tracker.observe_client_churn(_messages(20)) == 1.0


def test_bust_is_recorded_in_the_churn_depth_ring():
    # Anchor placement reads this ring. A head bust that never reaches it
    # would leave anchors placed against a prefix that no longer exists.
    tracker = _tracker()
    msgs = _messages(40)
    tracker.observe_client_churn(msgs, head_fingerprint="a")
    before = len(tracker.churn_depth_samples)

    tracker.observe_client_churn(msgs, head_fingerprint="b")

    assert len(tracker.churn_depth_samples) == before + 1
    assert tracker.churn_depth_samples[-1] == 0.0


def test_head_bust_counts_as_a_provably_dead_prefix():
    """A tools-array bust must reach prefix_known_dead, not just p_alive.

    _prefix_certainly_lapsed used to consider idle time alone. A head bust is
    not idle-related, so compression stayed gated on ordinary economics during
    the one turn where the whole transcript is re-billed anyway.
    """
    from headroom.proxy.handlers.anthropic import _prefix_certainly_lapsed

    class _Fresh:
        def peek_idle_seconds(self):
            return 0.0

    tracker = _Fresh()
    assert _prefix_certainly_lapsed(tracker, 3600.0) is False
    assert _prefix_certainly_lapsed(tracker, 3600.0, head_bust=True) is True


def test_head_bust_short_circuits_before_ttl_is_consulted():
    """A head bust is fatal even when the TTL is unknown or unset."""
    from headroom.proxy.handlers.anthropic import _prefix_certainly_lapsed

    class _NoIdle:
        def peek_idle_seconds(self):
            raise RuntimeError("no idle clock")

    assert _prefix_certainly_lapsed(_NoIdle(), None, head_bust=True) is True
    assert _prefix_certainly_lapsed(_NoIdle(), None) is False
