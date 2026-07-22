from __future__ import annotations

from headroom.proxy.handlers.anthropic import (
    SUBAGENT_CAP_POSITIVE_SIGNAL,
    _record_subagent_cap_rewrite,
    _subagent_cap_rewrite_log,
    _wire_subagent_cap_target,
    subagent_cap_rewrite_snapshot,
)

FABLE = "claude-fable-5"
SONNET = "claude-sonnet-5"


def _subagent_system() -> str:
    return f"You are Claude Code, an agentic CLI tool. The exact model ID is {FABLE}."


def test_fable_is_not_rewritten_when_wire_fallback_is_off() -> None:
    assert (
        _wire_subagent_cap_target(
            fallback_enabled=False,
            is_subagent_request=True,
            model=FABLE,
            system=_subagent_system(),
            cap=SONNET,
        )
        is None
    )


def test_fable_subagent_is_rewritten_only_when_wire_fallback_is_on() -> None:
    assert (
        _wire_subagent_cap_target(
            fallback_enabled=True,
            is_subagent_request=True,
            model=FABLE,
            system=_subagent_system(),
            cap=SONNET,
        )
        == SONNET
    )


def test_main_1m_fable_is_not_rewritten_even_with_fallback_on() -> None:
    system = f"You are Claude Code. The exact model ID is {FABLE}[1m]."
    assert (
        _wire_subagent_cap_target(
            fallback_enabled=True,
            is_subagent_request=False,
            model=FABLE,
            system=system,
            cap=SONNET,
        )
        is None
    )


def test_main_thread_shaped_request_without_marker_is_not_capped() -> None:
    # The lax detector (is_subagent_request) can be True on a marker-absence
    # false positive, but the strict positive-token match on the exact model id
    # is the only thing that authorizes a rewrite. A main-thread-shaped system
    # that never renders the bare id must pass through even when the lax flag
    # is wrong, because that flag is uncertain evidence, not proof.
    system = (
        "You are Claude Code, an agentic CLI tool, in the user's primary "
        "session. The marker rendering was dropped by a retry."
    )
    assert (
        _wire_subagent_cap_target(
            fallback_enabled=True,
            is_subagent_request=True,
            model=FABLE,
            system=system,
            cap=SONNET,
        )
        is None
    )


def test_true_subagent_shaped_request_is_still_rewritten() -> None:
    # A genuine subagent shape: the exact bare id present as a standalone
    # token, no "[1m]" marker anywhere. This is the positive signal, and it
    # must still trigger the rewrite when the fallback is on.
    system = (
        "You are a subagent handling one delegated task. "
        f"The exact model ID is {FABLE}."
    )
    assert (
        _wire_subagent_cap_target(
            fallback_enabled=True,
            is_subagent_request=True,
            model=FABLE,
            system=system,
            cap=SONNET,
        )
        == SONNET
    )


class TestRewriteVisibility:
    """Every applied cap rewrite must be countable and readable from /stats."""

    def setup_method(self) -> None:
        _subagent_cap_rewrite_log.clear()

    def test_recorded_rewrite_carries_before_after_and_signal(self) -> None:
        _record_subagent_cap_rewrite("req-1", FABLE, SONNET)
        snap = subagent_cap_rewrite_snapshot()
        assert snap["rewrite_count"] >= 1
        last = snap["recent_rewrites"][-1]
        assert last["request_id"] == "req-1"
        assert last["model_before"] == FABLE
        assert last["model_after"] == SONNET
        assert last["signal"] == SUBAGENT_CAP_POSITIVE_SIGNAL

    def test_recent_rewrites_ring_is_bounded(self) -> None:
        for i in range(30):
            _record_subagent_cap_rewrite(f"req-{i}", FABLE, SONNET)
        snap = subagent_cap_rewrite_snapshot()
        assert snap["rewrite_count"] >= 30
        assert len(snap["recent_rewrites"]) <= 20
