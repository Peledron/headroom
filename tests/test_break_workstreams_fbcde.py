"""Breaker pass over the workstream F/B/C/D/E changes plus the HYBRID_REBASE claim.

Chaos-engineering style: each test tries to falsify a stated contract with a
realistic fault, not a contrived one. Findings are reported to the harness,
this file does not modify or soften any source under test.
"""

from __future__ import annotations

import time

import pytest

from headroom.proxy.hybrid_mode import HybridModeConfig, HybridModeController
from headroom.proxy.replay_capture import redact
from headroom.proxy.touch_registry import TouchEntry, TouchRegistry
from headroom.proxy.cache_reconciliation import CacheReconciliationLog


# ---------------------------------------------------------------------------
# Target: HYBRID_REBASE reachability claim
# ---------------------------------------------------------------------------


def test_hybrid_rebase_unreachable_under_realistic_low_pressure():
    """Reproduce the production scenario: CCR deferral keeps context small,
    the cache prefix goes fully cold (TTL long expired, p_alive == 0), and
    the potential rebase gain is large and positive. The claim under test is
    that HYBRID_REBASE can fire in this scenario. It cannot: the economic
    gate at hybrid_mode.py requires context_pressure >= 0.50, a bar that a
    system whose entire job is keeping context small will rarely ever cross.
    """
    controller = HybridModeController("anthropic", HybridModeConfig())

    # Warm the prefix past min_warm_turns=3 with a realistic frozen/live
    # split: a long agent session where content has been deferred by CCR to
    # keep total context around 20% of a 200k-token window.
    total_tokens = 40_000
    context_limit = 200_000
    pressure = total_tokens / context_limit  # 0.20, realistic for a
    # deferral-managed session, nowhere near the 0.50 economic-rebase floor
    # or the 0.92 pressure-rebase floor.

    frozen_message_count = 40
    message_count = 44
    cached_suffix_tokens = 32_000
    # Session returns after the 5m TTL has long expired: hazard survival is
    # at the floor, cache is stone cold, a rebase truly is "close to free".
    p_alive = 0.0
    # A generous estimate: compressing the frozen prefix now would save 30%
    # of total tokens, comfortably above the 8% minimum_savings_fraction gate.
    estimated_savings_tokens = int(total_tokens * 0.30)
    expected_reads = 10.0

    decisions = []
    for _ in range(6):  # drive past min_warm_turns and any transient phase
        decision = controller.decide(
            frozen_message_count=frozen_message_count,
            message_count=message_count,
            total_tokens=total_tokens,
            estimated_savings_tokens=estimated_savings_tokens,
            cached_suffix_tokens=cached_suffix_tokens,
            expected_reads=expected_reads,
            p_alive=p_alive,
            context_pressure=pressure,
        )
        decisions.append(decision)

    # Compute the gain directly to show it clears minimum_net_gain_tokens.
    gain = controller.net_rebase_gain(
        estimated_savings_tokens=estimated_savings_tokens,
        cached_suffix_tokens=cached_suffix_tokens,
        expected_reads=expected_reads,
        p_alive=p_alive,
    )
    assert gain >= HybridModeConfig().minimum_net_gain_tokens, (
        f"test setup invalid: gain {gain} does not even clear the token "
        "threshold, so this is not a fair reproduction"
    )
    savings_fraction = estimated_savings_tokens / total_tokens
    assert savings_fraction >= HybridModeConfig().minimum_savings_fraction

    # Fixed 2026-07-20: the economic branch no longer carries a pressure
    # floor, so a session whose gain and savings_fraction clear their
    # thresholds rebases even at low pressure, exactly the CCR-deferral
    # regime that used to be structurally excluded.
    fired = any(d.should_rebase for d in decisions)
    assert fired, (
        "the economic rebase must fire at low pressure once gain and "
        "savings_fraction clear their thresholds and p_alive is at the floor"
    )


def test_hybrid_rebase_fires_immediately_and_then_respects_cooldown():
    """After the fix the first eligible decide() call rebases and the
    controller drops into cooldown, so repeated calls do not thrash."""
    controller = HybridModeController("anthropic", HybridModeConfig())
    total_tokens = 40_000
    common = dict(
        frozen_message_count=40,
        message_count=44,
        total_tokens=total_tokens,
        estimated_savings_tokens=int(total_tokens * 0.30),
        cached_suffix_tokens=32_000,
        expected_reads=10.0,
        p_alive=0.0,
    )
    # The aged gate is deliberate hysteresis, so the fire may come a few
    # turns in. It must come within the first handful of eligible turns,
    # and cooldown must then hold it off from thrashing.
    decisions = [controller.decide(context_pressure=0.20, **common) for _ in range(8)]
    fire_indexes = [i for i, d in enumerate(decisions) if d.should_rebase]
    assert fire_indexes, "the economic rebase must fire at low pressure"
    first_fire = fire_indexes[0]
    followups = decisions[first_fire + 1 : first_fire + 4]
    assert not any(d.should_rebase for d in followups), (
        "cooldown must prevent back-to-back rebases from thrashing the cache"
    )


def test_hybrid_rebase_gain_formula_can_be_hugely_negative_at_full_pressure_but_pressure_rebase_still_requires_positive_gain():
    """pressure_rebase requires gain > 0.0 even at 0.92+ pressure; a session
    with cached_suffix_tokens large and estimated_savings_tokens near zero
    (nothing to gain from compressing) can sit at 0.95 pressure and still
    never rebase outside the 0.98 emergency floor, because the write-premium
    term on the suffix alone drives gain negative when p_alive is high."""
    controller = HybridModeController("anthropic", HybridModeConfig())
    common = dict(
        frozen_message_count=40,
        message_count=44,
        total_tokens=100_000,
        estimated_savings_tokens=100,  # negligible savings
        cached_suffix_tokens=90_000,  # big warm suffix
        expected_reads=10.0,
        p_alive=1.0,  # cache fully warm right now
        context_pressure=0.95,  # over the pressure_rebase threshold
    )
    for _ in range(3):
        decision = controller.decide(**common)
    decision = controller.decide(**common)
    gain = controller.net_rebase_gain(
        estimated_savings_tokens=100,
        cached_suffix_tokens=90_000,
        expected_reads=10.0,
        p_alive=1.0,
    )
    assert gain < 0.0
    assert not decision.should_rebase


# ---------------------------------------------------------------------------
# Target: replay_capture redaction correctness
# ---------------------------------------------------------------------------


def test_redact_misses_hyphenated_api_key_header_variant():
    """A very common real header spelling, ``api-key`` (Azure/OpenAI-style,
    hyphenated rather than underscored or bare), is not in _SENSITIVE_KEYS
    and survives redact() verbatim."""
    payload = {"headers": {"api-key": "sk-live-abcdef0123456789"}}
    redacted = redact(payload)
    assert redacted["headers"]["api-key"] == "[redacted]", (
        f"expected redaction, got leaked secret: {redacted}"
    )


def test_redact_misses_credential_shaped_string_embedded_in_a_curl_command():
    """The realistic leak vector: an agent's Bash tool_use carries a curl
    command with a literal bearer token as a plain string value, not under a
    sensitive-looking key. redact() is purely key-based, so this leaks the
    credential into the replay capture file verbatim."""
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "command": (
                                "curl -H 'Authorization: Bearer sk-ant-api03-"
                                "REALSECRETVALUE1234567890' https://api.example.com"
                            )
                        },
                    }
                ],
            }
        ]
    }
    redacted = redact(payload)
    command = redacted["messages"][0]["content"][0]["input"]["command"]
    assert "REALSECRETVALUE1234567890" not in command, (
        f"credential leaked through an unkeyed string value: {command!r}"
    )


def test_redact_handles_nested_credentials_under_unusual_casing():
    """Sanity check the documented behaviour still holds for depth and case
    variance that the design explicitly claims to cover."""
    payload = {
        "outer": [
            {"inner": {"X-Api-Key": "topsecret"}},
            {"AUTHORIZATION": "Bearer abc"},
            {"deep": {"deeper": {"deepest": {"Anthropic-API-Key": "zzz"}}}},
        ]
    }
    redacted = redact(payload)
    assert redacted["outer"][0]["inner"]["X-Api-Key"] == "[redacted]"
    assert redacted["outer"][1]["AUTHORIZATION"] == "[redacted]"
    assert redacted["outer"][2]["deep"]["deeper"]["deepest"]["Anthropic-API-Key"] == "[redacted]"


def test_redact_misses_credentials_inside_a_tuple():
    """redact() only recurses into dict and list. A tuple anywhere in the
    request body (e.g. constructed by a non-JSON code path, or surviving a
    partial deep-copy) passes through unexamined, and any dict nested inside
    it keeps its secrets."""
    payload = {"weird": ({"authorization": "Bearer leaked-via-tuple"},)}
    redacted = redact(payload)
    # A tuple is returned unchanged by the scalar fallback branch, so the
    # nested dict inside it was never visited by redact().
    inner = redacted["weird"][0]
    assert inner["authorization"] == "[redacted]", (
        f"tuple-wrapped credential dict was never redacted: {inner}"
    )


# ---------------------------------------------------------------------------
# Target: touch_registry ski-rental budget accounting under failure
# ---------------------------------------------------------------------------


def test_mark_touched_burns_budget_on_transient_network_failure():
    """run_due_touches (server.py) calls mark_touched(key, refreshed=False)
    from its except branch on ANY exception, including a transient network
    timeout that has nothing to do with the touch itself being wrong. Two
    consecutive transient failures exhaust max_touches_per_entry=2 and the
    session can never be touched again until the full TTL lapses, silently
    disabling the exact protection this module exists to provide."""
    registry = TouchRegistry(max_touches_per_entry=2)
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": "k", "content-type": "application/json"}
    body = {
        "model": "claude-x",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
    }
    registry.record(url, headers, body)
    key = next(iter(registry.snapshot()))["session"]

    # Simulate two transient failures (what run_due_touches does on any
    # exception from _retry_request, e.g. a connect timeout).
    registry.mark_touched(key, refreshed=False)
    registry.mark_touched(key, refreshed=False)

    due = registry.replayable(now=time.time(), due_only=False)
    assert due == [], (
        "entry became permanently unreplayable after two unrelated network "
        "blips, with zero successful touches ever sent -- the ski-rental "
        "protection is disabled for up to a full TTL by pure bad luck"
    )


def test_touch_body_fallback_forwards_stream_flag_removed_but_preserves_tools():
    """touch_body/touch_body_fallback strip 'stream' (correct: a touch must
    not open an SSE connection) but do NOT strip 'tools' or 'tool_choice'.
    A touch replaying the last real request with tools attached and
    max_tokens=1 as the fallback could, on some upstreams, actually attempt
    a tool call in one generated token, which is not the no-op the docstring
    promises ('must not extend the conversation or trigger tools')."""
    registry = TouchRegistry()
    entry = TouchEntry(
        url="https://api.anthropic.com/v1/messages",
        headers={"x-api-key": "k"},
        body={
            "model": "claude-x",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
            "tools": [{"name": "run_shell", "description": "runs a shell command"}],
            "tool_choice": {"type": "auto"},
        },
    )
    fallback = registry.touch_body_fallback(entry)
    assert fallback["max_tokens"] == 1
    assert "stream" not in fallback
    # This assertion documents the gap: tools are still present on a
    # max_tokens=1 fallback call the module's own docstring says must not
    # trigger tools.
    assert "tools" in fallback, (
        "if this now fails, tools are being stripped and the finding below "
        "no longer applies"
    )


# ---------------------------------------------------------------------------
# Target: cache_reconciliation classification of planned vs unplanned busts
# ---------------------------------------------------------------------------


def test_unplanned_bust_flag_does_not_consult_transforms_applied():
    """A cache read collapse caused by headroom's OWN deliberate transform
    (e.g. a structural bust that intentionally forces a fresh 5m write) is
    indistinguishable, in this log, from a genuine unplanned provider-side
    bust. transforms_applied is recorded on the payload but never consulted
    by is_unplanned_bust/_record, so a self-inflicted, intentional bust
    inflates the unplanned_bust counter operators use to detect a real
    regression."""
    log = CacheReconciliationLog(log_path="/tmp/does-not-matter-not-written.jsonl")
    now = 1000.0
    log._record(
        session_key="s1",
        request_id="r1",
        model="claude-x",
        billed_cache_read=10_000,
        billed_cache_creation=0,
        alive_fraction=1.0,
        first_diverged_index=None,
        transforms=[],
        now=now,
    )
    # Next turn: headroom itself deliberately forced a fresh write via the
    # structural-bust path (transforms explicitly says so), well within the
    # TTL window, not a scheduled expiry.
    record = log._record(
        session_key="s1",
        request_id="r2",
        model="claude-x",
        billed_cache_read=100,  # far under half of the 10_000 predicted
        billed_cache_creation=9_000,
        alive_fraction=0.1,
        first_diverged_index=3,
        transforms=["structural_bust_requires_fresh_5m"],
        now=now + 5.0,  # 5 seconds later, nowhere near the 300s TTL
    )
    # Fixed 2026-07-20: a transform label naming a deliberate bust
    # (PLANNED_BUST_MARKERS) now excludes the record from the alarm counter.
    assert record.unplanned_bust is False, (
        "a bust the transforms list itself attributes to headroom must not "
        "count as unplanned"
    )
    assert "structural_bust_requires_fresh_5m" in record.transforms


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
