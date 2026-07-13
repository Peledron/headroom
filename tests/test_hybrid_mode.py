from __future__ import annotations

from headroom.proxy.hybrid_mode import HybridModeConfig, HybridModeController, HybridPhase


def _decide(controller: HybridModeController, **overrides: object):
    args = {
        "frozen_message_count": 4,
        "message_count": 5,
        "total_tokens": 20_000,
        "estimated_savings_tokens": 4_000,
        "cached_suffix_tokens": 12_000,
        "expected_reads": 20.0,
        "p_alive": 1.0,
        "context_pressure": 0.75,
        "write_multiplier": 1.25,
    }
    args.update(overrides)
    return controller.decide(**args)


def test_cold_prefix_establishes_without_freezing() -> None:
    controller = HybridModeController("anthropic")
    decision = _decide(controller, frozen_message_count=0, cached_suffix_tokens=0)
    assert decision.phase is HybridPhase.COLD_PREFIX
    assert decision.frozen_message_count == 0
    assert decision.should_rebase is False


def test_warm_prefix_only_exposes_live_delta() -> None:
    controller = HybridModeController("anthropic")
    decision = _decide(controller, estimated_savings_tokens=100, context_pressure=0.2)
    assert decision.phase is HybridPhase.LIVE_DELTA
    assert decision.frozen_message_count == 4
    assert decision.should_rebase is False


def test_rebase_requires_age_and_positive_provider_net_gain() -> None:
    controller = HybridModeController(
        "anthropic", HybridModeConfig(min_warm_turns=2, rebase_cooldown_turns=3)
    )
    first = _decide(controller, expected_reads=100.0)
    second = _decide(controller, expected_reads=100.0)
    assert first.should_rebase is False
    assert second.should_rebase is True
    assert second.phase is HybridPhase.REBASE_PENDING
    assert second.frozen_message_count == 0
    assert second.net_gain_tokens > 0


def test_rebase_cooldown_prevents_oscillation() -> None:
    controller = HybridModeController(
        "openai", HybridModeConfig(min_warm_turns=1, rebase_cooldown_turns=2)
    )
    rebase = _decide(controller, write_multiplier=1.0)
    cooldown = _decide(controller, write_multiplier=1.0)
    assert rebase.should_rebase is True
    assert cooldown.phase is HybridPhase.REBASE_COOLDOWN
    assert cooldown.should_rebase is False
    assert cooldown.frozen_message_count == 4


def test_low_gain_does_not_bust_a_warm_prefix() -> None:
    controller = HybridModeController("anthropic", HybridModeConfig(min_warm_turns=1))
    decision = _decide(
        controller,
        estimated_savings_tokens=100,
        cached_suffix_tokens=50_000,
        expected_reads=1.0,
        context_pressure=0.6,
    )
    assert decision.should_rebase is False
    assert decision.net_gain_tokens < 0


def test_hard_context_limit_can_force_one_rebase() -> None:
    controller = HybridModeController("anthropic", HybridModeConfig(min_warm_turns=99))
    decision = _decide(
        controller,
        estimated_savings_tokens=0,
        expected_reads=0.0,
        context_pressure=0.99,
    )
    assert decision.should_rebase is True
    assert decision.reason == "hard_context_limit"
