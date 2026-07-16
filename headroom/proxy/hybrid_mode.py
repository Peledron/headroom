"""State machine for cache-safe hybrid proxy compression.

Hybrid mode establishes a deterministic compressed prefix, keeps that prefix
byte-stable while the provider cache is warm, and compresses only the live
delta. A historical rebase is exceptional and must clear provider-specific
net-cost, minimum-age, hysteresis, and cooldown gates.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum


class HybridPhase(str, Enum):
    COLD_PREFIX = "cold_prefix"
    WARM_PREFIX = "warm_prefix"
    LIVE_DELTA = "live_delta"
    REBASE_PENDING = "rebase_pending"
    REBASE_COOLDOWN = "rebase_cooldown"


@dataclass(frozen=True, slots=True)
class HybridModeConfig:
    min_warm_turns: int = 3
    rebase_cooldown_turns: int = 8
    minimum_net_gain_tokens: float = 256.0
    minimum_savings_fraction: float = 0.08
    pressure_rebase_threshold: float = 0.92
    pressure_hard_limit: float = 0.98
    adaptive_ttl: bool = True
    subagent_ttl_5m: bool = True
    net_cost_mutations: bool = True
    strip_deep_reminders: bool = True
    mid_anchor: bool = True
    canon_model_id: bool = True

    @classmethod
    def from_environment(cls) -> HybridModeConfig:
        """Resolve legacy feature flags into one hybrid policy."""

        def enabled(name: str) -> bool:
            value = os.environ.get(name)
            return value != "0" if value is not None else True

        return cls(
            adaptive_ttl=enabled("HEADROOM_ADAPTIVE_TTL"),
            subagent_ttl_5m=enabled("HR_SUBAGENT_TTL_5M"),
            net_cost_mutations=enabled("HEADROOM_NET_COST_POLICY"),
            strip_deep_reminders=enabled("HR_STRIP_DEEP_REMINDERS"),
            mid_anchor=enabled("HR_MID_ANCHOR"),
            canon_model_id=enabled("HR_CANON_MODEL_ID"),
        )


@dataclass(frozen=True, slots=True)
class HybridModeDecision:
    phase: HybridPhase
    frozen_message_count: int
    should_rebase: bool = False
    net_gain_tokens: float = 0.0
    reason: str = ""
    generation: int = 0


class HybridModeController:
    """Session-scoped hybrid controller with explicit hysteresis."""

    _ECONOMICS = {
        "anthropic": (0.10, 1.25),
        "bedrock": (0.10, 1.25),
        "gemini": (0.10, 1.00),
        "openai": (0.50, 1.00),
    }

    def __init__(self, provider: str, config: HybridModeConfig | None = None):
        self.provider = provider
        self.config = config or HybridModeConfig.from_environment()
        self.phase = HybridPhase.COLD_PREFIX
        self.generation = 0
        self._warm_turns = 0
        self._cooldown_remaining = 0

    @staticmethod
    def _finite_nonnegative(value: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return number if math.isfinite(number) and number > 0.0 else 0.0

    def net_rebase_gain(
        self,
        *,
        estimated_savings_tokens: float,
        cached_suffix_tokens: float,
        expected_reads: float,
        p_alive: float,
        write_multiplier: float | None = None,
    ) -> float:
        """Return provider-priced gain in plain input-token cost units."""
        read, default_write = self._ECONOMICS.get(self.provider, (0.50, 1.00))
        write = self._finite_nonnegative(write_multiplier or default_write) or default_write
        saving = self._finite_nonnegative(estimated_savings_tokens)
        suffix = self._finite_nonnegative(cached_suffix_tokens)
        reads = self._finite_nonnegative(expected_reads)
        try:
            alive = float(p_alive)
        except (TypeError, ValueError, OverflowError):
            alive = 1.0
        if not math.isfinite(alive):
            alive = 1.0
        alive = min(1.0, max(0.0, alive))
        return saving * (write + read * (reads - 1.0)) - alive * (write - read) * (suffix + saving)

    def decide(
        self,
        *,
        frozen_message_count: int,
        message_count: int,
        total_tokens: int,
        estimated_savings_tokens: int,
        cached_suffix_tokens: int,
        expected_reads: float,
        p_alive: float,
        context_pressure: float,
        write_multiplier: float | None = None,
    ) -> HybridModeDecision:
        frozen = min(max(0, frozen_message_count), max(0, message_count))
        if frozen == 0 or cached_suffix_tokens <= 0:
            self.phase = HybridPhase.COLD_PREFIX
            self._warm_turns = 0
            return HybridModeDecision(
                phase=self.phase,
                frozen_message_count=0,
                reason="establish_prefix",
                generation=self.generation,
            )

        self._warm_turns += 1
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self.phase = HybridPhase.REBASE_COOLDOWN
            return HybridModeDecision(
                phase=self.phase,
                frozen_message_count=frozen,
                reason="rebase_cooldown",
                generation=self.generation,
            )

        savings_fraction = (
            max(0, estimated_savings_tokens) / total_tokens if total_tokens > 0 else 0.0
        )
        gain = self.net_rebase_gain(
            estimated_savings_tokens=estimated_savings_tokens,
            cached_suffix_tokens=cached_suffix_tokens,
            expected_reads=expected_reads,
            p_alive=p_alive,
            write_multiplier=write_multiplier,
        )
        pressure = context_pressure if math.isfinite(context_pressure) else 0.0
        aged = self._warm_turns >= self.config.min_warm_turns
        economic_rebase = (
            aged
            and savings_fraction >= self.config.minimum_savings_fraction
            and gain >= self.config.minimum_net_gain_tokens
            and pressure >= 0.50
        )
        pressure_rebase = aged and pressure >= self.config.pressure_rebase_threshold and gain > 0.0
        emergency_rebase = pressure >= self.config.pressure_hard_limit

        if economic_rebase or pressure_rebase or emergency_rebase:
            self.phase = HybridPhase.REBASE_PENDING
            self.generation += 1
            self._warm_turns = 0
            self._cooldown_remaining = self.config.rebase_cooldown_turns
            reason = "hard_context_limit" if emergency_rebase else "positive_net_gain"
            return HybridModeDecision(
                phase=self.phase,
                frozen_message_count=0,
                should_rebase=True,
                net_gain_tokens=gain,
                reason=reason,
                generation=self.generation,
            )

        self.phase = HybridPhase.LIVE_DELTA if frozen < message_count else HybridPhase.WARM_PREFIX
        return HybridModeDecision(
            phase=self.phase,
            frozen_message_count=frozen,
            net_gain_tokens=gain,
            reason="compress_live_delta" if frozen < message_count else "preserve_warm_prefix",
            generation=self.generation,
        )
