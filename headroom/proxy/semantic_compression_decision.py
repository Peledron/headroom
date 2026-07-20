"""Cost and safety gate for future extractive tool-result compression.

This module deliberately makes no content mutation. It centralizes the hard
exclusions and the net-gain calculation so any later compressor has to pass the
same cache-aware admission rule before it can be wired into the request path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticCompressionCandidate:
    block_type: str
    tokens_before: int
    estimated_tokens_after: int
    price_factor: float = 1.0
    expected_downstream_reads: float = 0.0
    compressor_cost_usd: float = 0.0
    latency_cost_usd: float = 0.0
    cache_invalidation_risk_usd: float = 0.0
    token_price_usd: float = 0.0
    informational: bool = False
    in_live_cached_prefix: bool = False
    structured_payload: bool = False


@dataclass(frozen=True)
class SemanticCompressionDecision:
    apply: bool
    gain_usd: float
    reason: str


def decide_semantic_compression(
    candidate: SemanticCompressionCandidate,
) -> SemanticCompressionDecision:
    """Apply the investigation's gain formula with non-negotiable exclusions."""
    if candidate.block_type in {"thinking", "tool_use"}:
        return SemanticCompressionDecision(False, 0.0, "protected_block_type")
    if candidate.block_type != "tool_result" or not candidate.informational:
        return SemanticCompressionDecision(False, 0.0, "not_allowlisted")
    if candidate.structured_payload:
        return SemanticCompressionDecision(False, 0.0, "structured_payload")
    if candidate.in_live_cached_prefix:
        return SemanticCompressionDecision(False, 0.0, "live_cached_prefix")

    tokens_saved = max(0, candidate.tokens_before - candidate.estimated_tokens_after)
    immediate = tokens_saved * candidate.token_price_usd * candidate.price_factor
    downstream = (
        tokens_saved
        * candidate.token_price_usd
        * max(0.0, candidate.expected_downstream_reads)
    )
    gain = (
        immediate
        + downstream
        - max(0.0, candidate.compressor_cost_usd)
        - max(0.0, candidate.latency_cost_usd)
        - max(0.0, candidate.cache_invalidation_risk_usd)
    )
    return SemanticCompressionDecision(gain > 0.0, gain, "positive_gain" if gain > 0 else "non_positive_gain")
