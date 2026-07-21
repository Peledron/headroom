from headroom.proxy.semantic_compression_decision import (
    SemanticCompressionCandidate,
    decide_semantic_compression,
)


def _candidate(**overrides):  # noqa: ANN003, ANN202
    values = {
        "block_type": "tool_result",
        "tokens_before": 1000,
        "estimated_tokens_after": 400,
        "token_price_usd": 0.000003,
        "expected_downstream_reads": 2,
        "informational": True,
    }
    values.update(overrides)
    return SemanticCompressionCandidate(**values)


def test_positive_gain_informational_tool_result_is_admitted() -> None:
    decision = decide_semantic_compression(_candidate())
    assert decision.apply
    assert decision.gain_usd > 0


def test_thinking_tool_use_and_live_prefix_are_never_admitted() -> None:
    assert not decide_semantic_compression(_candidate(block_type="thinking")).apply
    assert not decide_semantic_compression(_candidate(block_type="tool_use")).apply
    assert not decide_semantic_compression(_candidate(in_live_cached_prefix=True)).apply


def test_structured_and_negative_gain_candidates_are_rejected() -> None:
    assert not decide_semantic_compression(_candidate(structured_payload=True)).apply
    assert not decide_semantic_compression(
        _candidate(compressor_cost_usd=10.0)
    ).apply
