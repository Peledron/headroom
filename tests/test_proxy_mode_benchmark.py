"""Tests for local token/cache mode benchmark harness."""

from benchmarks.proxy_mode_benchmark import run_local_benchmark
from headroom.proxy.handlers.openai import _effective_openai_context_limit


def test_local_mode_benchmark_shows_compression_and_cache_tradeoff() -> None:
    results = run_local_benchmark(turns=6)

    baseline = results["baseline"]
    token = results["token"]
    cache = results["cache"]
    hybrid = results["hybrid"]

    assert token.total_tokens_saved > 0
    assert cache.total_tokens_saved > 0
    assert token.total_sent_tokens < baseline.total_sent_tokens
    assert cache.total_sent_tokens < baseline.total_sent_tokens
    assert hybrid.total_tokens_saved > 0
    assert hybrid.total_sent_tokens < baseline.total_sent_tokens

    # Cache mode should preserve prefix better than token mode.
    assert cache.total_cache_read_tokens >= token.total_cache_read_tokens
    assert hybrid.total_cache_read_tokens >= token.total_cache_read_tokens


def test_openai_local_mode_benchmark_exercises_token_and_hybrid_modes() -> None:
    results = run_local_benchmark(turns=6, provider="openai")

    baseline = results["baseline"]
    token = results["token"]
    hybrid = results["hybrid"]

    assert token.total_tokens_saved > 0
    assert hybrid.total_tokens_saved > 0
    assert token.total_sent_tokens < baseline.total_sent_tokens
    assert hybrid.total_sent_tokens < baseline.total_sent_tokens
    assert hybrid.total_cache_read_tokens >= token.total_cache_read_tokens


def test_codex_context_limit_is_smaller_than_api_model_limit() -> None:
    class Provider:
        @staticmethod
        def get_context_limit(model: str) -> int:
            return 1_050_000

    provider = Provider()
    assert _effective_openai_context_limit(provider, "gpt-5.6-sol", "codex") == 272_000
    assert _effective_openai_context_limit(provider, "gpt-5.6-sol", None) == 1_050_000
