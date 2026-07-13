from __future__ import annotations

from benchmarks.optical_compression_benchmark import run_optical_benchmark


def test_optical_benchmark_compares_all_requested_modes() -> None:
    results = run_optical_benchmark(model="gpt-4o-mini", paragraphs=120)
    assert set(results) == {
        "raw",
        "text",
        "optical_low",
        "optical_high",
        "hybrid_text_optical",
    }
    assert results["text"].billed_input_tokens < results["raw"].billed_input_tokens
    assert results["optical_low"].billed_input_tokens == results["raw"].billed_input_tokens
    assert results["optical_low"].pages == 0
