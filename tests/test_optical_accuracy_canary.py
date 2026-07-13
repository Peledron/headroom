from __future__ import annotations

import pytest

from benchmarks.optical_accuracy_canary import (
    CONTROLLED_HEADROOM_ENV,
    build_canary_corpus,
    build_cases,
    clean_worker_environment,
    run_accuracy_canary,
    score_answer,
)


def test_accuracy_canary_fixture_has_summary_and_sparse_exact_markers() -> None:
    corpus = build_canary_corpus()

    assert "Incident codename: Alder" in corpus
    assert "CITRINE-4821" in corpus
    assert "LANTERN-7734" in corpus
    assert "ORCHID-9056" in corpus
    assert "Deployment state transition 3: green" in corpus
    assert len(build_cases()) == 6


def test_accuracy_score_is_deterministic_and_json_fence_tolerant() -> None:
    expected = {"marker": "CITRINE-4821"}

    assert score_answer('{"marker":"CITRINE-4821"}', expected) == 1
    assert score_answer('```json\n{"marker":"citrine-4821"}\n```', expected) == 1
    assert score_answer('{"marker":"wrong"}', expected) == 0
    assert score_answer("not json", expected) == 0


def test_clean_worker_environment_removes_inherited_headroom_tuning() -> None:
    env = clean_worker_environment(
        {
            "PATH": "/bin",
            "OPENAI_API_KEY": "preserved-for-worker",
            "HEADROOM_MODE": "cache",
            "HEADROOM_UNKNOWN_EXPERIMENT": "1",
        }
    )

    assert env["PATH"] == "/bin"
    assert env["OPENAI_API_KEY"] == "preserved-for-worker"
    assert "HEADROOM_UNKNOWN_EXPERIMENT" not in env
    assert {key: env[key] for key in CONTROLLED_HEADROOM_ENV} == CONTROLLED_HEADROOM_ENV


def test_live_canary_requires_provider_credential(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_accuracy_canary(
            provider="openai",
            model="gpt-4o-mini",
            live=True,
            max_cases=1,
        )


@pytest.mark.parametrize(
    ("provider", "model"),
    (("openai", "gpt-4o-mini"), ("anthropic", "claude-haiku-4-5")),
)
def test_dry_run_reports_cost_and_all_modes_without_api_calls(provider: str, model: str) -> None:
    result = run_accuracy_canary(
        provider=provider,
        model=model,
        live=False,
        max_cases=1,
    )

    assert result["live"] is False
    assert set(result["results"]) == {"raw", "text", "optical", "text_optical"}
    assert result["results"]["raw"]["estimated_savings_pct"] == pytest.approx(0.0)
    assert result["results"]["text"]["estimated_input_tokens"] > 0
    assert result["results"]["text"]["correct_fields"] is None
    assert result["results"]["text"]["actual_input_tokens"] is None
