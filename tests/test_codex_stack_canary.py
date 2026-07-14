from __future__ import annotations

from benchmarks.codex_stack_canary import (
    RunResult,
    ToolEvent,
    TurnResult,
    _aggregate,
    _api_cost,
    _prepare_codex_home,
    _score,
    _usage_delta,
)


def _turn(**overrides: object) -> TurnResult:
    values: dict[str, object] = {
        "latency_seconds": 1.0,
        "input_tokens": 1_000_000,
        "cached_input_tokens": 250_000,
        "output_tokens": 100_000,
        "reasoning_output_tokens": 10_000,
        "cumulative_input_tokens": 1_000_000,
        "cumulative_cached_input_tokens": 250_000,
        "cumulative_output_tokens": 100_000,
        "cumulative_reasoning_output_tokens": 10_000,
        "tool_items": 2,
        "command_items": 1,
        "mcp_items": 1,
        "mcp_tools": ["tokensave.tokensave_context"],
        "commands": ["rtk rg pattern"],
        "tool_events": [
            ToolEvent(
                kind="mcp",
                name="tokensave.tokensave_context",
                status="completed",
                serialized_chars=500,
            )
        ],
        "tool_output_chars": 500,
        "response": "{}",
    }
    values.update(overrides)
    return TurnResult(**values)  # type: ignore[arg-type]


def test_score_requires_all_repository_facts() -> None:
    response = """{
      "codex_context_limit": 272000,
      "api_context_limit": 1050000,
      "context_selector": "_effective_openai_context_limit",
      "cold_prefix": "HybridPhase.COLD_PREFIX",
      "warm_prefix": ["HybridPhase.WARM_PREFIX", "HybridPhase.LIVE_DELTA"],
      "rebase_gates": ["economic_rebase", "pressure_rebase"],
      "cooldown": "HybridPhase.REBASE_COOLDOWN",
      "hard_limit": "hard_context_limit",
      "controller": "HybridModeController"
    }"""
    assert _score(response) == (9, 9)


def test_api_cost_uses_uncached_cached_and_output_rates() -> None:
    # Luna: $1/M uncached, $0.10/M cached, $6/M output.
    assert _api_cost("gpt-5.6-luna", [_turn()]) == 1.375


def test_usage_delta_converts_resumed_thread_counters() -> None:
    previous = _turn()
    current = _turn(
        input_tokens=1_400_000,
        cached_input_tokens=400_000,
        output_tokens=130_000,
        reasoning_output_tokens=15_000,
        cumulative_input_tokens=1_400_000,
        cumulative_cached_input_tokens=400_000,
        cumulative_output_tokens=130_000,
        cumulative_reasoning_output_tokens=15_000,
    )
    delta = _usage_delta(current, previous)
    assert delta.input_tokens == 400_000
    assert delta.cached_input_tokens == 150_000
    assert delta.output_tokens == 30_000
    assert delta.reasoning_output_tokens == 5_000


def test_prepare_codex_home_links_only_minimum_for_bare(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "auth.json").write_text("{}", encoding="utf-8")
    (source / "models_cache.json").write_text("{}", encoding="utf-8")
    (source / "config.toml").write_text("model = 'secret-config'", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(source))

    home = _prepare_codex_home(tmp_path / "nested" / "run", "bare")

    assert (home / "auth.json").is_symlink()
    assert (home / "models_cache.json").is_symlink()
    assert not (home / "config.toml").exists()


def test_aggregate_keeps_tool_layers_separate() -> None:
    run = RunResult(
        arm="full",
        model="gpt-5.6-luna",
        repeat=0,
        score=9,
        score_max=9,
        api_equivalent_usd=1.25,
        headroom_project="test-project",
        headroom_metrics=None,
        turns=[_turn()],
    )
    row = _aggregate([run])[0]
    assert row["accuracy_pct"] == 100.0
    assert row["command_items"] == 1
    assert row["mcp_items"] == 1
    assert row["mcp_tools"] == {"tokensave.tokensave_context": 1}
    assert row["tool_output_chars"] == 500
