from __future__ import annotations

import json

from benchmarks.claude_stack_canary import (
    HEADROOM_ANTHROPIC_URL,
    Usage,
    _parse_stream,
    _settings,
)


def test_usage_cost_uses_sonnet5_intro_cache_rates() -> None:
    usage = Usage(
        uncached_input=1_000_000,
        cache_write_5m=1_000_000,
        cache_write_1h=1_000_000,
        cache_read=1_000_000,
        output=1_000_000,
    )
    assert usage.api_equivalent_usd() == 18.7


def test_parse_stream_sums_usage_and_tools() -> None:
    stdout = "\n".join(
        (
            '{"type":"assistant","message":{"usage":{"input_tokens":2,'
            '"cache_creation_input_tokens":30,"cache_read_input_tokens":40,'
            '"cache_creation":{"ephemeral_5m_input_tokens":10,'
            '"ephemeral_1h_input_tokens":20},"output_tokens":5},'
            '"content":[{"type":"tool_use","name":"Bash"}]}}',
            '{"type":"result","result":"{}","total_cost_usd":0.1}',
        )
    )
    turn = _parse_stream(stdout)
    assert turn.usage.uncached_input == 2
    assert turn.usage.cache_write_5m == 10
    assert turn.usage.cache_write_1h == 20
    assert turn.usage.cache_read == 40
    assert turn.usage.output == 5
    assert turn.tool_names == ["Bash"]
    assert turn.provider_cost_usd == 0.1


def test_settings_select_headroom_or_direct_anthropic(tmp_path) -> None:
    settings = tmp_path / "settings.json"
    _settings(settings, headroom=True, budget_command="budget")
    assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"] == (
        HEADROOM_ANTHROPIC_URL
    )
    _settings(settings, headroom=False, budget_command="budget")
    assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"] == (
        "https://api.anthropic.com"
    )
