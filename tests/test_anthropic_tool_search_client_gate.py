from __future__ import annotations

from headroom.proxy.handlers.anthropic import (
    _ensure_claude_tool_search_history_compatibility,
    _guard_large_claude_tool_search,
    _should_inject_server_tool_search,
)


def test_claude_code_never_gets_proxy_owned_tool_search() -> None:
    assert not _should_inject_server_tool_search(
        provider_name="anthropic",
        anthropic_backend=None,
        client="claude-code",
        setting="true",
    )


def test_non_claude_direct_anthropic_client_can_opt_in() -> None:
    assert _should_inject_server_tool_search(
        provider_name="anthropic",
        anthropic_backend=None,
        client="opencode",
        setting="true",
    )


def test_backend_and_disabled_setting_remain_off() -> None:
    assert not _should_inject_server_tool_search(
        provider_name="anthropic",
        anthropic_backend=object(),
        client="opencode",
        setting="true",
    )
    assert not _should_inject_server_tool_search(
        provider_name="anthropic",
        anthropic_backend=None,
        client="opencode",
        setting="0",
    )


def test_large_claude_context_materializes_deferred_tools() -> None:
    tools = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {"name": "WebSearch", "defer_loading": True, "input_schema": {}},
        {"name": "Read", "input_schema": {}},
    ]
    guarded = _guard_large_claude_tool_search(
        tools,
        client="claude-code",
        input_tokens=200_000,
    )
    assert guarded is not tools
    assert [tool["name"] for tool in guarded] == [
        "tool_search_tool_regex",
        "WebSearch",
        "Read",
    ]
    assert all("defer_loading" not in tool for tool in guarded)


def test_referenced_search_tool_is_restored_without_deferral() -> None:
    tools = [
        {"name": "WebSearch", "defer_loading": True, "input_schema": {}},
        {"name": "Read", "input_schema": {}},
    ]
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "name": "tool_search_tool_regex",
                    "id": "srvtoolu_1",
                    "input": {"query": "WebSearch"},
                }
            ],
        }
    ]

    compatible = _ensure_claude_tool_search_history_compatibility(
        tools,
        messages,
        client="claude-code",
    )

    assert compatible[0] == {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }
    assert compatible[1:] == [
        {"name": "WebSearch", "input_schema": {}},
        {"name": "Read", "input_schema": {}},
    ]


def test_search_history_compatibility_is_noop_without_reference() -> None:
    tools = [{"name": "Read", "input_schema": {}}]
    assert (
        _ensure_claude_tool_search_history_compatibility(
            tools,
            [{"role": "user", "content": "continue"}],
            client="claude-code",
        )
        is tools
    )


def test_large_context_guard_leaves_small_and_non_claude_requests_unchanged() -> None:
    tools = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {"name": "WebSearch", "defer_loading": True, "input_schema": {}},
    ]
    assert (
        _guard_large_claude_tool_search(tools, client="claude-code", input_tokens=99_999)
        is tools
    )
    assert (
        _guard_large_claude_tool_search(tools, client="opencode", input_tokens=200_000)
        is tools
    )
