from __future__ import annotations

from headroom.cli.claude_tool_budget import evaluate, is_claude_exploration_tool


def _event(event: str, tool_name: str = "Bash") -> dict[str, object]:
    return {
        "session_id": "claude-session-1",
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": {"command": "rtk rg pattern"},
    }


def test_budget_resets_on_each_user_prompt(tmp_path) -> None:
    prompt_context = evaluate(_event("UserPromptSubmit"), state_dir=tmp_path)
    assert prompt_context is not None
    assert "2 calls" in prompt_context["hookSpecificOutput"]["additionalContext"]
    assert evaluate(_event("PreToolUse"), state_dir=tmp_path, budget=1) is not None
    denied = evaluate(_event("PreToolUse"), state_dir=tmp_path, budget=1)
    assert denied is not None
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    assert evaluate(_event("UserPromptSubmit"), state_dir=tmp_path) is not None
    allowed = evaluate(_event("PreToolUse"), state_dir=tmp_path, budget=1)
    assert allowed is not None
    assert "additionalContext" in allowed["hookSpecificOutput"]


def test_lifecycle_calls_do_not_consume_budget(tmp_path) -> None:
    assert evaluate(_event("UserPromptSubmit"), state_dir=tmp_path) is not None
    for tool_name in (
        "mcp__serena__activate_project",
        "mcp__serena__initial_instructions",
    ):
        assert evaluate(
            _event("PreToolUse", tool_name), state_dir=tmp_path, budget=1
        ) is None

    warning = evaluate(_event("PreToolUse"), state_dir=tmp_path, budget=1)
    assert warning is not None
    assert "additionalContext" in warning["hookSpecificOutput"]


def test_non_exploration_events_and_tools_are_ignored(tmp_path) -> None:
    assert evaluate(_event("PostToolUse"), state_dir=tmp_path) is None
    payload = _event("PreToolUse")
    payload["tool_input"] = {"command": "rtk proxy pytest -q"}
    assert evaluate(payload, state_dir=tmp_path) is None


def test_claude_builtin_readers_are_exploration() -> None:
    for tool_name in ("Read", "Grep", "Glob", "ToolSearch"):
        assert is_claude_exploration_tool(tool_name, {})
