from __future__ import annotations

from headroom.cli.codex_tool_budget import evaluate, is_exploration_tool


def _payload(turn: str, command: str = "rtk rg pattern") -> dict[str, object]:
    return {
        "session_id": "session-1",
        "turn_id": turn,
        "tool_name": "Bash",
        "tool_input": {"command": command},
    }


def test_classifies_reads_but_not_tests_or_edits() -> None:
    assert is_exploration_tool("Bash", {"command": "rtk proxy sed -n '1,20p' app.py"})
    assert is_exploration_tool("mcp__tokensave__tokensave_context", {})
    assert is_exploration_tool("mcp__serena__find_symbol", {})
    assert is_exploration_tool("tokensave.tokensave_context", {})
    assert is_exploration_tool("serena.find_symbol", {})
    assert not is_exploration_tool("mcp__serena__activate_project", {})
    assert not is_exploration_tool("mcp__serena__initial_instructions", {})
    assert not is_exploration_tool("serena.activate_project", {})
    assert not is_exploration_tool("serena.initial_instructions", {})
    assert not is_exploration_tool("Bash", {"command": "rtk uv run pytest -q"})
    assert not is_exploration_tool("apply_patch", {"command": "*** Begin Patch"})


def test_warns_on_last_allowed_call_and_denies_later_calls(tmp_path) -> None:
    assert evaluate(_payload("turn-1"), state_dir=tmp_path, budget=2) is None

    warning = evaluate(_payload("turn-1", "rtk sed -n 1,20p app.py"), state_dir=tmp_path, budget=2)
    assert warning is not None
    assert "additionalContext" in warning["hookSpecificOutput"]

    denial = evaluate(_payload("turn-1", "rtk rg another"), state_dir=tmp_path, budget=2)
    assert denial is not None
    assert denial["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_budget_resets_for_each_turn_and_ignores_non_reads(tmp_path) -> None:
    for turn in ("turn-1", "turn-2"):
        assert evaluate(_payload(turn), state_dir=tmp_path, budget=2) is None
    assert evaluate(
        _payload("turn-1", "rtk uv run pytest -q"), state_dir=tmp_path, budget=2
    ) is None
