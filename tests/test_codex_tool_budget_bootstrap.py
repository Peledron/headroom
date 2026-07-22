from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.cli import bootstrap_agent_tools


def test_codex_tool_budget_hook_preserves_existing_hooks(tmp_path: Path) -> None:
    hooks_path = tmp_path / "hooks.json"
    existing = {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "existing-cleaner"}],
                }
            ]
        },
        "unrelated": True,
    }
    hooks_path.write_text(json.dumps(existing), encoding="utf-8")

    assert bootstrap_agent_tools.ensure_codex_tool_budget_hook(hooks_path)
    assert not bootstrap_agent_tools.ensure_codex_tool_budget_hook(hooks_path)

    installed = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert installed["unrelated"] is True
    assert installed["hooks"]["PostToolUse"] == existing["hooks"]["PostToolUse"]
    groups = installed["hooks"]["PreToolUse"]
    assert len(groups) == 1
    assert groups[0]["matcher"] == bootstrap_agent_tools.CODEX_TOOL_BUDGET_MATCHER
    assert groups[0]["hooks"] == [
        {
            "type": "command",
            "command": bootstrap_agent_tools.CODEX_TOOL_BUDGET_COMMAND,
        }
    ]


def test_codex_tool_budget_hook_rejects_invalid_json(tmp_path: Path) -> None:
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text("not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid Codex hooks JSON"):
        bootstrap_agent_tools.ensure_codex_tool_budget_hook(hooks_path)


def test_codex_tool_budget_hook_updates_stale_matcher(tmp_path: Path) -> None:
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": bootstrap_agent_tools.CODEX_TOOL_BUDGET_COMMAND,
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    assert bootstrap_agent_tools.ensure_codex_tool_budget_hook(hooks_path)
    installed = json.loads(hooks_path.read_text(encoding="utf-8"))
    assert installed["hooks"]["PreToolUse"][0]["matcher"] == (
        bootstrap_agent_tools.CODEX_TOOL_BUDGET_MATCHER
    )
