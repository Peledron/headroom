from __future__ import annotations

import json
from pathlib import Path

from headroom.cli import bootstrap_agent_tools


def test_claude_budget_hooks_preserve_existing_settings_and_backup(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    original = {
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
    settings_path.write_text(json.dumps(original), encoding="utf-8")

    assert bootstrap_agent_tools.ensure_claude_tool_budget_hooks(settings_path)
    assert not bootstrap_agent_tools.ensure_claude_tool_budget_hooks(settings_path)

    backup_path = settings_path.with_suffix(".json.headroom-backup")
    assert json.loads(backup_path.read_text(encoding="utf-8")) == original
    installed = json.loads(settings_path.read_text(encoding="utf-8"))
    assert installed["unrelated"] is True
    assert installed["hooks"]["PostToolUse"] == original["hooks"]["PostToolUse"]
    for event in ("UserPromptSubmit", "PreToolUse"):
        groups = installed["hooks"][event]
        commands = [hook for group in groups for hook in group["hooks"]]
        assert commands == [
            {
                "type": "command",
                "command": bootstrap_agent_tools.CLAUDE_TOOL_BUDGET_COMMAND,
            }
        ]
