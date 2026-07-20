from __future__ import annotations

import importlib.util
from pathlib import Path

_HOOK_PATH = Path(__file__).parents[1] / ".claude" / "hooks" / "skill-main-thread-gate.py"
_SPEC = importlib.util.spec_from_file_location("skill_main_thread_gate", _HOOK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_claude_api_is_denied_on_main_thread() -> None:
    result = _MODULE.decision(
        {
            "tool_name": "Skill",
            "tool_input": {"skill": "claude-api"},
            "transcript_path": "/tmp/project/session.jsonl",
        }
    )
    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_claude_api_is_allowed_in_subagent_transcript() -> None:
    assert (
        _MODULE.decision(
            {
                "tool_name": "Skill",
                "tool_input": {"skill": "claude-api"},
                "transcript_path": "/tmp/project/session/subagents/agent-1.jsonl",
            }
        )
        is None
    )


def test_other_skills_are_untouched() -> None:
    assert (
        _MODULE.decision(
            {
                "tool_name": "Skill",
                "tool_input": {"skill": "engineering-baseline"},
                "transcript_path": "/tmp/project/session.jsonl",
            }
        )
        is None
    )
