#!/usr/bin/env python3
"""Block oversized skills on the main Claude Code thread."""

from __future__ import annotations

import json
import sys
from typing import Any

_SUBAGENT_ONLY_SKILLS = {"claude-api"}


def decision(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("tool_name") != "Skill":
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    skill = str(tool_input.get("skill", "")).strip().lower()
    if skill not in _SUBAGENT_ONLY_SKILLS:
        return None
    transcript_path = str(payload.get("transcript_path", ""))
    agent_id = str(payload.get("agent_id", ""))
    if "/subagents/" in transcript_path or agent_id:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"{skill} is subagent-only because loading it on the main thread "
                "adds an extremely large persistent context. Spawn a Sonnet or "
                "Haiku subagent with a narrow brief and invoke the skill there."
            ),
        }
    }


def main() -> None:
    payload = json.load(sys.stdin)
    result = decision(payload)
    if result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
