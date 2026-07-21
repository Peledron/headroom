"""Claude Code hooks that bound repository exploration per user prompt."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

from headroom.cli.codex_tool_budget import DEFAULT_BUDGET, is_exploration_tool

CLAUDE_EXPLORATION_TOOLS = frozenset({"Read", "Grep", "Glob", "ToolSearch"})


def is_claude_exploration_tool(tool_name: str, tool_input: Any) -> bool:
    """Include Claude's built-in repository readers in the shared budget."""
    return tool_name in CLAUDE_EXPLORATION_TOOLS or is_exploration_tool(
        tool_name, tool_input
    )


def _state_path(payload: dict[str, Any], state_dir: Path) -> Path:
    session_id = str(payload.get("session_id", "unknown-session"))
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:24]
    return state_dir / f"claude-tool-budget-{digest}.json"


def _read_count(stream: TextIO) -> int:
    stream.seek(0)
    try:
        value = json.load(stream)
    except (json.JSONDecodeError, OSError):
        return 0
    if not isinstance(value, dict):
        return 0
    count = value.get("exploration_calls", 0)
    return count if isinstance(count, int) and count >= 0 else 0


def _write_count(stream: TextIO, count: int) -> None:
    stream.seek(0)
    stream.truncate()
    json.dump({"exploration_calls": count}, stream)
    stream.flush()
    os.fsync(stream.fileno())


def evaluate(
    payload: dict[str, Any], *, state_dir: Path, budget: int = DEFAULT_BUDGET
) -> dict[str, Any] | None:
    """Evaluate a Claude hook event and return structured hook output."""
    event = str(payload.get("hook_event_name", ""))
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(payload, state_dir)

    if event == "UserPromptSubmit":
        with path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            _write_count(stream, 0)
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": (
                    f"Repository exploration budget for this prompt: {budget} calls. "
                    "Plan before calling tools, combine related files or searches into "
                    "each call, and synthesize immediately after the final call. Serena "
                    "activation and initial instructions do not consume this budget."
                ),
            }
        }

    if event != "PreToolUse":
        return None
    tool_name = str(payload.get("tool_name", ""))
    if not is_claude_exploration_tool(tool_name, payload.get("tool_input")):
        return None

    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        count = _read_count(stream)
        if count >= budget:
            reason = (
                "Hard stop: the repository exploration budget is exhausted. Do not "
                "try another tool or alternate tool name. Answer now from the evidence "
                "already collected."
            )
            return {
                "systemMessage": reason,
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }

        count += 1
        _write_count(stream, count)
        if count == budget:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": (
                        "This is the final repository exploration call allowed for "
                        "this user prompt. Use its result without further reads."
                    ),
                }
            }
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        budget = max(0, int(os.environ.get("HEADROOM_CLAUDE_READ_BUDGET", DEFAULT_BUDGET)))
    except ValueError:
        budget = DEFAULT_BUDGET
    state_dir = Path(
        os.environ.get(
            "HEADROOM_CLAUDE_HOOK_STATE_DIR",
            Path(tempfile.gettempdir()) / "headroom-claude-hooks",
        )
    )
    output = evaluate(payload, state_dir=state_dir, budget=budget)
    if output is not None:
        json.dump(output, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
