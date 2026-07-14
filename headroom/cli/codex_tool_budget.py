"""Codex PreToolUse hook that bounds repetitive repository exploration."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

DEFAULT_BUDGET = 2
EXPLORATION_COMMAND = re.compile(
    r"(?:^|[\s/])(rg|grep|sed|cat|head|tail|find|fd|ls|tree|sqlite3)(?:\s|$)",
    re.IGNORECASE,
)
EXPLORATION_MCP_PREFIXES = (
    "mcp__tokensave__",
    "mcp__serena__",
    "tokensave.",
    "serena.",
)


def is_exploration_tool(tool_name: str, tool_input: Any) -> bool:
    """Return whether a tool call consumes the bounded discovery budget."""
    if tool_name.startswith(EXPLORATION_MCP_PREFIXES):
        return True
    if tool_name != "Bash" or not isinstance(tool_input, dict):
        return False
    command = tool_input.get("command", tool_input.get("cmd", ""))
    return isinstance(command, str) and EXPLORATION_COMMAND.search(command) is not None


def _state_path(payload: dict[str, Any], state_dir: Path) -> Path:
    session_id = str(payload.get("session_id", "unknown-session"))
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:24]
    return state_dir / f"codex-tool-budget-{digest}.json"


def _read_state(stream: TextIO) -> dict[str, int]:
    stream.seek(0)
    try:
        value = json.load(stream)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): int(count) for key, count in value.items() if isinstance(count, int)}


def _write_state(stream: TextIO, state: dict[str, int]) -> None:
    stream.seek(0)
    stream.truncate()
    json.dump(state, stream, sort_keys=True)
    stream.flush()
    os.fsync(stream.fileno())


def evaluate(
    payload: dict[str, Any], *, state_dir: Path, budget: int = DEFAULT_BUDGET
) -> dict[str, Any] | None:
    """Evaluate one hook payload and return Codex hook output when needed."""
    tool_name = str(payload.get("tool_name", ""))
    if not is_exploration_tool(tool_name, payload.get("tool_input")):
        return None

    turn_id = str(payload.get("turn_id", "unknown-turn"))
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(payload, state_dir)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        state = _read_state(stream)
        count = state.get(turn_id, 0)
        if count >= budget:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "Repository exploration budget exhausted for this turn. "
                        "Synthesize from existing evidence. Do not retry with another "
                        "read, search, TokenSave, Serena, or SQLite call."
                    ),
                }
            }

        count += 1
        state = {turn_id: count}
        _write_state(stream, state)
        if count == budget:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": (
                        "This is the final repository exploration call allowed for "
                        "this turn. Use its result to answer without further reads."
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
        budget = max(0, int(os.environ.get("HEADROOM_CODEX_READ_BUDGET", DEFAULT_BUDGET)))
    except ValueError:
        budget = DEFAULT_BUDGET
    state_dir = Path(
        os.environ.get(
            "HEADROOM_CODEX_HOOK_STATE_DIR",
            Path(tempfile.gettempdir()) / "headroom-codex-hooks",
        )
    )
    output = evaluate(payload, state_dir=state_dir, budget=budget)
    if output is not None:
        json.dump(output, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
