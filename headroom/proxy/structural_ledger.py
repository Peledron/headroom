"""Deterministic structural-state ledger, rendered at a cache bust.

When the structural-bust path forces a fresh cache write (see
``_structural_bust_requires_fresh_5m`` in ``handlers/anthropic.py``), the
whole message prefix is about to be re-billed anyway, so it is a free place
to also summarize what the conversation had actually accumulated. This
module builds that summary from the request's own ``tool_use`` and
``tool_result`` content blocks: a file state table (last operation seen per
path), command outcomes, unresolved errors, and the newest user task line.

Pure and stateless. Nothing here is injected into the request or the model's
context, it only feeds a log line for operators, so a change in this module
can never alter what is forwarded upstream or bust a cache by itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_FILE_PATH_KEYS = ("file_path", "path", "notebook_path")
_FILE_TOOL_OPERATIONS = {
    "Write": "write",
    "Edit": "edit",
    "MultiEdit": "edit",
    "NotebookEdit": "edit",
    "Read": "read",
}
_COMMAND_TOOL_NAMES = ("Bash", "BashOutput")

# A conservative reading of a shell exit code out of tool_result text, e.g.
# "exit code: 1" or "Exit code 127". Deterministic: same text always yields
# the same signal, independent of when the parser runs.
_EXIT_CODE_RE = re.compile(r"exit code[:\s]+(-?\d+)", re.IGNORECASE)
_ERROR_MARKERS = ("error", "traceback", "exception", "failed")


@dataclass
class FileState:
    path: str
    last_operation: str
    last_seen_turn: int


@dataclass
class CommandOutcome:
    command: str
    exit_signal: str | None
    turn: int
    is_error: bool


@dataclass
class UnresolvedError:
    source: str
    text: str
    turn: int


@dataclass
class StructuralLedger:
    files: list[FileState] = field(default_factory=list)
    commands: list[CommandOutcome] = field(default_factory=list)
    unresolved_errors: list[UnresolvedError] = field(default_factory=list)
    newest_user_task: str | None = None

    def render(self) -> str:
        """Render a compact, single-purpose-per-line summary for the log."""
        lines: list[str] = []
        if self.newest_user_task:
            lines.append(f"task: {_truncate(self.newest_user_task)}")
        if self.files:
            files_str = ", ".join(
                f"{f.path}[{f.last_operation}@{f.last_seen_turn}]" for f in self.files
            )
            lines.append(f"files: {files_str}")
        if self.commands:
            commands_str = ", ".join(
                f"{_truncate(c.command, 60)}"
                f"{'(' + c.exit_signal + ')' if c.exit_signal else ''}"
                f"@{c.turn}"
                for c in self.commands
            )
            lines.append(f"commands: {commands_str}")
        if self.unresolved_errors:
            errors_str = " | ".join(
                f"{e.source}@{e.turn}: {_truncate(e.text, 80)}" for e in self.unresolved_errors
            )
            lines.append(f"unresolved_errors: {errors_str}")
        if not lines:
            return "structural_ledger: empty"
        return "structural_ledger: " + " ; ".join(lines)


def _truncate(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _content_text(content: Any) -> str:
    """Flatten a tool_result content field (string, or list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _first_file_path(tool_input: dict[str, Any]) -> str | None:
    for key in _FILE_PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _looks_like_error(text: str, is_error_flag: bool) -> bool:
    if is_error_flag:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _ERROR_MARKERS)


def build_structural_ledger(messages: list[Any]) -> StructuralLedger:
    """Walk request messages in order and build the ledger.

    ``messages`` is the Anthropic-shape list of {"role", "content"} dicts.
    A "turn" is the index of the message the block appears in, so ordering
    within a single tool_result content list does not need its own clock.
    """
    ledger = StructuralLedger()
    files: dict[str, FileState] = {}
    # tool_use_id -> (tool name, tool input), resolved when the matching
    # tool_result block for that id arrives later in the transcript.
    pending_tool_use: dict[str, tuple[str, dict[str, Any]]] = {}
    unresolved_by_source: dict[str, UnresolvedError] = {}

    for turn, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if role == "user" and isinstance(content, str) and content.strip():
            ledger.newest_user_task = content

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text" and role == "user":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    ledger.newest_user_task = text

            elif block_type == "tool_use":
                tool_id = block.get("id")
                name = block.get("name")
                tool_input = block.get("input")
                if isinstance(tool_id, str) and isinstance(name, str) and isinstance(
                    tool_input, dict
                ):
                    pending_tool_use[tool_id] = (name, tool_input)
                    # File tools update state immediately on the call, the
                    # matching tool_result decides success/failure only for
                    # commands and errors below.
                    if name in _FILE_TOOL_OPERATIONS:
                        path = _first_file_path(tool_input)
                        if path:
                            files[path] = FileState(
                                path=path,
                                last_operation=_FILE_TOOL_OPERATIONS[name],
                                last_seen_turn=turn,
                            )

            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id")
                is_error_flag = bool(block.get("is_error", False))
                text = _content_text(block.get("content"))
                matched = (
                    pending_tool_use.get(tool_use_id) if isinstance(tool_use_id, str) else None
                )
                name = matched[0] if matched else None
                tool_input = matched[1] if matched else {}
                source = name or (tool_use_id or "unknown")

                if name in _COMMAND_TOOL_NAMES:
                    command = tool_input.get("command")
                    command_str = command if isinstance(command, str) else "<unknown>"
                    exit_match = _EXIT_CODE_RE.search(text)
                    exit_signal = exit_match.group(1) if exit_match else None
                    is_error = _looks_like_error(text, is_error_flag) or (
                        exit_signal is not None and exit_signal != "0"
                    )
                    ledger.commands.append(
                        CommandOutcome(
                            command=command_str,
                            exit_signal=exit_signal,
                            turn=turn,
                            is_error=is_error,
                        )
                    )
                    if is_error:
                        unresolved_by_source[source] = UnresolvedError(
                            source=source, text=text or command_str, turn=turn
                        )
                    else:
                        unresolved_by_source.pop(source, None)
                else:
                    if _looks_like_error(text, is_error_flag):
                        unresolved_by_source[source] = UnresolvedError(
                            source=source, text=text, turn=turn
                        )
                    else:
                        unresolved_by_source.pop(source, None)

    ledger.files = sorted(files.values(), key=lambda f: (f.last_seen_turn, f.path))
    ledger.unresolved_errors = sorted(
        unresolved_by_source.values(), key=lambda e: (e.turn, e.source)
    )
    return ledger
