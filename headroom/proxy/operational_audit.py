"""Small in-process counters for invisible proxy decisions and repeated tool calls."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter, OrderedDict
from typing import Any

from headroom.proxy.touch_registry import session_fingerprint


class OperationalAudit:
    """Track model rewrites and duplicate tool calls without retaining payloads."""

    def __init__(self, max_sessions: int = 64) -> None:
        self._lock = threading.Lock()
        self._substitutions: Counter[str] = Counter()
        self._tool_calls_total = 0
        self._duplicate_tool_calls = 0
        self._duplicates_by_tool: Counter[str] = Counter()
        self._anthropic_fanout_requests = 0
        self._anthropic_internal_iterations = 0
        self._anthropic_internal_input_tokens = 0
        self._effort_lowerings = 0
        self._effort_pins = 0
        self._sessions: OrderedDict[str, tuple[set[str], set[str]]] = OrderedDict()
        self._max_sessions = max_sessions

    def record_model_substitution(self, source: str, target: str, reason: str) -> None:
        key = f"{source} -> {target} ({reason})"
        with self._lock:
            self._substitutions[key] += 1

    def observe_anthropic_tools(self, body: dict[str, Any]) -> None:
        session = session_fingerprint(body)
        messages = body.get("messages")
        if session is None or not isinstance(messages, list):
            return
        with self._lock:
            seen_ids, seen_signatures = self._sessions.pop(session, (set(), set()))
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    call_id = str(block.get("id") or "")
                    if call_id and call_id in seen_ids:
                        continue
                    if call_id:
                        seen_ids.add(call_id)
                    name = str(block.get("name") or "unknown")
                    canonical = json.dumps(
                        [name, block.get("input")],
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    signature = hashlib.sha256(canonical.encode()).hexdigest()
                    self._tool_calls_total += 1
                    if signature in seen_signatures:
                        self._duplicate_tool_calls += 1
                        self._duplicates_by_tool[name] += 1
                    else:
                        seen_signatures.add(signature)
            self._sessions[session] = (seen_ids, seen_signatures)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def record_anthropic_iteration_fanout(self, iterations: int, input_tokens: int) -> None:
        if iterations <= 1:
            return
        with self._lock:
            self._anthropic_fanout_requests += 1
            self._anthropic_internal_iterations += iterations
            self._anthropic_internal_input_tokens += max(0, input_tokens)

    def record_effort_routing(self, action: str) -> None:
        with self._lock:
            if action == "lowered":
                self._effort_lowerings += 1
            elif action == "pinned":
                self._effort_pins += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model_substitutions_total": sum(self._substitutions.values()),
                "model_substitutions": dict(self._substitutions),
                "tool_calls_observed": self._tool_calls_total,
                "duplicate_tool_calls": self._duplicate_tool_calls,
                "duplicate_tool_calls_by_tool": dict(self._duplicates_by_tool),
                "anthropic_fanout_requests": self._anthropic_fanout_requests,
                "anthropic_internal_iterations": self._anthropic_internal_iterations,
                "anthropic_internal_input_tokens": self._anthropic_internal_input_tokens,
                "effort_lowerings": self._effort_lowerings,
                "effort_pins": self._effort_pins,
            }
