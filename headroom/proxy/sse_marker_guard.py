"""Streaming counterpart of the response-side recovery-marker guard.

The buffered Anthropic path repairs marker mimicry on the complete response
object. Direct SSE forwards bytes as they arrive, which is how two fabricated
markers reached a client on 2026-07-18. This transformer closes that gap by
holding back only the events of an in-flight ``tool_use`` content block until
its ``content_block_stop`` arrives, then releasing the block verbatim,
expanded (valid marker), or replaced by a guard text block (fabricated
marker). Text deltas and every other event still stream through immediately.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("headroom.proxy")

_MARKER_PREFIXES = (
    "[Tool input masked:",
    "[Tool result masked:",
    "Retrieve original: hash=",
)
_HASH_RE = re.compile(r"hash=([0-9a-fA-F]{24})")


def _expand_value(
    value: Any, retrieve: Callable[[str], str | None]
) -> tuple[Any, str | None, bool]:
    """Return (expanded, missing_hash, changed) mirroring the buffered guard."""
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        changed = False
        for key, child in value.items():
            expanded, missing, child_changed = _expand_value(child, retrieve)
            if missing is not None:
                return value, missing, False
            changed = changed or child_changed
            out[key] = expanded
        return (out if changed else value), None, changed
    if isinstance(value, list):
        out_list: list[Any] = []
        changed = False
        for child in value:
            expanded, missing, child_changed = _expand_value(child, retrieve)
            if missing is not None:
                return value, missing, False
            changed = changed or child_changed
            out_list.append(expanded)
        return (out_list if changed else value), None, changed
    if not isinstance(value, str) or not any(p in value for p in _MARKER_PREFIXES):
        return value, None, False
    match = _HASH_RE.search(value)
    hash_key = match.group(1).lower() if match else "unknown"
    original = retrieve(hash_key) if match else None
    if original is None:
        return value, hash_key, False
    return original, None, True


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class SseToolUseMarkerGuard:
    """Feed raw SSE bytes, receive guarded SSE bytes.

    Only complete ``\\n\\n``-terminated events are released; the transformer
    never re-frames events it does not need to inspect, so pass-through
    traffic stays byte-identical.
    """

    def __init__(
        self,
        retrieve: Callable[[str], str | None],
        request_id: str,
        stored_hashes: Callable[[], list[str]] | None = None,
    ) -> None:
        self._retrieve = retrieve
        self._request_id = request_id
        self._stored_hashes = stored_hashes
        self._buffer = bytearray()
        # index -> {"raw": [bytes], "start": dict, "json": [str]}
        self._held: dict[int, dict[str, Any]] = {}
        self._blocked = False
        # Set once any tool_use block reaches the client intact. A blocked
        # stream cannot be retried the way a buffered response can, since its
        # earlier events are already gone, so a surviving call is the only way
        # the client's loop keeps turning without a human.
        self._released_tool_use = False

    def feed(self, chunk: bytes) -> bytes:
        self._buffer.extend(chunk)
        out = bytearray()
        while True:
            boundary = self._buffer.find(b"\n\n")
            if boundary < 0:
                break
            raw = bytes(self._buffer[: boundary + 2])
            del self._buffer[: boundary + 2]
            out.extend(self._handle_event(raw))
        return bytes(out)

    def flush(self) -> bytes:
        """Release everything still held. A truncated upstream stream must not
        swallow bytes, so unresolved blocks are forwarded verbatim."""
        out = bytearray()
        for index in sorted(self._held):
            for raw in self._held[index]["raw"]:
                out.extend(raw)
        if self._held:
            logger.warning(
                "[%s] SSE_MARKER_GUARD: stream ended with %d unresolved tool_use "
                "block(s), released verbatim",
                self._request_id,
                len(self._held),
            )
        self._held.clear()
        if self._buffer:
            out.extend(bytes(self._buffer))
            self._buffer.clear()
        return bytes(out)

    def _handle_event(self, raw: bytes) -> bytes:
        data = self._parse_data(raw)
        if data is None:
            return raw
        kind = data.get("type")
        index = data.get("index")
        if not isinstance(index, int):
            index = -1

        if kind == "content_block_start":
            block = data.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                self._held[index] = {"raw": [raw], "start": data, "json": []}
                return b""
            return raw

        if kind == "content_block_delta" and index in self._held:
            held = self._held[index]
            held["raw"].append(raw)
            delta = data.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                partial = delta.get("partial_json")
                if isinstance(partial, str):
                    held["json"].append(partial)
            return b""

        if kind == "content_block_stop" and index in self._held:
            held = self._held.pop(index)
            held["raw"].append(raw)
            return self._resolve_block(index, held)

        if kind == "message_delta" and self._blocked and not self._released_tool_use:
            delta = data.get("delta")
            if isinstance(delta, dict) and delta.get("stop_reason") == "tool_use":
                data["delta"] = {**delta, "stop_reason": "end_turn"}
                return _sse_event("message_delta", data)
            return raw

        return raw

    def _resolve_block(self, index: int, held: dict[str, Any]) -> bytes:
        joined = "".join(held["json"])
        if not any(p in joined for p in _MARKER_PREFIXES):
            self._released_tool_use = True
            return b"".join(held["raw"])

        try:
            parsed = json.loads(joined) if joined.strip() else {}
        except ValueError:
            parsed = None
        if parsed is None:
            match = _HASH_RE.search(joined)
            missing = match.group(1).lower() if match else "unknown"
            return self._blocked_block(index, missing)

        expanded, missing, changed = _expand_value(parsed, self._retrieve)
        if missing is not None:
            return self._blocked_block(index, missing)
        if not changed:
            self._released_tool_use = True
            return b"".join(held["raw"])

        self._released_tool_use = True
        logger.warning("[%s] SSE_MARKER_GUARD: expanded tool_use input", self._request_id)
        start = held["start"]
        block = dict(start.get("content_block") or {})
        block["input"] = {}
        out = bytearray()
        out.extend(
            _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": block},
            )
        )
        out.extend(
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(expanded),
                    },
                },
            )
        )
        out.extend(
            _sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": index}
            )
        )
        return bytes(out)

    def _blocked_block(self, index: int, missing_hash: str) -> bytes:
        from headroom.proxy.marker_recovery import blocked_message, nearest_stored_hashes

        self._blocked = True
        try:
            keys = self._stored_hashes() if self._stored_hashes is not None else []
            suggestions = nearest_stored_hashes(missing_hash, keys)
        except Exception:  # pragma: no cover - suggestions are never load bearing
            suggestions = []
        logger.warning(
            "[%s] SSE_MARKER_GUARD: blocked hash=%s near=%s",
            self._request_id,
            missing_hash,
            ",".join(suggestions) or "-",
        )
        message = blocked_message(missing_hash, suggestions=suggestions)
        out = bytearray()
        out.extend(
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        out.extend(
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": message},
                },
            )
        )
        out.extend(
            _sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": index}
            )
        )
        return bytes(out)

    @staticmethod
    def _parse_data(raw: bytes) -> dict[str, Any] | None:
        for line in raw.split(b"\n"):
            if line.startswith(b"data:"):
                try:
                    parsed = json.loads(line[5:].strip())
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
        return None
