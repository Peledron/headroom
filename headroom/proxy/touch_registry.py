"""Replay registry for TTL-extending cache touches.

A touch replays a session's last upstream request with ``max_tokens=0``, the
Anthropic-sanctioned no-generation cache-touch shape, so the provider re-reads
the cached prefix (0.1x price) instead of letting it expire and re-writing it
after an idle gap (2x price on the full prefix, roughly 30x the touch cost on
long sessions). Callers that hit an upstream that still rejects ``0`` with a
400 should retry once with ``max_tokens=1`` and log that the sanctioned shape
was refused. Only headroom still has the exact wire-form body and auth header
needed to hit the same prefix byte-identically, which is why this lives here
rather than in the desktop listener that triggers it.

Auth headers are retained in process memory only, keyed per session, capped,
and only replayed against the same upstream URL they arrived for.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

_KEPT_HEADERS = (
    "x-api-key",
    "authorization",
    "anthropic-version",
    "anthropic-beta",
    "content-type",
)

_TTL_SECONDS = {"5m": 300.0, "1h": 3600.0}
_WRITE_MULTIPLIER = {"5m": 1.25, "1h": 2.0}
_CACHE_READ_MULTIPLIER = 0.10

# Anthropic's own documented break-even: a cache write costs 1.25x (5m) or 2x
# (1h) against a 1x uncached read, and a cache hit costs 0.1x. The write
# premium pays for itself after this many subsequent hits (0.25 / 0.9 rounds
# up to 1 for the 5m tier, 1.0 / 0.9 rounds up to 2 for the 1h tier). These
# are fixed by the pricing model, not tunable, so they replace the previous
# ad-hoc single-hit assumption.
_BREAK_EVEN_HITS = {"5m": 1, "1h": 2}


@dataclass
class TouchEntry:
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    recorded_at: float = field(default_factory=time.time)
    touches_sent: int = 0
    ttl: str = "1h"


def _request_cache_ttl(body: dict[str, Any]) -> str:
    """Return the longest explicit Anthropic cache TTL in a request."""
    found = "5m"

    def visit(value: Any) -> None:
        nonlocal found
        if isinstance(value, dict):
            cache_control = value.get("cache_control")
            if isinstance(cache_control, dict) and cache_control.get("ttl") == "1h":
                found = "1h"
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(body)
    return found


def touch_break_even_age_seconds(ttl: str) -> float:
    """Ski-rental threshold where a touch is cheaper than risking a later bust.

    Uses the official break-even hit counts (see ``_BREAK_EVEN_HITS``): the
    entry becomes worth touching once its elapsed age would, if left to
    expire, forfeit more than that many hits' worth of the tier's read/write
    price gap.
    """
    ttl_seconds = _TTL_SECONDS.get(ttl, _TTL_SECONDS["1h"])
    write_cost = _WRITE_MULTIPLIER.get(ttl, _WRITE_MULTIPLIER["1h"])
    hits = _BREAK_EVEN_HITS.get(ttl, _BREAK_EVEN_HITS["1h"])
    return ttl_seconds * write_cost / (write_cost + hits * _CACHE_READ_MULTIPLIER)


def session_fingerprint(body: dict[str, Any]) -> str | None:
    """Stable identity for 'same conversation': the system head plus the
    first user message, which never change over a session's lifetime."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    system = body.get("system")
    head = json.dumps(
        [system if isinstance(system, (str, list)) else None, messages[0]],
        sort_keys=True,
        default=str,
    )[:4096]
    return hashlib.sha256(head.encode()).hexdigest()[:24]


class TouchRegistry:
    def __init__(self, max_sessions: int = 8, max_touches_per_entry: int = 2) -> None:
        self._entries: OrderedDict[str, TouchEntry] = OrderedDict()
        self._max_sessions = max_sessions
        self._max_touches = max_touches_per_entry

    def record(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> None:
        key = session_fingerprint(body)
        if key is None:
            return
        kept = {
            name: value
            for name, value in headers.items()
            if name.lower() in _KEPT_HEADERS and isinstance(value, str)
        }
        if not any(name.lower() in ("x-api-key", "authorization") for name in kept):
            return
        self._entries.pop(key, None)
        self._entries[key] = TouchEntry(
            url=url,
            headers=kept,
            body=copy.deepcopy(body),
            ttl=_request_cache_ttl(body),
        )
        while len(self._entries) > self._max_sessions:
            self._entries.popitem(last=False)

    def replayable(
        self,
        now: float | None = None,
        *,
        due_only: bool = False,
    ) -> list[tuple[str, TouchEntry]]:
        now = now if now is not None else time.time()
        out: list[tuple[str, TouchEntry]] = []
        for key in list(self._entries):
            entry = self._entries[key]
            age = now - entry.recorded_at
            if age >= _TTL_SECONDS.get(entry.ttl, _TTL_SECONDS["1h"]):
                del self._entries[key]
                continue
            if entry.touches_sent >= self._max_touches:
                continue
            if due_only and age < touch_break_even_age_seconds(entry.ttl):
                continue
            out.append((key, entry))
        return out

    def touch_body(self, entry: TouchEntry) -> dict[str, Any]:
        body = dict(entry.body)
        body["max_tokens"] = 0
        body.pop("stream", None)
        # A touch must not extend the conversation or trigger tools; the
        # cached prefix is untouched because breakpoints live in system,
        # tools, and messages, none of which change here.
        return body

    def touch_body_fallback(self, entry: TouchEntry) -> dict[str, Any]:
        """Body for the one-shot retry when an upstream rejects ``max_tokens=0``.

        Some upstreams still 400 on the sanctioned no-generation shape; callers
        should retry once with this body and log that the fallback fired.
        """
        body = self.touch_body(entry)
        body["max_tokens"] = 1
        # At max_tokens=1 the model may legally begin a tool_use block, which
        # the docstring contract rules out. tool_choice none keeps the tools
        # and system tiers refreshable while making tool emission impossible;
        # the messages tier may miss its refresh on this rare fallback path,
        # which is the lesser cost.
        if body.get("tools"):
            body["tool_choice"] = {"type": "none"}
        return body

    def mark_touched(self, key: str, *, refreshed: bool) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.touches_sent += 1
        if refreshed:
            entry.recorded_at = time.time()

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            {
                "session": key,
                "age_seconds": round(now - entry.recorded_at, 1),
                "touches_sent": entry.touches_sent,
                "ttl": entry.ttl,
                "break_even_age_seconds": round(touch_break_even_age_seconds(entry.ttl), 1),
                "url": entry.url,
            }
            for key, entry in self._entries.items()
        ]
