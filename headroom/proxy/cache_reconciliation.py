"""Closed-loop cache accounting: join predicted cache read against billed usage.

Log-only by design (workstream H). Nothing in this module mutates a request,
a gate decision, or a transform outcome. It only records what happened so an
unplanned cache bust is visible in /stats the day it happens, instead of
requiring a transcript audit weeks later. Every public entry point is safe to
call from the hot request path: no exception raised here escapes the caller.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("headroom.proxy")

DEFAULT_LOG_PATH = Path(os.path.expanduser("~/.headroom/logs/cache_reconciliation.jsonl"))
RING_SIZE = 200
RECENT_BUST_SIZE = 5
# The Anthropic 5-minute cache tier. A cold read after a longer idle gap is
# scheduled expiry, not an unplanned bust, and must not trip the alarm.
CACHE_TTL_SECONDS = 300.0
# Transform labels that mark a bust headroom chose on purpose. A cold read on
# a request carrying one of these is priced work, not an alarm condition.
PLANNED_BUST_MARKERS = ("hybrid_rebase", "structural_bust", "history_rebase")
# Anthropic cache tiers in seconds. The write ttl on a breakpoint sets how long
# a read stays cheap before scheduled expiry.
_TTL_LABEL_SECONDS = {"5m": 300.0, "1h": 3600.0}


def message_segment_ttl_seconds(body: dict | None) -> float:
    """The cache_control ttl of the message-history segment only.

    Reconciliation tracks whether the message-history prefix stayed warm. This
    proxy deliberately keeps the system and tools HEAD on the 1h tier so it stays
    cache-shared across sessions, even on a turn where the message tail is forced
    to 5m by the structural-bust or adaptive-ttl path. Scanning every breakpoint
    and taking the max would let that 1h HEAD mask a genuine 5m message tail, so
    a real 5m expiry would be misread as a within-window bust. Only the message
    segment governs history warmth, so only it is scanned here. Defaults to the
    Anthropic 5m default. Never raises.

    When the message segment carries multiple anchors at differing tiers, the max
    among them is used, since the most durable message anchor bounds how long any
    message-prefix read can still hit.
    """
    longest = CACHE_TTL_SECONDS
    try:
        if isinstance(body, dict):
            for msg in body.get("messages", []) or []:
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, list):
                    continue
                for block in content:
                    cc = block.get("cache_control") if isinstance(block, dict) else None
                    if isinstance(cc, dict):
                        label = cc.get("ttl")
                        seconds = (
                            _TTL_LABEL_SECONDS.get(label, CACHE_TTL_SECONDS)
                            if isinstance(label, str)
                            else CACHE_TTL_SECONDS
                        )
                        longest = max(longest, seconds)
    except Exception:
        return CACHE_TTL_SECONDS
    return longest


@dataclass(frozen=True)
class CacheReconciliationRecord:
    """One request's predicted-vs-billed cache accounting."""

    request_id: str
    model: str
    predicted_cache_read: int
    billed_cache_read: int
    billed_cache_creation: int
    alive_fraction: float
    first_diverged_index: int | None
    transforms: list[str] = field(default_factory=list)
    unplanned_bust: bool = False
    ttl_expired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "predicted_cache_read": self.predicted_cache_read,
            "billed_cache_read": self.billed_cache_read,
            "billed_cache_creation": self.billed_cache_creation,
            "alive_fraction": self.alive_fraction,
            "first_diverged_index": self.first_diverged_index,
            "transforms": list(self.transforms),
            "unplanned_bust": self.unplanned_bust,
            "ttl_expired": self.ttl_expired,
        }


def is_unplanned_bust(predicted_cache_read: int, billed_cache_read: int) -> bool:
    """A request the DP expected warm came back materially colder than that.

    Flags only when there was something to predict (predicted above zero) and
    the actual billed read landed under half of it. A predicted read of zero
    means there was no prior turn to compare against, not a bust.
    """
    return predicted_cache_read > 0 and billed_cache_read < predicted_cache_read / 2


def is_planned_bust(transforms: list[str] | None) -> bool:
    """True when a transform on this request deliberately paid for the bust."""
    if not transforms:
        return False
    return any(marker in label for label in transforms for marker in PLANNED_BUST_MARKERS)


class CacheReconciliationLog:
    """Per-session predicted-read chaining, a bounded ring, and a jsonl sink.

    Predicted cache read for v1 is deliberately simple: the previous request's
    billed read plus billed write for the same session is what should have
    been warm this turn. No token-level cost model, no TTL arithmetic, that
    lives in the priced gate this workstream is a precondition for.
    """

    def __init__(
        self,
        log_path: Path | str = DEFAULT_LOG_PATH,
        ring_size: int = RING_SIZE,
        recent_bust_size: int = RECENT_BUST_SIZE,
    ) -> None:
        self._log_path = Path(log_path)
        self._lock = threading.Lock()
        self._ring: deque[dict[str, Any]] = deque(maxlen=ring_size)
        self._recent_busts: deque[dict[str, Any]] = deque(maxlen=recent_bust_size)
        # session_key -> (billed_cache_read, billed_cache_creation, monotonic
        # timestamp) of its last request. The timestamp separates scheduled
        # 5m-TTL expiry from a genuine unplanned bust.
        self._session_prior: dict[str, tuple[int, int, float]] = {}
        self._request_count = 0
        self._unplanned_bust_count = 0
        self._ttl_expiry_count = 0

    def record(
        self,
        *,
        session_key: str,
        request_id: str,
        model: str,
        billed_cache_read: int,
        billed_cache_creation: int,
        alive_fraction: float,
        first_diverged_index: int | None,
        transforms: list[str] | None = None,
        now: float | None = None,
        ttl_seconds: float = CACHE_TTL_SECONDS,
    ) -> CacheReconciliationRecord:
        """Assemble, count, ring, and log one record. Never raises."""
        try:
            return self._record(
                now=now,
                session_key=session_key,
                request_id=request_id,
                model=model,
                billed_cache_read=billed_cache_read,
                billed_cache_creation=billed_cache_creation,
                alive_fraction=alive_fraction,
                first_diverged_index=first_diverged_index,
                transforms=transforms,
                ttl_seconds=ttl_seconds,
            )
        except Exception:
            logger.debug(
                "[%s] cache_reconciliation: record failed, skipping", request_id, exc_info=True
            )
            return CacheReconciliationRecord(
                request_id=request_id,
                model=model,
                predicted_cache_read=0,
                billed_cache_read=billed_cache_read,
                billed_cache_creation=billed_cache_creation,
                alive_fraction=alive_fraction,
                first_diverged_index=first_diverged_index,
                transforms=list(transforms or []),
                unplanned_bust=False,
            )

    def _record(
        self,
        *,
        session_key: str,
        request_id: str,
        model: str,
        billed_cache_read: int,
        billed_cache_creation: int,
        alive_fraction: float,
        first_diverged_index: int | None,
        transforms: list[str] | None,
        now: float | None = None,
        ttl_seconds: float = CACHE_TTL_SECONDS,
    ) -> CacheReconciliationRecord:
        billed_cache_read = max(0, int(billed_cache_read))
        billed_cache_creation = max(0, int(billed_cache_creation))
        if now is None:
            now = time.monotonic()
        # A 1h-tier write stays warm 12x longer than the 5m default, so a flat
        # threshold would flag a genuine 1h bust as benign expiry. Judge expiry
        # against the actual TTL the prior request was written with.
        ttl_seconds = ttl_seconds if ttl_seconds > 0 else CACHE_TTL_SECONDS
        with self._lock:
            prior = self._session_prior.get(session_key)
            predicted = (prior[0] + prior[1]) if prior else 0
            prior_age = (now - prior[2]) if prior else 0.0
            self._session_prior[session_key] = (billed_cache_read, billed_cache_creation, now)
        ttl_expired = prior is not None and prior_age > ttl_seconds
        unplanned_bust = (
            not ttl_expired
            and not is_planned_bust(transforms)
            and is_unplanned_bust(predicted, billed_cache_read)
        )
        record = CacheReconciliationRecord(
            request_id=request_id,
            model=model,
            predicted_cache_read=predicted,
            billed_cache_read=billed_cache_read,
            billed_cache_creation=billed_cache_creation,
            alive_fraction=alive_fraction,
            first_diverged_index=first_diverged_index,
            transforms=list(transforms or []),
            unplanned_bust=unplanned_bust,
            ttl_expired=ttl_expired,
        )
        payload = record.to_dict()
        with self._lock:
            self._request_count += 1
            self._ring.append(payload)
            if ttl_expired and is_unplanned_bust(predicted, billed_cache_read):
                self._ttl_expiry_count += 1
            if unplanned_bust:
                self._unplanned_bust_count += 1
                self._recent_busts.append(payload)
        self._append_jsonl(payload)
        return record

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        # Tolerate any IO failure silently, an unwritable log path must never
        # fail or slow the request it is trying to describe.
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            pass

    def snapshot(self) -> dict[str, Any]:
        """Counts and the last few bust records, shaped for /stats inclusion."""
        with self._lock:
            return {
                "requests": self._request_count,
                "unplanned_busts": self._unplanned_bust_count,
                "ttl_expiry_colds": self._ttl_expiry_count,
                "recent_unplanned_busts": list(self._recent_busts),
            }


_default_log: CacheReconciliationLog | None = None
_default_log_lock = threading.Lock()


def get_reconciliation_log() -> CacheReconciliationLog:
    """Process-wide singleton sink, lazily created on first use."""
    global _default_log
    if _default_log is None:
        with _default_log_lock:
            if _default_log is None:
                _default_log = CacheReconciliationLog()
    return _default_log
