"""Output tokens are the unmeasured half of the bill.

Every compression mechanism in this proxy works on the request. Observation
masking, CCR, diff-only re-reads, lineage matching and the anchor DP all price
against the input side, where a cached token costs 0.1x. Output is billed at
roughly five times an uncached input token, and nothing here touches it, so a
turn that mostly writes passes through the whole stack untouched and pays the
highest rate on the menu.

TOKEN_DIAG cannot carry this. It is emitted before the upstream call, where the
output count does not exist yet. The value is already parsed on the response
path for effort pricing, keyed by (model, effort). What was missing is a
reading of the distribution split by turn shape.

The split axis is whether the request is a tool continuation, meaning the
client is returning work mid-task, or a fresh ask. Those two shapes imply
different fixes. If continuations are the expensive ones, the lever is
narration during tool loops. If fresh asks are, the lever is response length on
direct questions. A single mean over both would hide which.

This module only counts. It changes no request and no response, and it holds no
prompt text.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Kept small because the recent window only has to be large enough for a
# stable median. Lifetime count, total and max are exact and unbounded.
RECENT_WINDOW = 512

TURN_SHAPE_CONTINUATION = "tool_continuation"
TURN_SHAPE_FRESH = "fresh_ask"
TURN_SHAPE_UNKNOWN = "unknown"


def classify_turn_shape(messages: Any) -> str:
    """Name the shape of the turn from the last message in the request.

    A user message carrying tool_result blocks is the client handing back work
    the model asked for, so the model is mid-task. Anything else that reaches
    the proxy with a last message present is treated as a fresh ask. An empty
    or unreadable message list is reported as unknown rather than guessed,
    since a wrong bucket is worse than a missing one.
    """

    if not isinstance(messages, list) or not messages:
        return TURN_SHAPE_UNKNOWN

    latest = messages[-1]
    if not isinstance(latest, dict):
        return TURN_SHAPE_UNKNOWN

    content = latest.get("content")
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    ):
        return TURN_SHAPE_CONTINUATION

    return TURN_SHAPE_FRESH


@dataclass
class ShapeStats:
    """Lifetime totals plus a bounded window for the median."""

    count: int = 0
    total: int = 0
    max_seen: int = 0
    recent: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))

    def observe(self, output_tokens: int) -> None:
        self.count += 1
        self.total += output_tokens
        self.max_seen = max(self.max_seen, output_tokens)
        self.recent.append(output_tokens)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def median(self) -> int:
        if not self.recent:
            return 0
        ordered = sorted(self.recent)
        return ordered[len(ordered) // 2]

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total": self.total,
            "mean": round(self.mean, 1),
            "median": self.median,
            "max": self.max_seen,
            "window": len(self.recent),
        }


@dataclass(frozen=True)
class OutputObservation:
    """What the handler logs for one response."""

    shape: str
    output_tokens: int
    count: int
    mean: float
    median: int
    max_seen: int

    def as_log_fields(self) -> str:
        return (
            f"shape={self.shape} out={self.output_tokens} "
            f"n={self.count} mean={self.mean:.0f} p50={self.median} max={self.max_seen}"
        )


class OutputTokenLedger:
    """Per-shape output token counts for the life of the process."""

    def __init__(self) -> None:
        self._shapes: dict[str, ShapeStats] = {}

    def observe(self, shape: str, output_tokens: int) -> OutputObservation | None:
        """Record one response. Returns None when there is nothing to record.

        A non-positive count means the response carried no usage block, which
        happens on errors and on some streaming paths. Counting those as zero
        would drag every mean down and make the distribution useless.
        """

        if not shape or not isinstance(output_tokens, int) or output_tokens <= 0:
            return None

        stats = self._shapes.setdefault(shape, ShapeStats())
        stats.observe(output_tokens)
        return OutputObservation(
            shape=shape,
            output_tokens=output_tokens,
            count=stats.count,
            mean=stats.mean,
            median=stats.median,
            max_seen=stats.max_seen,
        )

    def snapshot(self) -> dict[str, Any]:
        """Machine-readable stats, safe to serialize into the stats endpoint."""

        shapes = {name: stats.as_dict() for name, stats in self._shapes.items()}
        total_output = sum(stats.total for stats in self._shapes.values())
        total_count = sum(stats.count for stats in self._shapes.values())
        return {
            "shapes": shapes,
            "total_output_tokens": total_output,
            "responses": total_count,
        }

    def reset(self) -> None:
        self._shapes.clear()


_LEDGER = OutputTokenLedger()


def shared_output_ledger() -> OutputTokenLedger:
    """The process-wide ledger the request handlers write to."""

    return _LEDGER
