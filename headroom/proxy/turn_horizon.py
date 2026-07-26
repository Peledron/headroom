"""Estimate how many more turns will read this prefix, from what the session is doing.

Three gates now buy the future with present tokens. The masking gate spends a
rewrite to shrink a prefix, the effort router spends one to lower reasoning, and
the compaction advisor spends one to drop history. All three ask the same
question, how many more turns will read what I am about to rewrite, and all
three get the same answer from a Lindy estimator: a session that has run N turns
will probably run about N more.

Lindy is a decent prior and a poor predictor. It says the same thing about turn
40 of a one-line typo fix and turn 40 of a refactor across thirty files, which is
wrong in opposite directions. Every gate inherits that error, and because the
error is asymmetric in cost (an overestimate spends a rewrite that never repays,
an underestimate merely misses a saving) a wrong horizon is worse than a vague
one.

This estimates the horizon from observable difficulty instead. Nothing here is
semantic and nothing needs a trained language model. The signals are all free at
request time, and the labels are free too, which is the part that makes it work:

    A task boundary is its own label. When a new user ask arrives, every turn
    since the previous ask now knows exactly how many turns remained when it was
    taken. No annotation, no training set, no offline job. The session labels
    itself as it runs.

So each turn records its features and waits. At the next boundary the pending
turns are labelled and filed under their feature bucket. Later turns landing in
the same bucket read back the distribution of what actually happened.

Two deliberate conservatisms. Prediction uses a low quantile, not the mean,
because the cost of overestimating is a wasted rewrite and the cost of
underestimating is a missed saving, and those are not the same size. And the
estimator declines to answer until a bucket holds enough episodes, deferring to
the Lindy fallback rather than guessing from three samples.

What this is not. It does not know whether a task is conceptually hard, only
whether it is behaving like previous hard ones: long tool runs, high output
spend, retries, fast context growth. That correlation is enough to price a cache
decision, and it is worth being clear that a semantic difficulty model (BERT or
otherwise) would need the same episode table as its training data anyway. Build
the table first. It is useful on its own, and it is the prerequisite for
anything smarter.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

# Below this many labelled episodes a bucket is noise, so defer to the fallback.
MIN_EPISODES = 12

# Overestimating the horizon spends rewrites that never repay, underestimating
# only misses savings. Predict the pessimistic end of the observed distribution.
HORIZON_QUANTILE = 0.35

# A horizon nobody should trust, matching the other cost models' cap.
MAX_HORIZON = 40

# Turns wait this long for a boundary before being written off. A task that runs
# longer than this is not going to produce a useful short-horizon label.
MAX_PENDING_TURNS = 200

# Keep buckets bounded. Oldest episodes fall out first, so the estimate tracks
# what this workload is doing now rather than what it did last week.
MAX_EPISODES_PER_BUCKET = 64


@dataclass(frozen=True)
class TurnFeatures:
    """Observable difficulty signals for one turn. All free at request time.

    Deliberately coarse. These get bucketed immediately, so precision beyond the
    bucket edges buys nothing and would only fragment the table.
    """

    session_turn: int = 0
    turns_since_user_ask: int = 0
    output_tokens_ewma: float = 0.0
    prefix_growth_per_turn: float = 0.0
    error_fraction: float = 0.0

    def bucket(self) -> tuple[int, ...]:
        """Discretize into a table key.

        The bins are wide on purpose. A narrow grid gives every turn its own
        bucket, which never reaches MIN_EPISODES and so never predicts anything.
        """
        return (
            _bin(self.session_turn, (10, 30, 80, 200)),
            _bin(self.turns_since_user_ask, (3, 10, 30)),
            _bin(self.output_tokens_ewma, (500, 2000, 6000)),
            _bin(self.prefix_growth_per_turn, (500, 2000, 8000)),
            _bin(self.error_fraction, (0.01, 0.1)),
        )


def _bin(value: float, edges: tuple[float, ...]) -> int:
    """Index of the first edge ``value`` falls under, or len(edges) past the top."""
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


@dataclass
class _Pending:
    bucket: tuple[int, ...]
    turn: int


class TurnHorizonModel:
    """Learns remaining-turn distributions per difficulty bucket, in session.

    Thread-safe. Every method is best-effort bookkeeping that must never raise
    into a request path, so callers can treat a failure as "no opinion".
    """

    def __init__(self) -> None:
        self._episodes: dict[tuple[int, ...], list[int]] = {}
        self._pending: list[_Pending] = []
        self._lock = threading.Lock()

    def observe_turn(self, features: TurnFeatures) -> None:
        """Record a turn awaiting its label from the next task boundary."""
        with self._lock:
            self._pending.append(_Pending(features.bucket(), features.session_turn))
            if len(self._pending) > MAX_PENDING_TURNS:
                del self._pending[: len(self._pending) - MAX_PENDING_TURNS]

    def observe_task_boundary(self, session_turn: int) -> int:
        """Label every pending turn with how many turns it had left.

        Called when a new user ask arrives, which is the moment the previous
        task's length becomes known. Returns the number of episodes labelled.
        """
        with self._lock:
            labelled = 0
            for pending in self._pending:
                remaining = session_turn - pending.turn
                if remaining <= 0:
                    continue
                bucket = self._episodes.setdefault(pending.bucket, [])
                bucket.append(min(remaining, MAX_HORIZON))
                if len(bucket) > MAX_EPISODES_PER_BUCKET:
                    del bucket[: len(bucket) - MAX_EPISODES_PER_BUCKET]
                labelled += 1
            self._pending.clear()
            return labelled

    def predict(self, features: TurnFeatures) -> float | None:
        """Remaining turns for this turn's bucket, or None if not yet learnable."""
        with self._lock:
            episodes = self._episodes.get(features.bucket())
            if episodes is None or len(episodes) < MIN_EPISODES:
                return None
            ordered = sorted(episodes)
            index = int(HORIZON_QUANTILE * (len(ordered) - 1))
            return float(ordered[index])

    def episode_count(self, features: TurnFeatures | None = None) -> int:
        with self._lock:
            if features is None:
                return sum(len(v) for v in self._episodes.values())
            return len(self._episodes.get(features.bucket(), []))

    def export_state(self) -> dict[str, Any]:
        """Serialize for the cross-restart snapshot.

        Bucket indices and turn counts only. No conversation content, and
        nothing that identifies a session. Pending turns are dropped: their
        label depends on a boundary this process will no longer see.
        """
        with self._lock:
            return {
                "episodes": [
                    [list(bucket), list(values)] for bucket, values in self._episodes.items()
                ]
            }

    def restore_state(self, blob: dict[str, Any] | None) -> int:
        """Merge a snapshot. Returns the episodes restored.

        A bucket needs a dozen labelled episodes before it says anything, which
        is more than most sessions produce alone. Without this the table would
        restart empty every time and the estimator would never leave its
        fallback.
        """
        if not isinstance(blob, dict):
            return 0
        rows = blob.get("episodes")
        if not isinstance(rows, list):
            return 0
        restored = 0
        with self._lock:
            for row in rows:
                try:
                    raw_bucket, raw_values = row
                    bucket = tuple(int(x) for x in raw_bucket)
                    values = [int(v) for v in raw_values if 0 < int(v) <= MAX_HORIZON]
                except (TypeError, ValueError):
                    continue
                if not bucket or not values:
                    continue
                target = self._episodes.setdefault(bucket, [])
                target.extend(values)
                if len(target) > MAX_EPISODES_PER_BUCKET:
                    del target[: len(target) - MAX_EPISODES_PER_BUCKET]
                restored += len(values)
        return restored


def features_from_request(
    messages: list[Any],
    session_turn: int,
    prefix_tokens: int,
) -> TurnFeatures:
    """Read this turn's difficulty signals off the request that is already parsed.

    Everything here is a scan of the message array the handler is holding, so
    the estimator costs no extra tokens and no extra calls. Nothing semantic is
    read, only shape: how deep the current tool run is, how much the assistant
    is writing, how often tools are coming back as errors.
    """
    turns_since_ask = 0
    error_blocks = 0
    result_blocks = 0
    assistant_chars = 0
    assistant_msgs = 0
    found_ask = False

    for message in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            # A user message made of tool results is the tool run continuing.
            # One made of plain text is a person asking for something, which is
            # where the current task starts.
            if _is_tool_result_only(content):
                if not found_ask:
                    turns_since_ask += 1
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result_blocks += 1
                        if block.get("is_error"):
                            error_blocks += 1
            else:
                found_ask = True
        elif role == "assistant" and not found_ask:
            assistant_msgs += 1
            assistant_chars += _text_length(content)

    # Roughly four characters per token, the same approximation the rest of the
    # proxy uses for pre-tokenizer estimates.
    output_ewma = (assistant_chars / 4.0 / assistant_msgs) if assistant_msgs else 0.0
    growth = (prefix_tokens / session_turn) if session_turn > 0 else 0.0
    error_fraction = (error_blocks / result_blocks) if result_blocks else 0.0

    return TurnFeatures(
        session_turn=max(0, session_turn),
        turns_since_user_ask=turns_since_ask,
        output_tokens_ewma=output_ewma,
        prefix_growth_per_turn=growth,
        error_fraction=error_fraction,
    )


def _is_tool_result_only(content: Any) -> bool:
    """True when a user message carries nothing but tool results."""
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )


def _text_length(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                total += len(text)
    return total


_SHARED_HORIZON_MODEL = TurnHorizonModel()


def shared_horizon_model() -> TurnHorizonModel:
    """The process-wide horizon table, shared by every gate that prices a rewrite."""
    return _SHARED_HORIZON_MODEL


def expected_remaining_turns(
    features: TurnFeatures,
    fallback: float,
    model: TurnHorizonModel | None = None,
) -> tuple[float, str]:
    """Best available horizon, and where it came from.

    ``fallback`` is the caller's existing estimate, normally the Lindy figure
    from ``PrefixCacheTracker.expected_session_reads``. The learned value only
    wins where it has the evidence to; everywhere else the caller is no worse
    off than before. Returning the source alongside the number is what makes the
    two comparable in the logs, which is the only way to find out whether this
    actually beats Lindy on real traffic.
    """
    model = model or shared_horizon_model()
    try:
        learned = model.predict(features)
    except Exception:
        learned = None
    if learned is None:
        return max(0.0, min(fallback, MAX_HORIZON)), "lindy"
    return max(0.0, min(learned, MAX_HORIZON)), "learned"
