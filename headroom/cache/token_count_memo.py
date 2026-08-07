"""Per-session memoized message token counts (perf review F2).

Token mode does up to 6 full-transcript ``count_messages`` calls per request,
two of them synchronous on the event loop. The frozen prefix never changes
across turns, so only the delta (new/changed messages) needs a real count —
everything else is a cache hit keyed by message content.

Correctness relies on an algebraic identity::

    tokenizer.count_messages(messages)
        == sum(tokenizer.count_message(m) for m in messages) + tokenizer.REPLY_OVERHEAD

which holds for ``BaseTokenizer``'s default per-message loop (and every
subclass that just delegates to it — ``EstimatingTokenCounter``,
``TiktokenCounter``). This is the ONLY tokenizer family the proxy uses for
Claude models: ``tokenizers/registry.py``'s ``_create_anthropic`` always
returns ``EstimatingTokenCounter``. The identity does NOT hold for tokenizers
that encode the whole conversation jointly (``HuggingFaceTokenizer``'s
``apply_chat_template`` path, ``MistralTokenizer``'s
``encode_chat_completion`` path) — do not memoize with those.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Protocol


class _MessageCountingTokenizer(Protocol):
    REPLY_OVERHEAD: int

    def count_message(self, message: dict[str, Any]) -> int: ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int: ...


class TokenCountMemo:
    """LRU cache of per-message token counts, keyed by message content hash.

    Bound to one tokenizer instance at a time: counts computed under one
    tokenizer must never be served for another (different ratios/backends),
    so a tokenizer swap — e.g. the handler's count fail-open path downgrading
    to a fresh ``EstimatingTokenCounter`` mid-session — clears the memo and
    rebinds. The registry caches tokenizer instances per model, so on the
    normal path the binding is stable across a session's requests.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._counts: OrderedDict[str, int] = OrderedDict()
        # Strong reference on purpose: identity (`is`) comparison against a
        # weak/GC'd tokenizer could alias a new instance at the same address.
        self._bound_tokenizer: object | None = None

    def bind_or_reset(self, tokenizer: object) -> None:
        """Bind this memo to ``tokenizer``, clearing stale counts on a swap."""
        with self._lock:
            if self._bound_tokenizer is not tokenizer:
                if self._bound_tokenizer is not None:
                    self._counts.clear()
                self._bound_tokenizer = tokenizer

    @staticmethod
    def message_hash(message: dict[str, Any]) -> str:
        """Hash the whole message — role, content, tool_calls, function_call,
        name — every field ``Tokenizer.count_message`` sums over, so a memo
        hit guarantees the identical count. (Not reusing
        ``CompressionCache.content_hash``: that hashes tool_result CONTENT
        only, not the whole message.)
        """
        try:
            raw = json.dumps(message, sort_keys=True, ensure_ascii=False)
        except Exception:
            raw = repr(message)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]  # nosec B324

    def get(self, key: str) -> int | None:
        with self._lock:
            count = self._counts.get(key)
            if count is not None:
                self._counts.move_to_end(key)
            return count

    def put(self, key: str, count: int) -> None:
        with self._lock:
            if key in self._counts:
                del self._counts[key]
            self._counts[key] = count
            while len(self._counts) > self.max_entries:
                self._counts.popitem(last=False)

    def get_stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._counts)}


def count_messages_memoized(
    memo: TokenCountMemo,
    tokenizer: _MessageCountingTokenizer,
    messages: list[dict[str, Any]],
) -> int:
    """``tokenizer.count_messages(messages)``, memoizing each message's count.

    Only correct for additive tokenizers — see module docstring. Enforced at
    runtime via the ``ADDITIVE_COUNTS`` capability flag (``tokenizers/base``):
    non-additive or unflagged tokenizers get a plain (correct, unmemoized)
    ``count_messages`` call instead. The memo is also bound to the tokenizer
    instance — a swap (count fail-open downgrade) clears stale counts rather
    than serving numbers computed under a different tokenizer.
    """
    if not getattr(tokenizer, "ADDITIVE_COUNTS", False):
        return tokenizer.count_messages(messages)
    memo.bind_or_reset(tokenizer)
    total = 0
    for msg in messages:
        key = TokenCountMemo.message_hash(msg)
        count = memo.get(key)
        if count is None:
            count = tokenizer.count_message(msg)
            memo.put(key, count)
        total += count
    return total + tokenizer.REPLY_OVERHEAD
