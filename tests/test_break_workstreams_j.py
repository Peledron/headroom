"""Breaker pass on workstream J: bust-time CCR flush (anthropic.py) and the
tier-aware reconciliation TTL (cache_reconciliation.py + streaming.py).

Method: chaos engineering. Each test tries to falsify the contract the
writer claimed, not confirm the happy path. Findings are reported to the
fixer, not patched here -- this file only adds tests.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import anyio
import pytest
from fastapi import Request

from headroom.proxy.cache_reconciliation import (
    CacheReconciliationLog,
    is_planned_bust,
    is_unplanned_bust,
    message_segment_ttl_seconds,
)
from headroom.proxy.handlers.anthropic import (
    AnthropicHandlerMixin,
    _structural_bust_requires_fresh_5m,
)
from headroom.proxy.helpers import apply_session_sticky_ccr_tool, should_inject_ccr_tool
from headroom.proxy.models import ProxyConfig

# --------------------------------------------------------------------------- #
# Section 1: J1, bust-time CCR flush -- full-handler integration harness.     #
# --------------------------------------------------------------------------- #


class _DummyTokenizer:
    def count(self, messages) -> int:
        return 1

    def count_messages(self, messages) -> int:
        return 1

    def count_tokens(self, text) -> int:
        return 1


class _DummyMetrics:
    def __init__(self) -> None:
        self.stage_timings: list[tuple[str, dict]] = []

    async def record_request(self, **kwargs):
        return None

    async def record_stage_timings(self, path: str, timings: dict) -> None:
        self.stage_timings.append((path, timings))

    async def record_rate_limited(self, **kwargs) -> None:
        return None

    async def record_failed(self, **kwargs) -> None:
        return None

    def record_compression_failed(self, reason: str) -> None:
        return None


class _ResponseStub:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self._text = json.dumps(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-3-5-sonnet-latest",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    @property
    def text(self) -> str:
        return self._text

    @property
    def content(self) -> bytes:
        return self._text.encode("utf-8")

    def json(self) -> dict:
        return json.loads(self._text)


class _FakePrefixTracker:
    """Controllable double for PrefixCacheTracker.

    Exposes exactly the surface handle_anthropic_messages touches in
    token mode (non-hybrid, non-cache-preserving), with alive_fraction
    and frozen_message_count as free knobs so the structural-bust branch
    can be driven deterministically without a live PrefixCacheTracker.
    """

    def __init__(self, *, frozen_message_count: int, alive_fraction: float) -> None:
        self._frozen_message_count = frozen_message_count
        self._alive_fraction = alive_fraction
        self._cached_token_count = 5000
        self.compress_latched = False
        self.hybrid_controller = SimpleNamespace(
            config=SimpleNamespace(adaptive_ttl=False, subagent_ttl_5m=False)
        )

    def observe_client_churn(self, _messages, head_fingerprint=None):
        return self._alive_fraction

    def get_last_original_messages(self):
        return []

    def get_frozen_message_count(self):
        return self._frozen_message_count

    def get_last_forwarded_messages(self):
        return []

    def note_compression(self, *_a, **_k):
        return None

    def update_from_response(self, *_a, **_k):
        return None

    def record_request(self, *_a, **_k):
        return None

    def record_turn_gap(self, *_a, **_k):
        return None

    def recommended_ttl(self):
        return None

    def prefers_long_ttl(self):
        # The real tracker answers False below the TTL size floor, and this
        # double's prefix is 5000 tokens, well under it.
        return False

    def cached_token_count(self):
        return self._cached_token_count

    def latch_compress(self):
        self.compress_latched = True

    def recent_compression_ratio(self, *_a, **_k):
        return 0.8

    def conservative_compression_ratio(self, *_a, **_k):
        return 0.8

    def compression_ratio_stddev(self):
        return 0.0

    def turn_number(self):
        return 2


class _DummyAnthropicHandler(AnthropicHandlerMixin):
    """Minimal handler wired to drive handle_anthropic_messages end to end,
    capturing the outbound body so J1's CCR tool-injection decision can be
    observed on the actual wire payload, not a re-implementation of it."""

    ANTHROPIC_API_URL = "https://api.anthropic.com"

    def _extract_anthropic_cache_ttl_metrics(self, usage):
        return (0, 0)

    def __init__(
        self,
        *,
        session_id: str,
        frozen_message_count: int,
        alive_fraction: float,
        ccr_inject_tool: bool = True,
    ) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = ProxyConfig(
            optimize=False,
            image_optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
            mode="token",
            cache_enabled=False,
            rate_limit_enabled=False,
            fallback_enabled=False,
            fallback_provider=None,
            prefix_freeze_enabled=False,
            memory_enabled=False,
            ccr_inject_tool=ccr_inject_tool,
            ccr_inject_system_instructions=False,
        )
        self.usage_reporter = None
        self.anthropic_provider = SimpleNamespace(get_context_limit=lambda model: 200_000)
        self.anthropic_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.cache = None
        self.security = None
        self._upstream_status = 200
        self.ccr_context_tracker = None
        self.ccr_injector = None
        self.ccr_response_handler = None
        self.ccr_feedback = None
        self.ccr_batch_processor = None
        self.ccr_mcp_server = None
        self.traffic_learner = None
        self.tool_injector = None
        self.read_lifecycle_manager = None
        self.logger = SimpleNamespace(log=lambda *a, **k: None)
        self.request_logger = self.logger
        self.usage_observer = None
        self.image_compressor = None
        tracker = _FakePrefixTracker(
            frozen_message_count=frozen_message_count, alive_fraction=alive_fraction
        )
        # The handler resolves its tracker through ``resolve_tracker`` and only
        # falls back to ``get_or_create``. Both hand back the same object in the
        # real store, so the double gives them the same one.
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: session_id,
            peek_idle_seconds=lambda *a, **k: 0.0,
            get_or_create=lambda *a, **k: tracker,
            resolve_tracker=lambda *a, **k: tracker,
        )
        self.anthropic_pre_upstream_sem = None
        self.anthropic_pre_upstream_concurrency = 0

        import concurrent.futures as _cf
        import threading as _threading

        self._compression_executor = _cf.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="dummy-compress"
        )
        self.compression_max_workers = 2
        self._compression_in_flight = 0
        self._compression_in_flight_max = 0
        self._compression_leaked_threads = 0
        self._compression_metrics_lock = _threading.Lock()
        self.captured_bodies: list[dict] = []

    async def _run_compression_in_executor(self, fn, *, timeout):
        loop = asyncio.get_running_loop()

        def _wrapped():
            return fn()

        future = loop.run_in_executor(self._compression_executor, _wrapped)
        return await asyncio.wait_for(future, timeout=timeout)

    async def _record_request_outcome(self, outcome) -> None:
        from headroom.proxy.outcome import emit_request_outcome

        await emit_request_outcome(self, outcome)

    async def _next_request_id(self) -> str:
        return f"req-{id(object()):x}"

    def _extract_tags(self, headers):
        return {}

    async def _retry_request(
        self,
        method: str,
        url: str,
        headers: dict,
        body: dict,
        *,
        original_body_bytes: bytes | None = None,
        body_mutated: bool = True,
        mutation_reasons: list[str] | None = None,
        request_id: str | None = None,
        forwarder_name: str = "test_dummy",
        path_for_log: str | None = None,
        timeout=None,
    ):
        del original_body_bytes, body_mutated, mutation_reasons
        del request_id, forwarder_name, path_for_log, timeout
        self.captured_bodies.append(body)
        return _ResponseStub(status_code=self._upstream_status)

    def _get_compression_cache(self, session_id):
        return SimpleNamespace(
            apply_cached=lambda m: m,
            compute_frozen_count=lambda m: 0,
            mark_stable_from_messages=lambda *a, **k: None,
            should_defer_compression=lambda h: False,
            mark_stable=lambda h: None,
            content_hash=lambda c: "h",
            update_from_result=lambda *a, **k: None,
            _cache={},
            _stable_hashes=set(),
        )


def _build_request(body: dict, headers: dict[str, str]) -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


def _tokenizer_patch():
    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer

    class _Ctx:
        def __enter__(self):
            _tk.get_tokenizer = lambda model: _DummyTokenizer()
            return self

        def __exit__(self, *exc):
            _tk.get_tokenizer = orig_get

    return _Ctx()


def _drive_bust_turn(
    *,
    session_id: str,
    ccr_inject_tool: bool = True,
    monkeypatch: pytest.MonkeyPatch,
) -> _DummyAnthropicHandler:
    """One request shaped to force the STRUCTURAL-BUST branch: a frozen
    prefix (get_frozen_message_count > 0) plus a low observed
    alive_fraction, no CCR markers anywhere in the message text."""
    monkeypatch.delenv("HEADROOM_STRUCTURAL_BUST_TTL_5M", raising=False)
    monkeypatch.delenv("HEADROOM_STRUCTURAL_BUST_ALIVE_THRESHOLD", raising=False)
    handler = _DummyAnthropicHandler(
        session_id=session_id,
        frozen_message_count=3,
        alive_fraction=0.1,  # well under the 0.5 default threshold
        ccr_inject_tool=ccr_inject_tool,
    )
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three, no compression markers here"},
            ],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    with _tokenizer_patch():
        anyio.run(handler.handle_anthropic_messages, request)
    return handler


def _tool_names(body: dict) -> list[Any]:
    return [t.get("name") for t in body.get("tools", []) if isinstance(t, dict)]


class TestBustFlushOnFreshSession:
    """J1(a)/(d) plus the real bug this attack surfaced: the flush claims to
    inject the retrieval tool "into the forced write" but, for a session
    that has never run CCR and has no fresh markers this turn,
    apply_session_sticky_ccr_tool declines silently -- the log lies."""

    def test_flush_log_fires_but_tool_is_never_added(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("INFO", logger="headroom.proxy")
        handler = _drive_bust_turn(
            session_id="sess-fresh-bust-flush", monkeypatch=monkeypatch
        )
        assert len(handler.captured_bodies) == 1, "expected exactly one upstream call"
        body = handler.captured_bodies[0]

        # Fixed 2026-07-21: the dead tool-flush override and its misleading log
        # were removed. A fresh-session bust must NOT claim a tool flush, and
        # the tool stays absent because there are no markers to redeem.
        flush_logged = any(
            "flushing deferred tool injection into forced write" in r.getMessage()
            for r in caplog.records
        )
        assert not flush_logged, (
            "the false tool-flush log must no longer fire for a fresh session"
        )
        assert "headroom_retrieve" not in _tool_names(body), (
            "a fresh session with no markers has nothing to redeem, so the "
            "retrieve tool must stay absent even on a bust"
        )


class TestBustFlushOnPriorCCRSession:
    """Contrast case: once a session has already done CCR, sticky replay
    fires unconditionally, so the flush *does* land there. Establishes the
    fresh-session case above is the actual gap, not a universal one."""

    def test_flush_reinjects_sticky_tool_for_already_ccrd_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_id = "sess-primed-bust-flush"
        # Prime the session tracker exactly as a prior real CCR turn would:
        # one call with fresh compressed content records golden bytes.
        apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id=session_id,
            request_id="req-prime",
            existing_tools=[],
            has_compressed_content_this_turn=True,
        )
        handler = _drive_bust_turn(session_id=session_id, monkeypatch=monkeypatch)
        body = handler.captured_bodies[0]
        assert "headroom_retrieve" in _tool_names(body), (
            "sticky replay should reinject the tool on a bust turn for a "
            "session that has already done CCR, regardless of this turn's "
            "marker content"
        )


class TestBustFlushZeroFrozenIsNoOp:
    """J1(d): frozen_message_count == 0 means nothing was deferred, the
    flush must never fire, structural bust itself must never trigger."""

    def test_no_frozen_prefix_never_forces_structural_bust(self) -> None:
        assert not _structural_bust_requires_fresh_5m(0.0, 0)
        assert not _structural_bust_requires_fresh_5m(0.01, 0)

    def test_negative_frozen_message_count_never_forces_structural_bust(self) -> None:
        # Defensive: a caller bug that lets frozen_message_count go negative
        # must not accidentally satisfy `> 0`.
        assert not _structural_bust_requires_fresh_5m(0.1, -1)


class TestBustFlushMutualExclusionWithMarkerOverride:
    """J1(c): is_bust_flush and is_marker_override must never both fire --
    is_bust_flush is only computed when `not should_inject`, and
    is_marker_override implies should_inject=True already, so the two
    branches are exclusive by construction. Verified against the real
    should_inject_ccr_tool, and the is_bust_flush formula transcribed
    verbatim from anthropic.py:3217-3223."""

    @pytest.mark.parametrize(
        "frozen_message_count,has_new_compressed_content,flush_into_forced_write",
        [
            (3, True, True),  # marker override should already cover this
            (3, True, False),
            (3, False, True),  # pure flush case
            (3, False, False),  # neither fires
            (0, True, False),  # no frozen prefix at all
        ],
    )
    def test_exclusivity_holds(
        self, frozen_message_count, has_new_compressed_content, flush_into_forced_write
    ) -> None:
        should_inject, is_marker_override = should_inject_ccr_tool(
            configured_inject_tool=True,
            frozen_message_count=frozen_message_count,
            has_compressed_content=has_new_compressed_content,
        )
        # anthropic.py:3216-3223 verbatim:
        is_bust_flush = False
        if not should_inject and True and flush_into_forced_write:  # configured_inject_tool=True
            should_inject = True
            is_bust_flush = True
        assert not (is_bust_flush and is_marker_override), (
            f"double-fire: is_bust_flush={is_bust_flush} "
            f"is_marker_override={is_marker_override}"
        )


# --------------------------------------------------------------------------- #
# Section 2: J2, tier-aware reconciliation TTL.                               #
# --------------------------------------------------------------------------- #


class TestMaxCacheTtlSecondsMalformedBodies:
    """J2(a): must never raise, must default to 300 (5m) on garbage."""

    @pytest.mark.parametrize(
        "body",
        [
            None,
            [],
            "not a dict",
            42,
            {"system": "a bare string, not a list"},
            {"system": {"not": "a list"}},
            {"messages": "also a bare string"},
            {"messages": 5},
            {"messages": [1, 2, 3]},
            {"messages": [{"role": "user", "content": "plain string content"}]},
            {"messages": [{"role": "user", "content": [1, 2, [3, 4]]}]},
            {"messages": [{"role": "user", "content": [{"cache_control": "not a dict"}]}]},
            {"messages": [{"role": "user", "content": [{"cache_control": {"ttl": 42}}]}]},
            {"messages": [{"role": "user", "content": [{"cache_control": {"ttl": None}}]}]},
            {"messages": [{"role": "user", "content": [{"cache_control": {"ttl": ["1h"]}}]}]},
            {"tools": "not a list"},
            {"tools": 5},
            {"tools": [1, "x", None, {"cache_control": {"ttl": "1h"}}]},
            {"messages": [None, {"role": "user"}, {"content": None}]},
            {"system": [1, "x", None, {"cache_control": None}]},
        ],
    )
    def test_never_raises_and_defaults_to_5m(self, body) -> None:
        result = message_segment_ttl_seconds(body)
        assert isinstance(result, float)
        assert result >= 300.0

    def test_malformed_ttl_label_defaults_within_valid_structure(self) -> None:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "x", "cache_control": {"ttl": "9d"}}],
                }
            ]
        }
        assert message_segment_ttl_seconds(body) == 300.0

    def test_valid_1h_is_recognized(self) -> None:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "x", "cache_control": {"ttl": "1h"}}],
                }
            ]
        }
        assert message_segment_ttl_seconds(body) == 3600.0

    def test_no_cache_control_anywhere_defaults_to_5m(self) -> None:
        assert message_segment_ttl_seconds({"messages": [{"role": "user", "content": "hi"}]}) == 300.0


class TestMaxCacheTtlSecondsTierConflation:
    """Fixed 2026-07-21: message_segment_ttl_seconds scans only the messages
    segment, so the deliberately-1h system/tools HEAD (anthropic.py: "The
    system/tools HEAD is left untouched (still 1h)") no longer masks a
    5m-forced message tail. Reconciliation tracks message-history warmth, and
    that is exactly the segment whose ttl this returns."""

    def test_untouched_1h_system_head_does_not_mask_forced_5m_message_tail(self) -> None:
        body = {
            "system": [
                {
                    "type": "text",
                    "text": "You are Claude Code.",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            "tools": [
                {
                    "name": "bash",
                    "description": "run bash",
                    "input_schema": {"type": "object"},
                }
            ],
            "messages": [
                {"role": "user", "content": "turn 1"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "the frozen suffix, just re-billed at 5m",
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        }
                    ],
                },
            ],
        }
        result = message_segment_ttl_seconds(body)
        # Fixed: returns 300 (the 5m tier the message tail was just written
        # with), ignoring the untouched 1h system HEAD.
        assert result == 300.0, (
            "the 1h system HEAD must not mask the 5m-forced message tail; "
            "reconciliation needs the message segment's own tier"
        )


class TestRecordTtlBoundary:
    """J2(b): a 1h-tier session's cold read at 400s is unplanned, a 5m-tier
    session's cold read at 400s is scheduled expiry."""

    def test_1h_tier_cold_read_at_400s_is_unplanned_bust(self) -> None:
        log = CacheReconciliationLog(log_path="/dev/null")
        log.record(
            session_key="s1",
            request_id="r1",
            model="claude",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=0.0,
            ttl_seconds=3600.0,
        )
        record = log.record(
            session_key="s1",
            request_id="r2",
            model="claude",
            billed_cache_read=0,
            billed_cache_creation=10_000,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=400.0,
            ttl_seconds=3600.0,
        )
        assert record.ttl_expired is False
        assert record.unplanned_bust is True

    def test_5m_tier_cold_read_at_400s_is_scheduled_ttl_expiry(self) -> None:
        log = CacheReconciliationLog(log_path="/dev/null")
        log.record(
            session_key="s2",
            request_id="r1",
            model="claude",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=0.0,
            ttl_seconds=300.0,
        )
        record = log.record(
            session_key="s2",
            request_id="r2",
            model="claude",
            billed_cache_read=0,
            billed_cache_creation=10_000,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=400.0,
            ttl_seconds=300.0,
        )
        assert record.ttl_expired is True
        assert record.unplanned_bust is False

    def test_boundary_exactly_at_ttl_is_not_expired(self) -> None:
        # prior_age > ttl_seconds is strict, so prior_age == ttl_seconds
        # must NOT be flagged expired.
        log = CacheReconciliationLog(log_path="/dev/null")
        log.record(
            session_key="s3",
            request_id="r1",
            model="claude",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=0.0,
            ttl_seconds=300.0,
        )
        record = log.record(
            session_key="s3",
            request_id="r2",
            model="claude",
            billed_cache_read=0,
            billed_cache_creation=10_000,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=300.0,
            ttl_seconds=300.0,
        )
        assert record.ttl_expired is False
        assert record.unplanned_bust is True


class TestRecordTtlClamping:
    """J2(c): ttl_seconds <= 0 or NaN must clamp to the 300s default, never
    propagate into the comparison and produce nonsense (e.g. every read
    reads as instantly expired, or NaN comparisons silently always False
    in a way that masks a real bust)."""

    @pytest.mark.parametrize("bad_ttl", [0.0, -1.0, -300.0, float("nan")])
    def test_non_positive_or_nan_ttl_clamps_to_default(self, bad_ttl: float) -> None:
        log = CacheReconciliationLog(log_path="/dev/null")
        log.record(
            session_key="clamp",
            request_id="r1",
            model="claude",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=0.0,
            ttl_seconds=300.0,
        )
        # 200s later: under the clamped 300s default this must NOT be
        # ttl_expired (a bad ttl must not silently suppress bust detection
        # by making everything look expired, nor crash on NaN comparisons).
        record = log.record(
            session_key="clamp",
            request_id="r2",
            model="claude",
            billed_cache_read=0,
            billed_cache_creation=10_000,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=200.0,
            ttl_seconds=bad_ttl,
        )
        assert record.ttl_expired is False
        assert record.unplanned_bust is True

    def test_infinite_ttl_never_expires(self) -> None:
        # inf > 0 is True, so this is accepted verbatim (not clamped) --
        # confirm it does not crash and simply never expires.
        log = CacheReconciliationLog(log_path="/dev/null")
        log.record(
            session_key="inf",
            request_id="r1",
            model="claude",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=0.0,
            ttl_seconds=300.0,
        )
        record = log.record(
            session_key="inf",
            request_id="r2",
            model="claude",
            billed_cache_read=0,
            billed_cache_creation=10_000,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=1_000_000.0,
            ttl_seconds=float("inf"),
        )
        assert record.ttl_expired is False


class TestPlannedBustStillWinsWithinTtl:
    """J2(d): planned-bust marker exclusion must still win over a
    within-TTL cold read that would otherwise look unplanned."""

    def test_planned_marker_suppresses_unplanned_bust_within_ttl(self) -> None:
        log = CacheReconciliationLog(log_path="/dev/null")
        log.record(
            session_key="planned",
            request_id="r1",
            model="claude",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=0.0,
            ttl_seconds=300.0,
        )
        # Within TTL (only 10s elapsed) but billed_cache_read collapses --
        # would be unplanned_bust=True except a planned-bust transform is
        # present.
        record = log.record(
            session_key="planned",
            request_id="r2",
            model="claude",
            billed_cache_read=0,
            billed_cache_creation=10_000,
            alive_fraction=0.2,
            first_diverged_index=1,
            transforms=["structural_bust_5m_rewrite"],
            now=10.0,
            ttl_seconds=300.0,
        )
        assert record.ttl_expired is False
        assert is_planned_bust(record.transforms) is True
        assert record.unplanned_bust is False

    def test_same_scenario_without_marker_is_flagged(self) -> None:
        # Contrast: identical numbers, no planned-bust label -> flagged.
        log = CacheReconciliationLog(log_path="/dev/null")
        log.record(
            session_key="unplanned",
            request_id="r1",
            model="claude",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=0.0,
            ttl_seconds=300.0,
        )
        record = log.record(
            session_key="unplanned",
            request_id="r2",
            model="claude",
            billed_cache_read=0,
            billed_cache_creation=10_000,
            alive_fraction=0.2,
            first_diverged_index=1,
            transforms=[],
            now=10.0,
            ttl_seconds=300.0,
        )
        assert record.unplanned_bust is True


class TestIsUnplannedBustNullModel:
    """Null-model sanity: a predicted-zero session (no prior turn) must
    never be flagged, regardless of what gets billed."""

    def test_zero_predicted_is_never_a_bust(self) -> None:
        assert is_unplanned_bust(0, 0) is False
        assert is_unplanned_bust(0, 999_999) is False

    def test_exactly_half_predicted_is_not_yet_a_bust(self) -> None:
        # is_unplanned_bust requires billed < predicted/2 (strict).
        assert is_unplanned_bust(1000, 500) is False
        assert is_unplanned_bust(1000, 499) is True
