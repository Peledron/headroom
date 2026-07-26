"""Breaker coverage for the claim: ``_record_request_outcome`` already emits
the structured PERF line for completed Codex WS turns, so the manual
``logger.info(... PERF ...)`` block removed from
``handle_openai_responses_ws`` was truly redundant.

The writer's own regression test for this
(``tests/test_codex_ws_compression_scheduler.py::
test_codex_ws_emits_perf_log_with_cache_keys``) is ``pytest.skip``-ped and
the only other guard is a source-text grep (absence of
``_perf_input_tokens`` plus presence of ``_record_request_outcome(``) which
cannot detect a wrong PERF *format*, a missing emit under some code path, or
a duplicate emit. This file drives the real WS handler end to end with a
fake upstream/client pair (same fixtures pattern as
``tests/test_openai_codex_ws_lifecycle.py``) and inspects the actual log
records the funnel produces, then feeds the rendered log line through the
real ``benchmarks/codex_cache_audit.py`` parser to prove the two components
agree on wire format.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.ws_session_registry import WebSocketSessionRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]

AUDIT_SCRIPT = REPO_ROOT / "benchmarks" / "codex_cache_audit.py"
_SPEC = importlib.util.spec_from_file_location("audit_codex_cache_breaker", AUDIT_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)


# ── Fakes (self-contained; deliberately not imported from sibling test
#    modules so this file has no coupling to their private helpers). ──────


class _TokenCounter:
    def count_text(self, text: str) -> int:
        return len(text.split())


class _RealFunnelMetrics:
    """Records what ``emit_request_outcome``'s step 1 receives, nothing more."""

    def __init__(self) -> None:
        self.recorded_requests: list[dict] = []

    async def record_request(self, **kwargs) -> None:
        self.recorded_requests.append(dict(kwargs))

    async def record_stage_timings(self, path: str, timings: dict[str, float]) -> None:
        return None

    def inc_active_ws_sessions(self) -> None:
        return None

    def dec_active_ws_sessions(self) -> None:
        return None

    def inc_active_relay_tasks(self, n: int = 1) -> None:
        return None

    def dec_active_relay_tasks(self, n: int = 1) -> None:
        return None

    def record_ws_session_duration(self, duration_ms: float, cause: str) -> None:
        return None

    def record_codex_ws_frame(self, **kwargs) -> None:
        return None


class _RealFunnelHandler(OpenAIHandlerMixin):
    """Mirrors production ``HeadroomProxy`` shape closely enough to run
    ``handle_openai_responses_ws`` and, critically, uses the *real*
    ``_record_request_outcome`` -> ``emit_request_outcome`` funnel instead of
    a test double. If the funnel silently failed to log PERF for the Codex
    WS branch, this handler would reproduce the bug exactly as production
    would.
    """

    OPENAI_API_URL = "https://api.openai.com"

    def __init__(self) -> None:
        self.rate_limiter = None
        self.metrics = _RealFunnelMetrics()
        self.config = SimpleNamespace(
            optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
            log_full_messages=False,
            # Read when the WS forwarder builds its upstream header set. Absent
            # here the handler dies before any turn completes, and the PERF
            # assertions below fail for a reason that has nothing to do with
            # PERF emission.
            openai_extra_headers={},
        )
        self.usage_reporter = None
        self.openai_provider = SimpleNamespace(
            get_context_limit=lambda model: 128_000,
            get_token_counter=lambda model: _TokenCounter(),
        )
        self.openai_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        # cost_tracker=None and logger=None below make emit_request_outcome
        # skip steps 2 and 3, exactly as production does with --no-cost /
        # --no-request-logging. Step 4 (PERF) has no such off switch.
        self.cost_tracker = None
        self.logger = None
        self.memory_handler = None
        self.ws_sessions = WebSocketSessionRegistry()
        self.compression_executor_calls = 0

    async def _next_request_id(self) -> str:
        return "hr_breaker_ws_test"

    async def _run_compression_in_executor(self, fn, *, timeout: float):
        self.compression_executor_calls += 1
        return fn()

    async def _record_request_outcome(self, outcome) -> None:
        # Exact production wrapper (server.py:_record_request_outcome):
        # a thin pass-through to the free funnel function.
        from headroom.proxy.outcome import emit_request_outcome

        await emit_request_outcome(self, outcome)


class _FakeWebSocketDisconnect(Exception):
    pass


_FakeWebSocketDisconnect.__name__ = "WebSocketDisconnect_Fake"


class _FakeWebSocket:
    def __init__(
        self,
        frames: list[str],
        *,
        disconnect_after_n_sends: int,
    ) -> None:
        self.headers = {"authorization": "Bearer test"}
        self._frames = list(frames)
        self._disconnect_after_n_sends = disconnect_after_n_sends
        self.sent_text: list[str] = []
        self.closed = False
        self.close_code: int | None = None
        self._disconnect_event = __import__("asyncio").Event()
        self.client = SimpleNamespace(host="127.0.0.1", port=12345)

    async def accept(self, subprotocol=None, headers=None) -> None:
        return None

    async def receive_text(self) -> str:
        if self._frames:
            return self._frames.pop(0)
        await self._disconnect_event.wait()
        raise _FakeWebSocketDisconnect("client closed")

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)
        if len(self.sent_text) >= self._disconnect_after_n_sends:
            self._disconnect_event.set()

    async def send_bytes(self, data: bytes) -> None:
        return None

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed = True
        self.close_code = code


class _FakeHeaders:
    def __init__(self, pairs) -> None:
        self._pairs = list(pairs)

    def raw_items(self):
        return list(self._pairs)

    def items(self):
        return list(self._pairs)


class _FakeUpstream:
    def __init__(self, events: list[str]) -> None:
        self._events = list(events)
        self.sent: list[str] = []
        self.closed = False
        self.response = SimpleNamespace(headers=_FakeHeaders([]))

    async def __aenter__(self) -> _FakeUpstream:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for ev in self._events:
            yield ev


def _make_fake_websockets_module(upstream: _FakeUpstream):
    module = MagicMock()

    async def _connect(*args, **kwargs):
        return upstream

    module.connect = _connect
    module.Subprotocol = str
    return module


def _turn_frame(model: str = "gpt-5.4") -> str:
    return json.dumps({"type": "response.create", "response": {"model": model, "input": "hi"}})


def _completed_event(resp_id: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> str:
    return json.dumps(
        {
            "type": "response.completed",
            "response": {
                "id": resp_id,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_tokens_details": {"cached_tokens": cached_tokens},
                },
            },
        }
    )


def _created_event(resp_id: str) -> str:
    return json.dumps({"type": "response.created", "response": {"id": resp_id}})


class _DirectLogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def proxy_log_capture():
    handler = _DirectLogCapture()
    target = logging.getLogger("headroom.proxy")
    target.addHandler(handler)
    prior_level = target.level
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(prior_level)


@pytest.mark.asyncio
async def test_codex_ws_two_turns_emit_exactly_two_parseable_perf_lines(proxy_log_capture) -> None:
    """Falsifies (or confirms) the core claim of the diff.

    Two completed Codex Responses turns on one WS session must produce
    exactly two structured PERF lines through the real
    ``_record_request_outcome`` funnel (no manual emit, no duplicate, no
    silent drop), and those lines must be parseable by the real
    ``benchmarks/codex_cache_audit.py`` regexes with the correct per-turn
    (not cumulative) token deltas.
    """
    upstream_events = [
        _created_event("r_1"),
        _completed_event("r_1", input_tokens=1000, cached_tokens=800, output_tokens=20),
        _created_event("r_2"),
        _completed_event("r_2", input_tokens=1300, cached_tokens=900, output_tokens=30),
    ]
    upstream = _FakeUpstream(upstream_events)
    fake_ws_mod = _make_fake_websockets_module(upstream)
    client_ws = _FakeWebSocket(
        frames=[_turn_frame(), _turn_frame()],
        disconnect_after_n_sends=len(upstream_events),
    )
    handler = _RealFunnelHandler()

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    perf_lines = [
        record.getMessage()
        for record in proxy_log_capture.records
        if "PERF" in record.getMessage()
    ]

    assert len(perf_lines) == 2, (
        f"Expected exactly 2 PERF lines (one per completed Codex WS turn), "
        f"got {len(perf_lines)}: {perf_lines!r}. If this is 0, the P0 "
        "'Requests: 0' visibility bug is back (the claim that "
        "_record_request_outcome already emits PERF for this path is "
        "FALSE). If this is >2, the removed manual emit was NOT actually "
        "redundant duplication has reappeared through another path."
    )

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    rendered = "\n".join(formatter.format(r) for r in proxy_log_capture.records if "PERF" in r.getMessage())

    for line in rendered.splitlines():
        assert audit.PERF_RE.match(line), (
            f"benchmarks/codex_cache_audit.py's PERF_RE cannot parse the real "
            f"PERF line emitted by the funnel: {line!r}"
        )

    parsed = [audit._fields(audit.PERF_RE.match(line).group("fields")) for line in rendered.splitlines()]

    assert parsed[0]["tok_after"] == "1000"
    assert parsed[0]["cache_read"] == "800"
    assert parsed[0]["tok_out"] == "20"

    # Second turn's PERF line must carry this turn's OWN usage, not the
    # cumulative session total (1000 + 1300 = 2300). ws_*_total accumulators
    # are session-wide; the funnel's job is to diff against the previous
    # recorded baseline and report only the delta.
    assert parsed[1]["tok_after"] == "1300", (
        f"Second turn tok_after={parsed[1]['tok_after']!r}, expected 1300 "
        "(this turn's own input_tokens). A value of 2300 would mean the "
        "delta-tracking baseline reset or double-counted across turns."
    )
    assert parsed[1]["cache_read"] == "900"
    assert parsed[1]["tok_out"] == "30"


@pytest.mark.asyncio
async def test_codex_ws_perf_line_round_trips_through_audit_tool(tmp_path: Path, proxy_log_capture) -> None:
    """End-to-end: handler -> real logger format -> audit tool's own parser.

    This is the integration point the writer's skipped test never reached.
    A format drift between what the funnel emits and what the audit tool
    expects would silently zero out ``turns``/``input_tokens`` in every
    downstream report despite the log containing real traffic.
    """
    upstream_events = [
        _created_event("r_1"),
        _completed_event("r_1", input_tokens=25_000, cached_tokens=24_000, output_tokens=50),
    ]
    upstream = _FakeUpstream(upstream_events)
    fake_ws_mod = _make_fake_websockets_module(upstream)
    client_ws = _FakeWebSocket(frames=[_turn_frame()], disconnect_after_n_sends=len(upstream_events))
    handler = _RealFunnelHandler()

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        await handler.handle_openai_responses_ws(client_ws)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_path = tmp_path / "proxy.log"
    log_path.write_text(
        "\n".join(formatter.format(r) for r in proxy_log_capture.records) + "\n"
    )

    records, frames, agents = audit.parse_proxy_log(log_path)
    report = audit.summarize_log(
        records, frames, agents, min_prompt_tokens=20_000, priming_turns=0, bust_fraction=0.10
    )

    assert report["turns"] == 1, (
        f"Expected the audit tool to see 1 turn from a real handler-emitted "
        f"PERF line; saw {report['turns']}. This is the 'Requests: 0' "
        "regression surfacing through the actual tool the fix's own "
        "commentary invokes."
    )
    assert report["input_tokens"] == 25_000
    assert report["cached_input_tokens"] == 24_000

