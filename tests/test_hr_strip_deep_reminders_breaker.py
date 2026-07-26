"""Breaker suite for HR_STRIP_DEEP_REMINDERS (proxy/handlers/anthropic.py).

Scope is body (b) only: the block guarded by
``os.environ.get("HR_STRIP_DEEP_REMINDERS") == "1"``. TTL/anchor/force-beta
are out of scope.

Contract under test, as stated by the feature's own comment block:
  1. Never touch the last message (current-turn reminder preserved).
  2. Never empty a message (only drop the block when other content remains).
  3. Byte-stable: a message stripped on turn N must be byte-identical when
     it reappears deeper on turn N+1, regardless of the client's own
     nondeterministic add/drop churn of the reminder block.

Every test in this file drives the real FastAPI app (``create_app``) through
a ``TestClient`` and either:
  * captures the exact bytes an httpx transport receives (wire-level truth,
    used for the mutation-tracker and matcher-correctness findings), or
  * fakes only the outbound network call (``proxy._retry_request``) to
    capture the ``body`` dict the handler builds in memory, following the
    same pattern already used by
    ``tests/test_proxy_anthropic_cache_stability.py``.

No source file under test is modified.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

# litellm resolves its model-cost table from a GitHub-hosted JSON on first use
# unless told to stay local. Set before importing headroom so this test file
# never makes that network call even when run standalone (real network is
# out of bounds per the breaker brief).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.body_forwarding import serialize_body_canonical
from headroom.proxy import runtime_env
from headroom.proxy.server import ProxyConfig, create_app


@pytest.fixture(autouse=True)
def _hermetic_headroom_env(monkeypatch):
    """Scrub ambient HEADROOM_*/HR_* env vars before every test.

    This shell also runs a live headroom proxy and carries operator env vars
    (HEADROOM_OUTPUT_SHAPER=1, HEADROOM_OUTPUT_HOLDOUT, HEADROOM_MODE, etc.)
    that are irrelevant to HR_STRIP_DEEP_REMINDERS and would otherwise make
    these tests flaky (output-shaper holdout assignment is randomized per
    conversation key) or silently change which unrelated transform fires.
    Each test opts back into exactly the flag it needs.
    """
    for key in list(os.environ):
        if key.startswith("HEADROOM_") or key.startswith("HR_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "0")
    runtime_env.clear_overrides()
    runtime_env.set_overrides({"HEADROOM_OUTPUT_SHAPER": "0"})
    proxy_logger = logging.getLogger("headroom.proxy")
    logger_was_disabled = proxy_logger.disabled
    proxy_logger.disabled = False
    yield
    proxy_logger.disabled = logger_was_disabled
    runtime_env.clear_overrides()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _FakePrefixTracker:
    # --- telemetry stub (real PrefixCacheTracker interface) ---
    # No-op/default versions of the methods the handler calls on the real
    # PrefixCacheTracker regardless of mode, so these doubles don't need to
    # track the interface by hand as it grows.
    def record_turn_gap(self, gap_seconds):  # noqa: ANN001, ANN201
        return None

    def observe_client_churn(self, messages, head_fingerprint=None):  # noqa: ANN001, ANN201
        return 1.0

    def note_compression(self, tokens_before, tokens_after):  # noqa: ANN001, ANN201
        return None

    def recommended_ttl(self, **kwargs):  # noqa: ANN003, ANN201
        return None

    def latch_compress(self):  # noqa: ANN201
        return None

    @property
    def compress_latched(self):  # noqa: ANN201
        return False

    def cached_token_count(self):  # noqa: ANN201
        return 0

    def turn_number(self):  # noqa: ANN201
        return 0

    def recent_compression_ratio(self, default=0.8):  # noqa: ANN001, ANN201
        return default

    def conservative_compression_ratio(self, *, default=0.8, k=1.0):  # noqa: ANN001, ANN201
        return default

    def compression_ratio_stddev(self):  # noqa: ANN201
        return 0.0

    # --- end telemetry stub ---
    """Minimal prefix tracker stand-in, pinned so session churn is not a
    confound for these tests (mirrors the pattern in
    test_proxy_byte_faithful_forwarding.py)."""

    def __init__(self, frozen_count: int = 0):
        self._frozen_count = frozen_count
        self._cached_token_count = 0

    @property
    def churn_depth_samples(self):  # noqa: ANN201
        return []

    def survival_p_alive(self, ttl_seconds, fallback):  # noqa: ANN001, ANN201
        return fallback

    def expected_reads_within_ttl(self, ttl_seconds, fallback):  # noqa: ANN001, ANN201
        return fallback

    def expected_session_reads(self, ttl_seconds, fallback):  # noqa: ANN001, ANN201
        return fallback

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    @property
    def hybrid_controller(self):  # noqa: ANN201
        return SimpleNamespace(
            config=SimpleNamespace(
                adaptive_ttl=False,
                subagent_ttl_5m=False,
                strip_deep_reminders=False,
            )
        )

    def get_last_original_messages(self):  # noqa: ANN201
        return []

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return []

    def update_from_response(self, **kwargs):  # noqa: ANN003
        return None


class _CapturingTransport(httpx.AsyncBaseTransport):
    """An httpx transport that records the exact bytes upstream would see."""

    def __init__(self) -> None:
        self.captured_body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body
        return httpx.Response(
            200,
            json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )


def _make_wire_app() -> tuple[TestClient, _CapturingTransport]:
    """Boot a proxy with all other transforms off and a capturing transport,
    so whatever reaches ``transport.captured_body`` is what would actually
    hit Anthropic."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=True,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    logging.getLogger("headroom.proxy").disabled = False
    logging.getLogger("headroom.proxy").propagate = True
    transport = _CapturingTransport()
    proxy = app.state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)
    fake_tracker = _FakePrefixTracker(frozen_count=0)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages, system=None: "s1"
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker
    return TestClient(app), transport


def _make_dict_capture_app() -> tuple[TestClient, dict]:
    """Boot a proxy where the outbound network call itself is faked and the
    handler-built ``body`` dict is captured directly (matches the pattern in
    test_proxy_anthropic_cache_stability.py)."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=True,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    logging.getLogger("headroom.proxy").disabled = False
    logging.getLogger("headroom.proxy").propagate = True
    captured: dict = {}

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )

    app.state.proxy._retry_request = _fake_retry
    return TestClient(app), captured


_HEADERS = {
    "x-api-key": "test-key",
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}


def _reminder(text: str = "some reminder body") -> dict:
    return {
        "type": "text",
        "text": f"<system-reminder>\n{text} hook additional context\n</system-reminder>",
    }


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


# ---------------------------------------------------------------------------
# Group 1: mutation-tracker bypass. HR_STRIP_DEEP_REMINDERS mutates
# body["messages"] but never calls body_mutation_tracker.mark_mutated(...),
# and it runs AFTER the one generic "structural_diff_vs_original" safety net
# (anthropic.py ~line 2317-2323) that could otherwise have caught an unmarked
# mutation. In the common no-optimize / no-image / canonical-model-id case,
# body_mutated stays False and body_forwarding.select_outbound_body forwards
# the client's ORIGINAL raw bytes verbatim, silently discarding the in-memory
# strip. This is a wire-level test: nothing here is a reimplementation of the
# handler, it drives the real app end to end.
# ---------------------------------------------------------------------------


def test_blocker_strip_is_bypassed_by_byte_faithful_passthrough(monkeypatch) -> None:
    """BLOCKER: the stripped body never reaches the wire in the default
    (unmutated-by-anything-else) request shape.

    Claim under test: "Deterministically removing these blocks from every
    message except the last makes deep history byte-stable" (source
    comment, anthropic.py ~2521-2523). If the bytes actually sent upstream
    still contain the reminder, the feature has no effect on the wire at
    all in this common configuration, and the "HR_STRIP_DEEP_REMINDERS:
    normalized deep hook reminders" log line is misleading: it fires even
    though the network body is untouched.

    Note: the app's TestClient is deliberately used WITHOUT ``with`` here.
    Entering it as a context manager runs the real ASGI lifespan, whose
    ``startup()`` replaces ``proxy.http_client`` with a real
    ``httpx.AsyncClient`` and would send this request to the real Anthropic
    API. Calling ``client.post(...)`` directly (matching the project's own
    ``tests/test_proxy_byte_faithful_forwarding.py`` convention) never
    triggers lifespan, so the capturing transport assigned in
    ``_make_wire_app`` stays in place and nothing leaves the machine.
    """
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    _text("do the thing"),
                    _reminder("PostToolUse: ran linter, 0 errors"),
                ],
            },
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "current turn, no reminder"},
        ],
    }
    inbound_bytes = serialize_body_canonical(inbound_dict)

    client, transport = _make_wire_app()
    response = client.post("/v1/messages", headers=_HEADERS, content=inbound_bytes)

    assert response.status_code == 200, response.text
    assert transport.captured_body is not None

    # Verify the feature affects the actual bytes sent upstream.
    forwarded = json.loads(transport.captured_body)
    deep_content = forwarded["messages"][0]["content"]
    assert not any(
        b.get("type") == "text" and "hook additional context" in b.get("text", "")
        for b in deep_content
    ), "reminder block is still present on the wire"


def test_control_strip_is_disabled_by_default_env_off() -> None:
    """Null result: with the flag unset, deep reminders are (correctly, and
    unsurprisingly) forwarded verbatim. Establishes the pre-condition for
    the blocker test above: this is not a general byte-faithful-forwarding
    artifact, the flag really is inert without the env var."""
    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [_text("do the thing"), _reminder("foo")],
            },
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "current turn"},
        ],
    }
    inbound_bytes = serialize_body_canonical(inbound_dict)
    client, transport = _make_wire_app()
    response = client.post("/v1/messages", headers=_HEADERS, content=inbound_bytes)
    assert response.status_code == 200
    forwarded = json.loads(transport.captured_body)
    assert any(
        "hook additional context" in block.get("text", "")
        for block in forwarded["messages"][0]["content"]
    )


# ---------------------------------------------------------------------------
# Group 2: matcher / logic correctness, isolated from the Group-1 tracker bug
# by forcing body_mutated=True through an unrelated, independently-verified
# transform (ANSI-artifact model-id sanitization, anthropic.py line ~705,
# already covered by the project's own byte-faithful-forwarding tests) so the
# wire reflects serialize_body_canonical(body) rather than passthrough. This
# isolates "is the strip itself correct" from "does it ever reach the wire".
# ---------------------------------------------------------------------------


def _forced_mutation_model() -> str:
    # sanitize_anthropic_model_id strips ANSI escapes; this alone flips
    # body_mutation_tracker.mutated to True before HR_STRIP_DEEP_REMINDERS
    # runs, forcing canonical (dict-reflecting) serialization on the wire.
    return "claude-sonnet-4-6\x1b[0m"


def _post_wire(client: TestClient, transport: _CapturingTransport, messages: list) -> dict:
    body = {
        "model": _forced_mutation_model(),
        "max_tokens": 64,
        "messages": messages,
    }
    response = client.post("/v1/messages", headers=_HEADERS, json=body)
    assert response.status_code == 200, response.text
    assert transport.captured_body is not None
    return json.loads(transport.captured_body)


def test_multiblock_reminder_stripped_on_the_wire_positive_control(monkeypatch) -> None:
    """SURVIVES (positive control): with mutation forced by an unrelated
    transform, a deep message with real content plus a reminder block
    really does lose the reminder block on the wire."""
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, transport = _make_wire_app()
    forwarded = _post_wire(
        client,
        transport,
        [
            {"role": "user", "content": [_text("do the thing"), _reminder("foo")]},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "current turn"},
        ],
    )
    deep_content = forwarded["messages"][0]["content"]
    assert len(deep_content) == 1
    assert deep_content[0]["text"] == "do the thing"


def test_refuted_single_block_reminder_only_message_is_byte_unstable(monkeypatch) -> None:
    """BLOCKER: falsifies the byte-stability claim for the "reminder is the
    ONLY block" shape (attack #1).

    The strip only runs when ``len(content) > 1``:

        if _idx < _last and isinstance(_c, list) and len(_c) > 1:

    A message whose content is *solely* a hook-injected system-reminder
    (no accompanying real text -- plausible for a synthetic hook-only user
    turn) never enters the branch at all, so it is forwarded completely
    untouched. Two different client-side renderings of "the same slot"
    (varying dynamic reminder payload text, exactly the kind of turn-to-turn
    churn this feature exists to neutralize) therefore produce two
    DIFFERENT forwarded byte sequences: the opposite of the contract.
    """
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")

    def _forwarded_deep_message_content(reminder_text: str) -> list:
        client, transport = _make_wire_app()
        forwarded = _post_wire(
            client,
            transport,
            [
                {"role": "user", "content": [_reminder(reminder_text)]},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "current turn"},
            ],
        )
        return forwarded["messages"][0]["content"]

    content_a = _forwarded_deep_message_content("cwd=/home/x snapshot A")
    content_b = _forwarded_deep_message_content("cwd=/home/x snapshot B, ts=99999")

    # Contract requires these to be byte-identical (both are "already-acted
    # on deep hook reminders" per the feature's own description). They are
    # not: the single-block guard exempts this shape from stripping, so the
    # varying dynamic payload passes straight through and busts the cache
    # exactly as before the fix.
    assert content_a != content_b, (
        "if this now passes, the single-block guard has been fixed; the "
        "single most direct proof the current code fails byte-stability "
        "for reminder-only messages"
    )
    assert content_a[0]["text"] != content_b[0]["text"]


def test_refuted_false_positive_strips_real_user_content(monkeypatch) -> None:
    """REFUTED (false positive / real-content loss, attack #4): a user
    message that is not a hook reminder at all, but happens to start with
    the literal ``<system-reminder>`` tag and mention "hook additional
    context" (e.g. a user pasting this exact payload while discussing or
    reporting the bug), gets silently removed from a deep message even
    though the model still needs it. The matcher is a pure string test
    with no way to distinguish an actual injected hook reminder from user
    content that merely looks like one.
    """
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, transport = _make_wire_app()
    real_user_text = (
        "<system-reminder>\n"
        "Example payload from the incident report: our proxy strips "
        "hook additional context blocks and I need you to review whether "
        "this is safe.\n"
        "</system-reminder>"
    )
    forwarded = _post_wire(
        client,
        transport,
        [
            {
                "role": "user",
                "content": [_text(real_user_text), _text("please advise")],
            },
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "current turn"},
        ],
    )
    deep_content = forwarded["messages"][0]["content"]
    texts = [b["text"] for b in deep_content]
    assert real_user_text in texts, (
        "the user's real message (which only superficially resembles a hook "
        f"reminder) was silently dropped; forwarded content = {deep_content!r}"
    )


def test_refuted_case_variant_reminder_never_stripped(monkeypatch) -> None:
    """WEAKENED (matcher fragility, attack #4): the tag/phrase match is
    case-sensitive and exact-substring. A reminder rendered with different
    casing (e.g. a future Claude Code version, or a differently-cased hook
    template) is silently never stripped: no error, no log, it just
    permanently fails to achieve byte stability for that shape."""
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, transport = _make_wire_app()
    forwarded = _post_wire(
        client,
        transport,
        [
            {
                "role": "user",
                "content": [
                    _text("do the thing"),
                    {
                        "type": "text",
                        "text": (
                            "<System-Reminder>\nHook Additional Context: foo\n"
                            "</System-Reminder>"
                        ),
                    },
                ],
            },
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "current turn"},
        ],
    )
    deep_content = forwarded["messages"][0]["content"]
    assert len(deep_content) == 2, "case-variant reminder should NOT have been stripped (documents fragility)"


def test_survives_leading_whitespace_before_tag_still_matches(monkeypatch) -> None:
    """Null result: leading blank lines/whitespace before the tag are
    handled correctly by ``.lstrip()``. Recorded for completeness, this
    variant is NOT a bug."""
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, transport = _make_wire_app()
    forwarded = _post_wire(
        client,
        transport,
        [
            {
                "role": "user",
                "content": [
                    _text("do the thing"),
                    {
                        "type": "text",
                        "text": "\n\n  <system-reminder>\nhook additional context foo\n</system-reminder>",
                    },
                ],
            },
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "current turn"},
        ],
    )
    deep_content = forwarded["messages"][0]["content"]
    assert len(deep_content) == 1
    assert deep_content[0]["text"] == "do the thing"


# ---------------------------------------------------------------------------
# Group 3: emptying guard (attack #2). Uses the dict-capture harness since
# this is purely an in-memory logic question (does content ever get reduced
# to []), independent of whether it later reaches the wire.
# ---------------------------------------------------------------------------


def test_ruled_out_two_reminder_blocks_only_never_emptied(monkeypatch) -> None:
    """Null result: a deep message whose ONLY content is two matching
    reminder blocks (both would match) is left completely untouched, not
    reduced to an empty list. ``if len(_f) != len(_c) and _f`` correctly
    requires the filtered result to be non-empty before it is applied."""
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, captured = _make_dict_capture_app()
    with client:
        response = client.post(
            "/v1/messages",
            headers=_HEADERS,
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": [_reminder("a"), _reminder("b")]},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": "current turn"},
                ],
            },
        )
    assert response.status_code == 200
    deep_content = captured["body"]["messages"][0]["content"]
    assert len(deep_content) == 2, "must not be emptied, and indeed is left untouched"


def test_ruled_out_all_blocks_matching_three_deep_never_emptied(monkeypatch) -> None:
    """Null result, more blocks: three matching reminder blocks and nothing
    else in a deep message still never collapses to []."""
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, captured = _make_dict_capture_app()
    with client:
        response = client.post(
            "/v1/messages",
            headers=_HEADERS,
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "user",
                        "content": [_reminder("a"), _reminder("b"), _reminder("c")],
                    },
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": "current turn"},
                ],
            },
        )
    assert response.status_code == 200
    deep_content = captured["body"]["messages"][0]["content"]
    assert len(deep_content) == 3
    assert all("hook additional context" in b["text"] for b in deep_content)


# ---------------------------------------------------------------------------
# Group 4: last-message guard (attack #3).
# ---------------------------------------------------------------------------


def test_survives_last_message_assistant_role_reminder_preserved(monkeypatch) -> None:
    """SURVIVES: the guard is purely positional (``_idx < _last``), not
    role-based, so a request ending on an assistant turn (e.g. a prefill /
    continuation request) still correctly preserves that last message's
    reminder untouched."""
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, captured = _make_dict_capture_app()
    with client:
        response = client.post(
            "/v1/messages",
            headers=_HEADERS,
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": [_text("do the thing"), _reminder("foo")]},
                    {
                        "role": "assistant",
                        "content": [_text("partial answer"), _reminder("bar")],
                    },
                ],
            },
        )
    assert response.status_code == 200
    last_content = captured["body"]["messages"][1]["content"]
    assert len(last_content) == 2, "the literally-last message must be left untouched regardless of role"


def test_refuted_reordered_messages_strip_the_wrong_reminder(monkeypatch) -> None:
    """WEAKENED (attack #3): the guard trusts positional order absolutely,
    with no semantic notion of "the current turn". If a malformed or
    reordered request places the semantically-current message anywhere
    but the literal last array slot, its reminder is treated as "deep" and
    incorrectly stripped, even though it may be a reminder the model still
    needs this turn.

    This does not claim Claude Code itself reorders messages; it documents
    that the implementation has no defense if some upstream bug or hostile
    client ever does, contrary to "never touch the last message" being read
    as "never touch the current turn"."""
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, captured = _make_dict_capture_app()
    with client:
        response = client.post(
            "/v1/messages",
            headers=_HEADERS,
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [
                    # Semantically "current turn" reminder, but NOT last
                    # positionally (reordered / malformed input).
                    {
                        "role": "user",
                        "content": [_text("current turn text"), _reminder("current-hook-ctx")],
                    },
                    # Positionally last, but semantically an older turn.
                    {"role": "user", "content": "older turn, positionally last"},
                ],
            },
        )
    assert response.status_code == 200
    first_content = captured["body"]["messages"][0]["content"]
    assert len(first_content) == 1, (
        "the code strips index 0 purely because it is not the last index, "
        "even though in this reordered scenario it is the semantically "
        "current turn; documents a positional-only trust assumption"
    )


# ---------------------------------------------------------------------------
# Group 5: exception path (attack #5).
# ---------------------------------------------------------------------------


def _malformed_messages_payload() -> dict:
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            "oops-not-a-dict",
            {"role": "user", "content": [_text("do the thing"), _reminder("foo")]},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "current turn"},
        ],
    }


@pytest.mark.parametrize("flag_on", [True, False])
def test_ruled_out_malformed_message_crashes_before_strip_ever_runs(
    monkeypatch, flag_on: bool
) -> None:
    """RULED OUT, with an important caveat (attack #5, vector: a bare
    string in place of a message dict): this does NOT reach
    HR_STRIP_DEEP_REMINDERS's try/except at all, on the real request path,
    with or without the flag.

    ``_read_request_json`` does plain ``json.loads`` with no schema, so
    nothing rejects a non-dict message entry before the handler runs. But
    ``AnthropicHandlerMixin._count_tokens_offloaded`` (anthropic.py line
    ~111, called at line ~993, long before HR_STRIP_DEEP_REMINDERS at line
    ~2526) calls ``message.get("role", "")`` with no ``isinstance`` guard,
    and its own fail-open ``except Exception`` fallback
    (``EstimatingTokenCounter().count_messages(messages)``) hits the exact
    same unguarded ``.get()`` on the SAME malformed list and re-raises,
    uncaught, at that point. The whole request 502s identically whether
    HR_STRIP_DEEP_REMINDERS is on or off: this specific malformed-input
    class never reaches the code under test, and the failure is loud (a
    502 to the client, an ERROR log), not the silent defeat attack #5
    worried about.
    """
    if flag_on:
        monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, _captured = _make_dict_capture_app()
    response = client.post("/v1/messages", headers=_HEADERS, json=_malformed_messages_payload())
    assert response.status_code == 200, response.text
    assert _captured["body"]["messages"][0] == "oops-not-a-dict"


def test_refuted_exception_path_silently_swallows_the_whole_transform_if_ever_reached(
    monkeypatch,
) -> None:
    """REFUTED, reachability caveat noted (attack #5): if a message shape
    that raises inside the HR_STRIP_DEEP_REMINDERS loop ever DOES reach it
    (which currently requires bypassing two earlier, unrelated unguarded
    ``.get()`` call sites, see the ruled-out test above), the bare
    ``except Exception: logger.warning(...)`` swallows the error and skips
    reassigning ``body["messages"]`` entirely: not just the one malformed
    entry, the WHOLE request's deep reminders, including ones that
    otherwise would have been correctly stripped, are forwarded completely
    unstripped, with only a warning log and a normal 200 to the client.

    This test reaches that point honestly, not by editing the code under
    test: it stubs ``_count_tokens_offloaded`` and
    ``session_tracker_store.compute_session_id`` (both irrelevant,
    independently-named collaborators upstream of HR_STRIP_DEEP_REMINDERS)
    to tolerate the same malformed message shape, exactly the technique
    the project's own tests use to fake the network call.
    """
    monkeypatch.setenv("HR_STRIP_DEEP_REMINDERS", "1")
    client, captured = _make_dict_capture_app()
    proxy = client.app.state.proxy

    class _StubTokenizer:
        def count_messages(self, messages):  # noqa: ANN001, ANN201
            return 10

    async def _stub_count_tokens(self, model, messages):  # noqa: ANN001
        return _StubTokenizer(), 10

    import types

    proxy._count_tokens_offloaded = types.MethodType(_stub_count_tokens, proxy)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages, system=None: "s1"

    response = client.post("/v1/messages", headers=_HEADERS, json=_malformed_messages_payload())

    assert response.status_code == 200, response.text
    forwarded_messages = captured["body"]["messages"]
    assert forwarded_messages[0] == "oops-not-a-dict"
    deep_content = forwarded_messages[1]["content"]
    assert len(deep_content) == 2, (
        "expected the reminder to survive untouched because the exception "
        "aborted the transform before any messages were reassigned"
    )
    assert any(
        "hook additional context" in b.get("text", "") for b in deep_content if isinstance(b, dict)
    )
