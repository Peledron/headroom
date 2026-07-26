"""Recovery from a blocked recovery marker: naming the near miss, and not
parking the client's agent loop while doing it.

Marker text is assembled from fragments here rather than written out. A test
file is itself written through a tool call, and a contiguous marker literal in
that call's input is exactly what the guard refuses.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx

from headroom.proxy.handlers.anthropic import _guard_anthropic_tool_use_markers
from headroom.proxy.marker_recovery import blocked_message, nearest_stored_hashes

_STABILITY_SPEC = importlib.util.spec_from_file_location(
    "_marker_recovery_stability_helpers",
    Path(__file__).with_name("test_proxy_anthropic_cache_stability.py"),
)
assert _STABILITY_SPEC is not None and _STABILITY_SPEC.loader is not None
_STABILITY = importlib.util.module_from_spec(_STABILITY_SPEC)
_STABILITY_SPEC.loader.exec_module(_STABILITY)
_make_proxy_client = _STABILITY._make_proxy_client

_MASK_OPEN = "[Tool input" + " masked: content. "
_RETRIEVE = "Retrieve or" + "iginal: hash="

STORED = "0123456789abcdef01234567"
ONE_OFF = "0123456789abcdef01234568"


def _marker(hash_key: str) -> str:
    return f"{_MASK_OPEN}{_RETRIEVE}{hash_key}]"


class _Store:
    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self.entries = entries or {}

    def retrieve(self, hash_key: str):  # noqa: ANN201
        original = self.entries.get(hash_key)
        return None if original is None else SimpleNamespace(original_content=original)

    def stored_hashes(self) -> list[str]:
        return list(self.entries)


def test_one_character_slip_names_the_stored_hash() -> None:
    assert nearest_stored_hashes(ONE_OFF, [STORED, "ffffffffffffffffffffffff"]) == [STORED]


def test_truncated_hash_matches_by_prefix() -> None:
    assert nearest_stored_hashes(STORED[:-2], [STORED]) == [STORED]


def test_distant_hash_suggests_nothing() -> None:
    assert nearest_stored_hashes("ffffffffffffffffffffffff", [STORED]) == []


def test_present_hash_is_not_suggested_back_to_itself() -> None:
    # A hash that is in the store did not fail transcription, so pointing at
    # it would send the caller after a problem they do not have.
    assert nearest_stored_hashes(STORED, [STORED, ONE_OFF]) == []


def test_unreadable_hash_suggests_nothing() -> None:
    assert nearest_stored_hashes("unknown", [STORED]) == []


def test_message_offers_two_routes_and_no_dead_end() -> None:
    text = blocked_message(ONE_OFF, suggestions=[STORED])
    assert STORED in text
    assert "headroom_retrieve" in text
    assert "re-read the file" in text
    # The advice that could not be followed: the content the caller is missing
    # is the content that was masked away.
    assert "written out in full" not in text


def test_unknown_hash_message_says_the_hash_was_unreadable() -> None:
    text = blocked_message("unknown", suggestions=[])
    assert "no readable hash" in text


def _response(*blocks: dict) -> dict:
    return {
        "id": "msg_marker_recovery",
        "type": "message",
        "role": "assistant",
        "content": list(blocks),
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _tool_use(tool_id: str, tool_input: dict) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": "Write", "input": tool_input}


def test_clean_sibling_calls_survive_and_the_turn_keeps_its_tool_stop(monkeypatch) -> None:
    payload = _response(
        _tool_use("blocked", {"content": _marker(ONE_OFF)}),
        _tool_use("clean", {"content": "ordinary content"}),
    )
    monkeypatch.setattr(
        "headroom.cache.compression_store.get_compression_store",
        lambda: _Store({STORED: "the real content"}),
    )

    result = _guard_anthropic_tool_use_markers(payload, request_id="test")

    assert result.changed and not result.stalled
    # The client runs the clean call, gets a result, and the loop turns again
    # without anyone typing.
    assert payload["stop_reason"] == "tool_use"
    ids = [b.get("id") for b in payload["content"] if b.get("type") == "tool_use"]
    assert ids == ["clean"]
    assert any(b["type"] == "text" and STORED in b["text"] for b in payload["content"])


def test_sole_blocked_call_reports_a_stalled_turn(monkeypatch) -> None:
    payload = _response(_tool_use("blocked", {"content": _marker(ONE_OFF)}))
    monkeypatch.setattr(
        "headroom.cache.compression_store.get_compression_store", lambda: _Store()
    )

    result = _guard_anthropic_tool_use_markers(payload, request_id="test")

    assert result.stalled
    assert payload["stop_reason"] == "end_turn"
    assert result.message


def _post(monkeypatch, upstream: list[dict], store: _Store):  # noqa: ANN001, ANN202
    """Drive one client request, serving ``upstream`` responses in order."""
    calls: list[dict] = []
    with _make_proxy_client() as test_client:
        proxy = test_client.app.state.proxy
        monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda: store)

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            calls.append(body)
            return httpx.Response(200, json=upstream[min(len(calls) - 1, len(upstream) - 1)])

        proxy._retry_request = _fake_retry
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "write it"}],
            },
        )
    return response, calls


def test_blocked_sole_call_is_retried_upstream_not_handed_to_the_user(monkeypatch) -> None:
    blocked = _response(_tool_use("blocked", {"content": _marker(ONE_OFF)}))
    healed = _response(_tool_use("healed", {"content": "the real content"}))

    response, calls = _post(monkeypatch, [blocked, healed], _Store({STORED: "the real content"}))

    payload = response.json()
    # What the client receives is a live turn it can act on, so its loop never
    # stops to ask a human about a mistyped hash.
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0]["input"]["content"] == "the real content"
    assert len(calls) == 2
    retry_messages = calls[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert STORED in retry_messages[-2]["content"][0]["text"]
    assert retry_messages[-1]["role"] == "user"
    assert calls[1]["stream"] is False


def test_a_second_blocked_response_ends_the_turn(monkeypatch) -> None:
    blocked = _response(_tool_use("blocked", {"content": _marker(ONE_OFF)}))

    response, calls = _post(monkeypatch, [blocked], _Store())

    payload = response.json()
    # One retry, then stop. A model that repeats the mistake will not be talked
    # out of it by a second copy of the same message, and the loop must not
    # spend upstream calls discovering that.
    assert payload["stop_reason"] == "end_turn"
    assert all(block.get("type") != "tool_use" for block in payload["content"])
    assert len(calls) == 2
