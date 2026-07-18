"""Response-side recovery-marker guard regressions."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx

from headroom.proxy.handlers.anthropic import _guard_anthropic_tool_use_markers

_STABILITY_SPEC = importlib.util.spec_from_file_location(
    "_marker_guard_stability_helpers",
    Path(__file__).with_name("test_proxy_anthropic_cache_stability.py"),
)
assert _STABILITY_SPEC is not None and _STABILITY_SPEC.loader is not None
_STABILITY = importlib.util.module_from_spec(_STABILITY_SPEC)
_STABILITY_SPEC.loader.exec_module(_STABILITY)
_make_proxy_client = _STABILITY._make_proxy_client


class _Store:
    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self.entries = entries or {}

    def retrieve(self, hash_key: str):  # noqa: ANN201
        original = self.entries.get(hash_key)
        return None if original is None else SimpleNamespace(original_content=original)


def _response(tool_input: dict) -> dict:
    return {
        "id": "msg_marker_guard",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool-1",
                "name": "Write",
                "input": tool_input,
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _post_response(monkeypatch, upstream_json: dict, store: _Store):  # noqa: ANN001, ANN202
    with _make_proxy_client() as test_client:
        proxy = test_client.app.state.proxy
        monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda: store)

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(200, json=upstream_json)

        proxy._retry_request = _fake_retry
        return test_client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "write it"}],
            },
        )


def test_valid_hash_expansion_client_receives_original(monkeypatch) -> None:
    hash_key = "0123456789abcdef01234567"
    original = "the complete file contents"
    marker = f"[Tool input masked: content. Retrieve original: hash={hash_key}]"

    response = _post_response(
        monkeypatch,
        _response({"file_path": "/tmp/x", "content": marker}),
        _Store({hash_key: original}),
    )

    assert response.status_code == 200
    assert response.json()["content"][0]["input"]["content"] == original


def test_fabricated_hash_block_becomes_text_and_logs(monkeypatch) -> None:
    hash_key = "deadbeefdeadbeefdeadbeef"
    marker = f"[Tool input masked: content. Retrieve original: hash={hash_key}]"

    # The headroom.proxy logger does not propagate to root in the app
    # config, so capture must attach a handler to the logger itself.
    guard_records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            guard_records.append(record.getMessage())

    handler = _Capture()
    logging.getLogger("headroom.proxy").addHandler(handler)
    try:
        response = _post_response(monkeypatch, _response({"content": marker}), _Store())
    finally:
        logging.getLogger("headroom.proxy").removeHandler(handler)

    payload = response.json()
    assert payload["stop_reason"] == "end_turn"
    assert payload["content"][0]["type"] == "text"
    assert "tool_use" not in json.dumps(payload["content"])
    assert any("MARKER_GUARD: blocked" in m for m in guard_records)


def test_non_marker_inputs_untouched_byte_identically(monkeypatch) -> None:
    payload = _response({"content": "ordinary content", "nested": {"count": 2}})
    before = json.dumps(payload, separators=(",", ":")).encode()
    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda: _Store())

    assert not _guard_anthropic_tool_use_markers(payload, request_id="test")
    assert json.dumps(payload, separators=(",", ":")).encode() == before


def test_marker_in_nested_input_value_is_guarded(monkeypatch) -> None:
    hash_key = "abcdef0123456789abcdef01"
    marker = f"[Tool result masked: output. Retrieve original: hash={hash_key}]"
    payload = _response({"options": {"payload": ["safe", marker]}})
    monkeypatch.setattr(
        "headroom.cache.compression_store.get_compression_store",
        lambda: _Store({hash_key: "expanded nested value"}),
    )

    assert _guard_anthropic_tool_use_markers(payload, request_id="test")
    assert payload["content"][0]["input"]["options"]["payload"][1] == "expanded nested value"


def test_incident_2026_07_17_write_mimicry(monkeypatch) -> None:
    invented_hash = "999999999999999999999999"
    fabricated_file_contents = (
        f"[Tool input masked: content (12000 chars). Retrieve original: hash={invented_hash}]"
    )

    response = _post_response(
        monkeypatch,
        _response(
            {
                "file_path": "/tmp/incident-replay.md",
                "content": fabricated_file_contents,
            }
        ),
        _Store(),
    )

    payload = response.json()
    assert payload["stop_reason"] == "end_turn"
    assert all(block.get("type") != "tool_use" for block in payload["content"])
    assert invented_hash in payload["content"][0]["text"]
