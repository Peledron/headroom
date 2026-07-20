"""Streaming marker-guard transformer regressions."""

from __future__ import annotations

import json

from headroom.proxy.sse_marker_guard import SseToolUseMarkerGuard


def _event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _tool_use_stream(input_json: str, *, index: int = 1) -> list[bytes]:
    events = [
        _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": "tool-1", "name": "Write", "input": {}},
            },
        )
    ]
    for i in range(0, len(input_json), 7):
        events.append(
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": input_json[i : i + 7]},
                },
            )
        )
    events.append(
        _event("content_block_stop", {"type": "content_block_stop", "index": index})
    )
    return events


def _run(guard: SseToolUseMarkerGuard, events: list[bytes], chunk_size: int = 11) -> bytes:
    raw = b"".join(events)
    out = bytearray()
    for i in range(0, len(raw), chunk_size):
        out.extend(guard.feed(raw[i : i + chunk_size]))
    out.extend(guard.flush())
    return bytes(out)


def _blocks(out: bytes) -> list[dict]:
    parsed = []
    for frame in out.split(b"\n\n"):
        for line in frame.split(b"\n"):
            if line.startswith(b"data:"):
                parsed.append(json.loads(line[5:]))
    return parsed


def test_plain_tool_use_passes_byte_identically() -> None:
    events = _tool_use_stream(json.dumps({"file_path": "/tmp/x", "content": "hello world"}))
    guard = SseToolUseMarkerGuard(lambda h: None, request_id="t")
    assert _run(guard, events) == b"".join(events)


def test_text_deltas_stream_through_without_holding() -> None:
    text_event = _event(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    )
    guard = SseToolUseMarkerGuard(lambda h: None, request_id="t")
    assert guard.feed(text_event) == text_event


def test_valid_marker_is_expanded() -> None:
    hash_key = "0123456789abcdef01234567"
    marker = f"[Tool input masked: content. Retrieve original: hash={hash_key}]"
    events = _tool_use_stream(json.dumps({"file_path": "/tmp/x", "content": marker}))
    guard = SseToolUseMarkerGuard(
        lambda h: "restored body" if h == hash_key else None, request_id="t"
    )
    out = _blocks(_run(guard, events))
    deltas = [d for d in out if d.get("type") == "content_block_delta"]
    assert len(deltas) == 1
    restored = json.loads(deltas[0]["delta"]["partial_json"])
    assert restored == {"file_path": "/tmp/x", "content": "restored body"}


def test_fabricated_marker_becomes_text_block_and_flips_stop_reason() -> None:
    marker = "[Tool input masked: content (12000 chars). Retrieve original: hash=999999999999999999999999]"
    events = _tool_use_stream(json.dumps({"file_path": "/tmp/incident.md", "content": marker}))
    events.append(
        _event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
        )
    )
    guard = SseToolUseMarkerGuard(lambda h: None, request_id="t")
    out = _blocks(_run(guard, events))
    starts = [d for d in out if d.get("type") == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["text"]
    text = "".join(
        d["delta"]["text"]
        for d in out
        if d.get("type") == "content_block_delta" and d["delta"].get("type") == "text_delta"
    )
    assert "999999999999999999999999" in text
    (message_delta,) = [d for d in out if d.get("type") == "message_delta"]
    assert message_delta["delta"]["stop_reason"] == "end_turn"


def test_unparseable_input_with_marker_prefix_is_blocked() -> None:
    events = _tool_use_stream('{"content": "[Tool input masked: x. Retrieve original: hash=aaaaaaaaaaaaaaaaaaaaaaaa]"')
    guard = SseToolUseMarkerGuard(lambda h: None, request_id="t")
    out = _blocks(_run(guard, events))
    starts = [d for d in out if d.get("type") == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["text"]


def test_truncated_stream_releases_held_events_verbatim() -> None:
    events = _tool_use_stream(json.dumps({"content": "x"}))[:-1]
    guard = SseToolUseMarkerGuard(lambda h: None, request_id="t")
    assert _run(guard, events) == b"".join(events)


def test_multiple_blocks_only_tool_use_held() -> None:
    text_events = [
        _event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        ),
        _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ]
    tool_events = _tool_use_stream(json.dumps({"content": "plain"}), index=1)
    guard = SseToolUseMarkerGuard(lambda h: None, request_id="t")
    out = _run(guard, text_events + tool_events)
    assert out == b"".join(text_events + tool_events)
