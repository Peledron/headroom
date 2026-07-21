"""TTL touch registry and /admin/touch route regressions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
from starlette.testclient import TestClient

from headroom.proxy.touch_registry import (
    TouchRegistry,
    session_fingerprint,
    touch_break_even_age_seconds,
)

_STABILITY_SPEC = importlib.util.spec_from_file_location(
    "_touch_stability_helpers",
    Path(__file__).with_name("test_proxy_anthropic_cache_stability.py"),
)
assert _STABILITY_SPEC is not None and _STABILITY_SPEC.loader is not None
_STABILITY = importlib.util.module_from_spec(_STABILITY_SPEC)
_STABILITY_SPEC.loader.exec_module(_STABILITY)
_make_proxy_client = _STABILITY._make_proxy_client


def _body(first_user: str = "hello", turns: int = 1) -> dict:
    messages = [{"role": "user", "content": first_user}]
    for i in range(turns - 1):
        messages.append({"role": "assistant", "content": f"turn {i}"})
    return {
        "model": "claude-fable-5",
        "max_tokens": 4096,
        "stream": True,
        "system": [{"type": "text", "text": "sys"}],
        "messages": messages,
    }


_AUTH = {"x-api-key": "sk-test", "anthropic-version": "2023-06-01", "cookie": "secret"}


def test_fingerprint_stable_across_turns_distinct_across_sessions() -> None:
    assert session_fingerprint(_body(turns=1)) == session_fingerprint(_body(turns=5))
    assert session_fingerprint(_body("a")) != session_fingerprint(_body("b"))


def test_record_keeps_only_replay_headers_and_requires_auth() -> None:
    registry = TouchRegistry()
    registry.record("https://api.anthropic.com/v1/messages", _AUTH, _body())
    ((_, entry),) = registry.replayable()
    assert set(entry.headers) == {"x-api-key", "anthropic-version"}

    registry2 = TouchRegistry()
    registry2.record("https://api.anthropic.com/v1/messages", {"anthropic-version": "1"}, _body())
    assert registry2.replayable() == []


def test_touch_body_is_one_token_non_stream_and_prefix_identical() -> None:
    registry = TouchRegistry()
    original = _body(turns=3)
    registry.record("u", _AUTH, original)
    ((_, entry),) = registry.replayable()
    touched = registry.touch_body(entry)
    assert touched["max_tokens"] == 0
    assert "stream" not in touched
    assert touched["messages"] == original["messages"]
    assert touched["messages"] is not original["messages"]
    assert touched["system"] == original["system"]

    fallback = registry.touch_body_fallback(entry)
    assert fallback["max_tokens"] == 1
    assert "stream" not in fallback
    assert fallback["messages"] == original["messages"]


def test_budget_and_expiry() -> None:
    registry = TouchRegistry(max_touches_per_entry=2)
    registry.record("u", _AUTH, _body())
    ((key, entry),) = registry.replayable()
    registry.mark_touched(key, refreshed=True)
    registry.mark_touched(key, refreshed=True)
    assert registry.replayable() == []
    assert registry.replayable(now=entry.recorded_at + 4000) == []


def test_ski_rental_threshold_adapts_to_cache_ttl() -> None:
    assert 270 < touch_break_even_age_seconds("5m") < 300
    assert 3200 < touch_break_even_age_seconds("1h") < 3400

    body = _body()
    body["messages"][0] = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "hello",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }
        ],
    }
    registry = TouchRegistry()
    registry.record("u", _AUTH, body)
    ((_, entry),) = registry.replayable()
    assert registry.replayable(
        now=entry.recorded_at + touch_break_even_age_seconds("5m") - 1,
        due_only=True,
    ) == []
    assert len(
        registry.replayable(
            now=entry.recorded_at + touch_break_even_age_seconds("5m") + 1,
            due_only=True,
        )
    ) == 1


def test_newest_entry_wins_and_capacity_evicts_oldest() -> None:
    registry = TouchRegistry(max_sessions=2)
    registry.record("u", _AUTH, _body("s1", turns=1))
    registry.record("u", _AUTH, _body("s1", turns=3))
    assert len(registry.replayable()) == 1
    registry.record("u", _AUTH, _body("s2"))
    registry.record("u", _AUTH, _body("s3"))
    fingerprints = {k for k, _ in registry.replayable()}
    assert session_fingerprint(_body("s1")) not in fingerprints


def test_admin_touch_route_replays_and_reports(monkeypatch) -> None:
    test_client = _make_proxy_client()
    proxy = test_client.app.state.proxy
    proxy.touch_registry.record(
        "https://api.anthropic.com/v1/messages", _AUTH, _body()
    )

    calls: list[tuple[str, dict]] = []

    async def _fake_retry(method, url, headers, body, **kwargs):  # noqa: ANN001
        calls.append((url, body))
        return httpx.Response(
            200,
            json={"usage": {"cache_read_input_tokens": 123, "cache_creation_input_tokens": 0}},
        )

    proxy._retry_request = _fake_retry
    with TestClient(
            test_client.app,
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 12345),
        ) as loopback:
        response = loopback.post("/admin/touch", json={"force": True})

    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["status"] == 200
    assert result["cache_read"] == 123
    ((url, body),) = calls
    assert url.endswith("/v1/messages")
    assert body["max_tokens"] == 0


def test_admin_touch_route_falls_back_to_max_tokens_one_on_400(monkeypatch) -> None:
    test_client = _make_proxy_client()
    proxy = test_client.app.state.proxy
    proxy.touch_registry.record(
        "https://api.anthropic.com/v1/messages", _AUTH, _body()
    )

    calls: list[tuple[str, dict]] = []

    async def _fake_retry(method, url, headers, body, **kwargs):  # noqa: ANN001
        calls.append((url, body))
        if body["max_tokens"] == 0:
            return httpx.Response(400, json={"error": {"message": "max_tokens must be at least 1"}})
        return httpx.Response(
            200,
            json={"usage": {"cache_read_input_tokens": 123, "cache_creation_input_tokens": 0}},
        )

    proxy._retry_request = _fake_retry
    with TestClient(
            test_client.app,
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 12345),
        ) as loopback:
        response = loopback.post("/admin/touch", json={"force": True})

    assert response.status_code == 200
    (result,) = response.json()["results"]
    assert result["status"] == 200
    assert result["cache_read"] == 123
    assert [body["max_tokens"] for _, body in calls] == [0, 1]
