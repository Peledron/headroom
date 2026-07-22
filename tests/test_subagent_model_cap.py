"""Wire-level premium-model cap for detected sub-agent requests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx

_STABILITY_SPEC = importlib.util.spec_from_file_location(
    "_model_cap_stability_helpers",
    Path(__file__).with_name("test_proxy_anthropic_cache_stability.py"),
)
assert _STABILITY_SPEC is not None and _STABILITY_SPEC.loader is not None
_STABILITY = importlib.util.module_from_spec(_STABILITY_SPEC)
_STABILITY_SPEC.loader.exec_module(_STABILITY)
_make_proxy_client = _STABILITY._make_proxy_client

_UPSTREAM_OK = {
    "id": "msg_cap",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 2},
}


def _post(model: str, system: object, monkeypatch, env: str | None = None):  # noqa: ANN001, ANN202
    monkeypatch.setenv("HR_SUBAGENT_MODEL_CAP_WIRE_FALLBACK", "1")
    if env is not None:
        monkeypatch.setenv("HR_SUBAGENT_MODEL_CAP", env)
    seen: list[dict] = []
    with _make_proxy_client() as test_client:
        proxy = test_client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            seen.append(body if isinstance(body, dict) else json.loads(body))
            return httpx.Response(200, json=_UPSTREAM_OK)

        proxy._retry_request = _fake_retry
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": 32,
                "system": system,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert len(seen) == 1
    return seen[0]


def _subagent_system(model: str) -> str:
    return f"You are Claude Code, an agentic CLI tool. The exact model ID is {model}."


def test_fable_subagent_is_rewritten_to_default_cap(monkeypatch) -> None:
    body = _post("claude-fable-5", _subagent_system("claude-fable-5"), monkeypatch)
    assert body["model"] == "claude-sonnet-5"


def test_opus_subagent_is_rewritten(monkeypatch) -> None:
    body = _post("claude-opus-4-8", _subagent_system("claude-opus-4-8"), monkeypatch)
    assert body["model"] == "claude-sonnet-5"


def test_main_1m_session_is_never_capped(monkeypatch) -> None:
    model = "claude-fable-5"
    system = f"You are Claude Code. The exact model ID is {model}[1m]."
    body = _post(model, system, monkeypatch)
    assert body["model"] == model


def test_cheap_subagent_passes_through(monkeypatch) -> None:
    body = _post("claude-sonnet-5", _subagent_system("claude-sonnet-5"), monkeypatch)
    assert body["model"] == "claude-sonnet-5"


def test_cap_disabled_by_env_zero(monkeypatch) -> None:
    body = _post("claude-fable-5", _subagent_system("claude-fable-5"), monkeypatch, env="0")
    assert body["model"] == "claude-fable-5"


def test_non_claude_code_traffic_untouched(monkeypatch) -> None:
    body = _post("claude-fable-5", "You are a helpful poetry assistant.", monkeypatch)
    assert body["model"] == "claude-fable-5"


def test_main_thread_shaped_request_without_marker_is_not_capped(monkeypatch) -> None:
    # Marker-absence alone (a 429 retry that drops "[1m]", a system prompt that
    # grew past the boundary match) is not a positive subagent signal. Without
    # the exact bare model id ALSO appearing as a standalone token, the request
    # must pass through unmodified rather than being treated as "probably a
    # subagent" by default.
    model = "claude-fable-5"
    system = (
        "You are Claude Code, Anthropic's official CLI for Claude, operating in "
        "the user's primary session. Follow the user's instructions carefully."
    )
    body = _post(model, system, monkeypatch)
    assert body["model"] == model


def test_subagent_task_dispatch_shape_is_still_capped(monkeypatch) -> None:
    # A structurally different subagent shape (Task-tool dispatch framing
    # rather than the top-level CLI boilerplate) still carries the one signal
    # that matters: the bare model id, no "[1m]" marker.
    model = "claude-opus-4-8"
    system = (
        f"You are a subagent dispatched by the Task tool to complete one "
        f"delegated objective. The exact model ID is {model}. Report back a "
        "single result message when done."
    )
    body = _post(model, system, monkeypatch)
    assert body["model"] == "claude-sonnet-5"


def test_wire_fallback_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HR_SUBAGENT_MODEL_CAP_WIRE_FALLBACK", raising=False)
    monkeypatch.setenv("HR_SUBAGENT_MODEL_CAP", "claude-sonnet-5")
    seen: list[dict] = []
    with _make_proxy_client() as test_client:
        proxy = test_client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            seen.append(body if isinstance(body, dict) else json.loads(body))
            return httpx.Response(200, json=_UPSTREAM_OK)

        proxy._retry_request = _fake_retry
        response = test_client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-fable-5",
                "max_tokens": 32,
                "system": _subagent_system("claude-fable-5"),
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert seen[0]["model"] == "claude-fable-5"
