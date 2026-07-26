"""Native (Rust) content-detector failures must degrade to the pure-Python
detector instead of propagating out as an HTTP 500. Regression test for #1123."""

from __future__ import annotations

import asyncio

import pytest

from headroom.transforms import content_router as cr

# Patch the native detector via its string target ("headroom._core.detect_content_type")
# rather than a module alias captured at import time. content_router._detect_content does a
# fresh `from headroom._core import detect_content_type` on every call, and other tests pop
# headroom._core out of sys.modules (e.g. test_rust_core_smoke), which rebuilds the module
# object. A captured alias would then go stale and the patch would miss the live module —
# the control-flow tests would silently run the real detector and never see the exception.


@pytest.fixture(autouse=True)
def _reset_detect_module_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the native-detector circuit breaker (#575) before each test.

    The breaker is process-wide and one-way by design: the first native
    ``detect_content_type`` in this environment parks in a futex and trips the
    watchdog, after which every later call routes straight to regex. That is
    correct in production and fatal here, because these tests need the native
    branch reachable to observe what it raises. Without the reset they pass
    alone and fail in-suite, which is how they were found.
    """
    monkeypatch.setattr(cr, "_detect_native_unhealthy", False)
    monkeypatch.setattr(cr, "_detect_native_wedged", False)
    monkeypatch.setattr(cr, "_detect_backend_warned", False)
    monkeypatch.setattr(cr, "_detect_panic_warned", False)


def test_falls_back_on_rust_exception(monkeypatch):
    """An ordinary exception from the native detector degrades to regex."""

    def _boom(_content):
        raise RuntimeError("simulated native failure")

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr("headroom._core.detect_content_type", _boom)
    monkeypatch.setattr(cr, "_detect_panic_warned", False, raising=False)

    # Must not raise; returns a usable detection result from the regex path.
    result = cr._detect_content('{"a": 1, "b": [1, 2, 3]}')
    assert result is not None
    assert result.content_type is not None


def test_falls_back_on_baseexception_panic(monkeypatch):
    """A BaseException-derived panic (like pyo3's PanicException) is caught too."""

    class FakePanic(BaseException):
        pass

    def _panic(_content):
        raise FakePanic("simulated pyo3 panic")

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr("headroom._core.detect_content_type", _panic)
    monkeypatch.setattr(cr, "_detect_panic_warned", False, raising=False)

    result = cr._detect_content("some plain text content here")
    assert result is not None


def test_control_flow_exceptions_propagate(monkeypatch):
    """KeyboardInterrupt/SystemExit must not be swallowed by the fallback."""

    def _interrupt(_content):
        raise KeyboardInterrupt

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr("headroom._core.detect_content_type", _interrupt)
    monkeypatch.setattr(cr, "_detect_panic_warned", False, raising=False)

    with pytest.raises(KeyboardInterrupt):
        cr._detect_content("content")


def test_cancelled_error_propagates(monkeypatch):
    """asyncio.CancelledError must propagate, not be swallowed as a fallback."""

    def _cancel(_content):
        raise asyncio.CancelledError()

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr("headroom._core.detect_content_type", _cancel)
    monkeypatch.setattr(cr, "_detect_panic_warned", False, raising=False)

    with pytest.raises(asyncio.CancelledError):
        cr._detect_content("content")
