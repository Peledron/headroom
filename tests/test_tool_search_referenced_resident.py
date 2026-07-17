"""Referenced tools must never be deferred (Anthropic 400 on compaction)."""

from __future__ import annotations

from headroom.proxy.helpers import (
    inject_tool_search_deferral,
    referenced_tool_names,
)


def _tools(n: int = 12) -> list[dict]:
    return [{"name": f"Tool{i}", "input_schema": {}} for i in range(n)]


def test_referenced_tool_names_extraction() -> None:
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "TaskCreate", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]},
        {"role": "user", "content": "plain"},
    ]
    assert referenced_tool_names(messages) == frozenset({"taskcreate"})
    assert referenced_tool_names(None) == frozenset()


def test_referenced_tool_stays_resident() -> None:
    tools = _tools()
    referenced = frozenset({"tool3"})
    out = inject_tool_search_deferral(tools, referenced=referenced)
    by_name = {t.get("name"): t for t in out if isinstance(t, dict) and t.get("name")}
    assert not by_name["Tool3"].get("defer_loading")
    assert by_name["Tool5"].get("defer_loading")


def test_without_referenced_still_defers() -> None:
    out = inject_tool_search_deferral(_tools())
    deferred = [t for t in out if isinstance(t, dict) and t.get("defer_loading")]
    assert deferred
