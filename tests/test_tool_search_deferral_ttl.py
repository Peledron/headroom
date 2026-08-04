"""``inject_tool_search_deferral`` must not downgrade a 1h cache breakpoint to 5m.

Anthropic processes cache_control blocks in ``tools`` -> ``system`` -> ``messages``
order and rejects any request where a ``ttl='1h'`` block follows a ``ttl='5m'``
one. Deferral strips cache_control off every deferred tool and re-places a single
marker on the last resident tool, so keeping the wrong marker rewrites the tools
prefix from 1h down to 5m while the client's message breakpoints stay 1h:

    400 invalid_request_error
    messages.N.content.M.cache_control.ttl: a ttl='1h' cache_control block must
    not come after a ttl='5m' cache_control block.
"""

from __future__ import annotations

from headroom.proxy.helpers import inject_tool_search_deferral

_CORE = frozenset({"Bash", "Read", "Edit"})


def _tools_with_1h_before_bare() -> list[dict]:
    """13 tools: 3 resident core, then a 1h marker, then a bare (5m) marker after it."""
    tools: list[dict] = [{"name": name, "input_schema": {}} for name in ("Bash", "Read", "Edit")]
    tools.append(
        {
            "name": "mcp__alpha__one",
            "input_schema": {},
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    )
    tools.extend({"name": f"mcp__beta__{i}", "input_schema": {}} for i in range(8))
    tools.append(
        {
            "name": "mcp__gamma__last",
            "input_schema": {},
            "cache_control": {"type": "ephemeral"},
        }
    )
    return tools


def test_deferral_keeps_the_longest_ttl_it_stripped():
    out = inject_tool_search_deferral(_tools_with_1h_before_bare(), core_tools=_CORE)

    markers = [
        tool["cache_control"]
        for tool in out
        if isinstance(tool, dict) and tool.get("cache_control")
    ]
    assert markers, "deferral dropped every cache breakpoint"
    assert any(marker.get("ttl") == "1h" for marker in markers), (
        "the 1h breakpoint was downgraded to 5m by a later bare marker, which makes "
        f"the client's 1h message breakpoints illegal upstream. markers={markers}"
    )
