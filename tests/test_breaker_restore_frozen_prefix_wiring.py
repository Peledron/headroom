"""Breaker: is ``_restore_frozen_prefix`` (Anthropic) ever actually called?

``AnthropicHandlerMixin._restore_frozen_prefix`` (anthropic.py, defined next to
``_strict_previous_turn_frozen_count``) docstrings itself as: "Force frozen
prefix bytes to match the original request exactly." That is a strong,
unconditional, per-message restore guarantee, stronger than the append-only
"replay-or-bail" contract of ``overlay_cached_prefix``.

This file establishes the steady state a caller would assume ("the function
that claims to restore frozen bytes runs on the live request path") and then
falsifies it by AST-scanning the actual source for call sites, contrasting
Anthropic against the OpenAI handler which DOES wire the equivalent method in.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.openai import OpenAIHandlerMixin


def _call_sites_of(method_name: str, source: str) -> list[ast.Call]:
    """Return every ast.Call node anywhere in `source` invoking `method_name`,
    excluding the call implied by the `def` line itself."""
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name == method_name:
                hits.append(node)
    return hits


def test_anthropic_restore_frozen_prefix_has_zero_call_sites_in_module():
    """REAL DEFECT: `_restore_frozen_prefix` is defined in anthropic.py but is
    never invoked anywhere else in that module. It is unreachable dead code on
    the live Anthropic request path — in BOTH proxy modes, not just token
    mode. Grep-level ground truth (not a change-detector on formatting): an
    `ast.Call` node whose function name is `_restore_frozen_prefix` must exist
    somewhere in the module besides the `def` statement itself for the method
    to ever run. There is none.
    """
    src_path = Path(inspect.getfile(AnthropicHandlerMixin))
    source = src_path.read_text()
    calls = _call_sites_of("_restore_frozen_prefix", source)
    assert calls == [], (
        f"expected zero call sites of _restore_frozen_prefix in {src_path}, "
        f"found {len(calls)} at lines {[c.lineno for c in calls]}. If this now "
        f"fails because someone wired it in, this test should be replaced by a "
        f"behavioral test of the call site, not deleted."
    )
    # Sanity: the method still exists (else the claim itself would be moot in
    # a different, more obvious way).
    assert hasattr(AnthropicHandlerMixin, "_restore_frozen_prefix")


def test_openai_restore_frozen_prefix_DOES_have_a_call_site():
    """Contrast case / control: the OpenAI handler wires the equivalent method
    in at its PRE_SEND-adjacent cache-safety step. This proves the AST method
    above can find real call sites when they exist, and shows the Anthropic
    gap is a wiring omission, not a fact about all providers or a limitation
    of the detector.
    """
    src_path = Path(inspect.getfile(OpenAIHandlerMixin))
    source = src_path.read_text()
    calls = _call_sites_of("_restore_frozen_prefix", source)
    assert len(calls) >= 1, (
        "expected OpenAI handler to call _restore_frozen_prefix at least once; "
        "if this now fails, OpenAI's wiring regressed too and the contrast "
        "claim in this test needs updating, not silencing."
    )


def test_restore_frozen_prefix_only_referenced_by_benchmarks_and_openai_tests():
    """Cross-file confirmation: the ONLY places that call
    `AnthropicHandlerMixin._restore_frozen_prefix` at all in this repository
    are two benchmark scripts invoking the classmethod directly for
    simulation purposes, never the request handler. `tests/test_proxy_handler_helpers.py`
    exercises `OpenAIHandlerMixin._restore_frozen_prefix`, not Anthropic's.
    This corroborates the AST finding with an independent, coarser method
    (plain substring search across the repo) so the two do not share a blind
    spot.
    """
    repo_root = Path(inspect.getfile(AnthropicHandlerMixin)).parents[3]
    hits = []
    for py_file in repo_root.rglob("*.py"):
        # Skip venvs / build artifacts that might be sitting under the tree.
        parts = py_file.parts
        if any(p in (".venv", "target", "node_modules") for p in parts):
            continue
        try:
            text = py_file.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if "_restore_frozen_prefix" in text:
            hits.append(py_file)
    # Every hit must be one of: the two handler definitions, a benchmark, or a test.
    unexpected = [
        f
        for f in hits
        if f.name not in ("anthropic.py", "openai.py")
        and "benchmark" not in str(f)
        and "test" not in f.name
    ]
    assert unexpected == [], (
        f"_restore_frozen_prefix referenced from unexpected non-test/benchmark "
        f"files: {unexpected}. Expected only handler defs, benchmarks, and tests."
    )
    assert any("benchmark" in str(f) for f in hits), (
        "expected at least one benchmark script referencing _restore_frozen_prefix "
        "directly (simulation harness) — if this list changed, the finding's "
        "framing needs updating."
    )
