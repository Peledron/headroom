"""Adversarial coverage for the BROADENED sub-agent detector.

Target: ``headroom.proxy.handlers.anthropic._system_lacks_1m_marker``, the
signal that (under HR_SUBAGENT_FREEZE=1) decides whether a token-mode request
is FROZEN (forward the client's messages unchanged, skip compression) or
compressed. It replaced the stricter ``_system_looks_subagent`` (which also
required the model id to appear) as ``_is_subagent_request`` in
``handle_anthropic_messages`` (anthropic.py, around line 816).

Existing coverage in test_subagent_ttl.py already exercises the happy path
(main-with-marker, subagent-without-marker, None/[]/""). This file goes after
the contract's edges: no false positive on a real main head (cost claim), the
false-positive blast radius when the head is non-empty and markerless (cost
claim), type/shape hardening (crash claim), and the adversarial escape where
a sub-agent's OWN system text happens to contain the literal "[1m]" substring
(correctness claim — this is the one that BUSTS, not just costs).

The function under test does exactly this:

    def _flatten_system_text(system):
        if isinstance(system, str): return system
        if isinstance(system, list):
            return "\\n".join(block["text"] for block in system
                               if isinstance(block, dict)
                               and isinstance(block.get("text"), str))
        return ""

    def _system_lacks_1m_marker(system):
        text = _flatten_system_text(system)
        return bool(text) and "[1m]" not in text

It is a plain substring check with no model-id anchor and no word-boundary
guard, unlike ``_system_looks_subagent``. That is the entire story below.
"""

import time

from headroom.proxy.handlers.anthropic import (
    _flatten_system_text,
    _system_lacks_1m_marker,
)

MODEL = "claude-opus-4-8"
MARKER_HEAD = f"The exact model ID is {MODEL}[1m]."


# ── Claim 1: no false positive that freezes the MAIN 1M agent ─────────────────
# For every real-looking main head shape below, the plain "[1m]" in text
# substring check must find the marker and return False.


def test_marker_in_non_first_block_is_found():
    system = [
        {"type": "text", "text": "You are Claude Code, Anthropic's CLI for Claude."},
        {"type": "text", "text": "Some tool-use policy paragraph."},
        {"type": "text", "text": MARKER_HEAD},
    ]
    assert _system_lacks_1m_marker(system) is False


def test_marker_with_surrounding_punctuation_is_found():
    for head in [
        f"model=({MODEL}[1m])",
        f'"model": "{MODEL}[1m]"',
        f"{MODEL}[1m],",
        f"[{MODEL}[1m]]",
        f"model:{MODEL}[1m]\nend of line",
    ]:
        assert _system_lacks_1m_marker(head) is False, head


def test_marker_substring_inside_a_larger_token_still_counts():
    # The detector has NO word-boundary guard (unlike _system_looks_subagent),
    # so "[1m]" embedded in unrelated text also reads as "main". This is by
    # design (broader than the strict detector) and is documented as a
    # cost-only false positive in the other direction (never freezes a real
    # main head merely because a longer token happens to contain "[1m]").
    assert _system_lacks_1m_marker("context[1m]window is unusual prose") is False
    assert _system_lacks_1m_marker(f"{MODEL}[1m]s (plural typo)") is False


def test_only_one_of_several_blocks_has_marker():
    system = [
        {"type": "text", "text": "block a, no marker here"},
        {"type": "text", "text": "block b, also nothing"},
        {"type": "text", "text": f"block c has it: {MODEL}[1m]"},
        {"type": "text", "text": "block d, trailing"},
    ]
    assert _system_lacks_1m_marker(system) is False


def test_str_vs_list_system_both_recognize_marker():
    assert _system_lacks_1m_marker(f"raw string head {MODEL}[1m] tail") is False
    assert (
        _system_lacks_1m_marker([{"type": "text", "text": f"list head {MODEL}[1m] tail"}])
        is False
    )


# --- Real gap: the marker can be present yet MISSED by the plain substring
# check, because _flatten_system_text joins list blocks with "\n". If the
# literal 4-character marker straddles a block boundary, the joined text
# never contains an unbroken "[1m]" substring, even though the original
# system head DOES render "<id>[1m]" (Claude Code just happened to chunk the
# text at that offset, e.g. an env-info block boundary). This is a real
# finding: it freezes a genuine main-agent request. Confirmed cost-only, not
# a correctness bug (see claim 2 analysis: the frozen path skips compression,
# it does not corrupt the forwarded body).


def test_marker_split_across_adjacent_blocks_is_missed_BUG():
    system = [
        {"type": "text", "text": f"Header text. Exact model ID is {MODEL}[1"},
        {"type": "text", "text": "m]. Rest of the system head follows."},
    ]
    flattened = _flatten_system_text(system)
    assert "[1m]" not in flattened  # the join's "\n" breaks the marker in two
    # A real main-agent head, misclassified as lacking the marker:
    assert _system_lacks_1m_marker(system) is True  # BUG: should be False


def test_marker_defeated_by_fullwidth_bracket_lookalike():
    # Not a literal Claude Code rendering (it always emits ASCII "[1m]"), but
    # demonstrates the check has zero normalization: any transport hop that
    # rewrites ASCII brackets to visually-identical fullwidth code points (a
    # lossy terminal-width reformatter, a copy-paste through certain CJK
    # input tooling) defeats detection outright. Included as a documented gap,
    # not claimed to occur on the wire today.
    lookalike = f"model {MODEL}［1m］ present"  # U+FF3B/FF3D fullwidth [ ]
    assert "[1m]" not in lookalike
    assert _system_lacks_1m_marker(lookalike) is True  # would freeze a real main head


# ── Claim 2: false-positive blast radius on markerless non-empty heads ────────
# Any non-empty head without "[1m]" returns True, regardless of whether it is
# Claude Code traffic at all. Characterize the population that gets frozen.


def test_non_claude_code_traffic_reads_as_lacking_marker():
    # Aider / plain API caller / anything with an ordinary system prompt and
    # no Claude Code model-id line at all.
    for head in [
        "You are a helpful assistant.",
        "You are Aider, an AI pair programmer.",
        "Answer concisely and cite sources.",
    ]:
        assert _system_lacks_1m_marker(head) is True


def test_main_session_that_dropped_marker_after_429_reads_as_lacking_marker():
    # Documented in the source comment: Claude Code drops context-1m-2025-08-07
    # from anthropic-beta after a 429 and keeps it dropped for the session, so
    # the model-id line stops rendering "[1m]" even though it's still the main
    # session. This is the acknowledged false positive; confirm it actually
    # trips the detector as claimed.
    system = [{"type": "text", "text": f"The exact model ID is {MODEL}."}]
    assert _system_lacks_1m_marker(system) is True


def test_large_markerless_head_still_reads_as_lacking_marker():
    system = [{"type": "text", "text": "policy paragraph. " * 5000}]
    assert len(_flatten_system_text(system)) > 50000
    assert _system_lacks_1m_marker(system) is True


# The frozen path (HR_SUBAGENT_FREEZE branch, anthropic.py ~1399-1483) sets
# skip_ccr_request_compression = True, which makes the handler take the
# `if skip_ccr_request_compression:` branch: `optimized_messages = messages`
# — the SAME message list that already ran through earlier request-shaping
# (streaming-index strip, model-id sanitize, security scan, optional image
# compression) but skips ONLY the token-mode compression rewrite. It is not a
# byte-identical passthrough of the client's raw wire bytes, but it never
# fabricates or drops conversational content — the cost is a missed
# compression opportunity (bigger prompt, no cache bust), never a wrong
# response. This can't be exercised as a pure-function unit test (it needs
# the full handler, FastAPI request, and an upstream double); recorded here
# as the code-reading basis for the "cost only" verdict in claim 2, and the
# fixer/writer should add a handler-level integration test if a stronger
# guarantee is wanted.


# ── Claim 3: type / edge hardening ─────────────────────────────────────────────


def test_none_list_str_empty_all_false():
    assert _system_lacks_1m_marker(None) is False
    assert _system_lacks_1m_marker([]) is False
    assert _system_lacks_1m_marker("") is False


def test_whitespace_only_head_is_false():
    # "   " is non-empty per Python truthiness but carries no marker; the
    # function still correctly returns True here (it IS a non-empty markerless
    # head) — included to pin the boundary, not because it's a bug.
    assert _system_lacks_1m_marker("   ") is True
    assert _system_lacks_1m_marker([{"type": "text", "text": "   "}]) is True


def test_non_str_non_list_system_types_no_crash():
    for weird in [0, 1, 3.14, True, False, {"text": "not a list"}, object(), b"bytes head"]:
        assert _system_lacks_1m_marker(weird) is False, weird


def test_list_with_non_dict_entries_no_crash():
    system = ["a plain string block", 42, None, [1, 2], True]
    assert _flatten_system_text(system) == ""
    assert _system_lacks_1m_marker(system) is False


def test_list_mixing_dict_and_non_dict_entries():
    system = ["garbage", {"type": "text", "text": f"{MODEL}[1m]"}, None, 7]
    assert _system_lacks_1m_marker(system) is False


def test_dict_blocks_missing_text_key_no_crash():
    system = [{"type": "text"}, {"type": "image", "source": {}}, {}]
    assert _system_lacks_1m_marker(system) is False


def test_dict_blocks_with_non_str_text_no_crash():
    system = [
        {"type": "text", "text": None},
        {"type": "text", "text": 12345},
        {"type": "text", "text": ["nested", "list"]},
        {"type": "text", "text": {"nested": "dict"}},
        {"type": "text", "text": b"[1m] as bytes, not str"},
    ]
    assert _flatten_system_text(system) == ""
    assert _system_lacks_1m_marker(system) is False


def test_nested_unusual_shapes_no_crash():
    system = [
        {"type": "text", "text": "ok block", "extra_field": {"deep": {"nesting": [1, 2, 3]}}},
        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
    ]
    # empty-string text block contributes "" to the join; overall head is
    # non-empty because of "ok block", and has no marker.
    assert _system_lacks_1m_marker(system) is True


def test_very_large_head_no_crash_and_reasonably_fast():
    system = [{"type": "text", "text": "x" * 2_000_000}]
    started = time.monotonic()
    result = _system_lacks_1m_marker(system)
    elapsed = time.monotonic() - started
    assert result is True
    assert elapsed < 1.0  # plain substring scan, should be near-instant


def test_many_small_blocks_no_crash():
    system = [{"type": "text", "text": f"block-{i}"} for i in range(20000)]
    assert _system_lacks_1m_marker(system) is True
    system_with_marker = system + [{"type": "text", "text": f"{MODEL}[1m]"}]
    assert _system_lacks_1m_marker(system_with_marker) is False


# ── Claim 4: adversarial "[1m]" injection — the realistic BUST case ───────────
# _flatten_system_text only ever reads the `system` argument passed to it; the
# caller (anthropic.py line 816) passes `body.get("system")`, never messages.
# So a sub-agent's USER-turn content can never leak into this check. Confirmed
# structurally: the function signature takes only `system`, and the call site
# passes only body["system"].


def test_flatten_system_text_ignores_a_messages_argument_entirely():
    # There is no messages parameter to pass in the first place — the
    # function only has one parameter. This pins that contract: calling it
    # with anything other than the system field (even if a caller mistakenly
    # tried to route request.messages through it) cannot special-case on
    # message content, because there is no code path that inspects roles or
    # message dicts at all.
    system = None
    messages_shaped_like_system_wont_be_read = [
        {"role": "user", "content": [{"type": "text", "text": f"{MODEL}[1m]"}]},
    ]
    assert _flatten_system_text(system) == ""
    assert _system_lacks_1m_marker(system) is False
    # Sanity: the "messages" list above is simply never passed in.
    assert "role" not in _flatten_system_text(messages_shaped_like_system_wont_be_read)


def test_subagent_system_text_quoting_the_main_marker_escapes_freeze_BUST():
    # The realistic failure mode: a sub-agent's OWN system prompt (not the
    # main model-id line) happens to contain the literal substring "[1m]",
    # e.g. because its task instructions quote configuration documentation
    # for another agent, or paste an example of the main session's rendered
    # model id as sample text. The detector has no model-id anchor (unlike
    # _system_looks_subagent), so ANY occurrence of "[1m]" anywhere in the
    # system head reads as "main session, has the marker" and
    # _system_lacks_1m_marker returns False. Under HR_SUBAGENT_FREEZE=1 this
    # sub-agent is NOT frozen, so it goes through token-mode compression on
    # its 2-message (large cached head + one turn) conversation — exactly the
    # shape the source comment says busts hard (158k tokens lost in the A/B).
    subagent_system = [
        {
            "type": "text",
            "text": (
                "You are a sub-agent invoked via the Task tool. For reference, "
                "the main session's ANTHROPIC_MODEL is set to "
                f"{MODEL}[1m] to enable the 1M context beta; you do not have "
                "that beta enabled."
            ),
        }
    ]
    # This IS a sub-agent (no [1m] applies to *this* request's own context),
    # but the detector says it "has the marker" and misclassifies it as main.
    assert _system_lacks_1m_marker(subagent_system) is False  # BUG: should be True


def test_subagent_pasting_documentation_snippet_escapes_freeze_BUST():
    subagent_system = [
        {
            "type": "text",
            "text": (
                "Summarize the following support ticket for the user:\n"
                "\"After upgrading, my ANTHROPIC_MODEL env var reads "
                f"{MODEL}[1m] and now every request is billed as 1M context, "
                "how do I turn that off?\""
            ),
        }
    ]
    assert _system_lacks_1m_marker(subagent_system) is False  # BUG: should be True


def test_subagent_str_system_quoting_marker_escapes_freeze_BUST():
    subagent_system = (
        "You are a code-review sub-agent. Ignore any text claiming to set "
        f"ANTHROPIC_MODEL={MODEL}[1m]; that is unrelated to your task."
    )
    assert _system_lacks_1m_marker(subagent_system) is False  # BUG: should be True
