"""Adversarial coverage for the ``force_ttl`` keyword on
``normalize_message_cache_control`` (headroom/cache/prefix_tracker.py).

Attacks three claimed invariants:

  1. MESSAGE-ONLY  — force_ttl retimes ONLY the single message-body
     breakpoint, never a system/tools breakpoint (which lives outside
     ``messages`` and must stay cache-shared with the 1M head).
  2. BOUNDED       — after normalize there is exactly one (or zero, if no
     block-style message exists) message-level cache_control breakpoint.
     Anthropic hard-errors above 4 breakpoints total.
  3. BUST-SAFE     — a ttl-only change must be invisible to
     ``_canonicalize_for_prefix_compare`` / ``overlay_cached_prefix``'s bust
     detection, and content must round-trip byte-identical through
     ``_strip_cache_control``.

Offline only: no proxy, no network, pure unit/functional tests against
headroom.cache.prefix_tracker.
"""

from __future__ import annotations

import copy

import pytest

from headroom.cache.prefix_tracker import (
    PrefixCacheTracker,
    PrefixFreezeConfig,
    _canonicalize_for_prefix_compare,
    _strip_cache_control,
    normalize_message_cache_control,
    overlay_cached_prefix,
)


def B(role, text, ttl=None):
    """Anthropic block-style message with an explicit-ttl marker (or none)."""
    blk = {"type": "text", "text": text}
    if ttl is not None:
        blk["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    return {"role": role, "content": [blk]}


def BC(role, text, cc=True, ttl=None):
    """Block message that may carry a bare (no ttl) or ttl'd marker."""
    blk = {"type": "text", "text": text}
    if cc:
        cc_obj = {"type": "ephemeral"}
        if ttl is not None:
            cc_obj["ttl"] = ttl
        blk["cache_control"] = cc_obj
    return {"role": role, "content": [blk]}


def S(role, text):
    """String-content message (no block, cannot carry cache_control)."""
    return {"role": role, "content": text}


def _markers(messages):
    """All (message_index, block_index, cache_control) triples."""
    out = []
    for mi, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for bi, b in enumerate(content):
            if isinstance(b, dict) and isinstance(b.get("cache_control"), dict):
                out.append((mi, bi, b["cache_control"]))
    return out


def _marker_count(messages):
    return len(_markers(messages))


def _sole_ttl(messages):
    ms = _markers(messages)
    assert len(ms) == 1, f"expected exactly one breakpoint, got {len(ms)}: {ms}"
    return ms[0][2].get("ttl")


# ═══════════════════════════════════════════════════════════════════════════
# 1. MESSAGE-ONLY — force_ttl must never leak onto system/tools
# ═══════════════════════════════════════════════════════════════════════════


def test_force_ttl_signature_has_no_system_or_tools_parameter():
    """The function only accepts `messages`; there is structurally no channel
    for force_ttl to reach system/tools. Guard this against a future signature
    change that adds such a channel without a test noticing."""
    import inspect

    sig = inspect.signature(normalize_message_cache_control)
    params = set(sig.parameters)
    assert params == {"messages", "force_ttl"}, (
        f"normalize_message_cache_control gained new params {params - {'messages', 'force_ttl'}}; "
        "verify none of them can route a message-level ttl onto system/tools breakpoints"
    )


def test_force_ttl_does_not_touch_sibling_system_and_tools_objects():
    """Simulate a full request: system blocks + tool defs + messages, each
    carrying their OWN 1h breakpoint outside `messages`. Only `messages` is
    passed to normalize; system/tools objects must be byte-identical after."""
    system = [
        {"type": "text", "text": "SYS HEAD", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    tools = [
        {
            "name": "bash",
            "description": "d",
            "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    system_before = copy.deepcopy(system)
    tools_before = copy.deepcopy(tools)

    messages = [B("user", "a", ttl="1h"), B("user", "b", ttl="1h")]
    out = normalize_message_cache_control(messages, force_ttl="5m")

    assert system == system_before, "force_ttl on messages mutated the sibling system object"
    assert tools == tools_before, "force_ttl on messages mutated the sibling tools object"
    # And the message breakpoint really did get retimed (sanity: the isolation
    # above isn't vacuous because nothing changed at all).
    assert _sole_ttl(out) == "5m"


def test_role_system_smuggled_into_messages_list_still_gets_one_breakpoint():
    """Hostile input: a client or a buggy upstream transform puts a
    role="system" message inside the `messages` array (not the top-level
    `system` field). normalize_message_cache_control has no concept of role;
    it must still hold the same BOUNDED contract (exactly one message-level
    marker, placed on the true last block message), not silently exempt or
    duplicate a breakpoint for it."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "smuggled sys", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
        B("user", "a", ttl="1h"),
    ]
    out = normalize_message_cache_control(messages, force_ttl="5m")
    assert _marker_count(out) == 1
    # marker must be on the actual LAST message (index 1), not the smuggled one
    idxs = [mi for mi, _, _ in _markers(out)]
    assert idxs == [1], f"breakpoint landed on wrong message: {idxs}"


def test_tool_use_and_tool_result_blocks_only_get_forced_ttl_on_last():
    """Realistic Anthropic tool-loop shape: tool_use in assistant message,
    tool_result in the following user message. force_ttl must land on the
    single trailing block only, and tool payloads must be untouched."""
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}],
        },
    ]
    before = copy.deepcopy(messages)
    out = normalize_message_cache_control(messages, force_ttl="5m")
    assert _marker_count(out) == 1
    (mi, bi, cc) = _markers(out)[0]
    assert mi == 1 and bi == 0
    assert cc.get("ttl") == "5m"
    assert out[0]["content"][0]["input"] == {"cmd": "ls"}  # tool_use input untouched
    assert _strip_cache_control(out) == _strip_cache_control(before)


# ═══════════════════════════════════════════════════════════════════════════
# 2. BOUNDED — exactly one (or zero) message-level breakpoint, never >4/0-when-expected/2
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("force_ttl", ["1h", "5m", None, "", "bogus"])
def test_accumulated_markers_collapse_to_one_regardless_of_force_ttl(force_ttl):
    """5 pre-accumulated markers (the overlay pile-up scenario) must collapse
    to exactly 1 no matter what force_ttl is, never 0 or >1."""
    msgs = [
        BC("user", "a", ttl="1h"),
        BC("assistant", "b", ttl="1h"),
        BC("user", "c", ttl="1h"),
        BC("user", "d", ttl="1h"),
        BC("user", "e", ttl="1h"),
    ]
    out = normalize_message_cache_control(msgs, force_ttl=force_ttl)
    assert _marker_count(out) == 1, f"force_ttl={force_ttl!r} produced {_marker_count(out)} markers"


def test_string_only_conversation_yields_zero_breakpoints_even_with_force_ttl():
    """No block-style message exists anywhere: force_ttl has nothing to write
    onto. This must degrade to 0 breakpoints (safely within bound), not crash
    and not silently upgrade string content into a block to plant one."""
    msgs = [S("user", "a"), S("assistant", "b"), S("user", "c")]
    out = normalize_message_cache_control(msgs, force_ttl="5m")
    assert _marker_count(out) == 0
    assert out == msgs or out is msgs


def test_mixed_string_and_block_places_marker_on_last_BLOCK_message_only():
    """A trailing string-content message after a block message: the true
    conversation tail has no block to carry a marker, so the breakpoint stays
    on the last *block* message, not silently duplicated onto both."""
    msgs = [B("user", "a", ttl="1h"), S("assistant", "trailing string reply")]
    out = normalize_message_cache_control(msgs, force_ttl="1h")
    assert _marker_count(out) == 1
    (mi, _, cc) = _markers(out)[0]
    assert mi == 0
    assert cc.get("ttl") == "1h"


def test_empty_content_list_message_does_not_crash_or_get_marker():
    msgs = [B("user", "a", ttl="1h"), {"role": "user", "content": []}]
    out = normalize_message_cache_control(msgs, force_ttl="5m")
    assert _marker_count(out) == 1
    (mi, _, _) = _markers(out)[0]
    assert mi == 0  # the empty-content message can't carry it


def test_trailing_non_dict_block_does_not_get_marker_and_does_not_crash():
    """Hostile input: a content list whose last element is not a dict (e.g. a
    stray string leaked from a malformed client). Must not crash and must not
    silently coerce/mutate that element into a dict."""
    msgs = [
        B("user", "a", ttl="1h"),
        {"role": "user", "content": [{"type": "text", "text": "ok"}, "GARBAGE_NOT_A_DICT"]},
    ]
    out = normalize_message_cache_control(msgs, force_ttl="5m")
    assert _marker_count(out) == 1
    (mi, _, _) = _markers(out)[0]
    assert mi == 0
    assert out[1]["content"][-1] == "GARBAGE_NOT_A_DICT"


def test_empty_messages_list_is_a_noop():
    assert normalize_message_cache_control([], force_ttl="5m") == []


def test_never_exceeds_four_across_many_turns_with_force_ttl():
    conv = []
    for t in range(1, 20):
        conv = conv + [BC("user", f"turn-{t}", ttl="1h")]  # client marks newest
        conv = normalize_message_cache_control(conv, force_ttl="5m")
        assert _marker_count(conv) <= 4
    assert _marker_count(conv) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. BUST-SAFE — invisible to canonicalization / overlay bust detection
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("force_ttl", ["1h", "5m", None, "", "bogus", 3600, 0])
def test_canonicalize_ignores_force_ttl_choice(force_ttl):
    """No matter what force_ttl writes into the sole breakpoint's ttl field
    (even a nonsense/wrong-typed value), the cross-turn comparison key must
    be identical to the un-normalized original, because cache_control is
    stripped unconditionally at the key level, not the value level."""
    msgs = [B("user", "a", ttl="1h"), B("user", "b", ttl="1h")]
    out = normalize_message_cache_control(msgs, force_ttl=force_ttl)
    assert _canonicalize_for_prefix_compare(out) == _canonicalize_for_prefix_compare(msgs)
    assert _strip_cache_control(out) == _strip_cache_control(msgs)


def test_force_ttl_flip_between_turns_does_not_bust_canonicalize():
    """Turn-over-turn: force_ttl='5m' this turn, force_ttl='1h' next turn, on
    an append-only-extended conversation. The canonical compare (what
    extract_cache_stable_delta / overlay_cached_prefix key on) must see no
    difference from the ttl churn alone."""
    turn1_client = [B("user", "hello", ttl="1h")]
    turn1_out = normalize_message_cache_control(turn1_client, force_ttl="5m")
    assert _sole_ttl(turn1_out) == "5m"

    turn2_client = turn1_out + [B("user", "world", ttl="1h")]
    turn2_out = normalize_message_cache_control(turn2_client, force_ttl="1h")
    assert _sole_ttl(turn2_out) == "1h"

    # canonical prefix (first message) must still compare equal across the
    # ttl flip -- content is what the provider caches on, not ttl.
    assert _canonicalize_for_prefix_compare(turn2_out[:1]) == _canonicalize_for_prefix_compare(
        turn1_out[:1]
    )


def test_end_to_end_pipeline_force_ttl_never_busts_across_five_turns():
    """Full overlay_cached_prefix + normalize_message_cache_control pipeline
    (mirrors the handler's call order at anthropic.py:1632-1676), force_ttl
    pinned to '5m' every turn on a growing conversation with a compression
    step in between. Cache reads must never regress turn over turn."""

    def _compress(m):
        content = m["content"]
        if isinstance(content, list):
            blk = dict(content[0])
            blk["text"] = blk.get("text", "")[: max(1, len(blk.get("text", "")) // 2)]
            return {**m, "content": [blk]}
        return m

    def _freeze(original, frozen):
        return [
            (original[i] if i < frozen else _compress(original[i])) for i in range(len(original))
        ]

    def _toklen(m):
        content = m.get("content")
        if isinstance(content, list):
            return max(1, sum(len(b.get("text", "")) for b in content if isinstance(b, dict)))
        return max(1, len(str(content)))

    def _cache_read(fwd, prev_fwd):
        if not prev_fwd:
            return 0
        matched = 0
        for a, b in zip(fwd, prev_fwd):
            if _strip_cache_control([a]) == _strip_cache_control([b]):
                matched += _toklen(a)
            else:
                break
        return matched

    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))
    prev_fwd = None
    cur_client = []
    results = []
    for t in range(1, 6):
        cur_client = cur_client + [B("user", f"turn-{t}:" + "X" * 200, ttl="1h")]
        frozen = tracker.get_frozen_message_count()
        fwd = _freeze(cur_client, frozen)
        fwd = overlay_cached_prefix(
            fwd,
            cur_client,
            tracker.get_last_original_messages(),
            tracker.get_last_forwarded_messages(),
        )
        fwd = normalize_message_cache_control(fwd, force_ttl="5m")
        assert _marker_count(fwd) == 1
        assert _sole_ttl(fwd) == "5m"

        exp = sum(_toklen(m) for m in prev_fwd) if prev_fwd else 0
        act = _cache_read(fwd, prev_fwd)
        results.append((t, exp, act))

        counts = [_toklen(m) for m in fwd]
        tracker.update_from_response(
            act, sum(counts) - act, fwd, message_token_counts=counts, original_messages=cur_client
        )
        prev_fwd = fwd

    for t, exp, act in results[1:]:
        assert act >= exp, f"turn {t}: cache bust under force_ttl='5m' pipeline: expected {exp} read {act}"


def test_idempotent_across_turns_does_not_mutate_caller_input():
    """Repeated force_ttl='5m' calls must not mutate the previous turn's
    output list/dicts in place -- a caller (e.g. the tracker) holds onto
    `get_last_forwarded_messages()` copies and any in-place mutation would
    silently corrupt tracker-recorded state."""
    conv = [BC("user", "turn-1", ttl="1h")]
    out1 = normalize_message_cache_control(conv, force_ttl="5m")
    snapshot1 = copy.deepcopy(out1)

    conv2 = out1 + [BC("user", "turn-2", ttl="1h")]
    out2 = normalize_message_cache_control(conv2, force_ttl="5m")

    assert out1 == snapshot1, "normalize_message_cache_control mutated a previous call's output in place"
    assert _sole_ttl(out2) == "5m"


# ═══════════════════════════════════════════════════════════════════════════
# 4. force_ttl value matrix + kept_ttl precedence
# ═══════════════════════════════════════════════════════════════════════════


def test_force_ttl_none_preserves_client_1h():
    msgs = [B("user", "a", ttl="1h")]
    out = normalize_message_cache_control(msgs, force_ttl=None)
    assert _sole_ttl(out) == "1h"


def test_force_ttl_empty_string_falls_back_to_kept_ttl():
    """force_ttl is now validated against {'5m', '1h', None}. '' is not a
    member of that set, so it is rejected and treated as force_ttl=None:
    normalize falls back to the client's kept_ttl ('1h' here) instead of
    writing a bare (no-ttl-key) marker that would silently downgrade the
    client's paid-for 1h retention to Anthropic's 5m default."""
    msgs = [B("user", "a", ttl="1h")]
    out = normalize_message_cache_control(msgs, force_ttl="")
    ms = _markers(out)
    assert len(ms) == 1
    (_, _, cc) = ms[0]
    assert cc.get("ttl") == "1h", f"expected kept_ttl '1h' preserved, got {cc}"


def test_force_ttl_bogus_value_is_rejected_falls_back_to_kept_ttl():
    """force_ttl is now validated: any string that is not one of Anthropic's
    two valid ttl tiers ('5m'/'1h') or None is ignored rather than forwarded
    verbatim, which previously would have produced a request Anthropic
    rejects with a 400. normalize falls back to the client's kept_ttl."""
    msgs = [B("user", "a", ttl="1h")]
    out = normalize_message_cache_control(msgs, force_ttl="bogus")
    assert _sole_ttl(out) == "1h"  # fixed: bogus value rejected, kept_ttl used


def test_force_ttl_wrong_type_is_rejected_falls_back_to_kept_ttl():
    """A non-string force_ttl (e.g. an int from a misconfigured caller) is not
    a member of {'5m', '1h', None}, so validation now rejects it and falls
    back to the client's kept_ttl instead of writing the raw int into the
    ttl field."""
    msgs = [B("user", "a", ttl="1h")]
    out = normalize_message_cache_control(msgs, force_ttl=3600)
    assert _sole_ttl(out) == "1h"  # fixed: wrong-typed value rejected, kept_ttl used


def test_force_ttl_precedence_over_client_kept_ttl():
    """Client sent 1h; force_ttl='5m' must win (that's the entire point of
    the knob: downgrade a sub-agent's write tier below what the client
    itself would have chosen)."""
    msgs = [B("user", "a", ttl="1h"), B("user", "b", ttl="1h")]
    out = normalize_message_cache_control(msgs, force_ttl="5m")
    assert _sole_ttl(out) == "5m"


def test_force_ttl_none_after_prior_force_ttl_reads_back_headroom_own_value_not_client_intent():
    """Self-referential contamination: once headroom has forced a ttl on a
    prior turn (e.g. HR_SUBAGENT_TTL_5M fired), the ONLY marker left in the
    messages array by the time a LATER turn calls normalize(force_ttl=None)
    is headroom's OWN previously-written marker -- not the client's original
    intent, which was already stripped and overwritten on the earlier turn.

    kept_ttl has no way to distinguish "client's ttl" from "headroom's own
    prior artifact"; it reads whatever ttl currently sits on the sole
    breakpoint. If a later turn's caller passes force_ttl=None expecting to
    "preserve the client's ttl" (per the docstring), it actually preserves
    headroom's own earlier forced value -- so once forced, a session that
    later stops forcing (flag no longer applies, or reads default) sticks at
    the forced tier forever, never reverting to the client's real 1h
    preference, even though the client re-sends 1h on every new message.

    This models a client that (like many non-Claude-Code clients, per this
    file's own docstring at line ~120) does NOT re-mark the newest message
    every turn -- it only marks once and leaves the marker in place, so the
    overlay replays headroom's forced marker into the frozen region and the
    client's original ttl is unrecoverable.
    """
    # Turn 1: client sends 1h. A subagent-detection flag forces 5m.
    turn1_client = [B("user", "hello", ttl="1h")]
    turn1_out = normalize_message_cache_control(turn1_client, force_ttl="5m")
    assert _sole_ttl(turn1_out) == "5m"

    # Turn 2: the client's real preference is STILL 1h (it never changed its
    # mind), but this client does not re-mark new messages -- the new
    # message below carries NO cache_control at all, same as many non-Claude
    # Code clients. The subagent flag has stopped firing (force_ttl=None),
    # expecting kept_ttl to fall back to "the client's ttl".
    turn2_client = turn1_out + [{"role": "user", "content": [{"type": "text", "text": "world"}]}]
    turn2_out = normalize_message_cache_control(turn2_client, force_ttl=None)

    # BUG: this reads back headroom's own turn-1 artifact (5m), not the
    # client's real, never-revoked 1h intent.
    assert _sole_ttl(turn2_out) == "5m"


def test_kept_ttl_prefers_last_marker_by_index_when_multiple_present():
    """Pre-normalize state can (transiently, mid-pipeline) have more than one
    marker if the overlay replayed an old one and the client also marked its
    new tail message. Document that kept_ttl takes the marker at the HIGHEST
    message index, which is only "the client's newest intent" when the
    highest-index marker really is the client's own -- it is purely
    positional, not source-aware."""
    msgs = [
        BC("user", "old (headroom's from a prior turn)", ttl="5m"),
        BC("user", "new (client's this turn)", ttl="1h"),
    ]
    out = normalize_message_cache_control(msgs, force_ttl=None)
    assert _sole_ttl(out) == "1h"  # last-by-index wins here, matches "newest" by luck of ordering

    # Same content, order reversed: kept_ttl still takes the LAST index,
    # even though semantically that is now the "old" marker -- proving the
    # selection is positional, not based on which one is truly the client's.
    msgs_reordered = [
        BC("user", "new (client's this turn)", ttl="1h"),
        BC("user", "old (headroom's from a prior turn)", ttl="5m"),
    ]
    out2 = normalize_message_cache_control(msgs_reordered, force_ttl=None)
    assert _sole_ttl(out2) == "5m"
