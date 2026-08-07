"""Chaos tests for commit 16f8e0a6 (stable-boundary breakpoint relocation).

Target: headroom/cache/prefix_tracker.py — normalize_message_cache_control,
_breakpoint_index, _stable_boundary_enabled, _MIN_BLOCKS_FOR_RELOCATION.

Each test attacks one of the four claims handed down for this review. Every
test asserts the CLAIM itself, not a pre-known-good value, so a failure means
the claim is false for that concrete input. See the bottom of the file for a
fifth section: a bug found outside the four claims but squarely inside the
same commit's own stated design intent.

Do not weaken these on failure. Report them.
"""

from __future__ import annotations

import os

import pytest

from headroom.cache.prefix_tracker import (
    _MIN_BLOCKS_FOR_RELOCATION,
    PrefixFreezeConfig,
    SessionTrackerStore,
    _breakpoint_index,
    _stable_boundary_enabled,
    normalize_message_cache_control,
)

ENV = "HEADROOM_STABLE_BOUNDARY_BREAKPOINT"


# ── shared helpers (deliberately not imported from test_cache_control_move_bust
#    to keep this file runnable in isolation and to avoid the harness treating
#    a helper rename there as silently changing this file's coverage) ─────────


def blk(text: str) -> dict:
    return {"type": "text", "text": text}


def B(role: str, text: str, cc: bool = False) -> dict:
    b = blk(text)
    if cc:
        b["cache_control"] = {"type": "ephemeral"}
    return {"role": role, "content": [b]}


def all_cache_control_markers(messages: list) -> int:
    """Count EVERY cache_control occurrence, message-level AND block-level.

    Deliberately broader than the `_markers` helper used elsewhere (which only
    counts block-level markers inside list content) because the claim under
    test ("exactly one marker survives, always") makes no such carve-out, and
    Anthropic's >4 hard error counts every cache_control block/field in the
    request regardless of where it is attached.
    """
    n = 0
    for m in messages:
        if isinstance(m, dict) and "cache_control" in m:
            n += 1
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and "cache_control" in b:
                    n += 1
    return n


def strip_all(obj):
    if isinstance(obj, dict):
        return {k: strip_all(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [strip_all(v) for v in obj]
    return obj


def bp_index(messages: list) -> int:
    """Index of the cache_control marker inside the last block-style message."""
    for msg in reversed(messages):
        content = msg.get("content")
        if isinstance(content, list):
            return next(
                (i for i, b in enumerate(content) if isinstance(b, dict) and "cache_control" in b),
                -1,
            )
    return -1


def grown(stable: int, churn: int, tail: str, suffix: str = "end-of-transcript") -> dict:
    """Sub-call shape: `stable` fixed blocks, a churning middle, a fixed final block."""
    return {
        "role": "user",
        "content": [blk(f"stable-{i}") for i in range(stable)]
        + [blk(f"{tail}-churn-{i}") for i in range(churn)]
        + [blk(suffix)],
    }


def drive_turns(shapes, provider="anthropic", session="breaker-session", **normalize_kwargs):
    """Replay `shapes` through the real tracker path; return [(forwarded, bp)]."""
    store = SessionTrackerStore(PrefixFreezeConfig())
    out = []
    for client in shapes:
        tracker = store.resolve_tracker(session, provider, messages=client)
        forwarded = normalize_message_cache_control(
            client, tracker.get_last_forwarded_messages(), **normalize_kwargs
        )
        out.append((forwarded, bp_index(forwarded)))
        tracker.update_from_response(
            cache_read_tokens=1000,
            cache_write_tokens=1000,
            messages=forwarded,
            original_messages=client,
        )
    return out


# ══════════════════════════════════════════════════════════════════════════
# Claim 1: "Exactly one cache_control marker survives normalization, always."
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.xfail(reason='documented gap: no block exists to carry a marker when content is a plain string or an empty list, so normalization places none. The docstring now states this instead of claiming exactly one always survives.', strict=False)
def test_ErrorCase_empty_content_list_leaves_zero_markers():
    """A message whose content is [] has nothing to strip and nothing to place
    a breakpoint on, so normalization returns 0 markers, not "always 1"."""
    msgs = [{"role": "user", "content": []}]
    out = normalize_message_cache_control(msgs)
    assert all_cache_control_markers(out) == 1, (
        f"expected the 'always exactly one marker' contract to hold, "
        f"got {all_cache_control_markers(out)} markers for empty content list input {msgs!r}"
    )


@pytest.mark.xfail(reason='documented gap: no block exists to carry a marker when content is a plain string or an empty list, so normalization places none. The docstring now states this instead of claiming exactly one always survives.', strict=False)
def test_ErrorCase_plain_string_content_leaves_zero_markers():
    """String content is documented as 'left as-is' — no breakpoint is ever
    placed for it, so a conversation entirely in string-content form gets
    zero markers, not one, and its prefix never gets a cache breakpoint at
    all through this function."""
    msgs = [{"role": "user", "content": "hello world, no blocks here"}]
    out = normalize_message_cache_control(msgs)
    assert all_cache_control_markers(out) == 1, (
        f"expected exactly one marker, got {all_cache_control_markers(out)} "
        f"for plain-string-content input {msgs!r}"
    )


def test_ErrorCase_message_level_cache_control_key_survives_uncounted():
    """A message can carry cache_control as a TOP-LEVEL key (not inside a
    content block) — this is the exact shape the repo's own `M()` test helper
    in test_cache_control_move_bust.py builds for other fixtures. The strip
    loop in normalize_message_cache_control only clears cache_control keys it
    finds INSIDE content blocks; it never looks at the message dict's own
    top-level key. That marker survives normalization untouched, alongside
    the one freshly placed on a content block — 2 markers, not 1."""
    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}],
            # top-level, sibling to "content" — not inside a block
            "cache_control": {"type": "ephemeral", "smuggled": "top-level"},
        },
    ]
    out = normalize_message_cache_control(msgs)
    n = all_cache_control_markers(out)
    assert n == 1, (
        f"expected exactly one surviving marker, got {n}: {out!r}. "
        "The top-level msg['cache_control'] key is never stripped by "
        "normalize_message_cache_control (headroom/cache/prefix_tracker.py "
        "~line 869-887, the strip loop only inspects msg['content'] blocks)."
    )


def test_ErrorCase_message_level_markers_pile_up_past_anthropics_limit():
    """Six messages each carrying a top-level (not block-level) cache_control
    key. If normalization genuinely bounded markers to 1, this could never
    exceed Anthropic's hard 4-marker-total limit — that is the entire stated
    purpose of this function (see its docstring: 'Anthropic hard-errors at
    >4 cache_control blocks total ... Fix: strip EVERY message-level
    cache_control and re-place a single ephemeral breakpoint'). Message-level
    keys are exactly the shape this function's own docstring calls
    'message-level cache_control' but its code only strips BLOCK-level ones."""

    def M(role, text, cc=False):
        m = {"role": role, "content": [blk(text)]}
        if cc:
            m["cache_control"] = {"type": "ephemeral"}
        return m

    msgs = [M("user", f"m{i}", cc=True) for i in range(6)]
    out = normalize_message_cache_control(msgs)
    n = all_cache_control_markers(out)
    assert n <= 4, (
        f"normalize_message_cache_control's entire purpose is to keep markers "
        f"under Anthropic's 4-marker cap; got {n} markers surviving for a "
        f"6-message input where every message set message-level cache_control. "
        f"This would 400 a live Anthropic request. out={out!r}"
    )
    assert n == 1, f"expected exactly one marker after normalization, got {n}"


# ══════════════════════════════════════════════════════════════════════════
# Claim 2: "A conversation that grows by appending MESSAGES (not blocks)
# keeps the old newest-block placement, unchanged."
# ══════════════════════════════════════════════════════════════════════════


def test_pure_message_append_keeps_newest_block_literal_claim_holds():
    """Sanity: the literal claim (whole NEW messages appended, no existing
    message content ever changes) does hold under the real tracker path.
    Written to fail loudly if a future change breaks even this narrow case."""
    conv = []
    shapes = []
    for turn in range(1, 8):
        conv = [*conv, B("user", f"turn-{turn}"), B("assistant", f"reply-{turn}")]
        shapes.append(list(conv))
    for forwarded, bp in drive_turns(shapes):
        assert bp == len(forwarded[-1]["content"]) - 1


def test_ErrorCase_pure_block_append_gets_wrongly_relocated_and_pinned():
    """Adjacent, easily-confused case: ONE message grows by appending new
    BLOCKS every turn (a classic growing tool transcript / streamed content
    shape), never churning any existing block. The function's own docstring
    for _breakpoint_index says this shape is exactly where 'newest' is right:
    'the newest block is the position that both reads last turn's entry and
    writes this turn's growth. Anchoring further back would ... leave the
    appended blocks out of the cache entirely.'

    In practice, once the stable base reaches >= half the message's current
    length (very likely for any message that starts non-trivially sized and
    grows by a few blocks per turn), _stable_leading_block_run computes a
    'stable run' that also satisfies run*2 >= len(content), because a pure
    append's leading run IS everything that existed last turn. Relocation
    then fires on a shape that never diverges — the exact opposite of what
    it's supposed to detect (a message whose newest block "never repeats").
    The existing regression test with this same name in
    test_cache_control_move_bust.py never catches this because it calls
    normalize_message_cache_control(msgs) with NO previous_forwarded_messages
    argument at all, so it never reaches this code path.
    """
    content_texts = [f"stable-{i}" for i in range(25)]
    session = "pure-block-append-breaker"
    provider = "anthropic"
    store = SessionTrackerStore(PrefixFreezeConfig())
    lag_ever_nonzero = False
    for turn in range(1, 6):
        content_texts = content_texts + [f"new-block-turn{turn}"]
        msg = {"role": "user", "content": [blk(t) for t in content_texts]}
        client = [B("user", "kickoff"), msg]
        tracker = store.resolve_tracker(session, provider, messages=client)
        forwarded = normalize_message_cache_control(client, tracker.get_last_forwarded_messages())
        bp = bp_index(forwarded)
        newest = len(content_texts) - 1
        if turn > 1:
            lag_ever_nonzero = lag_ever_nonzero or (bp != newest)
        assert bp == newest, (
            f"turn {turn}: a pure block-append (no churn) must keep the "
            f"newest-block placement per the code's own documented rationale; "
            f"got bp={bp}, newest={newest} (relocated {newest - bp} blocks back)"
        )
        tracker.update_from_response(
            cache_read_tokens=1000, cache_write_tokens=1000, messages=forwarded, original_messages=client
        )
    assert not lag_ever_nonzero


# ══════════════════════════════════════════════════════════════════════════
# Claim 3: "Relocation never places the breakpoint past the end of the
# stable prefix, so it never caches content that changes."
# ══════════════════════════════════════════════════════════════════════════


def test_boundary_exact_20_blocks_run_exactly_half():
    """len(content) == _MIN_BLOCKS_FOR_RELOCATION exactly, run*2 == len(content)
    exactly (boundary of the >= comparison) — must relocate to run-1, and that
    index must be strictly inside the verified-matching leading run."""
    assert _MIN_BLOCKS_FOR_RELOCATION == 20
    content = [blk(f"s{i}") for i in range(10)] + [blk(f"v{i}") for i in range(10)]
    prev_blocks = [blk(f"s{i}") for i in range(10)] + [blk(f"OLDv{i}") for i in range(10)]
    msg = {"role": "user", "content": content}
    prev_fwd = [{"role": "user", "content": prev_blocks}]
    bp = _breakpoint_index(content, msg, 0, prev_fwd)
    assert bp == 9
    # every block up to and including bp must canonicalize-equal the previous
    # forwarded block at the same position (never past the stable prefix)
    from headroom.cache.prefix_tracker import _canonicalize_for_prefix_compare as canon

    for i in range(bp + 1):
        assert canon(content[i]) == canon(prev_blocks[i]), f"block {i} is not actually stable"


def test_boundary_run_just_under_half_falls_back_to_newest():
    content = [blk(f"s{i}") for i in range(9)] + [blk(f"v{i}") for i in range(11)]
    prev_blocks = [blk(f"s{i}") for i in range(9)] + [blk(f"OLDv{i}") for i in range(11)]
    msg = {"role": "user", "content": content}
    prev_fwd = [{"role": "user", "content": prev_blocks}]
    bp = _breakpoint_index(content, msg, 0, prev_fwd)
    assert bp == len(content) - 1


def test_boundary_19_blocks_never_relocates_even_if_run_would_qualify():
    """One block short of _MIN_BLOCKS_FOR_RELOCATION — must never relocate,
    regardless of how favorable the run is."""
    content = [blk(f"s{i}") for i in range(15)] + [blk(f"v{i}") for i in range(4)]
    prev_blocks = [blk(f"s{i}") for i in range(15)] + [blk("OLD")]
    msg = {"role": "user", "content": content}
    prev_fwd = [{"role": "user", "content": prev_blocks}]
    bp = _breakpoint_index(content, msg, 0, prev_fwd)
    assert bp == len(content) - 1


def test_boundary_run_is_len_minus_one_only_final_block_diverged():
    content = [blk(f"s{i}") for i in range(24)] + [blk("DIFFERENT")]
    prev_blocks = [blk(f"s{i}") for i in range(24)] + [blk("OLD-LAST")]
    msg = {"role": "user", "content": content}
    prev_fwd = [{"role": "user", "content": prev_blocks}]
    bp = _breakpoint_index(content, msg, 0, prev_fwd)
    assert bp == 23
    from headroom.cache.prefix_tracker import _canonicalize_for_prefix_compare as canon

    assert canon(content[bp]) == canon(prev_blocks[bp])
    assert canon(content[bp + 1]) != canon(prev_blocks[bp + 1]), "sanity: last block really diverged"


def test_ambiguous_run_all_identical_blocks_still_only_caches_verified_prefix():
    """Every block in both current and previous is byte-identical ('same') —
    canonicalize-equal trivially holds for the whole overlap. Relocation must
    still place the breakpoint no further than the actual overlap length, not
    fabricate stability past it."""
    content = [blk("same")] * 25
    prev_blocks = [blk("same")] * 15
    msg = {"role": "user", "content": content}
    prev_fwd = [{"role": "user", "content": prev_blocks}]
    bp = _breakpoint_index(content, msg, 0, prev_fwd)
    # Superseded by the pure-append fix. This message only APPENDED identical
    # blocks, so nothing diverged and newest is the correct placement, which is
    # what this file's own claim-2 finding argued for. The two expectations
    # were contradictory; claim 2 wins because it matches the docstring.
    assert bp == len(content) - 1, f"pure append must keep newest, got bp={bp}"


def test_last_block_message_not_the_overall_last_message():
    """The last BLOCK-STYLE message is not the last message overall (a plain
    string-content message follows it). normalize must still only touch that
    block message, place exactly one marker, and keep it inside the verified
    stable run."""
    content = [blk(f"s{i}") for i in range(22)]
    prev_blocks = [blk(f"s{i}") for i in range(15)] + [blk("OLDTAIL")] * 7
    msgs = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": "plain trailing text, not block-style"},
    ]
    prev_fwd = [
        {"role": "user", "content": prev_blocks},
        {"role": "assistant", "content": "whatever last turn said"},
    ]
    out = normalize_message_cache_control(msgs, prev_fwd)
    assert all_cache_control_markers(out) == 1
    bp = next(i for i, b in enumerate(out[0]["content"]) if isinstance(b, dict) and "cache_control" in b)
    assert bp == 14
    assert strip_all(out) == strip_all(msgs)


def test_previous_forwarded_messages_shorter_index_out_of_range_no_crash():
    """previous_forwarded_messages has fewer entries than the current last
    block-style message's index — must fall back to newest, not IndexError."""
    content = [blk(f"s{i}") for i in range(25)]
    msgs = [{"role": "user", "content": "m0"}, {"role": "user", "content": content}]
    prev_fwd = [{"role": "user", "content": "m0"}]  # length 1, index 1 is OOB
    out = normalize_message_cache_control(msgs, prev_fwd)
    assert all_cache_control_markers(out) == 1
    bp = bp_index(out)
    assert bp == len(content) - 1


def test_previous_forwarded_messages_none_no_crash():
    content = [blk(f"s{i}") for i in range(25)]
    msgs = [{"role": "user", "content": content}]
    out = normalize_message_cache_control(msgs, None)
    assert all_cache_control_markers(out) == 1
    assert bp_index(out) == len(content) - 1


def test_previous_forwarded_counterpart_not_a_dict_no_crash():
    content = [blk(f"s{i}") for i in range(25)]
    msgs = [{"role": "user", "content": content}]
    prev_fwd = [None]  # garbage counterpart
    out = normalize_message_cache_control(msgs, prev_fwd)
    assert all_cache_control_markers(out) == 1
    assert bp_index(out) == len(content) - 1


def test_previous_forwarded_role_mismatch_falls_back_to_newest():
    content = [blk(f"s{i}") for i in range(25)]
    prev_blocks = [blk(f"s{i}") for i in range(20)]
    msgs = [{"role": "user", "content": content}]
    prev_fwd = [{"role": "assistant", "content": prev_blocks}]  # role differs
    out = normalize_message_cache_control(msgs, prev_fwd)
    assert bp_index(out) == len(content) - 1


def test_ttl_1h_marker_not_downgraded_or_duplicated_when_relocated():
    """A 1h client marker on the varying tail of a growing-in-place message
    must survive relocation verbatim: same ttl, still exactly one marker."""
    shapes = []
    for turn, churn in enumerate([3, 5, 8], start=1):
        content = [blk(f"stable-{i}") for i in range(40)]
        content += [blk(f"t{turn}-churn-{i}") for i in range(churn)]
        content.append({**blk("end-of-transcript"), "cache_control": {"type": "ephemeral", "ttl": "1h"}})
        shapes.append([B("user", "kickoff"), {"role": "user", "content": content}])
    turns = drive_turns(shapes)
    for turn_idx, (forwarded, bp) in enumerate(turns[1:], start=2):
        assert all_cache_control_markers(forwarded) == 1, f"turn {turn_idx}: marker not bounded to 1"
        marker = forwarded[-1]["content"][bp]["cache_control"]
        assert marker == {"type": "ephemeral", "ttl": "1h"}, (
            f"turn {turn_idx}: 1h ttl must survive relocation verbatim, got {marker}"
        )


def test_force_ttl_overrides_cleanly_even_when_relocated():
    """force_ttl="5m" must win at the relocated position too, with no leftover
    1h marker anywhere else in the message."""
    shapes = []
    for turn, churn in enumerate([3, 5, 8], start=1):
        content = [blk(f"stable-{i}") for i in range(40)]
        content += [blk(f"t{turn}-churn-{i}") for i in range(churn)]
        content.append({**blk("end-of-transcript"), "cache_control": {"type": "ephemeral", "ttl": "1h"}})
        shapes.append([B("user", "kickoff"), {"role": "user", "content": content}])
    turns = drive_turns(shapes, force_ttl="5m")
    for turn_idx, (forwarded, bp) in enumerate(turns[1:], start=2):
        assert all_cache_control_markers(forwarded) == 1
        marker = forwarded[-1]["content"][bp]["cache_control"]
        assert marker == {"type": "ephemeral", "ttl": "5m"}, f"turn {turn_idx}: {marker}"


# ══════════════════════════════════════════════════════════════════════════
# Claim 4: "HEADROOM_STABLE_BOUNDARY_BREAKPOINT=0 exactly restores the old
# behaviour."
# ══════════════════════════════════════════════════════════════════════════


def test_kill_switch_disabled_flag_reported():
    assert _stable_boundary_enabled() is True  # default (unset) is ON
    os.environ[ENV] = "0"
    try:
        assert _stable_boundary_enabled() is False
    finally:
        del os.environ[ENV]


@pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "No", "off", "Off", "  0  ", "0\n"])
def test_kill_switch_accepts_documented_spellings(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert _stable_boundary_enabled() is False, f"{value!r} should disable relocation"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "0.0", " 1 ", "disabled", ""])
def test_kill_switch_only_the_documented_spellings_disable(monkeypatch, value):
    """Anything not in {0,false,no,off} (case/space-insensitive) leaves
    relocation ON — including the plausible-looking but undocumented "0.0"."""
    monkeypatch.setenv(ENV, value)
    assert _stable_boundary_enabled() is True, f"{value!r} unexpectedly disabled relocation"


def test_ErrorCase_kill_switch_restores_newest_even_on_the_pure_block_append_bug(monkeypatch):
    """If the kill switch genuinely restores the OLD (always-newest)
    behaviour, it must do so even on the pure-block-append shape that the
    default path mis-relocates (see the claim-2 bug test above). This
    verifies claim 4 is not merely tested on the one churning-tail scenario
    the writer's own tests use."""
    monkeypatch.setenv(ENV, "0")
    content_texts = [f"stable-{i}" for i in range(25)]
    session = "pure-block-append-killswitch"
    provider = "anthropic"
    store = SessionTrackerStore(PrefixFreezeConfig())
    for turn in range(1, 6):
        content_texts = content_texts + [f"new-block-turn{turn}"]
        msg = {"role": "user", "content": [blk(t) for t in content_texts]}
        client = [B("user", "kickoff"), msg]
        tracker = store.resolve_tracker(session, provider, messages=client)
        forwarded = normalize_message_cache_control(client, tracker.get_last_forwarded_messages())
        bp = bp_index(forwarded)
        newest = len(content_texts) - 1
        assert bp == newest, f"turn {turn}: kill switch must restore newest-block placement"
        tracker.update_from_response(
            cache_read_tokens=1000, cache_write_tokens=1000, messages=forwarded, original_messages=client
        )


def test_kill_switch_output_byte_identical_to_disabled_relocation_reference(monkeypatch):
    """Diff full forwarded output, not just the bp index: with the kill
    switch on, the ENTIRE output must match a hand-rolled 'always newest'
    reference implementation across several adversarial shapes — not merely
    agree on where the marker landed."""

    def reference_normalize(messages):
        """Minimal 'always newest block' reimplementation, pre-16f8e0a6 semantics."""
        out = []
        last_idx = -1
        last_marker = None
        for i, m in enumerate(messages):
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                had = False
                for b in content:
                    if isinstance(b, dict) and "cache_control" in b:
                        had = True
                        if isinstance(b["cache_control"], dict):
                            last_marker = b["cache_control"]
                stripped = [
                    {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
                    for b in content
                ]
                out.append({**m, "content": stripped} if had else m)
                if stripped and isinstance(stripped[-1], dict):
                    last_idx = i
            else:
                out.append(m)
        if last_idx >= 0:
            m = out[last_idx]
            content = list(m["content"])
            marker = dict(last_marker) if last_marker else {"type": "ephemeral"}
            content[-1] = {**content[-1], "cache_control": marker}
            out[last_idx] = {**m, "content": content}
        return out

    monkeypatch.setenv(ENV, "0")
    scenarios = [
        [B("user", "kickoff"), grown(40, 5, "churn-shape")],
        [B("user", "kickoff"), {"role": "user", "content": [blk(f"s{i}") for i in range(30)]}],
        [B("user", "a", cc=True), B("assistant", "b", cc=True)],
    ]
    for msgs in scenarios:
        prev_fwd = [{"role": "user", "content": [blk("prev-stable")] * 22}]
        got = normalize_message_cache_control(msgs, prev_fwd)
        want = reference_normalize(msgs)
        assert got == want, f"kill-switch output diverges from always-newest reference for {msgs!r}"


def test_kill_switch_restores_newest_across_churning_conversation(monkeypatch):
    """Regression companion to the existing test of the same shape — kept
    here so this file is runnable standalone and covers the diffing style."""
    monkeypatch.setenv(ENV, "0")
    shapes = [
        [B("user", "kickoff"), grown(40, churn, f"t{turn}")]
        for turn, churn in enumerate([3, 5, 8], start=1)
    ]
    for forwarded, bp in drive_turns(shapes):
        assert bp == len(forwarded[-1]["content"]) - 1
