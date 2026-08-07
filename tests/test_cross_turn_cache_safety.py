"""Cross-turn cache-safety invariant — the test class that catches cache busts.

Why the +150%-cache_create / +41%-cost bug slipped through: every prior cache
test was SINGLE-turn and used a fake tracker, so nobody exercised the real
multi-turn invariant that actually governs prompt-cache cost:

    Across append-only turns, the forwarded prefix must stay BYTE-IDENTICAL to
    what was forwarded (and cached) last turn — otherwise the provider re-creates
    the whole suffix (a cache bust) instead of reading it.

This simulates the provider's prefix cache (longest byte-identical leading run of
messages = cache_read; the rest = cache_create) and drives the REAL
``PrefixCacheTracker`` + the freeze model + ``overlay_cached_prefix`` over several
turns. It asserts the invariant directly, and proves the guard is load-bearing:
WITHOUT the overlay the freeze forwards the agent's original bytes and busts every
turn; WITH it the prefix stays stable.
"""

from headroom.cache.prefix_tracker import (
    PrefixCacheTracker,
    PrefixFreezeConfig,
    normalize_message_cache_control,
    overlay_cached_prefix,
)


def _toklen(m) -> int:
    return max(1, len(str(m.get("content", ""))))


def _compress(m):
    """Deterministic stand-in for a real compressor (kompress is deterministic
    per content via the result cache): shrink the content by half."""
    c = str(m.get("content", ""))
    return {**m, "content": c[: max(1, len(c) // 2)]}


def _apply_freeze(original, frozen_count):
    """Faithful model of pipeline.apply()'s freeze: the frozen prefix is
    forwarded as the agent's ORIGINAL bytes; everything else is compressed.
    (Mirrors content_router.py: `result_slots[i] = message` for i < frozen.)"""
    return [
        (original[i] if i < frozen_count else _compress(original[i])) for i in range(len(original))
    ]


def _provider_cache_read(forwarded, prev_forwarded):
    """Longest byte-identical leading run of messages the provider can serve from
    cache, in tokens. A single differing message breaks the prefix (bust)."""
    if not prev_forwarded:
        return 0
    matched = 0
    for a, b in zip(forwarded, prev_forwarded):
        if a == b:
            matched += _toklen(a)
        else:
            break
    return matched


def _drive_turns(*, use_overlay: bool, turns: int = 5):
    """Return per-turn (expected_cache_read, actual_cache_read). A bust is any
    turn where actual < expected (the previously-cached prefix wasn't reused)."""
    # min_cached_tokens=0 so freeze activates from turn 2 regardless of size.
    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))
    convo: list[dict] = []
    prev_forwarded: list[dict] | None = None
    out = []
    for t in range(1, turns + 1):
        # Append-only growth: one new large tool output per turn.
        convo = convo + [{"role": "user", "content": f"tool-output-turn-{t}:" + "X" * 400}]

        frozen = tracker.get_frozen_message_count()
        forwarded = _apply_freeze(convo, frozen)
        if use_overlay:
            forwarded = overlay_cached_prefix(
                forwarded,
                convo,
                tracker.get_last_original_messages(),
                tracker.get_last_forwarded_messages(),
            )

        expected_read = sum(_toklen(m) for m in prev_forwarded) if prev_forwarded else 0
        actual_read = _provider_cache_read(forwarded, prev_forwarded)
        out.append((expected_read, actual_read))

        counts = [_toklen(m) for m in forwarded]
        write = sum(counts) - actual_read
        tracker.update_from_response(
            actual_read, write, forwarded, message_token_counts=counts, original_messages=convo
        )
        prev_forwarded = forwarded
    return out


def test_freeze_busts_cache_every_turn_without_overlay():
    """Proves the test is load-bearing: the raw freeze path busts the cache."""
    results = _drive_turns(use_overlay=False)
    # From turn 2 on, a hit was expected but the prefix broke (actual < expected).
    busts = [exp > act for (exp, act) in results[1:]]
    assert any(busts), "expected the un-fixed freeze path to bust the prefix cache"


def test_overlay_keeps_prefix_byte_identical_no_bust():
    """The fix: every turn reuses the full previously-cached prefix — no bust."""
    results = _drive_turns(use_overlay=True)
    for exp, act in results[1:]:
        assert act >= exp, (
            f"cache bust: expected to read {exp} cached tokens but only read {act} "
            "— forwarded prefix diverged from last turn"
        )


def test_cache_create_stays_bounded_to_the_delta_with_overlay():
    """Cost proxy: with the fix, per-turn cache_create ≈ the new delta only, not
    the whole re-created prefix (which is what drove +150% cache_create)."""
    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))
    convo: list[dict] = []
    prev_forwarded: list[dict] | None = None
    creates = []
    for t in range(1, 6):
        convo = convo + [{"role": "user", "content": f"turn-{t}:" + "X" * 400}]
        frozen = tracker.get_frozen_message_count()
        forwarded = overlay_cached_prefix(
            _apply_freeze(convo, frozen),
            convo,
            tracker.get_last_original_messages(),
            tracker.get_last_forwarded_messages(),
        )
        read = _provider_cache_read(forwarded, prev_forwarded)
        counts = [_toklen(m) for m in forwarded]
        create = sum(counts) - read
        creates.append(create)
        tracker.update_from_response(
            read, create, forwarded, message_token_counts=counts, original_messages=convo
        )
        prev_forwarded = forwarded
    # Steady-state cache_create per turn should be ~one delta message, NOT growing
    # with conversation length. Assert the last turn creates no more than the
    # first (which had no cache to reuse).
    assert creates[-1] <= creates[0] + 1


def test_recorded_state_not_mutated_by_downstream_consumers():
    """Guard for the ref-return contract (perf review F1): get_last_original/
    forwarded_messages() now return the tracker's actual recorded lists, not
    deep copies. overlay_cached_prefix splices those refs directly into its
    output, so normalize_message_cache_control and memory injection may both
    receive dicts that ARE the recorded state. None of them may mutate in
    place — a future turn's replay depends on the recorded bytes staying
    exactly what was recorded at update_from_response time.
    """
    import copy

    from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

    tracker = PrefixCacheTracker("anthropic", PrefixFreezeConfig(min_cached_tokens=0))

    # Turn 1: a message carrying cache_control (normalize has something to
    # strip) so the recorded state exercises that mutation path too.
    turn1 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    counts = [_toklen(m) for m in turn1]
    tracker.update_from_response(
        0, sum(counts), turn1, message_token_counts=counts, original_messages=turn1
    )

    # Independent snapshot of turn 1's recorded state — the ground truth.
    snapshot_original = copy.deepcopy(tracker.get_last_original_messages())
    snapshot_forwarded = copy.deepcopy(tracker.get_last_forwarded_messages())

    # Turn 2: append-only extension driven through overlay + normalize +
    # memory injection — the three consumers the mutation audit covers.
    turn2_original = turn1 + [{"role": "user", "content": "turn 2"}]
    frozen = tracker.get_frozen_message_count()
    forwarded = overlay_cached_prefix(
        turn2_original,
        turn2_original,
        tracker.get_last_original_messages(),
        tracker.get_last_forwarded_messages(),
    )
    forwarded = normalize_message_cache_control(forwarded)
    forwarded = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        forwarded, "injected memory", frozen_message_count=frozen
    )

    assert tracker.get_last_original_messages() == snapshot_original, (
        "turn 1's recorded original messages were mutated by a downstream consumer"
    )
    assert tracker.get_last_forwarded_messages() == snapshot_forwarded, (
        "turn 1's recorded forwarded messages were mutated by a downstream consumer"
    )
