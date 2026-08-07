"""Adversarial tests for the loosened prefix-cache lineage match (#2085,
commit 16f8e0a6).

Target contract, straight from prefix_tracker.py:

* SessionTrackerStore.resolve_tracker must give two DIFFERENT conversations
  under one session id separate PrefixCacheTracker instances, so they never
  steal each other's frozen-prefix state and thrash the provider cache.
* The one relaxation added by 16f8e0a6, ``_is_message_continuation``, is
  supposed to be narrow enough that it only ever fires for one conversation's
  OWN newest message growing in place (the sub-call transcript shape), never
  for two unrelated conversations that happen to share a shape.

Every test here drives the real path: SessionTrackerStore(PrefixFreezeConfig())
-> resolve_tracker(session, provider, messages=...) [-> normalize_message_cache_control]
-> tracker.update_from_response(...), and compares id(tracker) across turns.
No previous-turn state is hand-built.
"""

from __future__ import annotations

import pytest

from headroom.cache.prefix_tracker import (
    _MIN_CONTINUATION_RUN_BLOCKS,
    _is_message_continuation,
    PrefixCacheTracker,
    PrefixFreezeConfig,
    SessionTrackerStore,
    normalize_message_cache_control,
)


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _shared_doc(n: int) -> list[dict]:
    """A quoted artifact (log excerpt, file content) two independent
    sub-agent calls both legitimately receive byte-identical, because a
    parent process injected the same context into both prompts."""
    return [_text(f"doc line {i}: " + "x" * 200) for i in range(n)]


def _sibling_history(
    kickoff: str,
    doc_blocks: list[dict],
    middle_blocks: list[dict],
    closing: dict,
) -> list[dict]:
    """The sub-call shape: one kickoff message plus one block-style message
    packing [shared doc] + [this call's own work] + [shared closing
    instruction]."""
    content = [*doc_blocks, *middle_blocks, closing]
    return [
        {"role": "user", "content": kickoff},
        {"role": "user", "content": content},
    ]


CLOSING = _text('Respond with a single JSON object: {"result": ...}. No prose.')


def _middle(label: str, n: int) -> list[dict]:
    return [_text(f"[{label}] investigate function {label}_{i}()") for i in range(n)]


class TestUnrelatedSiblingsShareATrackerViaSharedDocument:
    """Core finding: the guard's absolute-floor + fraction test was tuned
    against tiny shared boilerplate (see test_boilerplate_on_both_ends_...
    in test_prefix_tracker.py), but a REALISTIC shared artifact — a quoted
    document/log that is legitimately identical across two unrelated
    sub-agent calls — is exactly the kind of thing that clears an 8-block,
    half-of-message floor. Two conversations that never interacted end up
    sharing one PrefixCacheTracker."""

    @pytest.fixture
    def store(self):
        return SessionTrackerStore(PrefixFreezeConfig())

    def test_second_callers_first_ever_message_steals_the_first_callers_tracker(
        self, store
    ):
        sid = "shared-fallback-id"
        doc = _shared_doc(20)

        history_a1 = _sibling_history("kickoff", doc, _middle("A", 4), CLOSING)
        tracker_a = store.resolve_tracker(sid, "anthropic", messages=history_a1)

        # B has never sent a message before. This is its FIRST turn, a
        # wholly separate conversation — not a continuation of anything.
        history_b1 = _sibling_history("kickoff", doc, _middle("B", 4), CLOSING)
        tracker_b = store.resolve_tracker(sid, "anthropic", messages=history_b1)

        assert tracker_b is not tracker_a, (
            "REFUTED: B's first-ever message was misclassified as A's "
            "conversation growing in place and got handed A's tracker. "
            "Input: two histories sharing a 20-block quoted doc + identical "
            "closing block, differing only in a 4-block middle. "
            f"id(tracker_a)={id(tracker_a)} id(tracker_b)={id(tracker_b)}"
        )

    def test_hijack_then_orphans_the_true_owners_next_turn(self, store):
        """Worse than a one-off misclassification: once B's first message
        overwrites the session's lineage snapshot, A's own GENUINE next turn
        (a real append-only growth of history_a1) no longer matches anything
        recorded and is bounced onto a brand new, cold tracker. One false
        merge costs the true owner its frozen prefix too."""
        sid = "shared-fallback-id"
        doc = _shared_doc(20)

        history_a1 = _sibling_history("kickoff", doc, _middle("A", 4), CLOSING)
        tracker_a1 = store.resolve_tracker(sid, "anthropic", messages=history_a1)
        tracker_a1.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=9000,
            messages=history_a1,
            original_messages=history_a1,
        )

        history_b1 = _sibling_history("kickoff", doc, _middle("B", 4), CLOSING)
        tracker_b1 = store.resolve_tracker(sid, "anthropic", messages=history_b1)
        assert tracker_b1 is tracker_a1  # confirms the hijack from the test above
        tracker_b1.update_from_response(
            cache_read_tokens=9000,
            cache_write_tokens=500,
            messages=history_b1,
            original_messages=history_b1,
        )

        # A's real second turn: append-only growth of history_a1, exactly the
        # shape lineage matching exists to preserve.
        history_a2 = history_a1 + [
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "follow-up " + "z" * 200},
        ]
        tracker_a2 = store.resolve_tracker(sid, "anthropic", messages=history_a2)

        assert tracker_a2 is not tracker_a1, (
            "confirms the thrash: A's own continuation was orphaned onto a "
            "fresh tracker as a side effect of B's earlier false match"
        )
        assert tracker_a2.get_frozen_message_count() == 0, (
            "A's legitimate continuation lost its frozen prefix entirely "
            "and is paying a cold-start rewrite for a conversation that "
            "was never actually broken"
        )
        assert store.active_sessions == 3, (
            "one two-turn conversation (A) plus one one-turn conversation "
            "(B) produced THREE tracker lineages"
        )

    def test_corrupted_lineage_state_survives_the_full_pipeline(self, store):
        """Drives resolve_tracker -> normalize_message_cache_control ->
        update_from_response, the real call sequence in the handler, and
        checks that the hijacked tracker's recorded state is B's bytes, not
        A's — so any later cache-safety decision made off
        get_last_original_messages()/get_last_forwarded_messages() for what
        the caller believes is "A's session" is actually comparing against a
        different conversation's content."""
        sid = "shared-fallback-id"
        doc = _shared_doc(20)

        history_a1 = _sibling_history("kickoff", doc, _middle("A", 4), CLOSING)
        tracker = store.resolve_tracker(sid, "anthropic", messages=history_a1)
        forwarded_a1 = normalize_message_cache_control(history_a1)
        tracker.update_from_response(
            cache_read_tokens=0,
            cache_write_tokens=9000,
            messages=forwarded_a1,
            original_messages=history_a1,
        )

        history_b1 = _sibling_history("kickoff", doc, _middle("B", 4), CLOSING)
        hijacked = store.resolve_tracker(sid, "anthropic", messages=history_b1)
        assert hijacked is tracker
        forwarded_b1 = normalize_message_cache_control(history_b1)
        hijacked.update_from_response(
            cache_read_tokens=9000,
            cache_write_tokens=500,
            messages=forwarded_b1,
            original_messages=history_b1,
        )

        recorded = tracker.get_last_original_messages()
        assert recorded == history_b1
        assert recorded != history_a1, (
            "the tracker that A's session id resolves to now carries B's "
            "conversation content"
        )

    def test_ordering_decides_the_victim_two_way_race(self, store):
        """Chaos-engineering angle: which of two equally-unrelated callers
        keeps the original tracker and which one hijacks it is pure arrival
        order, not conversation identity. Reversing the order flips who
        wins, which is a concurrency/race hazard on top of the correctness
        bug: under real interleaving from parallel subagents the outcome is
        nondeterministic."""
        sid = "race"
        doc = _shared_doc(20)
        history_a1 = _sibling_history("kickoff", doc, _middle("A", 4), CLOSING)
        history_b1 = _sibling_history("kickoff", doc, _middle("B", 4), CLOSING)

        first = store.resolve_tracker(sid, "anthropic", messages=history_a1)
        second = store.resolve_tracker(sid, "anthropic", messages=history_b1)
        assert second is first  # order 1: A arrives first, B hijacks it

        store2 = SessionTrackerStore(PrefixFreezeConfig())
        first_r = store2.resolve_tracker(sid, "anthropic", messages=history_b1)
        second_r = store2.resolve_tracker(sid, "anthropic", messages=history_a1)
        assert second_r is first_r  # order 2: B arrives first, A hijacks it

    def test_three_way_fan_out_only_two_of_three_collide(self, store):
        """A third, later-arriving sibling with a different closing
        instruction correctly stays separate, which shows the failure is not
        "anything goes" — it's specifically that a shared tail plus a large
        shared prefix defeats the guard, while a merely-shared prefix does
        not."""
        sid = "shared-fallback-id"
        doc = _shared_doc(20)
        history_a = _sibling_history("kickoff", doc, _middle("A", 4), CLOSING)
        history_b = _sibling_history("kickoff", doc, _middle("B", 4), CLOSING)
        history_c = _sibling_history(
            "kickoff", doc, _middle("C", 4), _text("Respond in plain prose.")
        )

        t_a = store.resolve_tracker(sid, "anthropic", messages=history_a)
        t_b = store.resolve_tracker(sid, "anthropic", messages=history_b)
        t_c = store.resolve_tracker(sid, "anthropic", messages=history_c)

        assert t_b is t_a, "same closing instruction: still collides (the bug)"
        assert t_c is not t_a, "different closing instruction: correctly separate"


class TestContinuationRunBoundaries:
    """Exact off-by-one behaviour of the two guards
    (_MIN_CONTINUATION_RUN_BLOCKS=8, run*2 >= len(old))."""

    @staticmethod
    def _msg(blocks: list[dict]) -> dict:
        return {"role": "user", "content": blocks}

    def test_run_exactly_8_of_16_blocks_is_accepted(self):
        old = [_text(f"stable {i}") for i in range(8)] + [
            _text(f"old-tail {i}") for i in range(7)
        ] + [_text("FINAL")]
        new = [_text(f"stable {i}") for i in range(8)] + [
            _text(f"new-tail {i}") for i in range(7)
        ] + [_text("FINAL")]
        assert len(old) == 16 and len(new) == 16
        assert _is_message_continuation(self._msg(old), self._msg(new)) is True

    def test_run_exactly_8_of_17_blocks_is_rejected(self):
        """One extra churned block tips the fraction under 0.5: 8*2=16 < 17."""
        old = [_text(f"stable {i}") for i in range(8)] + [
            _text(f"old-tail {i}") for i in range(8)
        ] + [_text("FINAL")]
        new = [_text(f"stable {i}") for i in range(8)] + [
            _text(f"new-tail {i}") for i in range(8)
        ] + [_text("FINAL")]
        assert len(old) == 17 and len(new) == 17
        assert _is_message_continuation(self._msg(old), self._msg(new)) is False

    def test_run_of_7_is_rejected_by_the_absolute_floor_even_at_a_high_fraction(self):
        """7 stable blocks out of 14 clears the 0.5 fraction (7*2=14>=14) but
        not the absolute floor of 8 — the floor is meant to be the primary
        defence, confirm it actually is."""
        assert _MIN_CONTINUATION_RUN_BLOCKS == 8
        old = [_text(f"stable {i}") for i in range(7)] + [_text("FINAL")]
        new = [_text(f"stable {i}") for i in range(7)] + [_text("DIFFERENT-FINAL")]
        # also breaks on the final-block check, but construct a variant that
        # only tests the floor: same final block, one churned block between.
        old = [_text(f"stable {i}") for i in range(7)] + [
            _text("mid")
        ] + [_text("FINAL")]
        new = [_text(f"stable {i}") for i in range(7)] + [
            _text("mid-changed")
        ] + [_text("FINAL")]
        assert len(old) == 9
        assert _is_message_continuation(self._msg(old), self._msg(new)) is False


class TestPathologicalBlockShapes:
    """Malformed / boundary content shapes through the real resolve_tracker
    path, hunting a crash or a silent wrong-answer rather than a clean
    rejection."""

    @pytest.fixture
    def store(self):
        return SessionTrackerStore(PrefixFreezeConfig())

    def test_empty_content_list_is_rejected_not_crashed(self, store):
        sid = "s"
        first = [
            {"role": "user", "content": "kickoff"},
            {"role": "user", "content": []},
        ]
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        second = [
            {"role": "user", "content": "kickoff"},
            {"role": "user", "content": [_text("anything")]},
        ]
        # old content is empty -> `if not old` guard in _is_message_continuation
        # must reject cleanly, not raise or silently accept.
        result = store.resolve_tracker(sid, "anthropic", messages=second)
        assert result is not tracker

    def test_plain_string_content_never_enters_the_loose_path(self, store):
        """String content can't carry block-level growth; must always fall
        through to the strict/rewrite path (fresh tracker), never crash
        `_is_message_continuation`'s isinstance checks."""
        sid = "s"
        first = [
            {"role": "user", "content": "kickoff"},
            {"role": "user", "content": "a growing string that changes " + "x" * 50},
        ]
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        second = [
            {"role": "user", "content": "kickoff"},
            {"role": "user", "content": "a growing string that changes " + "y" * 50},
        ]
        assert store.resolve_tracker(sid, "anthropic", messages=second) is not tracker

    def test_dict_blocks_missing_type_key_do_not_crash_the_run_scan(self, store):
        sid = "s"
        doc = [{"text": f"line {i}"} for i in range(20)]  # no "type" key at all
        first = [
            {"role": "user", "content": "kickoff"},
            {"role": "user", "content": [*doc, {"text": "A"}, CLOSING]},
        ]
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        second = [
            {"role": "user", "content": "kickoff"},
            {"role": "user", "content": [*doc, {"text": "B"}, CLOSING]},
        ]
        # Same bug surface as the main finding, just with type-less blocks:
        # documents this doesn't require well-formed content blocks either.
        result = store.resolve_tracker(sid, "anthropic", messages=second)
        assert result is tracker, "type-less blocks still clear the loose-match floor"

    def test_final_block_equal_by_value_but_different_object_matches_correctly(
        self, store
    ):
        """Sanity check in the other direction: value equality (not identity)
        must be what decides the final-block check, since every request
        history is freshly deserialized JSON and never the same object twice.
        This one is expected to behave — included so a future switch to `is`
        comparison would be caught."""
        sid = "s"
        closing_1 = {"type": "text", "text": "same closing instruction"}
        closing_2 = {"type": "text", "text": "same closing instruction"}
        assert closing_1 is not closing_2
        assert closing_1 == closing_2
        doc = _shared_doc(20)
        first = _sibling_history("kickoff", doc, _middle("A", 4), closing_1)
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        second = _sibling_history("kickoff", doc, _middle("A", 5), closing_2)
        # This IS the same conversation growing (one more middle block, same
        # closing text) so reuse here is correct, not a bug.
        assert store.resolve_tracker(sid, "anthropic", messages=second) is tracker

    def test_shrinking_middle_block_count_gets_a_fresh_tracker(self, store):
        """No-blocks-lost guard: total block count must not go down, even
        when the run and the final block would otherwise pass."""
        sid = "s"
        doc = _shared_doc(20)
        first = _sibling_history("kickoff", doc, _middle("A", 8), CLOSING)
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        shrunk = _sibling_history("kickoff", doc, _middle("A", 2), CLOSING)
        assert store.resolve_tracker(sid, "anthropic", messages=shrunk) is not tracker

    def test_role_flip_on_the_growing_message_is_rejected(self, store):
        sid = "s"
        doc = _shared_doc(20)
        first = _sibling_history("kickoff", doc, _middle("A", 4), CLOSING)
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        flipped = _sibling_history("kickoff", doc, _middle("A", 5), CLOSING)
        flipped[1] = {**flipped[1], "role": "assistant"}
        assert store.resolve_tracker(sid, "anthropic", messages=flipped) is not tracker


class TestCanonicalizationBlindSpotInTheRunCompare:
    """_stable_leading_block_run (and therefore _is_message_continuation)
    reuses _canonicalize_for_prefix_compare, which by design drops an
    explicit list of "non-semantic" keys. Two blocks that differ ONLY in one
    of those keys canonicalize-equal even when the field is being used to
    carry real information the strict schema doesn't reserve it for. This
    widens what counts as a "byte-stable" leading run beyond what the
    provider's own cache key actually requires, which is the same class of
    risk as the shared-document finding above, just entering through
    canonicalization instead of through genuinely-identical bytes."""

    @pytest.fixture
    def store(self):
        return SessionTrackerStore(PrefixFreezeConfig())

    def test_blocks_differing_only_in_a_non_semantic_key_read_as_stable(self, store):
        """`state` is in _NON_SEMANTIC_KEYS. A tool-result-shaped block that
        legitimately uses `state` for real content (not transport metadata)
        is silently treated as identical across two different results."""
        sid = "s"

        def doc_with_state(tag: str) -> list[dict]:
            return [
                {"type": "text", "text": f"line {i}", "state": tag} for i in range(20)
            ]

        first = [
            {"role": "user", "content": "kickoff"},
            {
                "role": "user",
                "content": [*doc_with_state("draft"), _middle("A", 4)[0], CLOSING],
            },
        ]
        tracker = store.resolve_tracker(sid, "anthropic", messages=first)
        second = [
            {"role": "user", "content": "kickoff"},
            {
                "role": "user",
                # `state` flips from "draft" to "final" on every one of the 20
                # leading blocks — a real content change under most schemas —
                # yet the run is still computed as fully stable because
                # `state` never enters the comparison.
                "content": [*doc_with_state("final"), _middle("B", 4)[0], CLOSING],
            },
        ]
        result = store.resolve_tracker(sid, "anthropic", messages=second)
        assert result is tracker, (
            "documents the canonicalization blind spot: 20 blocks whose "
            "`state` field flipped on every one of them still register as a "
            "byte-stable leading run"
        )
