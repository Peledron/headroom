"""The closed-episode boundary is a heuristic, so it has to be measurable.

The age gate waits for the model to demonstrably move on, so its retrieval rate
is the floor: that is how often masking guesses wrong even when it is being
careful. These tests pin the accounting that lets the two gates be compared,
not any particular rate.
"""

from __future__ import annotations

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.transforms.observation_masking import (
    apply_candidates,
    closed_episode_turn,
    discover_candidates,
)


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _store(max_entries: int = 100) -> CompressionStore:
    return CompressionStore(max_entries=max_entries, enable_feedback=False)


def _mask(store: CompressionStore, gate: str, body: str) -> str:
    return store.store(original=body, compressed="", mask_gate=gate)


class TestStoreAccounting:
    def test_an_untracked_store_reports_no_gates(self):
        store = _store()
        store.store(original="x" * 100, compressed="")
        assert store.get_stats()["mask_gates"] == {}

    def test_a_masked_block_counts_against_its_own_gate(self):
        store = _store()
        _mask(store, "episode", "a" * 100)
        _mask(store, "age", "b" * 100)
        gates = store.get_stats()["mask_gates"]
        assert gates["episode"]["stored"] == 1
        assert gates["age"]["stored"] == 1
        assert gates["episode"]["retrieved"] == 0

    def test_a_never_retrieved_block_is_the_success_case(self):
        store = _store()
        _mask(store, "episode", "a" * 100)
        assert store.get_stats()["mask_gates"]["episode"]["retrieval_rate"] == 0.0

    def test_retrieval_moves_the_rate(self):
        store = _store()
        first = _mask(store, "episode", "a" * 100)
        _mask(store, "episode", "b" * 100)
        store.retrieve(first)
        gates = store.get_stats()["mask_gates"]
        assert gates["episode"]["retrieved"] == 1
        assert gates["episode"]["retrieval_rate"] == pytest.approx(0.5)

    def test_a_block_fetched_twice_is_still_one_block_that_was_needed(self):
        """Otherwise a single chatty retrieval loop reads as a broken gate."""
        store = _store()
        hash_key = _mask(store, "episode", "a" * 100)
        store.retrieve(hash_key)
        store.retrieve(hash_key)
        gates = store.get_stats()["mask_gates"]
        assert gates["episode"]["retrieved"] == 1
        assert gates["episode"]["retrievals"] == 2
        assert gates["episode"]["retrieval_rate"] == 1.0

    def test_re_storing_the_same_block_does_not_inflate_the_denominator(self):
        """The CCR mirror re-stores a live marker's hash on every turn."""
        store = _store()
        for _ in range(5):
            _mask(store, "episode", "a" * 100)
        assert store.get_stats()["mask_gates"]["episode"]["stored"] == 1

    def test_an_evicted_block_stays_in_the_denominator(self):
        """A block masked, never wanted, then aged out is the case that works.

        Deriving these counts from live entries would drop it and make every
        gate look worse the longer the session ran.
        """
        store = _store(max_entries=1)
        _mask(store, "episode", "a" * 100)
        _mask(store, "episode", "b" * 100)
        gates = store.get_stats()["mask_gates"]
        assert store.get_stats()["entry_count"] == 1
        assert gates["episode"]["stored"] == 2

    def test_a_miss_counts_nothing(self):
        store = _store()
        _mask(store, "episode", "a" * 100)
        assert store.retrieve("deadbeef" * 3) is None
        assert store.get_stats()["mask_gates"]["episode"]["retrieved"] == 0

    def test_the_two_gates_are_counted_apart(self):
        store = _store()
        episode = _mask(store, "episode", "a" * 100)
        _mask(store, "age", "b" * 100)
        store.retrieve(episode)
        gates = store.get_stats()["mask_gates"]
        assert gates["episode"]["retrieval_rate"] == 1.0
        assert gates["age"]["retrieval_rate"] == 0.0


class TestGateAttribution:
    """The gate recorded has to be the gate that actually admitted the block."""

    def _conversation(self) -> list[dict]:
        bulk = "L" * 4_000
        return [
            {"role": "user", "content": [{"type": "text", "text": "read the log"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat log"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": bulk}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "done, it is empty"}]},
            {"role": "user", "content": [{"type": "text", "text": "now rename the module"}]},
        ]

    def test_a_block_from_a_finished_task_is_attributed_to_the_episode_gate(self):
        messages = self._conversation()
        boundary = closed_episode_turn(messages)
        assert boundary is not None

        candidates = discover_candidates(
            messages,
            count_tokens=_count_tokens,
            mask_after_turns=99,  # age gate cannot admit anything
            episode_closed_turn=boundary,
        )
        assert [c.closed_episode for c in candidates] == [True]

        store = _store()
        apply_candidates(messages, candidates, compression_store=store)
        gates = store.get_stats()["mask_gates"]
        assert gates["episode"]["stored"] == 1
        assert "age" not in gates

    def test_the_same_block_is_attributed_to_the_age_gate_without_a_boundary(self):
        messages = self._conversation()
        candidates = discover_candidates(
            messages,
            count_tokens=_count_tokens,
            mask_after_turns=0,
            episode_closed_turn=None,
        )
        assert candidates
        assert not any(c.closed_episode for c in candidates)

        store = _store()
        apply_candidates(messages, candidates, compression_store=store)
        gates = store.get_stats()["mask_gates"]
        assert gates["age"]["stored"] == len(candidates)
        assert "episode" not in gates

    def test_a_mid_task_turn_closes_no_episode(self):
        """The last user message carries tool_result, so nothing is finished."""
        messages = self._conversation()[:3]
        assert closed_episode_turn(messages) is None
