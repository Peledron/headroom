from __future__ import annotations

from copy import deepcopy

import pytest

from headroom.transforms.observation_masking import (
    apply_candidates,
    discover_candidates,
    masking_gate_gain,
)


def _count(text: str) -> int:
    if text.startswith("[Tool result masked:"):
        return 1
    return len(text.split())


def _messages(payload: str, later_turns: int = 3) -> list[dict]:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": payload}],
        },
    ]
    for turn in range(later_turns):
        messages.extend(
            [
                {"role": "assistant", "content": f"turn {turn}"},
                {"role": "user", "content": "continue"},
            ]
        )
    return messages


class _Store:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

    def store(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("store unavailable")
        return kwargs["explicit_hash"]


@pytest.mark.parametrize("later_turns, expected", [(2, 0), (3, 1), (4, 1)])
def test_age_threshold(later_turns: int, expected: int):
    candidates = discover_candidates(
        _messages("word " * 20, later_turns),
        count_tokens=_count,
        mask_min_tokens=10,
    )
    assert len(candidates) == expected


@pytest.mark.parametrize("tokens, expected", [(9, 0), (10, 1), (11, 1)])
def test_size_threshold(tokens: int, expected: int):
    candidates = discover_candidates(
        _messages("word " * tokens),
        count_tokens=_count,
        mask_min_tokens=10,
    )
    assert len(candidates) == expected


def test_placeholder_and_ccr_content():
    payload = "\n  first\tline with spacing  \n" + "word " * 30
    candidates = discover_candidates(
        _messages(payload), count_tokens=_count, mask_min_tokens=10
    )
    candidate = candidates[0]
    assert "tool=Read" in candidate.marker
    assert f"tokens={_count(payload)}" in candidate.marker
    assert f"bytes={len(payload.encode('utf-8'))}" in candidate.marker
    assert 'head="first line with spacing"' in candidate.marker
    assert f"Retrieve original: hash={candidate.content_hash}" in candidate.marker

    store = _Store()
    result = apply_candidates(_messages(payload), candidates, compression_store=store)
    assert result.masked_count == 1
    assert store.calls[0]["compression_strategy"] == "observation_masking"
    assert store.calls[0]["original"] == payload


def test_idempotence_and_existing_recovery_marker():
    payload = "word " * 30
    messages = _messages(payload)
    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10)
    first = apply_candidates(messages, candidates, compression_store=_Store())
    frozen = deepcopy(first.messages)
    second_candidates = discover_candidates(
        first.messages, count_tokens=_count, mask_min_tokens=10
    )
    assert second_candidates == []
    assert first.messages == frozen

    recovery = _messages("old. Retrieve original: hash=0123456789abcdef01234567 " + payload)
    assert discover_candidates(recovery, count_tokens=_count, mask_min_tokens=10) == []


def test_non_string_and_store_failure_leave_messages_unchanged():
    messages = _messages("word " * 30)
    messages[1]["content"].append(
        {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text"}]}
    )
    original = deepcopy(messages)
    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10)
    store = _Store(fail=True)
    result = apply_candidates(messages, candidates, compression_store=store)
    assert result.messages == original
    assert result.messages is messages
    assert result.masked_count == 0


class _Policy:
    def net_mutation_gain(self, delta_t, suffix_tokens, expected_reads, p_alive, write_multiplier):  # noqa: ANN001, ANN201
        return delta_t * expected_reads - p_alive * suffix_tokens * write_multiplier


def test_gate_uses_exact_masking_delta():
    candidates = discover_candidates(
        _messages("word " * 30), count_tokens=_count, mask_min_tokens=10
    )
    gain = masking_gate_gain(
        candidates,
        compression_policy=_Policy(),
        suffix_tokens=10,
        expected_reads=4,
        p_alive=0.5,
        write_multiplier=2,
    )
    assert gain == candidates[0].tokens_saved * 4 - 10
