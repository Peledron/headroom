"""Adversarial coverage for the turn-2 observation-masking changes.

These tests intentionally exercise malformed-but-serializable Anthropic block
shapes and the batch application boundary.  They do not change product code.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from headroom.proxy.hybrid_mode import HybridModeConfig
from headroom.transforms.observation_masking import (
    _MASKABLE_INPUT_KEYS,
    apply_candidates,
    discover_candidates,
)


def _count(text: str) -> int:
    if text.startswith("[Tool result masked:") or text.startswith("[Tool input masked:"):
        return 1
    return len(text.split())


def _plain_count(text: str) -> int:
    return len(text.split())


def _aged_tool_use(
    input_value: dict[str, object], *, tool_id: str = "tool-1"
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Write",
                    "input": input_value,
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}],
        },
    ]
    for turn in range(3):
        messages.extend(
            [
                {"role": "assistant", "content": f"turn {turn}"},
                {"role": "user", "content": "continue"},
            ]
        )
    return messages


class _Store:
    def __init__(self, *, fail_originals: set[str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_originals = fail_originals or set()

    def store(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        if kwargs["original"] in self.fail_originals:
            raise OSError("simulated store failure")
        return str(kwargs["explicit_hash"])


def test_input_candidates_ignore_non_strings_and_copy_nested_input() -> None:
    shared_input: dict[str, object] = {
        "content": "payload " * 30,
        "new_string": 23,
        "file_path": "/tmp/example.py",
    }
    messages = _aged_tool_use(shared_input)
    original = deepcopy(messages)

    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert [candidate.input_key for candidate in candidates] == ["content"]

    result = apply_candidates(messages, candidates, compression_store=_Store())
    assert result.masked_count == 1
    assert messages == original
    assert result.messages[0]["content"][0]["input"] is not shared_input
    assert result.messages[0]["content"][0]["input"]["new_string"] == 23


def test_malformed_result_block_with_input_only_masks_result_content() -> None:
    messages = _aged_tool_use({"content": "unused " * 30})
    block = messages[1]["content"][0]
    block["input"] = {"content": "input " * 30}
    block["content"] = "result " * 30

    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert [(candidate.message_index, candidate.block_index, candidate.input_key) for candidate in candidates] == [
        (0, 0, "content"),
        (1, 0, None),
    ]
    result = apply_candidates(messages, candidates, compression_store=_Store())
    assert result.messages[1]["content"][0]["input"]["content"] == "input " * 30


def test_partial_store_failure_does_not_repeat_admitted_input_on_fresh_discovery() -> None:
    old = "old " * 30
    new = "new " * 30
    messages = _aged_tool_use({"old_string": old, "new_string": new})
    first_candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    store = _Store(fail_originals={new})
    first = apply_candidates(messages, first_candidates, compression_store=store)
    assert first.masked_count == 1

    second_candidates = discover_candidates(first.messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert [candidate.input_key for candidate in second_candidates] == ["new_string"]
    second_store = _Store()
    second = apply_candidates(first.messages, second_candidates, compression_store=second_store)
    assert second.masked_count == 1
    assert [call["original"] for call in second_store.calls] == [new]


def test_input_marker_economics_are_evaluated_per_key() -> None:
    messages = _aged_tool_use({"content": "one two three four five six seven eight nine"})
    candidates = discover_candidates(messages, count_tokens=_plain_count, mask_min_tokens=0, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert candidates == []


def test_minimum_zero_still_rejects_empty_and_tiny_payloads() -> None:
    for payload in ("", "one"):
        messages = _aged_tool_use({"content": payload})
        assert discover_candidates(messages, count_tokens=_plain_count, mask_min_tokens=0, mask_input_keys=_MASKABLE_INPUT_KEYS) == []


def test_threshold_env_parsing_clamps_negatives_and_accepts_large_ints(monkeypatch) -> None:
    monkeypatch.setenv("HR_MASK_AFTER_TURNS", "-4")
    monkeypatch.setenv("HR_MASK_MIN_TOKENS", "999999999999999999999999")
    config = HybridModeConfig.from_environment()
    assert config.mask_after_turns == 0
    assert config.mask_min_tokens == 999999999999999999999999


@pytest.mark.xfail(
    strict=True,
    reason="duplicate IDs overwrite old source-turn metadata with the newest occurrence",
)
def test_duplicate_tool_id_does_not_suppress_an_old_input_candidate() -> None:
    messages = _aged_tool_use({"content": "old " * 30}, tool_id="duplicate")
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "duplicate",
                    "name": "Write",
                    "input": {"content": "young " * 30},
                }
            ],
        }
    )
    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10, mask_input_keys=_MASKABLE_INPUT_KEYS)
    assert [(candidate.message_index, candidate.input_key) for candidate in candidates] == [
        (0, "content")
    ]
