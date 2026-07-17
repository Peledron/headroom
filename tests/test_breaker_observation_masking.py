"""Adversarial checks for observation masking.  Product code is deliberately untouched."""

from __future__ import annotations

from copy import deepcopy

import pytest

from headroom.transforms.observation_masking import apply_candidates, discover_candidates


def _count(text: str) -> int:
    return len(text.split())


def _marker_count(text: str) -> int:
    return 1 if text.startswith("[Tool result masked:") else _count(text)


def _messages(content: object, *, tool_id: str = "call-1", later_turns: int = 3) -> list[dict]:
    messages: list[dict] = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": "Read"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}]},
    ]
    messages.extend({"role": "assistant", "content": "next"} for _ in range(later_turns))
    return messages


class _Store:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def store(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        return kwargs["explicit_hash"]


def test_breaker_surrogate_content_does_not_crash_discovery() -> None:
    # Unpaired surrogates cannot round-trip the CCR store, so the block is
    # skipped whole: no crash, no candidate, no mutation.
    messages = _messages("word " * 20 + "\ud800", later_turns=3)
    original = deepcopy(messages)
    candidates = discover_candidates(messages, count_tokens=_count, mask_min_tokens=10)
    assert candidates == []
    assert messages == original


def test_breaker_exact_threshold_masks() -> None:
    # Threshold is inclusive, provided the marker is smaller than the content
    # (a marker bigger than the original is rejected on economics).
    candidates = discover_candidates(
        _messages("x " * 10), count_tokens=_marker_count, mask_min_tokens=10
    )
    assert len(candidates) == 1


def test_breaker_last_assistant_tool_use_never_masks() -> None:
    messages = [
        {"role": "assistant", "content": "older"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "last", "content": "x " * 30}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "last", "name": "Read"}]},
    ]
    assert discover_candidates(messages, count_tokens=_count, mask_min_tokens=10) == []


def test_breaker_orphan_tool_result_never_masks() -> None:
    messages = _messages("x " * 30, tool_id="known", later_turns=3)
    messages[1]["content"][0]["tool_use_id"] = "orphan"
    assert discover_candidates(messages, count_tokens=_count, mask_min_tokens=10) == []


def test_breaker_recovery_uses_marker_hash_and_is_idempotent() -> None:
    messages = _messages("x " * 30)
    candidates = discover_candidates(messages, count_tokens=_marker_count, mask_min_tokens=10)
    store = _Store()
    result = apply_candidates(messages, candidates, compression_store=store)
    assert store.calls[0]["original"] == "x " * 30
    assert store.calls[0]["explicit_hash"] in result.messages[1]["content"][0]["content"]
    second = discover_candidates(result.messages, count_tokens=_marker_count, mask_min_tokens=10)
    assert second == []
    assert apply_candidates(result.messages, second, compression_store=store).messages == result.messages
    assert len(store.calls) == 1


@pytest.mark.parametrize(
    "content",
    [
        "",
        [{"type": "text", "text": "x " * 30}, {"type": "text", "text": "y " * 30}],
        [{"type": "image", "source": {}}],
    ],
)
def test_breaker_empty_multiblock_and_nontext_remain_untouched(content: object) -> None:
    messages = _messages(content)
    original = deepcopy(messages)
    assert discover_candidates(messages, count_tokens=_count, mask_min_tokens=10) == []
    assert messages == original


def test_breaker_single_text_block_masks_and_preserves_cache_control() -> None:
    # Claude Code's real tool_result shape: a one-element text-block list,
    # sometimes carrying cache_control, which the marker block must keep.
    payload = "x " * 30
    content = [{"type": "text", "text": payload, "cache_control": {"type": "ephemeral"}}]
    messages = _messages(content)
    candidates = discover_candidates(messages, count_tokens=_marker_count, mask_min_tokens=10)
    assert len(candidates) == 1
    assert candidates[0].text_block is True
    store = _Store()
    result = apply_candidates(messages, candidates, compression_store=store)
    masked = result.messages[1]["content"][0]["content"]
    assert masked[0]["text"].startswith("[Tool result masked:")
    assert masked[0]["cache_control"] == {"type": "ephemeral"}
    assert store.calls[0]["original"] == payload
    # source untouched, second pass idempotent
    assert messages[1]["content"][0]["content"][0]["text"] == payload
    assert discover_candidates(result.messages, count_tokens=_marker_count, mask_min_tokens=10) == []
