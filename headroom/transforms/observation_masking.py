"""Cache-gated masking of old, large Anthropic tool observations."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

MASK_PREFIX = "[Tool result masked:"
INPUT_MASK_PREFIX = "[Tool input masked:"
_RECOVERY_MARKER = "Retrieve original: hash="
_EXCERPT_CHARS = 120

# tool_use input keys that carry bulk content worth masking. Deliberately a
# short allowlist: these are file bodies and edit payloads that are
# recoverable from the CCR store (and usually from disk), never control
# arguments the model reasons about later.
_MASKABLE_INPUT_KEYS = ("content", "file_text", "new_string", "old_string")


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    """One eligible Anthropic tool result and its prepared marker."""

    message_index: int
    block_index: int
    tool_use_id: str
    tool_name: str
    original: str
    marker: str
    content_hash: str
    original_tokens: int
    marker_tokens: int
    original_bytes: int
    input_key: str | None = None
    """None targets tool_result content; a key name targets that field of a
    tool_use block's input dict instead."""
    text_block: bool = False
    """True when the tool_result content is the Claude Code block-list form
    [{"type": "text", "text": ...}] rather than a plain string. The marker
    replaces the inner text, preserving sibling keys like cache_control."""

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.marker_tokens)


@dataclass(frozen=True, slots=True)
class MaskResult:
    messages: list[dict[str, Any]]
    masked_count: int = 0
    tokens_saved: int = 0
    bytes_saved: int = 0


def masking_gate_gain(
    candidates: list[MaskCandidate],
    *,
    compression_policy: Any,
    suffix_tokens: int,
    expected_reads: float,
    p_alive: float,
    write_multiplier: float,
) -> float:
    """Price an exact masking batch through the shared mutation formula."""
    delta_t = sum(candidate.tokens_saved for candidate in candidates)
    return float(
        compression_policy.net_mutation_gain(
            delta_t,
            suffix_tokens,
            expected_reads,
            p_alive,
            write_multiplier,
        )
    )


def _already_compact(content: str) -> bool:
    stripped = content.lstrip()
    return (
        stripped.startswith(MASK_PREFIX)
        or stripped.startswith(INPUT_MASK_PREFIX)
        or _RECOVERY_MARKER in content
        or stripped.startswith("[Read content stale:")
        or stripped.startswith("[Read content superseded:")
        or stripped.startswith("[Read content matured:")
    )


def _excerpt(content: str) -> str:
    line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    normalized = " ".join(line.split())
    if len(normalized) > _EXCERPT_CHARS:
        normalized = normalized[: _EXCERPT_CHARS - 3] + "..."
    return json.dumps(normalized, ensure_ascii=True)


def _tool_metadata(messages: list[dict[str, Any]]) -> tuple[dict[str, tuple[str, int]], int]:
    metadata: dict[str, tuple[str, int]] = {}
    assistant_turn = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_use_id = block.get("id")
                if isinstance(tool_use_id, str) and tool_use_id:
                    name = block.get("name")
                    metadata[tool_use_id] = (
                        name if isinstance(name, str) and name else "unknown",
                        assistant_turn,
                    )
        assistant_turn += 1
    return metadata, assistant_turn


def discover_candidates(
    messages: list[dict[str, Any]],
    *,
    count_tokens: Callable[[str], int],
    mask_after_turns: int = 3,
    mask_min_tokens: int = 400,
) -> list[MaskCandidate]:
    """Prepare eligible markers without mutating messages or writing CCR."""
    metadata, assistant_turns = _tool_metadata(messages)
    candidates: list[MaskCandidate] = []

    def prepare(
        message_index: int,
        block_index: int,
        tool_use_id: str,
        tool_name: str,
        source_turn: int,
        original: str,
        input_key: str | None,
        text_block: bool = False,
    ) -> MaskCandidate | None:
        if _already_compact(original):
            return None
        if assistant_turns - source_turn - 1 < max(0, mask_after_turns):
            return None
        original_tokens = max(0, int(count_tokens(original)))
        if original_tokens < max(0, mask_min_tokens):
            return None
        try:
            encoded = original.encode("utf-8")
        except UnicodeEncodeError:
            # Unpaired surrogates cannot round-trip through the CCR
            # store, so the block is left untouched.
            return None
        original_bytes = len(encoded)
        content_hash = hashlib.sha256(encoded).hexdigest()[:24]
        if input_key is None:
            marker = (
                f"[Tool result masked: tool={tool_name}, tokens={original_tokens}, "
                f"bytes={original_bytes}, head={_excerpt(original)}. Not deleted, recover "
                f"the full text by requesting: Retrieve original: hash={content_hash}]"
            )
        else:
            marker = (
                f"[Tool input masked: tool={tool_name}, key={input_key}, "
                f"tokens={original_tokens}, bytes={original_bytes}, "
                f"head={_excerpt(original)}. Not deleted, recover the full text by "
                f"requesting: Retrieve original: hash={content_hash}]"
            )
        marker_tokens = max(0, int(count_tokens(marker)))
        if marker_tokens >= original_tokens:
            return None
        return MaskCandidate(
            message_index=message_index,
            block_index=block_index,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            original=original,
            marker=marker,
            content_hash=content_hash,
            original_tokens=original_tokens,
            marker_tokens=marker_tokens,
            original_bytes=original_bytes,
            input_key=input_key,
            text_block=text_block,
        )

    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        is_assistant = message.get("role") == "assistant"
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                original = block.get("content")
                text_block = False
                if isinstance(original, list):
                    # Claude Code sends tool_result content as a block list,
                    # almost always a single text block. Mask that inner text;
                    # multi-block or non-text lists stay untouched.
                    if (
                        len(original) == 1
                        and isinstance(original[0], dict)
                        and original[0].get("type") == "text"
                        and isinstance(original[0].get("text"), str)
                    ):
                        original = original[0]["text"]
                        text_block = True
                    else:
                        continue
                if not isinstance(original, str):
                    continue
                tool_use_id = block.get("tool_use_id")
                if not isinstance(tool_use_id, str):
                    tool_use_id = ""
                tool_name, source_turn = metadata.get(
                    tool_use_id, ("unknown", assistant_turns)
                )
                candidate = prepare(
                    message_index,
                    block_index,
                    tool_use_id,
                    tool_name,
                    source_turn,
                    original,
                    None,
                    text_block,
                )
                if candidate is not None:
                    candidates.append(candidate)
            elif block_type == "tool_use" and is_assistant:
                tool_use_id = block.get("id")
                if not isinstance(tool_use_id, str) or tool_use_id not in metadata:
                    continue
                block_input = block.get("input")
                if not isinstance(block_input, dict):
                    continue
                tool_name, source_turn = metadata[tool_use_id]
                for input_key in _MASKABLE_INPUT_KEYS:
                    original = block_input.get(input_key)
                    if not isinstance(original, str):
                        continue
                    candidate = prepare(
                        message_index,
                        block_index,
                        tool_use_id,
                        tool_name,
                        source_turn,
                        original,
                        input_key,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return candidates


def apply_candidates(
    messages: list[dict[str, Any]],
    candidates: list[MaskCandidate],
    *,
    compression_store: Any,
) -> MaskResult:
    """Persist and apply an admitted batch, retaining originals on store failure."""
    if not candidates:
        return MaskResult(messages=messages)
    replacements: dict[tuple[int, int, str | None], MaskCandidate] = {}
    for candidate in candidates:
        try:
            stored_hash = compression_store.store(
                original=candidate.original,
                compressed="",
                tool_name=candidate.tool_name,
                tool_call_id=candidate.tool_use_id,
                compression_strategy="observation_masking",
                explicit_hash=candidate.content_hash,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("observation_masking: CCR store failed for %s: %s", candidate.tool_use_id, exc)
            continue
        if stored_hash != candidate.content_hash:
            logger.warning("observation_masking: CCR store returned an unexpected hash")
            continue
        replacements[
            (candidate.message_index, candidate.block_index, candidate.input_key)
        ] = candidate
    if not replacements:
        return MaskResult(messages=messages)

    output: list[dict[str, Any]] = []
    tokens_saved = 0
    bytes_saved = 0
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            output.append(message)
            continue
        new_content = list(content)
        changed = False
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            candidate = replacements.get((message_index, block_index, None))
            if candidate is not None:
                block_content = block.get("content")
                if candidate.text_block:
                    if (
                        isinstance(block_content, list)
                        and len(block_content) == 1
                        and isinstance(block_content[0], dict)
                        and block_content[0].get("text") == candidate.original
                    ):
                        block = {
                            **block,
                            "content": [
                                {**block_content[0], "text": candidate.marker}
                            ],
                        }
                        new_content[block_index] = block
                        tokens_saved += candidate.tokens_saved
                        bytes_saved += candidate.original_bytes - len(
                            candidate.marker.encode("utf-8")
                        )
                        changed = True
                elif block_content == candidate.original:
                    block = {**block, "content": candidate.marker}
                    new_content[block_index] = block
                    tokens_saved += candidate.tokens_saved
                    bytes_saved += candidate.original_bytes - len(
                        candidate.marker.encode("utf-8")
                    )
                    changed = True
            for input_key in _MASKABLE_INPUT_KEYS:
                candidate = replacements.get((message_index, block_index, input_key))
                if candidate is None:
                    continue
                block_input = block.get("input")
                if (
                    not isinstance(block_input, dict)
                    or block_input.get(input_key) != candidate.original
                ):
                    continue
                block = {
                    **block,
                    "input": {**block_input, input_key: candidate.marker},
                }
                new_content[block_index] = block
                tokens_saved += candidate.tokens_saved
                bytes_saved += candidate.original_bytes - len(
                    candidate.marker.encode("utf-8")
                )
                changed = True
        output.append({**message, "content": new_content} if changed else message)
    return MaskResult(
        messages=output,
        masked_count=len(replacements),
        tokens_saved=tokens_saved,
        bytes_saved=max(0, bytes_saved),
    )
