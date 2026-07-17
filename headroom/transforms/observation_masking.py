"""Cache-gated masking of old, large Anthropic tool observations."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

MASK_PREFIX = "[Tool result masked:"
_RECOVERY_MARKER = "Retrieve original: hash="
_EXCERPT_CHARS = 80


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
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            original = block.get("content")
            if not isinstance(original, str) or _already_compact(original):
                continue
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                tool_use_id = ""
            tool_name, source_turn = metadata.get(tool_use_id, ("unknown", assistant_turns))
            if assistant_turns - source_turn - 1 < max(0, mask_after_turns):
                continue
            original_tokens = max(0, int(count_tokens(original)))
            if original_tokens < max(0, mask_min_tokens):
                continue
            try:
                encoded = original.encode("utf-8")
            except UnicodeEncodeError:
                # Unpaired surrogates cannot round-trip through the CCR
                # store, so the block is left untouched.
                continue
            original_bytes = len(encoded)
            content_hash = hashlib.sha256(encoded).hexdigest()[:24]
            marker = (
                f"[Tool result masked: tool={tool_name}, tokens={original_tokens}, "
                f"bytes={original_bytes}, head={_excerpt(original)}. "
                f"Retrieve original: hash={content_hash}]"
            )
            marker_tokens = max(0, int(count_tokens(marker)))
            if marker_tokens >= original_tokens:
                continue
            candidates.append(
                MaskCandidate(
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
                )
            )
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
    replacements: dict[tuple[int, int], MaskCandidate] = {}
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
        replacements[(candidate.message_index, candidate.block_index)] = candidate
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
            candidate = replacements.get((message_index, block_index))
            if candidate is None or not isinstance(block, dict):
                continue
            if block.get("content") != candidate.original:
                continue
            new_content[block_index] = {**block, "content": candidate.marker}
            tokens_saved += candidate.tokens_saved
            bytes_saved += candidate.original_bytes - len(candidate.marker.encode("utf-8"))
            changed = True
        output.append({**message, "content": new_content} if changed else message)
    return MaskResult(
        messages=output,
        masked_count=len(replacements),
        tokens_saved=tokens_saved,
        bytes_saved=max(0, bytes_saved),
    )
