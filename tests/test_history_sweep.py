from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from headroom.proxy.hybrid_mode import HybridModeConfig
from headroom.transforms.observation_masking import (
    discover_candidates,
    sweep_history,
)


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return len(text.split())


class _Router:
    def compress(self, text: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            compressed="summary retained facts",
            strategy_used=SimpleNamespace(value="text"),
        )


class _Store:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def store(self, **entry: object) -> str:
        self.entries.append(entry)
        return str(entry["explicit_hash"])


def _old_result_messages(text: str) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool-1", "content": text}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "next"}]},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": [{"type": "text", "text": "now"}]},
    ]


def test_bust_floor_discovery_rerun_finds_mid_size_result() -> None:
    text = " ".join(f"token{i}" for i in range(100))
    messages = _old_result_messages(text)

    steady = discover_candidates(
        messages,
        count_tokens=_Tokenizer().count_text,
        mask_after_turns=1,
        mask_min_tokens=150,
    )
    bust = discover_candidates(
        messages,
        count_tokens=_Tokenizer().count_text,
        mask_after_turns=1,
        mask_min_tokens=HybridModeConfig().mask_min_tokens_at_bust,
    )

    assert steady == []
    assert len(bust) == 1


def test_sweep_compresses_eligible_result_and_stores_ccr() -> None:
    original = " ".join(f"record{i}" for i in range(100))
    messages = _old_result_messages(original)
    store = _Store()

    result = sweep_history(
        messages,
        router=_Router(),
        tokenizer=_Tokenizer(),
        compression_store=store,
    )

    swept = result.messages[1]["content"][0]["content"]
    assert result.swept_count == 1
    assert len(swept.encode()) <= len(original.encode()) * 0.85
    assert f"<<ccr:{store.entries[0]['explicit_hash']}" in swept
    assert store.entries[0]["compression_strategy"] == "history_sweep"
    assert store.entries[0]["original"] == original


def test_assistant_text_is_default_off_and_opt_in_for_old_blocks() -> None:
    original = " ".join(f"assistant{i}" for i in range(100))
    messages: list[dict[str, object]] = [
        {"role": "assistant", "content": [{"type": "text", "text": original}]},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": [{"type": "text", "text": "two"}]},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": [{"type": "text", "text": "four"}]},
        {"role": "user", "content": "five"},
        {"role": "assistant", "content": [{"type": "text", "text": "six"}]},
        {"role": "user", "content": "seven"},
        {"role": "assistant", "content": [{"type": "text", "text": "eight"}]},
    ]

    default = sweep_history(
        messages,
        router=_Router(),
        tokenizer=_Tokenizer(),
        compression_store=_Store(),
    )
    opted_in = sweep_history(
        messages,
        router=_Router(),
        tokenizer=_Tokenizer(),
        compression_store=_Store(),
        sweep_assistant_text=True,
    )

    assert default.messages is messages
    assert default.swept_count == 0
    assert opted_in.swept_count == 1
    assert opted_in.messages[0]["content"][0]["text"] != original


def test_sweep_is_idempotent() -> None:
    messages = _old_result_messages(" ".join(f"row{i}" for i in range(100)))
    store = _Store()
    first = sweep_history(
        messages,
        router=_Router(),
        tokenizer=_Tokenizer(),
        compression_store=store,
    )
    second = sweep_history(
        first.messages,
        router=_Router(),
        tokenizer=_Tokenizer(),
        compression_store=store,
    )

    assert second.messages is first.messages
    assert second.swept_count == 0
    assert len(store.entries) == 1


def test_unadmitted_path_can_forward_byte_identical_messages() -> None:
    messages = _old_result_messages("small result")
    before = deepcopy(messages)
    candidates = discover_candidates(
        messages,
        count_tokens=_Tokenizer().count_text,
        mask_after_turns=1,
        mask_min_tokens=150,
    )

    assert candidates == []
    assert messages == before
