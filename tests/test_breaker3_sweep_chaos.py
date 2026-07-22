"""Breaker: chaos tests for observation_masking.sweep_history and the
bust-floor rerun logic wired in handlers/anthropic.py.

Scope is request-side only (see brief). Existing coverage:
tests/test_history_sweep.py. This file targets the six chaos surfaces from
the brief: cache-determinism, the 15% boundary and marker idempotence, CCR
store failure mid-batch, the bust-floor economic-flip gap, the
sweep_assistant_text mimicry surface, and unicode edge content.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from headroom.proxy.hybrid_mode import HybridModeConfig
from headroom.transforms.observation_masking import (
    discover_candidates,
    masking_gate_gain,
    sweep_history,
)


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return len(text.split())


class _FixedRouter:
    """Deterministic router: same input always yields the same output."""

    def __init__(self, compressed: str) -> None:
        self._compressed = compressed

    def compress(self, text: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            compressed=self._compressed,
            strategy_used=SimpleNamespace(value="text"),
        )


class _CounterRouter:
    """Router whose output changes on every call, simulating a
    time-dependent or randomized compressor."""

    def __init__(self) -> None:
        self.calls = 0

    def compress(self, text: str, **_: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(
            compressed=f"summary retained facts call={self.calls}",
            strategy_used=SimpleNamespace(value="text"),
        )


class _Store:
    def __init__(self, fail_after: int | None = None) -> None:
        self.entries: list[dict[str, object]] = []
        self.fail_after = fail_after

    def store(self, **entry: object) -> str:
        if self.fail_after is not None and len(self.entries) >= self.fail_after:
            raise RuntimeError("simulated CCR store outage")
        self.entries.append(entry)
        return str(entry["explicit_hash"])


def _old_result_messages(text: str, tool_use_id: str = "tool-1") -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_use_id, "name": "Bash"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "next"}]},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": [{"type": "text", "text": "now"}]},
    ]


def _multi_result_messages(texts: dict[str, str]) -> list[dict[str, object]]:
    """N tool calls, each aged past the sweep floor, then several more
    assistant turns to keep everything comfortably eligible."""
    messages: list[dict[str, object]] = []
    for tool_use_id, _ in texts.items():
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_use_id, "name": "Bash"}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": texts[tool_use_id],
                    }
                ],
            }
        )
    for _ in range(3):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "x"}]})
        messages.append({"role": "user", "content": "continue"})
    return messages


# ---------------------------------------------------------------------------
# 1. Determinism / cache safety
# ---------------------------------------------------------------------------


def test_sweep_history_same_input_twice_is_byte_identical() -> None:
    original = " ".join(f"record{i}" for i in range(100))
    messages = _old_result_messages(original)
    router = _FixedRouter("summary retained facts")

    result_a = sweep_history(
        deepcopy(messages),
        router=router,
        tokenizer=_Tokenizer(),
        compression_store=_Store(),
    )
    result_b = sweep_history(
        deepcopy(messages),
        router=router,
        tokenizer=_Tokenizer(),
        compression_store=_Store(),
    )

    assert result_a.swept_count == result_b.swept_count == 1
    assert result_a.messages == result_b.messages


def test_sweep_history_trusts_router_determinism_no_internal_guard() -> None:
    """FINDING: sweep_history has no guard against a nondeterministic
    compressor. If the router's output varies call to call (time-dependent
    summarizer, randomized truncation, etc.), sweeping byte-identical input
    twice yields byte-different markers, which busts the provider prefix
    cache on every turn that re-sweeps the same content. This is exactly
    the failure mode principle 1 warns about, and sweep_history provides no
    detection or rejection of it."""
    original = " ".join(f"record{i}" for i in range(100))
    messages = _old_result_messages(original)
    router = _CounterRouter()

    result_a = sweep_history(
        deepcopy(messages),
        router=router,
        tokenizer=_Tokenizer(),
        compression_store=_Store(),
    )
    result_b = sweep_history(
        deepcopy(messages),
        router=router,
        tokenizer=_Tokenizer(),
        compression_store=_Store(),
    )

    assert result_a.swept_count == result_b.swept_count == 1
    # The two sweeps of the SAME input produced DIFFERENT bytes.
    assert result_a.messages != result_b.messages


# ---------------------------------------------------------------------------
# 2. The 15% minimum threshold, boundary and idempotence
# ---------------------------------------------------------------------------


def test_savings_fraction_exact_boundary_is_accepted() -> None:
    """replacement_bytes == encoded * 0.85 exactly must pass (the guard is
    a strict '>' rejection), not be rejected as an off-by-one."""
    # 13-word, 80-byte original chosen so that 80 * 0.85 = 68 is an exact
    # integer byte target: 15% savings, right at the minimum_savings_fraction
    # default of 0.15.
    import hashlib

    original = " ".join(f"word{i}" for i in range(13))
    encoded = original.encode("utf-8")
    enc_len = len(encoded)
    assert enc_len == 80
    content_hash = hashlib.sha256(encoded).hexdigest()[:24]
    marker = f"<<ccr:{content_hash},string,{enc_len}B>>"
    marker_len = len(marker.encode("utf-8"))
    # replacement = f"{compressed.rstrip()}\n{marker}", want total == 68.
    target_total = 68
    compressed_len = target_total - 1 - marker_len  # -1 for the newline
    assert compressed_len > 0, "test construction requires marker under 67 bytes"
    compressed = "b" * compressed_len

    messages = _old_result_messages(original)
    router = _FixedRouter(compressed)
    result = sweep_history(
        messages,
        router=router,
        tokenizer=_Tokenizer(),
        compression_store=_Store(),
        minimum_savings_fraction=0.15,
    )
    assert result.swept_count == 1, (
        "exact-boundary savings (85/100 = 15%) should be accepted, not "
        "rejected by the strict '>' comparison"
    )


def test_savings_fraction_one_byte_over_boundary_is_rejected() -> None:
    import hashlib

    original = " ".join(f"word{i}" for i in range(13))
    encoded = original.encode("utf-8")
    enc_len = len(encoded)
    assert enc_len == 80
    content_hash = hashlib.sha256(encoded).hexdigest()[:24]
    marker = f"<<ccr:{content_hash},string,{enc_len}B>>"
    marker_len = len(marker.encode("utf-8"))
    target_total = 69  # one byte over the 68-byte (15%) boundary
    compressed_len = target_total - 1 - marker_len
    compressed = "b" * compressed_len

    messages = _old_result_messages(original)
    router = _FixedRouter(compressed)
    result = sweep_history(
        messages,
        router=router,
        tokenizer=_Tokenizer(),
        compression_store=_Store(),
        minimum_savings_fraction=0.15,
    )
    assert result.swept_count == 0


def test_second_sweep_pass_is_idempotent_on_own_marker() -> None:
    original = " ".join(f"record{i}" for i in range(100))
    messages = _old_result_messages(original)
    router = _FixedRouter("summary retained facts")

    first = sweep_history(
        messages, router=router, tokenizer=_Tokenizer(), compression_store=_Store()
    )
    assert first.swept_count == 1

    second = sweep_history(
        first.messages, router=router, tokenizer=_Tokenizer(), compression_store=_Store()
    )
    assert second.swept_count == 0
    assert second.messages == first.messages


def test_nested_ccr_marker_anywhere_in_text_blocks_the_whole_block() -> None:
    """FINDING (missed optimization, not correctness): _already_compact
    treats ANY occurrence of the substring '<<ccr:' anywhere in the text as
    'already compact' and skips the block entirely, even when the marker
    sits in the middle of an otherwise large, uncompressed, eligible
    result (e.g. a grep hit that happens to quote a previously-swept
    fragment verbatim, or paste of prior tool output containing a marker).
    Such a block is permanently ineligible for sweeping regardless of its
    size, with no fallback to compress the marker-free portions."""
    quoted_marker = "<<ccr:abcdef0123456789abcdef01,string,42B>>"
    original = (
        "some earlier output happened to contain " + quoted_marker + " "
        + " ".join(f"record{i}" for i in range(200))
    )
    messages = _old_result_messages(original)
    router = _FixedRouter("summary retained facts")

    result = sweep_history(
        messages, router=router, tokenizer=_Tokenizer(), compression_store=_Store()
    )
    assert result.swept_count == 0


# ---------------------------------------------------------------------------
# 3. CCR store failure mid-batch
# ---------------------------------------------------------------------------


def test_store_failure_mid_batch_skips_block_and_keeps_prior_work() -> None:
    """FIXED contract (was: uncaught exception discarding the batch): a
    store failure on one block is caught per-block, that block is left
    unswept, earlier successfully-stored blocks stay applied, and the
    call returns a consistent MaskResult instead of raising."""
    texts = {
        "tool-1": " ".join(f"alpha{i}" for i in range(100)),
        "tool-2": " ".join(f"beta{i}" for i in range(100)),
    }
    messages = _multi_result_messages(texts)
    router = _FixedRouter("summary retained facts")
    store = _Store(fail_after=1)  # first store() succeeds, second raises

    result = sweep_history(
        messages, router=router, tokenizer=_Tokenizer(), compression_store=store
    )

    swept = [
        b["content"]
        for m in result.messages
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    ccr_marked = [c for c in swept if isinstance(c, str) and "<<ccr:" in c]
    untouched = [c for c in swept if isinstance(c, str) and "<<ccr:" not in c]
    assert len(ccr_marked) == 1, "first block swept and kept"
    assert len(untouched) == 1, "failing block left as original"
    assert result.swept_count == 1

def test_bust_floor_rerun_finds_candidates_never_priced_by_admitting_gain() -> None:
    """FINDING (design-intent question, flagged for judgement): when
    admission is structural (client_prefix_alive_fraction == 0, a rebase,
    or p_alive <= 0.05 -- handlers/anthropic.py L1728-1733), the gain-gate
    branch at L1734 is skipped entirely, so masking_gate_gain is NEVER
    computed for the wider mask_min_tokens_at_bust candidate set that
    replaces the original candidates at L1772-1782. This test shows the
    bust-floor set can contain candidates whose own priced gain, if it
    were computed, is negative (e.g. a write-multiplier cost that exceeds
    the small dT of many near-floor 60-token candidates with few expected
    remaining reads). Whether "the cache is already dead so price is moot"
    is correct design intent for the STRUCTURAL branch depends on the p_alive
    <= 0.05 case still forwarding a live prefix worth some future reads;
    that branch is not truly zero-value like the alive_fraction==0 case.
    Flagging for the fixer/design owner to confirm intent."""
    steady_floor = HybridModeConfig().mask_min_tokens
    bust_floor = HybridModeConfig().mask_min_tokens_at_bust
    text = " ".join(f"token{i}" for i in range(75))  # ~75 tokens: between floors
    messages = _old_result_messages(text)

    steady_candidates = discover_candidates(
        messages, count_tokens=_Tokenizer().count_text, mask_after_turns=1,
        mask_min_tokens=steady_floor,
    )
    bust_candidates = discover_candidates(
        messages, count_tokens=_Tokenizer().count_text, mask_after_turns=1,
        mask_min_tokens=bust_floor,
    )
    assert steady_candidates == []
    assert len(bust_candidates) == 1

    # Price the bust-floor candidate with a low expected-reads, low-p_alive,
    # 1h write multiplier: a plausible near-death-prefix scenario.
    from headroom.transforms.compression_policy import CompressionPolicy

    policy = CompressionPolicy(
        live_zone_only=False,
        cache_aligner_enabled=False,
        volatile_token_threshold=0,
        max_lossy_ratio=1.0,
        toin_read_only=False,
    )
    gain = masking_gate_gain(
        bust_candidates,
        compression_policy=policy,
        suffix_tokens=50_000,
        expected_reads=0.3,
        p_alive=0.04,
        write_multiplier=2.0,
    )
    # The candidate this test constructs prices negative under realistic
    # near-dead-prefix economics, yet the structural admission path in
    # anthropic.py applies bust-floor candidates unconditionally without
    # ever calling masking_gate_gain on them.
    assert gain < 0.0


# ---------------------------------------------------------------------------
# 5. sweep_assistant_text: turn-age boundary and the mimicry surface
# ---------------------------------------------------------------------------


def _messages_with_assistant_text_age(age_turns_after: int, *, block_content: bool) -> list[dict[str, object]]:
    """Build a transcript where the target assistant text message is
    followed by exactly `age_turns_after` further assistant turns."""
    messages: list[dict[str, object]] = []
    target_text = " ".join(f"assistantword{i}" for i in range(100))
    if block_content:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        )
    else:
        messages.append({"role": "assistant", "content": target_text})
    for i in range(age_turns_after):
        messages.append({"role": "user", "content": f"reply{i}"})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"turn{i}"}]})
    return messages


@pytest.mark.parametrize("block_content", [False, True])
def test_sweep_assistant_text_age_boundary_exactly_three(block_content: bool) -> None:
    router = _FixedRouter("summary retained facts")

    # age 2: assistant_turns - assistant_turn - 1 == 2 -> NOT eligible (< 3)
    messages_age2 = _messages_with_assistant_text_age(2, block_content=block_content)
    result_age2 = sweep_history(
        messages_age2, router=router, tokenizer=_Tokenizer(), compression_store=_Store(),
        sweep_assistant_text=True,
    )
    assert result_age2.swept_count == 0, "age 2 (below the 3-turn floor) must not sweep"

    # age 3: assistant_turns - assistant_turn - 1 == 3 -> eligible (>= 3)
    messages_age3 = _messages_with_assistant_text_age(3, block_content=block_content)
    result_age3 = sweep_history(
        messages_age3, router=router, tokenizer=_Tokenizer(), compression_store=_Store(),
        sweep_assistant_text=True,
    )
    assert result_age3.swept_count == 1, "age exactly 3 must sweep (boundary is inclusive)"

    # age 4: comfortably eligible
    messages_age4 = _messages_with_assistant_text_age(4, block_content=block_content)
    result_age4 = sweep_history(
        messages_age4, router=router, tokenizer=_Tokenizer(), compression_store=_Store(),
        sweep_assistant_text=True,
    )
    assert result_age4.swept_count == 1


def test_sweep_assistant_text_plants_ccr_marker_in_assistant_authored_position() -> None:
    """FINDING (judgement: same authorship class as the mimicry incident,
    severity flagged for review): the 2026-07-17 mimicry incident
    (hybrid_mode.py L50-53) was about masked MODEL-AUTHORED content
    teaching the model to fabricate markers, because masking plants marker
    text in a position the model can imitate on its next turn. Tool
    RESULTS are documented as safe because they are environment-authored
    (observation_masking.py module docstring L19-27). Assistant TEXT is
    model-authored, the exact class the docstring calls out as unsafe for
    tool_use inputs. sweep_assistant_text defaults to False specifically
    for this reason, but when a caller opts in, sweep_history plants a
    literal '<<ccr:...>>' marker as the tail of an assistant-authored
    message with no distinguishing signal from a tool-result marker, and
    (from reading handlers/anthropic.py L1660-1686) the response-side
    marker guard's buffering gate is conditioned on mask_tool_inputs /
    tool_use blocks, not on assistant text blocks: whether the guard
    validates a swept assistant-text marker on the way back out is unclear
    from the request-side code alone and should be confirmed against
    headroom/ccr/response_handler.py before shipping sweep_assistant_text=1
    to any tier."""
    messages = _messages_with_assistant_text_age(3, block_content=False)
    router = _FixedRouter("summary retained facts")
    result = sweep_history(
        messages, router=router, tokenizer=_Tokenizer(), compression_store=_Store(),
        sweep_assistant_text=True,
    )
    assert result.swept_count == 1
    swept_message = result.messages[0]
    assert swept_message["role"] == "assistant"
    assert "<<ccr:" in swept_message["content"]


# ---------------------------------------------------------------------------
# 6. Unicode: surrogates, zero-width, RTL
# ---------------------------------------------------------------------------


def test_sweep_history_unpaired_surrogate_is_skipped_not_crashed() -> None:
    original = "prefix " * 60 + "\ud83d" + " suffix " * 60  # unpaired high surrogate
    messages = _old_result_messages(original)
    router = _FixedRouter("summary retained facts")

    result = sweep_history(
        messages, router=router, tokenizer=_Tokenizer(), compression_store=_Store()
    )
    assert result.swept_count == 0
    # Original content is untouched, not corrupted.
    content = messages[1]["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    assert block["content"] == original


def test_sweep_history_zero_width_and_rtl_content_round_trips() -> None:
    zwsp = "​"
    rtl_word = "אבג"  # Hebrew aleph-bet-gimel
    rtl_override = "‮"
    original = zwsp.join(
        [f"line{i}{rtl_word}{rtl_override}filler text padding out this record" for i in range(200)]
    )
    messages = _old_result_messages(original)
    router = _FixedRouter("summary retained facts" + zwsp)

    result = sweep_history(
        messages, router=router, tokenizer=_Tokenizer(), compression_store=_Store()
    )
    assert result.swept_count == 1
    swept_content = result.messages[1]["content"][0]["content"]
    assert swept_content.startswith("summary retained facts")
    assert "<<ccr:" in swept_content
