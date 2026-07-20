"""Chaos tests for the cache-safe effort routing change.

Targets: request_uses_prompt_caching false negatives (the money bug, a false
negative here means route_effort mutates output_config.effort on a request
that actually carries a cache_control breakpoint, busting the Anthropic
prompt cache), the byte-identical-body claim on the pinned path, effort
decision propagation edge cases, and audit counter integrity under
concurrency.
"""

from __future__ import annotations

import copy
import json
import threading
from typing import Any

import pytest

from headroom.proxy.operational_audit import OperationalAudit
from headroom.proxy.output_shaper import (
    OutputShaperSettings,
    ShapeResult,
    TurnKind,
    request_uses_prompt_caching,
    route_effort,
    shape_request,
)

ENABLED = OutputShaperSettings(enabled=True)


def _mechanical_messages(extra_content: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    tool_result_content: Any = "ok"
    msgs = [
        {"role": "user", "content": "fix the bug in foo.py"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading the file."},
                {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01", "content": tool_result_content}
            ],
        },
    ]
    if extra_content is not None:
        msgs[-1]["content"] = extra_content
    return msgs


# ---------------------------------------------------------------------------
# request_uses_prompt_caching: hunt false negatives
# ---------------------------------------------------------------------------


class TestFalseNegatives:
    def test_none_body_does_not_raise(self):
        # Defensive contract per brief: "never raises". None is not a dict.
        assert request_uses_prompt_caching(None) is False  # type: ignore[arg-type]

    def test_tools_none_does_not_raise_and_is_false(self):
        assert request_uses_prompt_caching({"tools": None}) is False

    def test_tools_containing_none_entries_does_not_raise(self):
        body = {"tools": [None, {"name": "x", "cache_control": {"type": "ephemeral"}}]}
        assert request_uses_prompt_caching(body) is True

    def test_system_none_does_not_raise(self):
        assert request_uses_prompt_caching({"system": None}) is False

    def test_system_dict_form_is_detected(self):
        # A single system block given as a bare dict (not wrapped in a list)
        # is not a documented shape, but a client or intermediate proxy could
        # send it. The detector accepts it so a breakpoint there still pins.
        body = {"system": {"type": "text", "text": "Sys.", "cache_control": {"type": "ephemeral"}}}
        assert request_uses_prompt_caching(body) is True

    def test_messages_none_does_not_raise(self):
        assert request_uses_prompt_caching({"messages": None}) is False

    def test_message_none_entry_does_not_raise(self):
        body = {"messages": [None, {"role": "user", "content": "hi"}]}
        assert request_uses_prompt_caching(body) is False

    def test_content_block_none_entry_does_not_raise(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        None,
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                    ],
                }
            ]
        }
        assert request_uses_prompt_caching(body) is True

    def test_cache_control_value_none_still_counts_as_present(self):
        # Presence of the key is the documented signal, not truthiness of
        # the value. A client (or a buggy upstream rewrite) sending an
        # explicit null must still be treated as a live breakpoint.
        body = {"tools": [{"name": "x", "cache_control": None}]}
        assert request_uses_prompt_caching(body) is True

    def test_cache_control_on_tool_result_block_itself(self):
        body = {
            "messages": _mechanical_messages(
                extra_content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "ok",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            )
        }
        assert request_uses_prompt_caching(body) is True

    def test_cache_control_nested_inside_tool_result_content_list_is_detected(self):
        # tool_result blocks can themselves carry a "content" array of
        # sub-blocks (text/image). Anthropic's public examples place
        # cache_control on the outer block, but nothing stops a client
        # library from using the nested placement, so the detector must
        # look one level deep. A false negative here lets route_effort
        # mutate effort on a cached request and bust the full prefix.
        body = {
            "messages": _mechanical_messages(
                extra_content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": [
                            {
                                "type": "text",
                                "text": "ok",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                ]
            )
        }
        assert request_uses_prompt_caching(body) is True

    def test_last_message_string_content_with_earlier_message_cached_still_true(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                    ],
                },
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "next turn"},
            ]
        }
        assert request_uses_prompt_caching(body) is True

    def test_extra_top_level_body_keys_are_ignored_safely(self):
        body = {
            "metadata": {"cache_control": {"type": "ephemeral"}},
            "tools": [{"name": "x"}],
        }
        # cache_control outside tools/system/messages is not a real
        # Anthropic cache breakpoint location, correctly ignored.
        assert request_uses_prompt_caching(body) is False


# ---------------------------------------------------------------------------
# Byte-identical body claim on the pinned path
# ---------------------------------------------------------------------------


class TestPinnedBodyByteIdentical:
    def test_route_effort_pinned_body_byte_identical(self):
        body = {
            "output_config": {"effort": "xhigh"},
            "tools": [{"name": "read_file", "cache_control": {"type": "ephemeral"}}],
            "messages": _mechanical_messages(),
            "thinking": {"type": "adaptive"},
        }
        before = json.dumps(body, sort_keys=True)
        labels, decision = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        after = json.dumps(body, sort_keys=True)
        assert decision == "pinned"
        assert labels == []
        assert before == after

    def test_shape_request_pinned_effort_only_body_byte_identical_with_steering_off(self):
        # Isolate the effort lever from the unrelated verbosity lever by
        # forcing level 0, so a body diff can only come from route_effort.
        body = {
            "output_config": {"effort": "xhigh"},
            "system": [
                {"type": "text", "text": "Sys.", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": _mechanical_messages(),
        }
        before = copy.deepcopy(body)
        result = shape_request(body, ENABLED, level_override=0)
        assert result.effort_decision == "pinned"
        assert body == before
        assert result.changed is False
        assert result.labels == []


# ---------------------------------------------------------------------------
# Effort decision propagation edge cases
# ---------------------------------------------------------------------------


class TestEffortDecisionEdgeCases:
    def test_output_config_absent_gives_none_decision(self):
        body: dict[str, Any] = {"messages": []}
        labels, decision = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert labels == []
        assert decision is None

    def test_output_config_wrong_type_string(self):
        body = {"output_config": "xhigh"}
        labels, decision = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert labels == []
        assert decision is None
        assert body["output_config"] == "xhigh"

    def test_output_config_wrong_type_list(self):
        body = {"output_config": ["xhigh"]}
        labels, decision = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert labels == []
        assert decision is None

    def test_effort_already_at_floor_with_cache_control_gives_none_not_pinned(self):
        # No lowering would apply regardless of caching, so the decision
        # must be None, not "pinned" -- "pinned" implies a suppressed edit.
        body = {
            "output_config": {"effort": "low"},
            "tools": [{"name": "x", "cache_control": {"type": "ephemeral"}}],
        }
        labels, decision = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert labels == []
        assert decision is None

    def test_unknown_effort_string_with_cache_control_gives_none_not_pinned(self):
        body = {
            "output_config": {"effort": "turbo"},
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        }
        labels, decision = route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert labels == []
        assert decision is None
        assert body["output_config"]["effort"] == "turbo"

    def test_non_mechanical_turn_with_cache_control_gives_none_not_pinned(self):
        body = {
            "output_config": {"effort": "xhigh"},
            "tools": [{"name": "x", "cache_control": {"type": "ephemeral"}}],
        }
        for kind in (TurnKind.NEW_USER_ASK, TurnKind.ERROR_CONTINUATION, TurnKind.UNKNOWN):
            labels, decision = route_effort(body, kind, ENABLED)
            assert labels == []
            assert decision is None

    def test_shape_result_default_effort_decision_is_none(self):
        assert ShapeResult().effort_decision is None


# ---------------------------------------------------------------------------
# Audit counters: concurrency and snapshot stability
# ---------------------------------------------------------------------------


class TestOperationalAuditEffortRouting:
    def test_unknown_action_is_ignored(self):
        audit = OperationalAudit()
        audit.record_effort_routing("bogus")
        snap = audit.snapshot()
        assert snap["effort_lowerings"] == 0
        assert snap["effort_pins"] == 0

    def test_empty_string_and_none_like_actions_ignored(self):
        audit = OperationalAudit()
        audit.record_effort_routing("")
        audit.record_effort_routing("Lowered")  # case-sensitive mismatch
        audit.record_effort_routing("PINNED")
        snap = audit.snapshot()
        assert snap["effort_lowerings"] == 0
        assert snap["effort_pins"] == 0

    def test_snapshot_keys_present_and_exact(self):
        audit = OperationalAudit()
        snap = audit.snapshot()
        assert "effort_lowerings" in snap
        assert "effort_pins" in snap
        assert snap["effort_lowerings"] == 0
        assert snap["effort_pins"] == 0

    def test_concurrent_recording_no_lost_updates(self):
        audit = OperationalAudit()
        n_threads = 16
        per_thread = 200

        def worker(action: str) -> None:
            for _ in range(per_thread):
                audit.record_effort_routing(action)

        threads = []
        for i in range(n_threads):
            action = "lowered" if i % 2 == 0 else "pinned"
            t = threading.Thread(target=worker, args=(action,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = audit.snapshot()
        expected_each = (n_threads // 2) * per_thread
        assert snap["effort_lowerings"] == expected_each
        assert snap["effort_pins"] == expected_each

    def test_snapshot_is_a_stable_copy_not_live_view(self):
        audit = OperationalAudit()
        audit.record_effort_routing("lowered")
        snap1 = audit.snapshot()
        audit.record_effort_routing("lowered")
        assert snap1["effort_lowerings"] == 1


# ---------------------------------------------------------------------------
# Legacy clamp removal: confirm nothing else references it
# ---------------------------------------------------------------------------


class TestLegacyClampFullyRemoved:
    def test_output_shaper_module_has_no_legacy_symbol(self):
        import headroom.proxy.output_shaper as shaper_mod

        assert not hasattr(shaper_mod, "LEGACY_THINKING_FLOOR")
        assert not hasattr(shaper_mod, "clamp_legacy_thinking_budget")
        assert "LEGACY_THINKING_FLOOR" not in shaper_mod.__all__

    def test_output_effort_policy_module_has_no_legacy_symbol(self):
        import headroom.proxy.output_effort_policy as policy_mod

        assert not hasattr(policy_mod, "LEGACY_THINKING_FLOOR")
        assert not hasattr(policy_mod, "clamp_legacy_thinking_budget")

    def test_thinking_budget_untouched_by_route_effort_on_mechanical_turn(self):
        # The legacy lever is gone entirely: thinking.budget_tokens must
        # survive route_effort unchanged regardless of value.
        body = {
            "output_config": {"effort": "xhigh"},
            "thinking": {"type": "enabled", "budget_tokens": 1},
        }
        route_effort(body, TurnKind.MECHANICAL_CONTINUATION, ENABLED)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
