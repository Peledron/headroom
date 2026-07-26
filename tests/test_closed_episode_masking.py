"""A finished task's tool results stop waiting out the turn-age gate.

The age gate is a proxy for "the model has moved on". A new user ask is the
direct evidence: whatever the previous task produced is finished work, however
recent it is in turns. Masking it then is the cheap half of compaction, one
thread at a time as it goes cold, rather than the whole history at once when
the context window fills.

These pin what counts as a closed task, and the two things the relaxation is
not allowed to touch: the size floor, and tool inputs (the 2026-07-17 marker
mimicry incident).
"""

from __future__ import annotations

from typing import Any

from headroom.transforms.observation_masking import (
    closed_episode_turn,
    discover_candidates,
)


def _count(text: str) -> int:
    return max(1, len(text) // 4)


BIG = "x" * 4000


def _assistant_call(tool_use_id: str, command: str = "ls") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "Bash",
                "input": {"command": command},
            }
        ],
    }


def _tool_result(tool_use_id: str, text: str = BIG) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
        ],
    }


def _ask(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _finished_task_then_new_ask() -> list[dict[str, Any]]:
    return [
        _ask("first task"),
        _assistant_call("call-1"),
        _tool_result("call-1"),
        {"role": "assistant", "content": "done with the first task"},
        _ask("now do something else"),
    ]


class TestClosedEpisodeTurn:
    def test_a_new_user_ask_closes_the_task_before_it(self):
        assert closed_episode_turn(_finished_task_then_new_ask()) == 2

    def test_tool_results_coming_back_close_nothing(self):
        messages = [
            _ask("first task"),
            _assistant_call("call-1"),
            _tool_result("call-1"),
        ]
        assert closed_episode_turn(messages) is None

    def test_the_opening_ask_has_no_task_behind_it(self):
        assert closed_episode_turn([_ask("first task")]) is None

    def test_no_user_message_at_all(self):
        assert closed_episode_turn([{"role": "assistant", "content": "hi"}]) is None

    def test_an_empty_conversation(self):
        assert closed_episode_turn([]) is None


class TestAgeGateRelaxation:
    def test_a_recent_result_waits_out_the_gate_mid_task(self):
        messages = _finished_task_then_new_ask()[:-1]  # no new ask yet
        found = discover_candidates(
            messages, count_tokens=_count, mask_after_turns=3, mask_min_tokens=100
        )
        assert found == []

    def test_the_same_result_is_eligible_once_the_task_closes(self):
        messages = _finished_task_then_new_ask()
        found = discover_candidates(
            messages,
            count_tokens=_count,
            mask_after_turns=3,
            mask_min_tokens=100,
            episode_closed_turn=closed_episode_turn(messages),
        )
        assert [c.tool_use_id for c in found] == ["call-1"]

    def test_without_the_boundary_the_gate_still_holds(self):
        messages = _finished_task_then_new_ask()
        found = discover_candidates(
            messages, count_tokens=_count, mask_after_turns=3, mask_min_tokens=100
        )
        assert found == []

    def test_the_size_floor_is_not_relaxed(self):
        messages = _finished_task_then_new_ask()
        messages[2] = _tool_result("call-1", "tiny")
        found = discover_candidates(
            messages,
            count_tokens=_count,
            mask_after_turns=3,
            mask_min_tokens=100,
            episode_closed_turn=closed_episode_turn(messages),
        )
        assert found == []

    def test_work_after_the_boundary_still_waits(self):
        """Only the closed task is freed; the live one keeps its age gate.

        The boundary is passed in rather than derived: mid-task the derivation
        answers None (the test below pins that), so the two rules have to be
        exercised separately to show the relaxation stops at the boundary.
        """
        messages = _finished_task_then_new_ask() + [
            _assistant_call("call-2"),
            _tool_result("call-2"),
        ]
        found = discover_candidates(
            messages,
            count_tokens=_count,
            mask_after_turns=3,
            mask_min_tokens=100,
            episode_closed_turn=2,
        )
        assert [c.tool_use_id for c in found] == ["call-1"]

    def test_mid_task_nothing_is_freed(self):
        """Work handed back mid-task closes no episode, so nothing is relaxed."""
        messages = _finished_task_then_new_ask() + [
            _assistant_call("call-2"),
            _tool_result("call-2"),
        ]
        assert closed_episode_turn(messages) is None

    def test_tool_inputs_never_ride_the_relaxation(self):
        """Marker text in a tool input is what caused the 2026-07-17 incident."""
        messages = _finished_task_then_new_ask()
        messages[1] = _assistant_call("call-1", command=BIG)
        found = discover_candidates(
            messages,
            count_tokens=_count,
            mask_after_turns=3,
            mask_min_tokens=100,
            mask_input_keys=("command",),
            episode_closed_turn=closed_episode_turn(messages),
        )
        assert all(c.input_key is None for c in found)
