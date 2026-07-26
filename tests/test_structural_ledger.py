"""Tests for the structural-state ledger built from tool_use/tool_result blocks."""

from __future__ import annotations

from headroom.proxy.structural_ledger import build_structural_ledger


def _tool_use(tool_id: str, name: str, **input_kwargs):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": input_kwargs}


def _tool_result(tool_id: str, text: str, *, is_error: bool = False):
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "is_error": is_error,
        "content": [{"type": "text", "text": text}],
    }


def test_file_created_then_deleted_tracks_last_operation():
    messages = [
        {"role": "user", "content": "build a thing"},
        {
            "role": "assistant",
            "content": [_tool_use("t1", "Write", file_path="/tmp/a.py", content="x")],
        },
        {"role": "user", "content": [_tool_result("t1", "File created")]},
        {
            "role": "assistant",
            "content": [
                _tool_use(
                    "t2", "Bash", command="rm /tmp/a.py"
                )
            ],
        },
        {"role": "user", "content": [_tool_result("t2", "removed")]},
    ]
    ledger = build_structural_ledger(messages)
    assert len(ledger.files) == 1
    assert ledger.files[0].path == "/tmp/a.py"
    assert ledger.files[0].last_operation == "write"
    # A later Edit on the same path should overwrite the earlier operation.
    messages.append(
        {
            "role": "assistant",
            "content": [_tool_use("t3", "Edit", file_path="/tmp/a.py")],
        }
    )
    ledger2 = build_structural_ledger(messages)
    assert ledger2.files[0].last_operation == "edit"
    assert ledger2.files[0].last_seen_turn == 5


def test_command_failing_then_passing_clears_unresolved_error():
    messages = [
        {"role": "user", "content": "run tests"},
        {"role": "assistant", "content": [_tool_use("c1", "Bash", command="pytest")]},
        {"role": "user", "content": [_tool_result("c1", "exit code: 1\nFAILED test_x")]},
        {"role": "assistant", "content": [_tool_use("c2", "Bash", command="pytest")]},
        {"role": "user", "content": [_tool_result("c2", "exit code: 0\nall passed")]},
    ]
    ledger = build_structural_ledger(messages)
    assert len(ledger.commands) == 2
    assert ledger.commands[0].exit_signal == "1"
    assert ledger.commands[0].is_error is True
    assert ledger.commands[1].exit_signal == "0"
    assert ledger.commands[1].is_error is False
    # The passing rerun clears the unresolved error for that source.
    assert ledger.unresolved_errors == []


def test_error_with_no_later_success_stays_unresolved():
    messages = [
        {"role": "user", "content": "deploy it"},
        {"role": "assistant", "content": [_tool_use("c1", "Bash", command="deploy.sh")]},
        {"role": "user", "content": [_tool_result("c1", "Traceback: something broke")]},
    ]
    ledger = build_structural_ledger(messages)
    assert len(ledger.unresolved_errors) == 1
    assert ledger.unresolved_errors[0].source == "Bash"
    assert "broke" in ledger.unresolved_errors[0].text


def test_newest_user_task_is_verbatim_last_line():
    messages = [
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second, more specific task"},
    ]
    ledger = build_structural_ledger(messages)
    assert ledger.newest_user_task == "second, more specific task"


def test_render_is_deterministic_and_non_empty_only_with_data():
    empty_ledger = build_structural_ledger([])
    assert empty_ledger.render() == "structural_ledger: empty"

    messages = [
        {"role": "user", "content": "task line"},
        {"role": "assistant", "content": [_tool_use("t1", "Write", file_path="/tmp/x")]},
        {"role": "user", "content": [_tool_result("t1", "ok")]},
    ]
    first = build_structural_ledger(messages).render()
    second = build_structural_ledger(messages).render()
    assert first == second
    assert "task line" in first
    assert "/tmp/x" in first


def test_ledger_render_wired_only_into_the_structural_bust_branch():
    """The render call must live inside the bust conditional, not run
    unconditionally. Anything looser risks a log call that fires every
    request (log-only workstreams must never sit on the hot path)."""
    import inspect

    from headroom.proxy.handlers import anthropic as anthropic_handler

    source = inspect.getsource(anthropic_handler)
    calls = source.count("structural_ledger.build_structural_ledger(")
    assert calls == 1, "expected exactly one call site for the ledger render"

    bust_log_index = source.index('"[%s] STRUCTURAL-BUST: alive_fraction=%.2f forcing fresh %s write"')
    call_index = source.index("structural_ledger.build_structural_ledger(")

    # The render call sits shortly after the bust log line (same
    # conditional block), not scattered elsewhere in the handler.
    assert bust_log_index < call_index < bust_log_index + 2000


def test_non_command_tool_error_is_tracked_and_clearable():
    messages = [
        {"role": "user", "content": "read a file"},
        {"role": "assistant", "content": [_tool_use("r1", "Read", file_path="/tmp/missing")]},
        {"role": "user", "content": [_tool_result("r1", "Error: file not found", is_error=True)]},
    ]
    ledger = build_structural_ledger(messages)
    assert len(ledger.unresolved_errors) == 1

    messages.append(
        {"role": "assistant", "content": [_tool_use("r2", "Read", file_path="/tmp/missing")]}
    )
    messages.append({"role": "user", "content": [_tool_result("r2", "contents here")]})
    ledger2 = build_structural_ledger(messages)
    assert ledger2.unresolved_errors == []
