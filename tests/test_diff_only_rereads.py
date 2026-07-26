"""Re-reads of an already-read file forward as a diff, not full content.

Measured 2026-07-25 over 1175 recorded turns: unmasked tool results at or
above 500 tokens are 36.2 percent of all genuinely appended content, and the
appended tail is what sets the read-to-write ceiling. A re-read whose file
moved by a few lines appends the whole file again. Sending the delta turns an
L-byte append into a d-byte one.

The base read must survive for the diff to mean anything, so these tests also
pin that the stale/superseded pass leaves it alone.
"""

from __future__ import annotations

from headroom.config import ReadLifecycleConfig
from headroom.transforms.read_lifecycle import ReadLifecycleManager, ReadState

PHRASE = "Retrieve original: hash="


def _read_call(tool_id: str, path: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": "Read", "input": {"file_path": path}}
        ],
    }


def _read_result(tool_id: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": text}],
    }


def _file(n_lines: int, changed_line: int | None = None) -> str:
    lines = []
    for i in range(1, n_lines + 1):
        body = "changed content here" if i == changed_line else f"original line body {i}"
        lines.append(f"{i:6d}\t{body}")
    return "\n".join(lines)


def _conversation(before: str, after: str) -> list[dict]:
    return [
        {"role": "user", "content": "open it"},
        _read_call("call_a", "/repo/app.py"),
        _read_result("call_a", before),
        {"role": "assistant", "content": "read it"},
        {"role": "user", "content": "open it again"},
        _read_call("call_b", "/repo/app.py"),
        _read_result("call_b", after),
    ]


def _manager(**overrides) -> ReadLifecycleManager:
    config = ReadLifecycleConfig(enabled=True, diff_rereads=True, **overrides)
    return ReadLifecycleManager(config)


def _result_text(messages: list[dict], tool_id: str) -> str:
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("tool_use_id") == tool_id:
                return block.get("content", "")
    raise AssertionError(f"no tool_result for {tool_id}")


def test_a_one_line_change_forwards_as_a_diff():
    """The whole point: a 200-line file that moved one line appends a few lines."""
    before = _file(200)
    after = _file(200, changed_line=97)
    result = _manager().apply(_conversation(before, after))

    assert result.reads_diffed == 1
    forwarded = _result_text(result.messages, "call_b")
    assert "changed content here" in forwarded, (
        "The diff must carry the new bytes, otherwise the model cannot see the change."
    )
    assert len(forwarded) < len(after) // 2
    assert "original line body 5" not in forwarded, (
        "Unchanged regions are the saving. If they survive, nothing was saved."
    )


def test_b_the_diff_base_is_not_masked_away():
    """A diff against bytes that were replaced by a marker is unusable."""
    before = _file(200)
    after = _file(200, changed_line=97)
    result = _manager().apply(_conversation(before, after))

    base = _result_text(result.messages, "call_a")
    assert base == before, (
        "The earlier read is the diff base. Superseding it deletes the only "
        "copy of the bytes the diff is expressed against."
    )


def test_c_the_full_text_stays_retrievable():
    """The model must be able to recover the real file if the diff is not enough."""
    before = _file(200)
    after = _file(200, changed_line=97)
    result = _manager().apply(_conversation(before, after))

    forwarded = _result_text(result.messages, "call_b")
    assert PHRASE in forwarded
    assert len(result.ccr_hashes) == 1
    assert result.ccr_hashes[0] in forwarded


def test_d_a_rewritten_file_falls_through_to_full_content():
    """A diff the size of the file is worse than the file: more bytes, less clear."""
    before = _file(120)
    after = "\n".join(f"{i:6d}\tcompletely different {i}" for i in range(1, 121))
    result = _manager().apply(_conversation(before, after))

    assert result.reads_diffed == 0
    assert _result_text(result.messages, "call_b") == after


def test_e_the_flag_defaults_off():
    """This rewrites content the model reasons against. Off until piloted."""
    assert ReadLifecycleConfig().diff_rereads is False

    before = _file(200)
    after = _file(200, changed_line=97)
    config = ReadLifecycleConfig(enabled=True)
    result = ReadLifecycleManager(config).apply(_conversation(before, after))
    assert result.reads_diffed == 0


def test_f_only_the_newest_read_is_diffed():
    """A diff against a diff cannot be reconstructed.

    Three reads of one file: the middle one must stay verbatim so it can serve
    as a base, and only the last may become a diff.
    """
    v1 = _file(200)
    v2 = _file(200, changed_line=97)
    v3 = _file(200, changed_line=150)
    messages = _conversation(v1, v2) + [
        {"role": "assistant", "content": "once more"},
        _read_call("call_c", "/repo/app.py"),
        _read_result("call_c", v3),
    ]
    result = _manager().apply(messages)

    assert result.reads_diffed == 1
    assert PHRASE not in _result_text(result.messages, "call_b"), (
        "The base of the newest diff must be literal text, not another marker."
    )
    assert PHRASE in _result_text(result.messages, "call_c")


def test_g_a_partial_read_is_not_diffed_against_a_different_window():
    """Different offsets are different slices. Their delta is not a file change."""
    manager = _manager()
    messages = [
        {"role": "user", "content": "head then tail"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_a",
                    "name": "Read",
                    "input": {"file_path": "/repo/app.py", "offset": 1, "limit": 100},
                }
            ],
        },
        _read_result("call_a", _file(100)),
        {"role": "assistant", "content": "now the tail"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_b",
                    "name": "Read",
                    "input": {"file_path": "/repo/app.py", "offset": 400, "limit": 100},
                }
            ],
        },
        _read_result("call_b", _file(100, changed_line=50)),
    ]
    result = manager.apply(messages)
    assert result.reads_diffed == 0


def test_h_a_tiny_reread_is_left_alone():
    """Below min_size_bytes the marker prose costs more than the content."""
    result = _manager(min_size_bytes=100_000).apply(_conversation(_file(200), _file(200, 97)))
    assert result.reads_diffed == 0


def test_i_an_unchanged_reread_is_left_to_the_superseded_path():
    """Byte-identical repeats measure at 0.1 percent. The existing path owns them."""
    same = _file(200)
    result = _manager().apply(_conversation(same, same))
    assert result.reads_diffed == 0


def test_j_a_stale_base_still_serves_as_a_diff_base():
    """An edit between the reads makes the base wrong, not unusable.

    The model reconstructs current = base + diff, and the diff carries the
    edit. Suppressing the stale marker here is the price of the append saving.
    """
    before = _file(200)
    after = _file(200, changed_line=97)
    messages = [
        {"role": "user", "content": "open it"},
        _read_call("call_a", "/repo/app.py"),
        _read_result("call_a", before),
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_edit",
                    "name": "Edit",
                    "input": {
                        "file_path": "/repo/app.py",
                        "old_string": "original line body 97",
                        "new_string": "changed content here",
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_edit", "content": "ok"}],
        },
        _read_call("call_b", "/repo/app.py"),
        _read_result("call_b", after),
    ]
    result = _manager().apply(messages)

    assert result.reads_diffed == 1
    assert _result_text(result.messages, "call_a") == before
    assert ReadState.STALE.value not in _result_text(result.messages, "call_a")
