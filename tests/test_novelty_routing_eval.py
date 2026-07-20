"""Tests for headroom.evals.novelty_routing_eval.

Every fixture is synthetic JSONL content written to pytest tmp_path, built
to match the schema documented in the module and the orchestrating task's
investigation notes. No test reads real ~/.claude/projects data and no test
makes a network call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from headroom.evals import novelty_routing_eval as nre


def _write_session(root: Path, project: str, session_id: str, lines: list[dict]) -> Path:
    proj_dir = root / project
    proj_dir.mkdir(parents=True, exist_ok=True)
    path = proj_dir / f"{session_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    return path


def _tool_use(tc_id: str, name: str, input_: dict) -> dict:
    return {
        "type": "assistant",
        "isSidechain": False,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tc_id, "name": name, "input": input_}],
        },
    }


def _tool_result(tc_id: str, text: str, sidechain: bool = False) -> dict:
    return {
        "type": "user",
        "isSidechain": sidechain,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": text}],
        },
    }


def _assistant_text(text: str, sidechain: bool = False) -> dict:
    return {
        "type": "assistant",
        "isSidechain": sidechain,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


# ---------------------------------------------------------------------------
# Step 0: corpus survey gate
# ---------------------------------------------------------------------------


def test_smoke_test_only_flips_for_tiny_corpus(tmp_path):
    lines = [
        _tool_use("t1", "Read", {"file_path": "/a.py"}),
        _tool_result("t1", "small file contents"),
    ]
    _write_session(tmp_path, "proj1", "session1", lines)

    report = nre.run_eval(tmp_path, bootstrap_resamples=5)

    assert report["smoke_test_only"] is True
    assert report["corpus_survey"]["sessions"] == 1
    assert report["corpus_survey"]["tool_outputs"] == 1


def test_smoke_test_only_false_for_large_enough_corpus(tmp_path):
    for i in range(35):
        lines = []
        for j in range(30):
            tc_id = f"t{i}_{j}"
            lines.append(_tool_use(tc_id, "Read", {"file_path": f"/f{j}.py"}))
            lines.append(_tool_result(tc_id, f"contents of file {j} in session {i}"))
        _write_session(tmp_path, "proj1", f"session{i}", lines)

    report = nre.run_eval(tmp_path, bootstrap_resamples=5)

    assert report["smoke_test_only"] is False
    assert report["corpus_survey"]["sessions"] == 35
    assert report["corpus_survey"]["tool_outputs"] == 35 * 30


# ---------------------------------------------------------------------------
# Step 1: loader, sidechain and compaction boundary handling
# ---------------------------------------------------------------------------


def test_loader_separates_sidechain_from_main_thread(tmp_path):
    lines = [
        _tool_use("m1", "Read", {"file_path": "/main.py"}),
        _tool_result("m1", "main thread content"),
        _tool_use("s1", "Read", {"file_path": "/sub.py"}),
        _tool_result("s1", "sidechain content", sidechain=True),
    ]
    # Mark the sidechain tool_use as sidechain too.
    lines[2]["isSidechain"] = True
    path = _write_session(tmp_path, "proj1", "session1", lines)

    session = nre.load_session(path)
    tool_outputs = [e for e in session.events if e.kind == "tool_output"]
    assert len(tool_outputs) == 2
    main_outputs = [e for e in tool_outputs if not e.is_sidechain]
    side_outputs = [e for e in tool_outputs if e.is_sidechain]
    assert len(main_outputs) == 1
    assert len(side_outputs) == 1
    assert main_outputs[0].text == "main thread content"
    assert side_outputs[0].text == "sidechain content"


def test_loader_flags_compaction_boundary_variants(tmp_path):
    lines = [
        _tool_use("t1", "Read", {"file_path": "/a.py"}),
        _tool_result("t1", "content a"),
        {"type": "summary", "summary": "compacted so far"},
        {"type": "system", "subtype": "compact_boundary"},
        {
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "resuming"}],
                "isCompactSummary": True,
            },
        },
        _tool_use("t2", "Read", {"file_path": "/b.py"}),
        _tool_result("t2", "content b"),
    ]
    path = _write_session(tmp_path, "proj1", "session1", lines)

    session = nre.load_session(path)
    boundary_kinds = [e.kind for e in session.events if e.kind == "compaction_boundary"]
    assert len(boundary_kinds) == 3
    tool_outputs = [e for e in session.events if e.kind == "tool_output"]
    assert len(tool_outputs) == 2


# ---------------------------------------------------------------------------
# Step 2: labeler rules, each triggered independently
# ---------------------------------------------------------------------------


def test_rule_refetch_fires_and_is_main_thread_only(tmp_path):
    lines = [
        _tool_use("t1", "Read", {"file_path": "/a.py"}),
        _tool_result("t1", "content of a"),
        _assistant_text("looking at other things"),
        _tool_use("t2", "Read", {"file_path": "/a.py"}),
        _tool_result("t2", "content of a again"),
    ]
    path = _write_session(tmp_path, "proj1", "session1", lines)
    session = nre.load_session(path)
    labels = nre.label_session(session, {})

    first = labels[0]
    assert first.needed_later is True
    assert "refetch" in first.rules_fired


def test_rule_refetch_ignores_sidechain_refetch(tmp_path):
    lines = [
        _tool_use("t1", "Read", {"file_path": "/a.py"}),
        _tool_result("t1", "content of a"),
        _tool_use("t2", "Read", {"file_path": "/a.py"}),
        _tool_result("t2", "content of a again", sidechain=True),
    ]
    lines[2]["isSidechain"] = True
    path = _write_session(tmp_path, "proj1", "session1", lines)
    session = nre.load_session(path)
    labels = nre.label_session(session, {})

    main_output_labels = [
        lbl
        for lbl, event in zip(labels, [e for e in session.events if e.kind == "tool_output"])
        if not event.is_sidechain
    ]
    assert main_output_labels[0].needed_later is False


def test_rule_ccr_hash_retrieve_fires(tmp_path):
    lines = [
        _tool_use("t1", "Bash", {"command": "ls"}),
        _tool_result("t1", "12 items compressed into pointer, hash=deadbeef1234"),
        _tool_use("t2", "headroom_retrieve", {"key": "deadbeef1234"}),
        _tool_result("t2", "restored content"),
    ]
    path = _write_session(tmp_path, "proj1", "session1", lines)
    session = nre.load_session(path)
    labels = nre.label_session(session, {})

    first = labels[0]
    assert first.needed_later is True
    assert "ccr_retrieve" in first.rules_fired


def test_rule_ccr_hash_retrieve_does_not_fire_without_matching_call(tmp_path):
    lines = [
        _tool_use("t1", "Bash", {"command": "ls"}),
        _tool_result("t1", "12 items compressed into pointer, hash=deadbeef1234"),
        _tool_use("t2", "headroom_retrieve", {"key": "somethingelse"}),
        _tool_result("t2", "unrelated content"),
    ]
    path = _write_session(tmp_path, "proj1", "session1", lines)
    session = nre.load_session(path)
    labels = nre.label_session(session, {})

    assert labels[0].needed_later is False


def test_rule_rare_token_overlap_fires(tmp_path):
    lines = [
        _tool_use("t1", "Bash", {"command": "cat config"}),
        _tool_result("t1", "the setting zzznonceword12345 controls retry behavior"),
        _assistant_text("based on zzznonceword12345 and retry behavior, I'll change it"),
    ]
    path = _write_session(tmp_path, "proj1", "session1", lines)
    session = nre.load_session(path)
    # Force the shared tokens to be "rare" via an explicit doc-freq map.
    doc_freq = {"zzznonceword12345": 0.0001, "retry": 0.0001, "behavior": 0.0001}
    labels = nre.label_session(session, doc_freq)

    assert labels[0].needed_later is True
    assert "rare_token_overlap" in labels[0].rules_fired


def test_rule_rare_token_overlap_does_not_fire_for_common_tokens(tmp_path):
    lines = [
        _tool_use("t1", "Bash", {"command": "cat config"}),
        _tool_result("t1", "some common words here"),
        _assistant_text("some common words here too"),
    ]
    path = _write_session(tmp_path, "proj1", "session1", lines)
    session = nre.load_session(path)
    doc_freq = {"some": 0.9, "common": 0.9, "words": 0.9, "here": 0.9}
    labels = nre.label_session(session, doc_freq)

    assert labels[0].needed_later is False


# ---------------------------------------------------------------------------
# Step 3: audit flags
# ---------------------------------------------------------------------------


def test_rule_concentration_flag_triggers():
    labels = [
        nre.LabeledEvent(
            session_id="s1",
            event_index=i,
            normalized_target=f"read:/f{i}.py",
            needed_later=True,
            rules_fired=["refetch"],
            snippet="x",
        )
        for i in range(10)
    ]
    labels.append(
        nre.LabeledEvent(
            session_id="s1",
            event_index=10,
            normalized_target="read:/other.py",
            needed_later=True,
            rules_fired=["rare_token_overlap"],
            snippet="x",
        )
    )
    audit = nre.audit_labels(labels)
    assert audit["rule_concentration_flag"] is not None
    assert audit["rule_concentration_flag"]["rule"] == "refetch"


def test_rule_concentration_flag_absent_when_balanced():
    labels = []
    for i in range(5):
        labels.append(
            nre.LabeledEvent(
                session_id="s1",
                event_index=i,
                normalized_target=f"read:/f{i}.py",
                needed_later=True,
                rules_fired=["refetch"],
                snippet="x",
            )
        )
    for i in range(5, 10):
        labels.append(
            nre.LabeledEvent(
                session_id="s1",
                event_index=i,
                normalized_target=f"read:/g{i}.py",
                needed_later=True,
                rules_fired=["rare_token_overlap"],
                snippet="x",
            )
        )
    audit = nre.audit_labels(labels)
    assert audit["rule_concentration_flag"] is None


def test_degenerate_session_rate_flag():
    all_positive = [
        nre.LabeledEvent(
            session_id="all_pos",
            event_index=i,
            normalized_target="x",
            needed_later=True,
            rules_fired=["refetch"],
            snippet="x",
        )
        for i in range(5)
    ]
    all_negative = [
        nre.LabeledEvent(
            session_id="all_neg",
            event_index=i,
            normalized_target="x",
            needed_later=False,
            rules_fired=[],
            snippet="x",
        )
        for i in range(5)
    ]
    mixed = [
        nre.LabeledEvent(
            session_id="mixed",
            event_index=i,
            normalized_target="x",
            needed_later=(i % 2 == 0),
            rules_fired=["refetch"] if i % 2 == 0 else [],
            snippet="x",
        )
        for i in range(6)
    ]
    audit = nre.audit_labels(all_positive + all_negative + mixed)
    assert audit["degenerate_session_count"] == 2


# ---------------------------------------------------------------------------
# Step 4: baselines, including real dedup_blocks and ReadLifecycleManager calls
# ---------------------------------------------------------------------------


def _repetitive_session(tmp_path: Path) -> Path:
    """A session with an old, then-repeated, large tool output plus enough
    turns after it that mask-by-age would catch it, so the production
    baselines and the age baseline all have something to act on."""
    big_text = "\n".join(f"line {i} of a long file body" for i in range(30))
    lines = [
        _tool_use("t1", "Read", {"file_path": "/big.py"}),
        _tool_result("t1", big_text),
    ]
    for i in range(50):
        lines.append(_assistant_text(f"turn {i} filler"))
    lines.append(_tool_use("t2", "Read", {"file_path": "/big.py"}))
    lines.append(_tool_result("t2", big_text))
    return _write_session(tmp_path, "proj1", "session1", lines)


def test_baselines_produce_sane_metrics(tmp_path):
    path = _repetitive_session(tmp_path)
    session = nre.load_session(path)
    labels = nre.label_session(session, {})
    labels_by_session = {session.session_id: {lbl.event_index: lbl.needed_later for lbl in labels}}

    results = nre.run_baselines([session], labels_by_session, bootstrap_resamples=20)
    names = {b.name for b in results}
    assert "mask_nothing" in names
    assert "mask_by_age_5" in names
    assert "random_matched_rate" in names
    assert "production_dedup" in names
    assert "production_read_lifecycle" in names

    mask_nothing = next(b for b in results if b.name == "mask_nothing")
    assert mask_nothing.mask_rate == 0.0
    assert mask_nothing.per_1000_point == 0.0

    for b in results:
        assert 0.0 <= b.mask_rate <= 1.0
        assert b.per_1000_point >= 0.0


def test_production_dedup_baseline_actually_masks_repeated_block(tmp_path):
    path = _repetitive_session(tmp_path)
    session = nre.load_session(path)
    masked = nre._mask_by_dedup(session)
    # The second, verbatim-repeated big Read output should be folded by the
    # real dedup_blocks() call.
    tool_outputs = [e for e in session.events if e.kind == "tool_output"]
    assert len(masked) >= 1
    assert tool_outputs[-1].index in masked


def test_production_read_lifecycle_baseline_runs_without_crashing(tmp_path):
    path = _repetitive_session(tmp_path)
    session = nre.load_session(path)
    # Just confirm this calls the real ReadLifecycleManager end to end and
    # returns a set of indices (possibly empty) rather than raising.
    masked = nre._mask_by_read_lifecycle(session)
    assert isinstance(masked, set)


def test_bootstrap_ci_shape_and_session_granularity(tmp_path):
    sessions = []
    for i in range(6):
        lines = [
            _tool_use(f"t{i}", "Read", {"file_path": f"/f{i}.py"}),
            _tool_result(f"t{i}", f"content {i}"),
            _tool_use(f"t{i}b", "Read", {"file_path": f"/f{i}.py"}),
            _tool_result(f"t{i}b", f"content {i} again"),
        ]
        path = _write_session(tmp_path, "proj1", f"session{i}", lines)
        sessions.append(nre.load_session(path))

    rare_doc_freq = nre._rare_token_doc_freq(sessions)
    labels_by_session = {}
    for session in sessions:
        labels = nre.label_session(session, rare_doc_freq)
        labels_by_session[session.session_id] = {lbl.event_index: lbl.needed_later for lbl in labels}

    masked_by_session = {s.session_id: nre._mask_by_age(s, 0) for s in sessions}
    point = nre._per_1000_needed_but_masked(sessions, labels_by_session, masked_by_session)
    ci_low, ci_high = nre._bootstrap_ci(
        sessions, labels_by_session, masked_by_session, seed=123, resamples=200
    )

    assert ci_low <= point <= ci_high


# ---------------------------------------------------------------------------
# --with-embeddings off means no eager import of the embedding backend
# ---------------------------------------------------------------------------


def test_module_does_not_import_embedding_backend_at_top_level():
    assert "headroom.relevance.embedding" not in sys.modules


def test_run_eval_works_without_embeddings_module_available(tmp_path, monkeypatch):
    lines = [
        _tool_use("t1", "Read", {"file_path": "/a.py"}),
        _tool_result("t1", "small file contents"),
    ]
    _write_session(tmp_path, "proj1", "session1", lines)

    # Simulate the embeddings backend being unimportable, steps 0-4 must
    # still run cleanly since they never touch it.
    monkeypatch.setitem(sys.modules, "headroom.relevance.embedding", None)
    report = nre.run_eval(tmp_path, bootstrap_resamples=5)
    assert "corpus_survey" in report
    assert "embeddings_extension" not in report


def test_embedding_stub_raises_or_marks_not_implemented():
    pytest.importorskip("headroom.relevance.embedding", reason="embedding module not present")
    result = nre._embedding_stub_report()
    assert result["implemented"] is False
    assert set(result["candidates"]) == {
        "candidate_1_cosine_similarity",
        "candidate_2_query_aware_variant",
        "candidate_3_comparison_plumbing",
        "candidate_4_verdict_section",
    }
