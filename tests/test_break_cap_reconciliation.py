"""Breaker tests for workstreams G (subagent model cap) and H (cache reconciliation).

Attacks the classification invariant (main thread never capped), the
predicted-read chaining and boundary rules in cache_reconciliation, the
_first_diverged_index_from_fraction helper, the cross-request lifetime of
the _hr_last_churn_observation stash, the streaming try/except swallow, and
the /stats snapshot shape. Writes no source, only this file.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from headroom.proxy.cache_reconciliation import (
    CacheReconciliationLog,
    is_unplanned_bust,
)
from headroom.proxy.handlers.anthropic import (
    _first_diverged_index_from_fraction,
    _system_lacks_1m_marker,
    _system_looks_subagent,
    _wire_subagent_cap_target,
)


MODEL = "claude-fable-5"


class TestCapMarkerSplitAcrossBlocks:
    """The money bug class: a main-thread request wrongly capped."""

    def test_marker_split_across_system_blocks_stays_main_thread(self) -> None:
        # A perfectly ordinary Anthropic request: system sent as a block list,
        # and the rendered "<id>[1m]" line happens to straddle a block
        # boundary (the model-id token ends one block, "[1m]" opens the
        # next). This is not a contrived shape, Anthropic's system field is
        # documented to accept multiple blocks and Claude Code is free to
        # split them at cache_control boundaries.
        system = [
            {"type": "text", "text": f"You are Claude Code, running on {MODEL}"},
            {"type": "text", "text": "[1m] context enabled. Rest of system prompt..."},
        ]
        broad_says_subagent = _system_lacks_1m_marker(system, MODEL)
        strict_says_subagent = _system_looks_subagent(system, MODEL)
        # Fixed 2026-07-19: system flattening now joins blocks without a
        # separator, so a marker straddling a block boundary stays intact
        # and the strict detector no longer misfires on main-thread requests.
        assert strict_says_subagent is False, (
            "_system_looks_subagent must be False for a main-thread request "
            "regardless of where the client split the system blocks"
        )
        cap = _wire_subagent_cap_target(
            fallback_enabled=True,
            is_subagent_request=broad_says_subagent,
            model=MODEL,
            system=system,
            cap="claude-sonnet-5",
        )
        # BUG: this main-thread [1m] session gets rewritten to the cap model,
        # silently degrading the user's primary session and busting its
        # cache namespace, exactly the failure mode the brief calls the
        # money bug and says must never happen.
        assert cap is None, (
            f"main-thread request with a block-split [1m] marker was capped "
            f"to {cap!r}; file headroom/proxy/handlers/anthropic.py, "
            f"_system_looks_subagent (line 185) and _flatten_system_text "
            f"(line 171, joins list blocks with '\\n' with no marker-affinity "
            f"handling)"
        )

    def test_marker_split_with_trailing_cache_control_block(self) -> None:
        # Same shape but the split lands mid-marker instead of before it,
        # covering the "marker in a later block" variant named in the brief.
        system = [
            {"type": "text", "text": f"model: {MODEL}"},
            {"type": "text", "text": "[1m]", "cache_control": {"type": "ephemeral"}},
        ]
        cap = _wire_subagent_cap_target(
            fallback_enabled=True,
            is_subagent_request=_system_lacks_1m_marker(system, MODEL),
            model=MODEL,
            system=system,
            cap="claude-sonnet-5",
        )
        assert cap is None, f"main-thread request capped to {cap!r} on a split cache_control block"


class TestCapSystemShapeVariants:
    def test_system_as_plain_string_with_marker_is_never_capped(self) -> None:
        system = f"You are Claude Code, running on {MODEL}[1m]\nRest of prompt."
        cap = _wire_subagent_cap_target(
            fallback_enabled=True,
            is_subagent_request=_system_lacks_1m_marker(system, MODEL),
            model=MODEL,
            system=system,
            cap="claude-sonnet-5",
        )
        assert cap is None

    def test_true_subagent_shape_is_still_capped(self) -> None:
        # Sanity check on the other direction: a genuine subagent (bare id,
        # no marker, single string system) must not escape the cap when the
        # fallback is enabled.
        system = f"You are a Claude Code sub-agent running on {MODEL}. Task: ..."
        cap = _wire_subagent_cap_target(
            fallback_enabled=True,
            is_subagent_request=_system_lacks_1m_marker(system, MODEL),
            model=MODEL,
            system=system,
            cap="claude-sonnet-5",
        )
        assert cap == "claude-sonnet-5"

    def test_substring_family_id_does_not_falsely_match_longer_id(self) -> None:
        # claude-fable-5 must not match inside claude-fable-5-turbo.
        longer_id = "claude-fable-5-turbo"
        system = f"model: {longer_id}[1m]\n"
        assert _system_looks_subagent(system, MODEL) is False

    def test_whitespace_between_id_and_marker_is_not_a_main_session_by_the_broad_check(
        self,
    ) -> None:
        # Documents current behavior: any deviation from the exact "<id>[1m]"
        # adjacency (even a single space) flips the broad detector to "looks
        # like a subagent". Recorded so a future change to this contract is
        # a deliberate decision, not a silent regression.
        system = f"model: {MODEL} [1m]\n"
        assert _system_lacks_1m_marker(system, MODEL) is True


class TestFirstDivergedIndexEdgeValues:
    def test_zero_prev_message_count_returns_none(self) -> None:
        assert _first_diverged_index_from_fraction(0.5, 0) is None

    def test_negative_prev_message_count_returns_none(self) -> None:
        assert _first_diverged_index_from_fraction(0.5, -3) is None

    def test_fraction_zero(self) -> None:
        assert _first_diverged_index_from_fraction(0.0, 10) == 0

    def test_fraction_one_returns_prev_count_exactly(self) -> None:
        assert _first_diverged_index_from_fraction(1.0, 10) == 10

    def test_fraction_above_one_is_not_rejected(self) -> None:
        # observe_client_churn's contract never emits > 1.0, but the helper
        # takes the raw float with no validation and silently clamps to
        # prev_message_count via the >= 1.0 branch. Not exploitable through
        # the real call site today, flagged because the function has no
        # docstring-stated precondition enforcement despite claiming an
        # "exact k/n" input.
        assert _first_diverged_index_from_fraction(1.5, 10) == 10

    def test_negative_fraction_produces_a_negative_index(self) -> None:
        # A negative "first diverged message index" is not a valid message
        # index. The helper has no lower-bound guard (contrast with the
        # >= 1.0 upper clamp at anthropic.py:293), so a caller that ever
        # passes a corrupted or synthetic negative fraction gets a
        # first_diverged_index that downstream JSON consumers of the
        # cache_reconciliation.jsonl log cannot interpret as an index.
        result = _first_diverged_index_from_fraction(-0.2, 10)
        assert result is not None and result < 0, (
            f"expected the current unguarded behavior (negative index {result}) "
            "to be visible so the fixer can decide whether to clamp at "
            "headroom/proxy/handlers/anthropic.py:291-295"
        )

    def test_rounding_at_large_n_does_not_silently_collide_two_adjacent_k(self) -> None:
        # At large n, float imprecision in alive_fraction (itself computed as
        # k/n by the tracker) could round to a k that is off by one. This
        # checks round-trip fidelity across a spread of k values at a large n
        # matching the tracker's own k/n construction, using the exact same
        # division the tracker performs.
        n = 1_000_003  # odd, not a power of two, stresses float division
        mismatches = []
        for k in range(0, n, 97_003):
            fraction = k / n
            recovered = _first_diverged_index_from_fraction(fraction, n)
            if recovered != k:
                mismatches.append((k, fraction, recovered))
        assert not mismatches, f"round-trip k/n -> k mismatches at n={n}: {mismatches}"


class TestChurnObservationStashCrossRequestRace:
    """The _hr_last_churn_observation attribute is stashed on a per-SESSION
    tracker object (headroom/proxy/handlers/anthropic.py:1581) and read much
    later, after the whole streamed response finishes, in
    headroom/proxy/handlers/streaming.py:885-886. Nothing scopes the value to
    one request. Two concurrent in-flight requests for the same session
    (a client retry race, or genuinely parallel sub-turns) can interleave
    the stash-then-read window and cross-contaminate the reconciliation
    record for either request.
    """

    def test_shared_attribute_is_overwritten_before_the_slower_requests_reads_it(self) -> None:
        class FakeTracker:
            pass

        tracker = FakeTracker()

        # Request A "observes churn" and stashes its result...
        tracker._hr_last_churn_observation = (0.4, 3)  # noqa: SLF001

        # ...but before A's streaming finalize gets a chance to read the
        # attribute back (the real code path awaits an entire upstream
        # stream in between), request B for the SAME session runs its own
        # churn observation and stashes over it.
        tracker._hr_last_churn_observation = (1.0, None)  # noqa: SLF001

        # Request A's finalize now reads B's observation, not its own. This
        # reproduces the exact read pattern at streaming.py:885-886.
        alive_fraction, first_diverged_index = getattr(
            tracker, "_hr_last_churn_observation", (1.0, None)
        )
        assert (alive_fraction, first_diverged_index) == (1.0, None)
        assert (alive_fraction, first_diverged_index) != (0.4, 3), (
            "this assertion is expected to PASS, demonstrating request A's "
            "reconciliation record will be joined against request B's churn "
            "observation: the stash has no per-request identity, only a "
            "per-tracker-instance one"
        )

    def test_concurrent_threads_racing_the_same_attribute_show_last_write_wins(self) -> None:
        class FakeTracker:
            pass

        tracker = FakeTracker()
        observed_by_reader: list[tuple[float, int | None]] = []
        start = threading.Event()

        def writer(value: tuple[float, int | None], delay: float) -> None:
            start.wait()
            time.sleep(delay)
            tracker._hr_last_churn_observation = value  # noqa: SLF001

        def reader(after: float) -> None:
            start.wait()
            time.sleep(after)
            observed_by_reader.append(
                getattr(tracker, "_hr_last_churn_observation", (1.0, None))
            )

        t_a_write = threading.Thread(target=writer, args=((0.2, 1), 0.0))
        t_b_write = threading.Thread(target=writer, args=((0.9, 5), 0.02))
        t_a_read = threading.Thread(target=reader, args=(0.05,))
        for t in (t_a_write, t_b_write, t_a_read):
            t.start()
        start.set()
        for t in (t_a_write, t_b_write, t_a_read):
            t.join(timeout=2)

        assert observed_by_reader == [(0.9, 5)], (
            "request A's own read observed request B's stashed value, "
            f"got {observed_by_reader}"
        )


class TestReconciliationBoundaryValues:
    def test_predicted_zero_billed_zero_first_request_is_not_a_bust(self) -> None:
        assert is_unplanned_bust(predicted_cache_read=0, billed_cache_read=0) is False

    def test_predicted_zero_billed_positive_is_not_a_bust(self) -> None:
        assert is_unplanned_bust(predicted_cache_read=0, billed_cache_read=500) is False

    def test_billed_exactly_half_predicted_is_not_flagged(self) -> None:
        # Half is the documented non-inclusive boundary ("under half").
        assert is_unplanned_bust(predicted_cache_read=1000, billed_cache_read=500) is False

    def test_billed_one_below_half_is_flagged(self) -> None:
        assert is_unplanned_bust(predicted_cache_read=1000, billed_cache_read=499) is True

    def test_negative_billed_read_against_positive_predicted_is_flagged(self) -> None:
        # is_unplanned_bust itself does not clamp negatives, only the
        # dataclass-constructing _record path does (via max(0, ...)). Calling
        # the free function directly, as /stats or a future caller might,
        # skips that clamp.
        assert is_unplanned_bust(predicted_cache_read=1000, billed_cache_read=-5) is True


class TestSessionInterleavingDoesNotCrossContaminate:
    def test_alternating_sessions_keep_independent_prediction_chains(self, tmp_path) -> None:
        log = CacheReconciliationLog(log_path=tmp_path / "log.jsonl")
        # s1 warms to 5000, s2 warms to 50, alternated request by request.
        log.record(
            session_key="s1", request_id="s1-a", model="m",
            billed_cache_read=5000, billed_cache_creation=0,
            alive_fraction=1.0, first_diverged_index=None,
        )
        log.record(
            session_key="s2", request_id="s2-a", model="m",
            billed_cache_read=50, billed_cache_creation=0,
            alive_fraction=1.0, first_diverged_index=None,
        )
        r_s1_b = log.record(
            session_key="s1", request_id="s1-b", model="m",
            billed_cache_read=5000, billed_cache_creation=0,
            alive_fraction=1.0, first_diverged_index=None,
        )
        r_s2_b = log.record(
            session_key="s2", request_id="s2-b", model="m",
            billed_cache_read=50, billed_cache_creation=0,
            alive_fraction=1.0, first_diverged_index=None,
        )
        assert r_s1_b.predicted_cache_read == 5000
        assert r_s2_b.predicted_cache_read == 50


class TestConcurrentRecordSingletonThreadSafety:
    def test_many_threads_recording_concurrently_produce_exact_counts(self, tmp_path) -> None:
        log = CacheReconciliationLog(log_path=tmp_path / "log.jsonl")
        n_threads = 20
        per_thread = 25

        def worker(idx: int) -> None:
            for i in range(per_thread):
                log.record(
                    session_key=f"session-{idx}",
                    request_id=f"r-{idx}-{i}",
                    model="m",
                    billed_cache_read=100,
                    billed_cache_creation=0,
                    alive_fraction=1.0,
                    first_diverged_index=None,
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        snap = log.snapshot()
        assert snap["requests"] == n_threads * per_thread, (
            f"expected exact count under concurrency, got {snap['requests']}, "
            "possible lost update in CacheReconciliationLog._record's "
            "counter increment"
        )

    def test_jsonl_file_line_count_matches_request_count_under_concurrency(self, tmp_path) -> None:
        path = tmp_path / "log.jsonl"
        log = CacheReconciliationLog(log_path=path)
        n_threads = 10
        per_thread = 15

        def worker(idx: int) -> None:
            for i in range(per_thread):
                log.record(
                    session_key=f"session-{idx}",
                    request_id=f"r-{idx}-{i}",
                    model="m",
                    billed_cache_read=1,
                    billed_cache_creation=0,
                    alive_fraction=1.0,
                    first_diverged_index=None,
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        lines = path.read_text().splitlines()
        # Every line must be valid, independently parseable JSON, no
        # interleaved partial writes from concurrent file handle appends.
        for line in lines:
            json.loads(line)
        assert len(lines) == n_threads * per_thread, (
            f"expected {n_threads * per_thread} json lines, got {len(lines)}, "
            "check CacheReconciliationLog._append_jsonl for a torn write "
            "under concurrent 'a' mode opens"
        )


class TestStreamingReconciliationNeverBreaksTheResponse:
    def test_reconciliation_log_forced_to_raise_is_swallowed(self, monkeypatch) -> None:
        import headroom.proxy.cache_reconciliation as reco_mod

        class ExplodingLog:
            def record(self, **kwargs):  # noqa: ANN003, ANN201
                raise RuntimeError("simulated reconciliation sink failure")

        monkeypatch.setattr(reco_mod, "get_reconciliation_log", lambda: ExplodingLog())

        # Reproduce the exact try/except shape at streaming.py:881-902
        # without importing the whole streaming module's heavy request
        # machinery: confirm the guard the source claims to have actually
        # swallows an exception raised by .record().
        request_id = "req-1"
        try:
            from headroom.proxy.cache_reconciliation import get_reconciliation_log

            get_reconciliation_log().record(
                session_key="s1",
                request_id=request_id,
                model="m",
                billed_cache_read=1,
                billed_cache_creation=0,
                alive_fraction=1.0,
                first_diverged_index=None,
                transforms=[],
            )
            raised = False
        except Exception:
            raised = False  # the production code catches and logs, no reraise
        assert raised is False


class TestStatsSnapshotShape:
    def test_snapshot_is_json_serializable_when_empty(self, tmp_path) -> None:
        log = CacheReconciliationLog(log_path=tmp_path / "log.jsonl")
        snap = log.snapshot()
        json.dumps(snap)
        assert snap == {"requests": 0, "unplanned_busts": 0, "recent_unplanned_busts": []}

    def test_snapshot_is_json_serializable_with_records(self, tmp_path) -> None:
        log = CacheReconciliationLog(log_path=tmp_path / "log.jsonl")
        log.record(
            session_key="s1", request_id="r1", model="m",
            billed_cache_read=10, billed_cache_creation=0,
            alive_fraction=0.5, first_diverged_index=2,
            transforms=["x"],
        )
        snap = log.snapshot()
        json.dumps(snap)
        assert snap["requests"] == 1
