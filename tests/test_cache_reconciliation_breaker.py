"""Adversarial tests for headroom/proxy/cache_reconciliation.py.

Targets the module's own stated hard contract: log-only, never raises on the
hot request path. Also probes _session_head/_session_prior for unbounded
growth and for state corruption on partial failure. Does not modify
cache_reconciliation.py or any existing test. Some tests below are expected
to fail: they encode the contract the module claims and let the real
behaviour disagree with it, per the breaker mandate. Each such test says so
in its docstring.
"""

from __future__ import annotations

import threading
from pathlib import Path

from headroom.proxy.cache_reconciliation import (
    MAX_TRACKED_SESSIONS,
    CacheReconciliationLog,
    head_fingerprints,
)


def _log(tmp_path: Path, **kwargs) -> tuple[CacheReconciliationLog, Path]:
    path = tmp_path / "cache_reconciliation.jsonl"
    return CacheReconciliationLog(log_path=path, **kwargs), path


# ---------------------------------------------------------------------------
# Section 2: the never-raises contract on head_fingerprints / _fingerprint.
# ---------------------------------------------------------------------------


class TestHeadFingerprintsNeverRaises:
    def test_self_referential_dict_in_tools(self) -> None:
        circular: dict = {}
        circular["self"] = circular
        tools_fp, system_fp, count = head_fingerprints({"tools": [circular]})
        assert count == 1
        assert tools_fp is None  # json cannot encode it, degrades cleanly

    def test_self_referential_list_in_system(self) -> None:
        circular: list = []
        circular.append(circular)
        tools_fp, system_fp, count = head_fingerprints({"system": circular})
        assert system_fp is None

    def test_deeply_nested_structure_does_not_raise(self) -> None:
        deep: dict = {}
        cur = deep
        for _ in range(50_000):
            cur["x"] = {}
            cur = cur["x"]
        # Must not raise RecursionError out of head_fingerprints, and must not
        # hang. A degraded (None) fingerprint is an acceptable outcome.
        tools_fp, _, count = head_fingerprints({"tools": [deep]})
        assert count == 1

    def test_object_with_raising_str_degrades_to_none(self) -> None:
        class Bad:
            def __str__(self) -> str:
                raise ValueError("nope")

        tools_fp, _, count = head_fingerprints({"tools": [Bad()]})
        assert count == 1
        assert tools_fp is None

    def test_unicode_surrogate_in_tool_name_does_not_raise(self) -> None:
        tools_fp, _, count = head_fingerprints({"tools": [{"name": "\udc80\ud800bad"}]})
        assert count == 1
        assert isinstance(tools_fp, str)

    def test_tools_is_not_a_list(self) -> None:
        for bad_tools in ("not-a-list", {"a": 1}, 12345, 3.14, True, b"bytes"):
            tools_fp, _, count = head_fingerprints({"tools": bad_tools})
            assert count is None, f"count should be None for tools={bad_tools!r}"

    def test_body_is_not_a_dict(self) -> None:
        for bad_body in (None, [], "string", 42, object()):
            assert head_fingerprints(bad_body) == (None, None, None)  # type: ignore[arg-type]

    def test_non_string_dict_keys_in_tools_do_not_raise(self) -> None:
        # A dict with mixed int/str keys sorts fine under sort_keys=True only
        # if json can coerce them; if it can't, _fingerprint must still not
        # raise out to the caller.
        weird = {1: "a", "b": 2, (3, 4): "tuple-key-is-invalid-json"}
        tools_fp, _, count = head_fingerprints({"tools": [weird]})
        assert count == 1
        # Either degrades to None (tuple keys are not valid dict keys for
        # json.dumps) or produces a hash; either is acceptable, it must not
        # raise, which not raising here already proves.

    def test_huge_tools_array_completes_quickly(self) -> None:
        import time

        big_tools = [{"name": f"t{i}", "description": "x" * 100} for i in range(100_000)]
        start = time.perf_counter()
        tools_fp, _, count = head_fingerprints({"tools": big_tools})
        elapsed = time.perf_counter() - start
        assert count == 100_000
        assert elapsed < 2.0, f"head_fingerprints took {elapsed:.2f}s on 100k tools"

    def test_nan_and_infinity_values_do_not_raise(self) -> None:
        tools_fp, _, count = head_fingerprints(
            {"tools": [{"name": "x", "score": float("nan"), "bound": float("inf")}]}
        )
        assert count == 1
        assert isinstance(tools_fp, str)


# ---------------------------------------------------------------------------
# Section 2b: record() itself, including inputs that reach past
# head_fingerprints (transforms, anchor_depths).
# ---------------------------------------------------------------------------


class TestRecordNeverRaises:
    def test_malformed_body_types_do_not_raise(self, tmp_path: Path) -> None:
        log, _ = _log(tmp_path)
        for bad_body in ([1, 2, 3], "a string body", 42, object()):
            record = log.record(
                session_key="s",
                request_id="r",
                model="m",
                billed_cache_read=1,
                billed_cache_creation=1,
                alive_fraction=1.0,
                first_diverged_index=None,
                body=bad_body,  # type: ignore[arg-type]
            )
            assert record.tools_fingerprint is None

    def test_anchor_depths_with_uncomparable_elements_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """record() must not raise, and it does not: the outer try/except
        catches the TypeError from sorted() on a mixed-type list. But the
        session-chaining state (_session_prior, _session_head) is mutated
        BEFORE that sorted() call runs, inside the same lock-protected
        section, so a request that "fails" from the caller's point of view
        (uncounted, not in the ring, predicted_cache_read forced to 0) has
        already silently contributed its billed_cache_read/billed_cache_creation
        and tools/system fingerprints to what the NEXT request in the same
        session will be compared and chained against.

        This test encodes the correct contract (a request that produced no
        visible record must also leave no trace in session-chaining state)
        and is expected to FAIL against current behaviour.
        """
        log, _ = _log(tmp_path)
        r1 = log.record(
            session_key="s1",
            request_id="r1",
            model="m",
            billed_cache_read=1000,
            billed_cache_creation=200,
            alive_fraction=1.0,
            first_diverged_index=None,
            anchor_depths=[1, "bad"],  # type: ignore[list-item]  # int vs str: sorted() raises
        )
        assert r1.predicted_cache_read == 0  # never raises, contract holds so far

        requests_before_r2 = log.snapshot()["requests"]
        assert requests_before_r2 == 0, "r1 was not counted, as the caller would expect"

        r2 = log.record(
            session_key="s1",
            request_id="r2",
            model="m",
            billed_cache_read=1150,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        # r2 should predict zero: from the log's own accounting, r1 never
        # happened (requests stayed at 0). Instead the failed r1's billed
        # data leaked into session_prior and is chained into r2's prediction.
        assert r2.predicted_cache_read == 0, (
            f"session state was mutated by a request that record() itself "
            f"reports as never having happened: predicted={r2.predicted_cache_read}, "
            f"expected 0 because r1 was not counted"
        )

    def test_hostile_transforms_iterable_does_not_raise_out_of_record(
        self, tmp_path: Path
    ) -> None:
        """A transforms value whose iteration fails must not escape record().

        The failure mode this guards is the fallback rebuilding the same value
        that just failed. When the primary path raises on `list(transforms)`,
        an `except` handler that also calls `list(transforms)` raises a second
        time with nothing left to catch it, so a method documented never to
        raise takes down the hot request path. Materializing the list once, up
        front, is what makes the contract hold: both paths then share a value
        that has already survived.
        """

        class HostileTransforms:
            def __bool__(self) -> bool:
                return True

            def __iter__(self):
                raise RuntimeError("boom mid-iteration")

        log, _ = _log(tmp_path)
        record = log.record(
            session_key="s2",
            request_id="r1",
            model="m",
            billed_cache_read=5000,
            billed_cache_creation=300,
            alive_fraction=1.0,
            first_diverged_index=None,
            transforms=HostileTransforms(),  # type: ignore[arg-type]
        )
        assert record.transforms == []

    def test_body_get_raising_before_the_lock_is_clean(self, tmp_path: Path) -> None:
        """Control case: when the malformed input raises BEFORE the
        lock-protected session-state mutation (head_fingerprints runs first),
        record() degrades cleanly with no state corruption. This is expected
        to PASS, distinguishing this from the anchor_depths case above.
        """

        class HostileBody(dict):
            def get(self, *a, **k):  # type: ignore[override]
                raise RuntimeError("body.get exploded")

        log, _ = _log(tmp_path)
        record = log.record(
            session_key="s3",
            request_id="r1",
            model="m",
            billed_cache_read=10,
            billed_cache_creation=5,
            alive_fraction=1.0,
            first_diverged_index=None,
            body=HostileBody(),
        )
        assert record.predicted_cache_read == 0
        assert log.snapshot()["requests"] == 0
        assert log._session_prior == {}  # noqa: SLF001


# ---------------------------------------------------------------------------
# Section 3: unbounded growth of _session_prior / _session_head.
# ---------------------------------------------------------------------------


class TestSessionStateGrowth:
    def test_session_dicts_are_bounded_across_many_distinct_sessions(
        self, tmp_path: Path
    ) -> None:
        """A proxy runs for weeks and accumulates one session key per
        conversation. Neither _session_prior nor _session_head is ever
        evicted: both grow one entry per distinct session_key forever, with
        no ring, no TTL sweep, and no cap, unlike _ring/_recent_busts which
        are bounded deques.

        Both are now capped at MAX_TRACKED_SESSIONS and evict least-recently
        used keys, so the count tracks the cap rather than the number of
        sessions the process has ever seen. Evicting a key costs one
        prediction, not correctness: the next request on that session chains
        from zero and reads as a cold start.
        """
        log, _ = _log(tmp_path)
        n_sessions = 5000
        for i in range(n_sessions):
            log.record(
                session_key=f"session-{i}",
                request_id=f"r{i}",
                model="m",
                billed_cache_read=1,
                billed_cache_creation=1,
                alive_fraction=1.0,
                first_diverged_index=None,
                body={"tools": [{"name": "x"}]},
            )
        assert len(log._session_prior) <= MAX_TRACKED_SESSIONS, (  # noqa: SLF001
            f"_session_prior holds {len(log._session_prior)} entries after "  # noqa: SLF001
            f"{n_sessions} distinct sessions, cap is {MAX_TRACKED_SESSIONS}"
        )
        assert len(log._session_head) <= MAX_TRACKED_SESSIONS, (  # noqa: SLF001
            f"_session_head holds {len(log._session_head)} entries after "  # noqa: SLF001
            f"{n_sessions} distinct sessions, cap is {MAX_TRACKED_SESSIONS}"
        )
        # The survivors must be the most recent sessions, not an arbitrary
        # subset: evicting a live session to keep a dead one would trade a
        # bounded map for a useless one.
        assert f"session-{n_sessions - 1}" in log._session_prior  # noqa: SLF001
        assert "session-0" not in log._session_prior  # noqa: SLF001

    def test_session_dicts_grow_in_lockstep_and_never_shrink(self, tmp_path: Path) -> None:
        """Documents the actual (buggy) growth behaviour precisely, so the
        magnitude is measurable rather than argued. Expected to PASS: it is
        the closed-form description of the leak, not a contract violation.
        """
        log, _ = _log(tmp_path)
        for i in range(2000):
            log.record(
                session_key=f"s-{i}",
                request_id=f"r{i}",
                model="m",
                billed_cache_read=1,
                billed_cache_creation=1,
                alive_fraction=1.0,
                first_diverged_index=None,
                body={"tools": [{"name": "x"}]},
            )
        assert len(log._session_prior) == 2000  # noqa: SLF001
        assert len(log._session_head) == 2000  # noqa: SLF001
        assert log.snapshot()["requests"] == 2000
        # Bounded structures for comparison: the ring and recent-busts deque
        # do NOT grow past their configured size.
        assert len(log._ring) <= 200  # noqa: SLF001


# ---------------------------------------------------------------------------
# Concurrency: verify the lock actually protects both dicts together.
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_record_calls_do_not_raise_or_lose_updates(self, tmp_path: Path) -> None:
        log, _ = _log(tmp_path)
        errors: list[BaseException] = []
        n_threads = 16
        n_per_thread = 200

        def worker(tid: int) -> None:
            for i in range(n_per_thread):
                try:
                    log.record(
                        session_key=f"s{tid % 4}",
                        request_id=f"r{tid}-{i}",
                        model="m",
                        billed_cache_read=i,
                        billed_cache_creation=1,
                        alive_fraction=1.0,
                        first_diverged_index=None,
                        body={"tools": [{"name": str(i % 3)}]},
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"{len(errors)} exceptions escaped record() under concurrency: {errors[:3]}"
        assert log.snapshot()["requests"] == n_threads * n_per_thread

    def test_session_head_and_session_prior_stay_paired_under_concurrency(
        self, tmp_path: Path
    ) -> None:
        """Both dicts are written inside the same lock in _record, so for
        any session key ever recorded, both dicts must have an entry, never
        just one (which would indicate the two mutations desynchronized).
        """
        log, _ = _log(tmp_path)

        def worker(tid: int) -> None:
            for i in range(100):
                log.record(
                    session_key=f"sess-{tid}",
                    request_id=f"r{tid}-{i}",
                    model="m",
                    billed_cache_read=i,
                    billed_cache_creation=1,
                    alive_fraction=1.0,
                    first_diverged_index=None,
                    body={"tools": [{"name": str(i)}]},
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert set(log._session_prior.keys()) == set(log._session_head.keys())  # noqa: SLF001
