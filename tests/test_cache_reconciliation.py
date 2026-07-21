"""Closed-loop cache accounting: record assembly, chaining, ring bounds, IO tolerance."""

from __future__ import annotations

import json

from headroom.proxy.cache_reconciliation import (
    CacheReconciliationLog,
    is_unplanned_bust,
)


def _log(tmp_path, **kwargs):  # noqa: ANN001, ANN201
    path = tmp_path / "cache_reconciliation.jsonl"
    return CacheReconciliationLog(log_path=path, **kwargs), path


class TestRecordAssembly:
    def test_first_request_in_a_session_predicts_zero(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        record = log.record(
            session_key="s1",
            request_id="r1",
            model="claude-fable-5",
            billed_cache_read=1000,
            billed_cache_creation=200,
            alive_fraction=1.0,
            first_diverged_index=None,
            transforms=["masking_gate"],
        )
        assert record.predicted_cache_read == 0
        assert record.billed_cache_read == 1000
        assert record.billed_cache_creation == 200
        assert record.transforms == ["masking_gate"]
        assert record.unplanned_bust is False

    def test_negative_billed_usage_is_clamped(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        record = log.record(
            session_key="s1",
            request_id="r1",
            model="m",
            billed_cache_read=-5,
            billed_cache_creation=-1,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        assert record.billed_cache_read == 0
        assert record.billed_cache_creation == 0


class TestPredictedReadChaining:
    def test_second_request_predicts_prior_read_plus_write(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        log.record(
            session_key="s1",
            request_id="r1",
            model="m",
            billed_cache_read=1000,
            billed_cache_creation=200,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        r2 = log.record(
            session_key="s1",
            request_id="r2",
            model="m",
            billed_cache_read=1150,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        assert r2.predicted_cache_read == 1200

    def test_different_sessions_do_not_share_chains(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        log.record(
            session_key="s1",
            request_id="r1",
            model="m",
            billed_cache_read=5000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        r2 = log.record(
            session_key="s2",
            request_id="r2",
            model="m",
            billed_cache_read=10,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        assert r2.predicted_cache_read == 0


class TestUnplannedBustFlagging:
    def test_billed_under_half_of_predicted_is_flagged(self) -> None:
        assert is_unplanned_bust(predicted_cache_read=1000, billed_cache_read=400)

    def test_billed_at_or_above_half_of_predicted_is_not_flagged(self) -> None:
        assert not is_unplanned_bust(predicted_cache_read=1000, billed_cache_read=500)
        assert not is_unplanned_bust(predicted_cache_read=1000, billed_cache_read=900)

    def test_zero_predicted_is_never_a_bust(self) -> None:
        assert not is_unplanned_bust(predicted_cache_read=0, billed_cache_read=0)

    def test_chained_record_flags_a_real_bust(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        log.record(
            session_key="s1",
            request_id="r1",
            model="m",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        r2 = log.record(
            session_key="s1",
            request_id="r2",
            model="m",
            billed_cache_read=100,
            billed_cache_creation=9_000,
            alive_fraction=0.1,
            first_diverged_index=1,
        )
        assert r2.predicted_cache_read == 10_000
        assert r2.unplanned_bust is True

        snap = log.snapshot()
        assert snap["requests"] == 2
        assert snap["unplanned_busts"] == 1
        assert snap["recent_unplanned_busts"][-1]["request_id"] == "r2"


class TestRingBounds:
    def test_recent_busts_ring_stays_bounded(self, tmp_path) -> None:
        log, _ = _log(tmp_path, recent_bust_size=3)
        # Seed a warm prediction, then bust repeatedly.
        log.record(
            session_key="s1",
            request_id="seed",
            model="m",
            billed_cache_read=10_000,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        for i in range(10):
            log.record(
                session_key="s1",
                request_id=f"bust{i}",
                model="m",
                billed_cache_read=0,
                billed_cache_creation=10_000,
                alive_fraction=0.0,
                first_diverged_index=0,
            )
        snap = log.snapshot()
        assert snap["unplanned_busts"] == 10
        assert len(snap["recent_unplanned_busts"]) == 3
        assert snap["recent_unplanned_busts"][-1]["request_id"] == "bust9"

    def test_ring_size_bounds_total_stored_records(self, tmp_path) -> None:
        log, _ = _log(tmp_path, ring_size=5)
        for i in range(20):
            log.record(
                session_key="s1",
                request_id=f"r{i}",
                model="m",
                billed_cache_read=1,
                billed_cache_creation=0,
                alive_fraction=1.0,
                first_diverged_index=None,
            )
        assert len(log._ring) == 5  # noqa: SLF001
        assert log.snapshot()["requests"] == 20


class TestIOFailureTolerance:
    def test_unwritable_log_path_does_not_raise(self, tmp_path) -> None:
        # A path whose parent is a file, not a directory, can never be
        # mkdir'd or opened for append. record() must swallow that silently.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        bad_path = blocker / "cache_reconciliation.jsonl"
        log = CacheReconciliationLog(log_path=bad_path)

        record = log.record(
            session_key="s1",
            request_id="r1",
            model="m",
            billed_cache_read=1,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        assert record.request_id == "r1"
        assert log.snapshot()["requests"] == 1

    def test_successful_write_produces_one_json_line(self, tmp_path) -> None:
        log, path = _log(tmp_path)
        log.record(
            session_key="s1",
            request_id="r1",
            model="m",
            billed_cache_read=1,
            billed_cache_creation=0,
            alive_fraction=1.0,
            first_diverged_index=None,
        )
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["request_id"] == "r1"


class TestTtlAwareness:
    def _warm(self, log, key="s1", now=0.0):  # noqa: ANN001, ANN201
        return log.record(
            session_key=key,
            request_id="r1",
            model="m",
            billed_cache_read=1000,
            billed_cache_creation=200,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=now,
        )

    def test_cold_read_after_ttl_gap_is_expiry_not_bust(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        self._warm(log, now=0.0)
        record = log.record(
            session_key="s1",
            request_id="r2",
            model="m",
            billed_cache_read=0,
            billed_cache_creation=1200,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=400.0,
        )
        assert record.ttl_expired is True
        assert record.unplanned_bust is False
        snap = log.snapshot()
        assert snap["unplanned_busts"] == 0
        assert snap["ttl_expiry_colds"] == 1

    def test_cold_read_within_ttl_still_counts_as_bust(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        self._warm(log, now=0.0)
        record = log.record(
            session_key="s1",
            request_id="r2",
            model="m",
            billed_cache_read=0,
            billed_cache_creation=1200,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=60.0,
        )
        assert record.ttl_expired is False
        assert record.unplanned_bust is True

    def test_warm_read_after_long_gap_is_not_flagged_at_all(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        self._warm(log, now=0.0)
        record = log.record(
            session_key="s1",
            request_id="r2",
            model="m",
            billed_cache_read=1200,
            billed_cache_creation=50,
            alive_fraction=1.0,
            first_diverged_index=None,
            now=400.0,
        )
        assert record.unplanned_bust is False
        snap = log.snapshot()
        assert snap["ttl_expiry_colds"] == 0


class TestTierAwareTtl:
    def test_1h_tier_cold_read_at_400s_is_a_real_bust_not_expiry(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        log.record(session_key="s", request_id="r1", model="m",
                   billed_cache_read=1000, billed_cache_creation=100,
                   alive_fraction=1.0, first_diverged_index=None,
                   ttl_seconds=3600.0, now=0.0)
        rec = log.record(session_key="s", request_id="r2", model="m",
                         billed_cache_read=0, billed_cache_creation=1200,
                         alive_fraction=1.0, first_diverged_index=None,
                         ttl_seconds=3600.0, now=400.0)
        assert rec.ttl_expired is False
        assert rec.unplanned_bust is True

    def test_5m_tier_cold_read_at_400s_is_scheduled_expiry(self, tmp_path) -> None:
        log, _ = _log(tmp_path)
        log.record(session_key="s", request_id="r1", model="m",
                   billed_cache_read=1000, billed_cache_creation=100,
                   alive_fraction=1.0, first_diverged_index=None,
                   ttl_seconds=300.0, now=0.0)
        rec = log.record(session_key="s", request_id="r2", model="m",
                         billed_cache_read=0, billed_cache_creation=1200,
                         alive_fraction=1.0, first_diverged_index=None,
                         ttl_seconds=300.0, now=400.0)
        assert rec.ttl_expired is True
        assert rec.unplanned_bust is False


class TestMaxCacheTtlSeconds:
    def test_reads_1h_from_message_segment(self) -> None:
        from headroom.proxy.cache_reconciliation import message_segment_ttl_seconds
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]}
        assert message_segment_ttl_seconds(body) == 3600.0

    def test_defaults_to_5m_when_no_ttl(self) -> None:
        from headroom.proxy.cache_reconciliation import message_segment_ttl_seconds
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert message_segment_ttl_seconds(body) == 300.0

    def test_none_body_is_5m(self) -> None:
        from headroom.proxy.cache_reconciliation import message_segment_ttl_seconds
        assert message_segment_ttl_seconds(None) == 300.0

    def test_1h_system_head_does_not_mask_5m_message_tail(self) -> None:
        # The breaker's finding 6: the deliberately-1h system HEAD must not
        # inflate the message segment's real 5m tier.
        from headroom.proxy.cache_reconciliation import message_segment_ttl_seconds
        body = {
            "system": [{"type": "text", "text": "s",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "m",
                 "cache_control": {"type": "ephemeral", "ttl": "5m"}}]}],
        }
        assert message_segment_ttl_seconds(body) == 300.0
