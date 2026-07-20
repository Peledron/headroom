from __future__ import annotations

from headroom.proxy.handlers.streaming import _anthropic_iteration_metrics
from headroom.proxy.operational_audit import OperationalAudit


def test_iteration_metrics_match_aggregated_anthropic_usage() -> None:
    usage = {
        "iterations": [
            {
                "input_tokens": 2,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 100_000,
            },
            {
                "input_tokens": 120_000,
                "cache_read_input_tokens": 100_000,
                "cache_creation_input_tokens": 0,
            },
        ]
    }
    assert _anthropic_iteration_metrics(usage) == {
        "internal_iteration_count": 2,
        "internal_iteration_input_tokens": 120_002,
        "internal_iteration_cache_read_tokens": 100_000,
        "internal_iteration_cache_write_tokens": 100_000,
    }


def test_operational_audit_exposes_iteration_fanout() -> None:
    audit = OperationalAudit()
    audit.record_anthropic_iteration_fanout(7, 1_408_085)
    snapshot = audit.snapshot()
    assert snapshot["anthropic_fanout_requests"] == 1
    assert snapshot["anthropic_internal_iterations"] == 7
    assert snapshot["anthropic_internal_input_tokens"] == 1_408_085
