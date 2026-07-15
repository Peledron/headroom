from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "codex_cache_audit.py"
SPEC = importlib.util.spec_from_file_location("audit_codex_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _line(ts: str, body: str) -> str:
    return f"2026-07-11 {ts} - headroom.proxy - INFO - {body}\n"


def test_log_audit_uses_openai_cache_semantics_and_deduplicates(tmp_path: Path) -> None:
    log = tmp_path / "proxy.log"
    perf = (
        "[hr_session] PERF model=gpt-5.6 msgs=2 tok_before=30000 tok_after=30000 "
        "tok_saved=0 cache_read=27000 cache_write=3000 cache_hit_pct=90 "
        "opt_ms=0 total_ms=10 tok_out=100 transforms=none client=codex"
    )
    legacy_duplicate = (
        "[hr_session] PERF model=gpt-5.6 msgs=2 tok_before=30000 tok_after=30000 "
        "tok_saved=0 cache_read=27000 cache_write=3000 cache_hit_pct=90 "
        "opt_ms=0 transforms=none client=codex"
    )
    log.write_text(
        _line("20:00:00,000", perf)
        + _line("20:00:00,001", legacy_duplicate)
        + _line(
            "20:00:00,002",
            "[hr_session] WS /v1/responses frame passthrough "
            "reason=router_no_compression frame=3 bytes=1234 type=response.create "
            "auth_mode=oauth model=gpt-5.6",
        )
    )

    records, frames, agents = audit.parse_proxy_log(log)
    report = audit.summarize_log(
        records,
        frames,
        agents,
        min_prompt_tokens=20_000,
        priming_turns=0,
        bust_fraction=0.1,
    )

    assert report["turns"] == 1
    assert report["legacy_duplicate_perf_lines_removed"] == 1
    assert report["cached_input_tokens"] == 27_000
    assert report["inferred_uncached_input_tokens"] == 3_000
    assert report["semantics"]["provider_reported_cache_write_tokens"] is None
    assert report["passthrough_reasons"] == {"router_no_compression": 1}


def _snapshot(event: str, frame: int, body: dict) -> dict:
    return {
        "event": event,
        "request_id": "hr_req",
        "session_id": "session",
        "body": body,
        "metadata": {"frame": frame},
    }


def test_wire_audit_pairs_frames_and_reports_exact_mutations(tmp_path: Path) -> None:
    wire = tmp_path / "wire"
    wire.mkdir()
    inbound_body = {
        "type": "response.create",
        "response": {
            "model": "gpt-5.6",
            "instructions": "stable",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "x" * 600,
                }
            ],
        },
    }
    outbound_body = json.loads(json.dumps(inbound_body))
    outbound_body["response"]["input"][0]["output"] = "compressed"
    (wire / "1.json").write_text(
        json.dumps(_snapshot("ws_inbound_first_frame", 1, inbound_body))
    )
    (wire / "2.json").write_text(
        json.dumps(_snapshot("ws_upstream_client_frame", 1, outbound_body))
    )

    report = audit.summarize_wire(wire)

    assert report["paired_forwarded_frames"] == 1
    assert report["proxy_mutated_frames"] == 1
    assert report["proxy_mutations"][0]["changed_paths"] == ["$.input[0].output"]
    assert report["eligible_tool_outputs"]["above_512_bytes"] == 1


def test_wire_audit_reports_cache_setting_drift(tmp_path: Path) -> None:
    wire = tmp_path / "wire"
    wire.mkdir()
    first = {"type": "response.create", "response": {"model": "gpt", "instructions": "a"}}
    second = {"type": "response.create", "response": {"model": "gpt", "instructions": "b"}}
    (wire / "1.json").write_text(json.dumps(_snapshot("ws_inbound_first_frame", 1, first)))
    (wire / "2.json").write_text(json.dumps(_snapshot("ws_inbound_client_frame", 2, second)))

    report = audit.summarize_wire(wire)

    assert report["cache_setting_drift_events"] == [
        {
            "session_id": "session",
            "prior_frame": 1,
            "frame": 2,
            "changed_fields": ["instructions"],
        }
    ]

