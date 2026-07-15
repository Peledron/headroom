#!/usr/bin/env python3
"""Audit Codex Responses cache use and Headroom request mutations.

OpenAI reports cached input tokens, not cache writes. This tool therefore
labels the residual ``input_tokens - cached_tokens`` as inferred uncached
input. It also understands Headroom's legacy duplicate Codex WebSocket PERF
emission and de-duplicates matching records within a short time window.

Wire snapshots are optional. When supplied, the audit pairs inbound Codex
``response.create`` frames with the corresponding forwarded frame and reports
the exact JSON paths Headroom changed. It also reports drift in cache-relevant
request settings across consecutive inbound frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PERF_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2},\d{3}).*"
    r"\[(?P<request_id>hr_[^\]]+)\] PERF (?P<fields>.*)$"
)
FRAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2},\d{3}).*"
    r"\[(?P<request_id>hr_[^\]]+)\] WS /v1/responses "
    r"(?P<kind>frame passthrough|compressed) (?P<fields>.*)$"
)
WS_HEADER_RE = re.compile(
    r"event=proxy_inbound_websocket request_id=(?P<request_id>hr_\S+).*headers=(?P<headers>\{.*\})$"
)
USER_AGENT_RE = re.compile(r'"user-agent": "(?P<user_agent>[^"]+)"')
KV_RE = re.compile(r"(?P<key>[a-zA-Z_]+)=(?P<value>[^\s]+)")

OUTPUT_ITEM_TYPES = {
    "custom_tool_call_output",
    "function_call_output",
    "local_shell_call_output",
    "apply_patch_call_output",
}
CACHE_SETTING_FIELDS = (
    "model",
    "instructions",
    "tools",
    "reasoning",
    "text",
    "tool_choice",
    "parallel_tool_calls",
)


def _fields(text: str) -> dict[str, str]:
    return {match.group("key"): match.group("value") for match in KV_RE.finditer(text)}


def _integer(fields: dict[str, str], key: str) -> int:
    try:
        return max(0, int(fields.get(key, "0")))
    except ValueError:
        return 0


def _timestamp(date: str, time_text: str) -> datetime:
    return datetime.strptime(f"{date} {time_text}", "%Y-%m-%d %H:%M:%S,%f")


@dataclass(frozen=True)
class PerfRecord:
    timestamp: datetime
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    inferred_uncached_input_tokens: int
    tokens_saved: int
    messages: int

    @property
    def cached_fraction(self) -> float | None:
        if self.input_tokens <= 0:
            return None
        return self.cached_input_tokens / self.input_tokens

    @property
    def legacy_duplicate_key(self) -> tuple[Any, ...]:
        return (
            self.request_id,
            self.model,
            self.input_tokens,
            self.cached_input_tokens,
            self.inferred_uncached_input_tokens,
            self.tokens_saved,
        )


@dataclass(frozen=True)
class FrameRecord:
    timestamp: datetime
    request_id: str
    kind: str
    reason: str
    auth_mode: str
    model: str
    frame: int
    bytes_before: int
    tokens_saved: int


def parse_proxy_log(path: Path) -> tuple[list[PerfRecord], list[FrameRecord], dict[str, str]]:
    perf: list[PerfRecord] = []
    frames: list[FrameRecord] = []
    user_agents: dict[str, str] = {}

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            user_agent_match = USER_AGENT_RE.search(line)
            if user_agent_match and "codex" in user_agent_match.group("user_agent").lower():
                user_agents.setdefault(
                    f"observed:{len(user_agents) + 1}", user_agent_match.group("user_agent")
                )
            perf_match = PERF_RE.match(line)
            if perf_match:
                values = _fields(perf_match.group("fields"))
                model = values.get("model", "unknown")
                client = values.get("client", "")
                if client != "codex" and not model.startswith(("gpt-", "o1", "o3", "o4")):
                    continue
                input_tokens = _integer(values, "tok_after")
                cached = _integer(values, "cache_read")
                residual = _integer(values, "cache_write")
                # The OpenAI handler derives this residual from input-cached.
                # Recompute when an older log omitted or misreported it.
                inferred_uncached = max(input_tokens - cached, 0)
                # Fall back to the handler-reported residual ONLY when the token
                # fields are absent so a recompute is impossible. A present but
                # divergent residual must not override the honest input-cached
                # value: the residual is the field more likely to be corrupted.
                if input_tokens == 0 and residual:
                    inferred_uncached = residual
                perf.append(
                    PerfRecord(
                        timestamp=_timestamp(perf_match.group("date"), perf_match.group("time")),
                        request_id=perf_match.group("request_id"),
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=_integer(values, "tok_out"),
                        cached_input_tokens=cached,
                        inferred_uncached_input_tokens=inferred_uncached,
                        tokens_saved=_integer(values, "tok_saved"),
                        messages=_integer(values, "msgs"),
                    )
                )
                continue

            frame_match = FRAME_RE.match(line)
            if frame_match:
                values = _fields(frame_match.group("fields"))
                frames.append(
                    FrameRecord(
                        timestamp=_timestamp(
                            frame_match.group("date"), frame_match.group("time")
                        ),
                        request_id=frame_match.group("request_id"),
                        kind=frame_match.group("kind"),
                        reason=values.get("reason", "applied"),
                        auth_mode=values.get("auth_mode", "unknown"),
                        model=values.get("model", "unknown"),
                        frame=_integer(values, "frame"),
                        bytes_before=_integer(values, "bytes"),
                        tokens_saved=_integer(values, "tokens_saved"),
                    )
                )
                continue

            header_match = WS_HEADER_RE.search(line)
            if header_match:
                try:
                    headers = json.loads(header_match.group("headers"))
                except json.JSONDecodeError:
                    continue
                user_agent = headers.get("user-agent")
                if isinstance(user_agent, str):
                    user_agents[header_match.group("request_id")] = user_agent

    return perf, frames, user_agents


# The legacy double emission wrote two byte-identical PERF lines from the same
# handler microseconds apart, so a tight window catches them. Distinct real turns
# on one session are seconds apart; a wide window would collapse them and hide
# genuine busts, so keep this small.
def deduplicate_perf(records: Iterable[PerfRecord], window_ms: int = 50) -> tuple[list[PerfRecord], int]:
    kept: list[PerfRecord] = []
    last_seen: dict[tuple[Any, ...], datetime] = {}
    duplicates = 0
    for record in sorted(records, key=lambda item: item.timestamp):
        prior = last_seen.get(record.legacy_duplicate_key)
        if prior is not None:
            delta_ms = (record.timestamp - prior).total_seconds() * 1000
            if 0 <= delta_ms <= window_ms:
                duplicates += 1
                continue
        kept.append(record)
        last_seen[record.legacy_duplicate_key] = record.timestamp
    return kept, duplicates


def summarize_log(
    perf_records: list[PerfRecord],
    frames: list[FrameRecord],
    user_agents: dict[str, str],
    *,
    min_prompt_tokens: int,
    priming_turns: int,
    bust_fraction: float,
) -> dict[str, Any]:
    perf, duplicate_count = deduplicate_perf(perf_records)
    by_session: dict[str, list[PerfRecord]] = defaultdict(list)
    for record in perf:
        by_session[record.request_id].append(record)

    mature: list[PerfRecord] = []
    for records in by_session.values():
        ordered = sorted(records, key=lambda item: item.timestamp)
        mature.extend(
            record
            for index, record in enumerate(ordered, start=1)
            if index > priming_turns
            and record.input_tokens >= min_prompt_tokens
            # A turn with no input tokens has no cached_fraction, so it can never
            # be counted as a bust. Excluding it here keeps it out of the mature
            # denominator too, instead of silently diluting the bust rate.
            and record.cached_fraction is not None
        )

    total_input = sum(record.input_tokens for record in perf)
    total_cached = sum(record.cached_input_tokens for record in perf)
    total_uncached = sum(record.inferred_uncached_input_tokens for record in perf)
    busts = [
        record
        for record in mature
        if record.cached_fraction is not None and record.cached_fraction < bust_fraction
    ]
    auth_modes = Counter(frame.auth_mode for frame in frames)
    reasons = Counter(frame.reason for frame in frames if frame.kind == "frame passthrough")
    modified_frames = [frame for frame in frames if frame.kind == "compressed"]

    classification_warnings = []
    for request_id, user_agent in sorted(user_agents.items()):
        modes = {frame.auth_mode for frame in frames if frame.request_id == request_id}
        if "codex-tui/" in user_agent.lower() and "oauth" in modes:
            classification_warnings.append(
                {
                    "request_id": request_id,
                    "observed_auth_mode": "oauth",
                    "expected_auth_mode": "subscription",
                    "user_agent_family": "codex-tui",
                }
            )
    # No cross-session fallback: a codex-tui UA appearing somewhere in the log
    # together with an unrelated oauth frame elsewhere is not evidence that any
    # codex-tui session was misclassified. Only per-request attribution above
    # can warrant a warning.

    return {
        "semantics": {
            "cached_input_tokens": "reported by OpenAI",
            "inferred_uncached_input_tokens": "input_tokens - cached_input_tokens",
            "provider_reported_cache_write_tokens": None,
        },
        "sessions": len(by_session),
        "turns": len(perf),
        "legacy_duplicate_perf_lines_removed": duplicate_count,
        "input_tokens": total_input,
        "cached_input_tokens": total_cached,
        "inferred_uncached_input_tokens": total_uncached,
        "weighted_cached_fraction": total_cached / total_input if total_input else None,
        "tokens_saved": sum(record.tokens_saved for record in perf),
        "mature_turn_rule": {
            "exclude_first_turns_per_session": priming_turns,
            "minimum_input_tokens": min_prompt_tokens,
            "bust_cached_fraction_below": bust_fraction,
        },
        "mature_turns": len(mature),
        "mature_busts": len(busts),
        "mature_bust_rate": len(busts) / len(mature) if mature else None,
        "frames": len(frames),
        "modified_frames": len(modified_frames),
        "frame_tokens_saved": sum(frame.tokens_saved for frame in modified_frames),
        "passthrough_reasons": dict(sorted(reasons.items())),
        "auth_modes": dict(sorted(auth_modes.items())),
        "classification_warnings": classification_warnings,
    }


def _unwrap_response(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    response = body.get("response")
    if body.get("type") == "response.create" and isinstance(response, dict):
        return response
    return body


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _diff_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_diff_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        paths = []
        common = min(len(left), len(right))
        for index in range(common):
            paths.extend(_diff_paths(left[index], right[index], f"{prefix}[{index}]"))
        if len(left) != len(right):
            paths.append(f"{prefix}[{common}:]")
        return paths
    return [] if left == right else [prefix]


def _frame_key(snapshot: dict[str, Any]) -> tuple[str, int] | None:
    session = snapshot.get("session_id") or snapshot.get("request_id")
    metadata = snapshot.get("metadata")
    frame = metadata.get("frame") if isinstance(metadata, dict) else None
    if isinstance(session, str) and isinstance(frame, int):
        return session, frame
    return None


def _eligible_outputs(payload: dict[str, Any]) -> dict[str, int]:
    items = payload.get("input")
    if not isinstance(items, list):
        items = payload.get("messages")
    if not isinstance(items, list):
        return {"items": 0, "bytes": 0, "above_512_bytes": 0, "below_512_bytes": 0}
    outputs = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") not in OUTPUT_ITEM_TYPES:
            continue
        output = item.get("output")
        if isinstance(output, str):
            outputs.append(len(output.encode("utf-8", errors="replace")))
    return {
        "items": len(outputs),
        "bytes": sum(outputs),
        "above_512_bytes": sum(size >= 512 for size in outputs),
        "below_512_bytes": sum(size < 512 for size in outputs),
    }


def summarize_wire(wire_dir: Path) -> dict[str, Any]:
    inbound: dict[tuple[str, int], dict[str, Any]] = {}
    outbound: dict[tuple[str, int], dict[str, Any]] = {}
    parse_errors = 0
    snapshots = 0

    for path in sorted(wire_dir.glob("*.json")):
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parse_errors += 1
            continue
        snapshots += 1
        key = _frame_key(snapshot)
        if key is None:
            continue
        event = snapshot.get("event")
        if event in {"ws_inbound_first_frame", "ws_inbound_client_frame"}:
            inbound[key] = snapshot
        elif event == "ws_upstream_client_frame":
            outbound[key] = snapshot

    mutations = []
    cache_setting_drift = []
    eligibility = Counter()
    inbound_by_session: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for key, snapshot in inbound.items():
        payload = _unwrap_response(snapshot.get("body"))
        if payload is None:
            continue
        inbound_by_session[key[0]].append((key[1], payload))
        eligibility.update(_eligible_outputs(payload))
        sent = outbound.get(key)
        sent_payload = _unwrap_response(sent.get("body")) if sent else None
        if sent_payload is None:
            continue
        paths = _diff_paths(payload, sent_payload)
        if paths:
            mutations.append(
                {
                    "session_id": key[0],
                    "frame": key[1],
                    "changed_paths": paths,
                }
            )

    for session_id, frames in inbound_by_session.items():
        ordered = sorted(frames)
        for (prior_frame, prior), (frame, current) in zip(ordered, ordered[1:]):
            changed = []
            for field in CACHE_SETTING_FIELDS:
                if _digest(prior.get(field)) != _digest(current.get(field)):
                    changed.append(field)
            if changed:
                cache_setting_drift.append(
                    {
                        "session_id": session_id,
                        "prior_frame": prior_frame,
                        "frame": frame,
                        "changed_fields": changed,
                    }
                )

    return {
        "snapshots": snapshots,
        "parse_errors": parse_errors,
        "inbound_response_create_frames": len(inbound),
        "paired_forwarded_frames": len(set(inbound) & set(outbound)),
        "proxy_mutated_frames": len(mutations),
        "proxy_mutations": mutations,
        "cache_setting_drift_events": cache_setting_drift,
        "eligible_tool_outputs": dict(eligibility),
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def render_human(report: dict[str, Any]) -> str:
    log = report["log"]
    lines = [
        "Codex Responses cache audit",
        f"sessions={log['sessions']} turns={log['turns']} "
        f"deduped_perf={log['legacy_duplicate_perf_lines_removed']}",
        f"input={log['input_tokens']:,} cached={log['cached_input_tokens']:,} "
        f"inferred_uncached={log['inferred_uncached_input_tokens']:,} "
        f"cached_fraction={_percent(log['weighted_cached_fraction'])}",
        f"mature_turns={log['mature_turns']} mature_busts={log['mature_busts']} "
        f"bust_rate={_percent(log['mature_bust_rate'])}",
        f"frames={log['frames']} modified={log['modified_frames']} "
        f"tokens_saved={log['tokens_saved']}",
        f"passthrough_reasons={json.dumps(log['passthrough_reasons'], sort_keys=True)}",
        f"auth_modes={json.dumps(log['auth_modes'], sort_keys=True)}",
    ]
    for warning in log["classification_warnings"]:
        lines.append(
            "WARNING auth classification: "
            f"{warning['request_id']} {warning['user_agent_family']} "
            f"observed={warning['observed_auth_mode']} "
            f"expected={warning['expected_auth_mode']}"
        )
    if "wire" in report:
        wire = report["wire"]
        lines.extend(
            [
                f"wire_snapshots={wire['snapshots']} paired_frames={wire['paired_forwarded_frames']}",
                f"proxy_mutated_frames={wire['proxy_mutated_frames']} "
                f"setting_drift_events={len(wire['cache_setting_drift_events'])}",
                f"eligible_tool_outputs={json.dumps(wire['eligible_tool_outputs'], sort_keys=True)}",
            ]
        )
    lines.append(
        "note: inferred_uncached is not a provider-reported cache write; "
        "OpenAI reports cached input only"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--wire-dir", type=Path)
    parser.add_argument("--min-prompt-tokens", type=int, default=20_000)
    parser.add_argument("--priming-turns", type=int, default=2)
    parser.add_argument("--bust-fraction", type=float, default=0.10)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.bust_fraction <= 1:
        raise SystemExit("--bust-fraction must be between 0 and 1")
    perf, frames, user_agents = parse_proxy_log(args.proxy_log)
    report: dict[str, Any] = {
        "log": summarize_log(
            perf,
            frames,
            user_agents,
            min_prompt_tokens=max(0, args.min_prompt_tokens),
            priming_turns=max(0, args.priming_turns),
            bust_fraction=args.bust_fraction,
        )
    }
    if args.wire_dir is not None:
        report["wire"] = summarize_wire(args.wire_dir)
    if args.as_json:
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(render_human(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

