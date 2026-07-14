"""Compare bare Codex, Headroom-only, and the complete local agent stack.

The bare and Headroom-only arms run with temporary Codex homes containing only
subscription authentication and the model catalog.  They ignore user config,
rules, hooks, MCP servers, skills, and global instructions.  The full arm uses
the real Codex config from a temporary home so session artifacts are discarded.

Costs use OpenAI's standard API price table as an API-equivalent comparison.
ChatGPT subscription usage is reported separately from the rate-card estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

PRICING_SOURCE = "https://platform.openai.com/docs/pricing"
PRICING_USD_PER_1M: dict[str, tuple[float, float, float]] = {
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
    "gpt-5.6-terra": (2.50, 0.25, 15.00),
    "gpt-5.6-luna": (1.00, 0.10, 6.00),
}
ARMS = ("bare", "headroom", "full")
HEADROOM_ONLY_CONFIG = """\
model_provider = "headroom"

[model_providers.headroom]
name = "OpenAI via Headroom proxy"
base_url = "http://127.0.0.1:8787/v1"
requires_openai_auth = true
supports_websockets = true

[model_providers.headroom.env_http_headers]
X-Headroom-Project = "HEADROOM_PROJECT"
"""
PROMPTS = (
    """Inspect headroom/proxy/handlers/openai.py and the relevant tests. Find the
effective context limit for gpt-5.6-sol when the client is Codex, the generic
OpenAI API limit for the same slug, and the selector function that distinguishes
them. Use repository tools as appropriate. Do not modify files. Keep notes for
the next turn and answer with the three facts.""",
    """Now inspect headroom/proxy/hybrid_mode.py and tests/test_hybrid_mode.py.
Explain the cold-prefix behavior, warm-prefix behavior, normal rebase gates,
cooldown behavior, and the hard-context-limit exception. Use repository tools
as appropriate. Do not modify files. Keep the exact class or function names.""",
    """Return one compact JSON object only. It must have these keys:
codex_context_limit, api_context_limit, context_selector, cold_prefix,
warm_prefix, rebase_gates, cooldown, hard_limit, controller. Use numeric limits
and exact symbol names where applicable. Do not use markdown fences.""",
)


@dataclass
class ToolEvent:
    kind: str
    name: str
    status: str
    serialized_chars: int


@dataclass
class TurnResult:
    latency_seconds: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    cumulative_input_tokens: int
    cumulative_cached_input_tokens: int
    cumulative_output_tokens: int
    cumulative_reasoning_output_tokens: int
    tool_items: int
    command_items: int
    mcp_items: int
    mcp_tools: list[str]
    commands: list[str]
    tool_events: list[ToolEvent]
    tool_output_chars: int
    response: str


@dataclass
class RunResult:
    arm: str
    model: str
    repeat: int
    score: int
    score_max: int
    api_equivalent_usd: float
    headroom_project: str
    headroom_metrics: dict[str, Any] | None
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None

    @property
    def accuracy_pct(self) -> float:
        return 100.0 * self.score / self.score_max if self.score_max else 0.0


def _headroom_stats() -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8787/stats", timeout=5) as response:
            return cast(dict[str, Any], json.load(response))
    except Exception:
        return None


def _snapshot(stats: dict[str, Any] | None) -> dict[str, Any] | None:
    if not stats:
        return None
    summary = stats.get("summary", {})
    codex_ws = summary.get("codex_ws", {})
    rate = stats.get("codex_rate_limits", {}).get("primary") or {}
    return {
        "api_requests": summary.get("api_requests"),
        "mode": summary.get("mode"),
        "codex_units_total": codex_ws.get("units_total"),
        "codex_units_modified": codex_ws.get("units_modified"),
        "codex_tokens_saved": codex_ws.get("tokens_saved"),
        "subscription_used_percent": rate.get("used_percent"),
        "subscription_window_minutes": rate.get("window_minutes"),
        "subscription_resets_at": rate.get("resets_at"),
    }


def _project_metrics(stats: dict[str, Any] | None, project: str) -> dict[str, Any] | None:
    if not stats:
        return None
    projects = stats.get("savings", {}).get("per_project", {})
    metrics = projects.get(project)
    return dict(metrics) if isinstance(metrics, dict) else None


def _link_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        destination.symlink_to(source, target_is_directory=source.is_dir())


def _prepare_codex_home(root: Path, arm: str) -> Path:
    source = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    home = root / f"codex-{arm}"
    home.mkdir(parents=True)
    for name in ("auth.json", "models_cache.json"):
        _link_if_present(source / name, home / name)
    if not (home / "auth.json").exists():
        raise RuntimeError(f"Codex subscription auth not found at {source / 'auth.json'}")

    if arm == "full":
        for name in (
            "config.toml",
            "hooks.json",
            "AGENTS.md",
            "RTK.md",
            "skills",
            "plugins",
        ):
            _link_if_present(source / name, home / name)
    elif arm == "headroom":
        (home / "config.toml").write_text(HEADROOM_ONLY_CONFIG, encoding="utf-8")
    return home


def _arm_flags(arm: str) -> list[str]:
    if arm == "full":
        return ["--dangerously-bypass-hook-trust"]
    if arm == "bare":
        return ["--ignore-user-config", "--ignore-rules"]
    return ["--ignore-rules"]


def _parse_events(stdout: str) -> tuple[str | None, TurnResult]:
    thread_id: str | None = None
    usage: dict[str, int] = {}
    response = ""
    item_types: list[str] = []
    mcp_tools: list[str] = []
    commands: list[str] = []
    tool_events: list[ToolEvent] = []
    tool_output_chars = 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        item_type = str(item.get("type", ""))
        item_types.append(item_type)
        if item_type == "agent_message":
            response = str(item.get("text", ""))
        else:
            tool_output_chars += len(json.dumps(item, sort_keys=True))
        serialized_chars = len(json.dumps(item, sort_keys=True))
        if item_type == "mcp_tool_call":
            server = str(item.get("server", "unknown"))
            tool = str(item.get("tool", item.get("name", "unknown")))
            name = f"{server}.{tool}"
            mcp_tools.append(name)
            tool_events.append(
                ToolEvent(
                    kind="mcp",
                    name=name,
                    status=str(item.get("status", "unknown")),
                    serialized_chars=serialized_chars,
                )
            )
        if item_type == "command_execution":
            command = item.get("command")
            if isinstance(command, str):
                commands.append(command)
                tool_events.append(
                    ToolEvent(
                        kind="command",
                        name=command[:500],
                        status=str(item.get("status", "unknown")),
                        serialized_chars=serialized_chars,
                    )
                )

    tool_types = {
        "command_execution",
        "mcp_tool_call",
        "function_call",
        "tool_call",
        "file_change",
    }
    result = TurnResult(
        latency_seconds=0.0,
        input_tokens=int(usage.get("input_tokens", 0)),
        cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        reasoning_output_tokens=int(usage.get("reasoning_output_tokens", 0)),
        cumulative_input_tokens=int(usage.get("input_tokens", 0)),
        cumulative_cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
        cumulative_output_tokens=int(usage.get("output_tokens", 0)),
        cumulative_reasoning_output_tokens=int(usage.get("reasoning_output_tokens", 0)),
        tool_items=sum(item_type in tool_types for item_type in item_types),
        command_items=item_types.count("command_execution"),
        mcp_items=item_types.count("mcp_tool_call"),
        mcp_tools=mcp_tools,
        commands=commands,
        tool_events=tool_events,
        tool_output_chars=tool_output_chars,
        response=response,
    )
    return thread_id, result


def _usage_delta(current: TurnResult, previous: TurnResult | None) -> TurnResult:
    """Convert Codex's cumulative thread counters to per-turn usage."""
    if previous is None:
        return current
    return replace(
        current,
        input_tokens=max(
            0, current.cumulative_input_tokens - previous.cumulative_input_tokens
        ),
        cached_input_tokens=max(
            0,
            current.cumulative_cached_input_tokens
            - previous.cumulative_cached_input_tokens,
        ),
        output_tokens=max(
            0, current.cumulative_output_tokens - previous.cumulative_output_tokens
        ),
        reasoning_output_tokens=max(
            0,
            current.cumulative_reasoning_output_tokens
            - previous.cumulative_reasoning_output_tokens,
        ),
    )


def _run_turn(
    *,
    arm: str,
    model: str,
    project: Path,
    codex_home: Path,
    prompt: str,
    thread_id: str | None,
    headroom_project: str,
    timeout: int,
) -> tuple[str, TurnResult]:
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("codex executable not found")
    if thread_id is None:
        command = [
            codex,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--cd",
            str(project),
            *_arm_flags(arm),
            prompt,
        ]
    else:
        command = [
            codex,
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
            "--model",
            model,
            *_arm_flags(arm),
            thread_id,
            prompt,
        ]

    env = {key: value for key, value in os.environ.items() if not key.startswith("HEADROOM_")}
    env.update(
        {
            "CODEX_HOME": str(codex_home),
            "HEADROOM_PROJECT": headroom_project,
            "NO_COLOR": "1",
        }
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"codex exited {completed.returncode}: {message[-2000:]}")
    new_thread_id, result = _parse_events(completed.stdout)
    result.latency_seconds = elapsed
    resolved_thread = new_thread_id or thread_id
    if not resolved_thread:
        raise RuntimeError("Codex output did not contain a thread id")
    return resolved_thread, result


def _score(response: str) -> tuple[int, int]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return 0, 9
    if not isinstance(payload, dict):
        return 0, 9

    def field(name: str) -> str:
        return json.dumps(payload.get(name, ""), sort_keys=True).lower()

    gates = field("rebase_gates")
    checks = (
        payload.get("codex_context_limit") == 272_000,
        payload.get("api_context_limit") == 1_050_000,
        "_effective_openai_context_limit" in field("context_selector"),
        "cold_prefix" in field("cold_prefix"),
        "warm_prefix" in field("warm_prefix") and "live_delta" in field("warm_prefix"),
        (
            "economic_rebase" in gates and "pressure_rebase" in gates
        )
        or ("min_warm" in gates and "gain" in gates),
        "rebase_cooldown" in field("cooldown"),
        any(
            marker in field("hard_limit")
            for marker in ("hard_context_limit", "emergency_rebase", "pressure_hard_limit")
        ),
        "hybridmodecontroller" in field("controller"),
    )
    return sum(checks), len(checks)


def _api_cost(model: str, turns: Sequence[TurnResult]) -> float:
    input_price, cached_price, output_price = PRICING_USD_PER_1M[model]
    total = 0.0
    for turn in turns:
        cached = min(turn.input_tokens, turn.cached_input_tokens)
        uncached = turn.input_tokens - cached
        total += (
            uncached * input_price
            + cached * cached_price
            + turn.output_tokens * output_price
        ) / 1_000_000
    return total


def _run_one(
    *,
    arm: str,
    model: str,
    repeat: int,
    project: Path,
    temp_root: Path,
    timeout: int,
) -> RunResult:
    headroom_project = f"stack-canary-{model}-{repeat}-{arm}"
    codex_home = _prepare_codex_home(temp_root / f"{model}-{repeat}", arm)
    turns: list[TurnResult] = []
    thread_id: str | None = None
    previous_cumulative: TurnResult | None = None
    try:
        for prompt in PROMPTS:
            thread_id, cumulative = _run_turn(
                arm=arm,
                model=model,
                project=project,
                codex_home=codex_home,
                prompt=prompt,
                thread_id=thread_id,
                headroom_project=headroom_project,
                timeout=timeout,
            )
            turn = _usage_delta(cumulative, previous_cumulative)
            previous_cumulative = cumulative
            turns.append(turn)
        score, score_max = _score(turns[-1].response)
        return RunResult(
            arm=arm,
            model=model,
            repeat=repeat,
            score=score,
            score_max=score_max,
            api_equivalent_usd=_api_cost(model, turns),
            headroom_project=headroom_project,
            headroom_metrics=_project_metrics(_headroom_stats(), headroom_project),
            turns=turns,
        )
    except Exception as exc:
        return RunResult(
            arm=arm,
            model=model,
            repeat=repeat,
            score=0,
            score_max=9,
            api_equivalent_usd=_api_cost(model, turns),
            headroom_project=headroom_project,
            headroom_metrics=_project_metrics(_headroom_stats(), headroom_project),
            turns=turns,
            error=str(exc),
        )


def _aggregate(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({(result.model, result.arm) for result in results})
    for model, arm in groups:
        selected = [result for result in results if result.model == model and result.arm == arm]
        turns = [turn for result in selected for turn in result.turns]
        tool_breakdown: dict[str, dict[str, int]] = {}
        for turn in turns:
            for event in turn.tool_events:
                key = f"{event.kind}:{event.name}"
                entry = tool_breakdown.setdefault(key, {"calls": 0, "serialized_chars": 0})
                entry["calls"] += 1
                entry["serialized_chars"] += event.serialized_chars
        rows.append(
            {
                "model": model,
                "arm": arm,
                "runs": len(selected),
                "failures": sum(result.error is not None for result in selected),
                "accuracy_pct": round(statistics.mean(result.accuracy_pct for result in selected), 2),
                "mean_latency_seconds": round(
                    statistics.mean(sum(turn.latency_seconds for turn in result.turns) for result in selected),
                    3,
                ),
                "mean_input_tokens": round(
                    statistics.mean(sum(turn.input_tokens for turn in result.turns) for result in selected),
                    1,
                ),
                "mean_cached_input_tokens": round(
                    statistics.mean(
                        sum(turn.cached_input_tokens for turn in result.turns) for result in selected
                    ),
                    1,
                ),
                "mean_output_tokens": round(
                    statistics.mean(sum(turn.output_tokens for turn in result.turns) for result in selected),
                    1,
                ),
                "mean_api_equivalent_usd": round(
                    statistics.mean(result.api_equivalent_usd for result in selected), 6
                ),
                "tool_items": sum(turn.tool_items for turn in turns),
                "command_items": sum(turn.command_items for turn in turns),
                "mcp_items": sum(turn.mcp_items for turn in turns),
                "mcp_tools": dict(Counter(tool for turn in turns for tool in turn.mcp_tools)),
                "tool_breakdown": tool_breakdown,
                "tool_output_chars": sum(turn.tool_output_chars for turn in turns),
            }
        )
    return rows


def run_canary(
    project: Path,
    models: Sequence[str],
    arms: Sequence[str],
    repeats: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    unknown = sorted(set(models) - PRICING_USD_PER_1M.keys())
    if unknown:
        raise ValueError(f"missing official pricing for: {', '.join(unknown)}")
    unknown_arms = sorted(set(arms) - set(ARMS))
    if unknown_arms:
        raise ValueError(f"unknown arms: {', '.join(unknown_arms)}")
    before = _snapshot(_headroom_stats())
    results: list[RunResult] = []
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="headroom-codex-stack-") as temp_dir:
        temp_root = Path(temp_dir)
        for model in models:
            for repeat in range(repeats):
                order = list(arms)
                rng.shuffle(order)
                for arm in order:
                    run_root = temp_root / f"{model}-{repeat}-{arm}"
                    run_root.mkdir(parents=True)
                    result = _run_one(
                        arm=arm,
                        model=model,
                        repeat=repeat,
                        project=project,
                        temp_root=run_root,
                        timeout=timeout,
                    )
                    results.append(result)
                    print(
                        f"{model} {arm} repeat={repeat + 1}: "
                        f"accuracy={result.accuracy_pct:.1f}% "
                        f"input={sum(turn.input_tokens for turn in result.turns):,} "
                        f"latency={sum(turn.latency_seconds for turn in result.turns):.1f}s "
                        f"error={result.error or '-'}",
                        flush=True,
                    )
    after = _snapshot(_headroom_stats())
    return {
        "schema_version": 1,
        "project": str(project),
        "models": list(models),
        "arms": list(arms),
        "repeats": repeats,
        "seed": seed,
        "pricing_source": PRICING_SOURCE,
        "pricing_usd_per_1m": {
            model: {"input": prices[0], "cached_input": prices[1], "output": prices[2]}
            for model, prices in PRICING_USD_PER_1M.items()
        },
        "subscription_note": (
            "Codex subscription utilization is an observed whole-run window delta. "
            "OpenAI does not publish a fixed Plus token or credit pool."
        ),
        "headroom_before": before,
        "headroom_after": after,
        "aggregate": _aggregate(results),
        "runs": [
            {
                **{key: value for key, value in asdict(result).items() if key != "turns"},
                "accuracy_pct": result.accuracy_pct,
                "turns": [asdict(turn) for turn in result.turns],
            }
            for result in results
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--arm", action="append", choices=ARMS, dest="arms")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    models = args.models or ["gpt-5.6-luna"]
    arms = args.arms or list(ARMS)
    report = run_canary(
        args.project.resolve(),
        models,
        arms,
        args.repeats,
        args.seed,
        args.timeout,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "aggregate": report["aggregate"],
                    "headroom_before": report["headroom_before"],
                    "headroom_after": report["headroom_after"],
                    "json_out": str(args.json_out),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(rendered)
    return 1 if any(run["error"] for run in report["runs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
