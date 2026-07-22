#!/usr/bin/env python3
"""Compare Claude's bounded full stack with and without the Headroom proxy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.codex_stack_canary import PROMPTS, _score

PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
SONNET5_INTRO_PRICES = {
    "uncached_input": 2.00,
    "cache_write_5m": 2.50,
    "cache_write_1h": 4.00,
    "cache_read": 0.20,
    "output": 10.00,
}
ARMS = ("full-no-headroom", "full-headroom")
HEADROOM_ANTHROPIC_URL = "http://127.0.0.1:8787"


@dataclass
class Usage:
    uncached_input: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0
    output: int = 0

    def add(self, raw: dict[str, Any]) -> None:
        self.uncached_input += int(raw.get("input_tokens", 0) or 0)
        self.cache_read += int(raw.get("cache_read_input_tokens", 0) or 0)
        creation = raw.get("cache_creation", {})
        if not isinstance(creation, dict):
            creation = {}
        five_minute = int(creation.get("ephemeral_5m_input_tokens", 0) or 0)
        one_hour = int(creation.get("ephemeral_1h_input_tokens", 0) or 0)
        unspecified = int(raw.get("cache_creation_input_tokens", 0) or 0)
        self.cache_write_5m += five_minute + max(0, unspecified - five_minute - one_hour)
        self.cache_write_1h += one_hour
        self.output += int(raw.get("output_tokens", 0) or 0)

    def api_equivalent_usd(self) -> float:
        prices = SONNET5_INTRO_PRICES
        total = (
            self.uncached_input * prices["uncached_input"]
            + self.cache_write_5m * prices["cache_write_5m"]
            + self.cache_write_1h * prices["cache_write_1h"]
            + self.cache_read * prices["cache_read"]
            + self.output * prices["output"]
        )
        return total / 1_000_000


@dataclass
class Turn:
    latency_seconds: float
    provider_cost_usd: float
    usage: Usage
    tool_names: list[str] = field(default_factory=list)
    result: str = ""


def _parse_stream(stdout: str) -> Turn:
    usage = Usage()
    tools: list[str] = []
    result = ""
    provider_cost = 0.0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "assistant":
            message = event.get("message", {})
            if not isinstance(message, dict):
                continue
            raw_usage = message.get("usage", {})
            if isinstance(raw_usage, dict):
                usage.add(raw_usage)
            content = message.get("content", [])
            if isinstance(content, list):
                tools.extend(
                    str(block.get("name", "unknown"))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                )
        elif event.get("type") == "result":
            result = str(event.get("result", ""))
            provider_cost = float(event.get("total_cost_usd", 0.0) or 0.0)
    return Turn(
        latency_seconds=0.0,
        provider_cost_usd=provider_cost,
        usage=usage,
        tool_names=tools,
        result=result,
    )


def _settings(
    path: Path,
    *,
    headroom: bool,
    budget_command: str,
    headroom_url: str = HEADROOM_ANTHROPIC_URL,
) -> None:
    config: dict[str, Any] = {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": budget_command}],
                }
            ],
            "PreToolUse": [
                {
                    "matcher": (
                        "Bash|Read|Grep|Glob|mcp__tokensave__.*|mcp__serena__.*"
                    ),
                    "hooks": [{"type": "command", "command": budget_command}],
                }
            ],
        }
    }
    config["env"] = {
        "ANTHROPIC_BASE_URL": (
            headroom_url if headroom else "https://api.anthropic.com"
        )
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def _run_arm(
    project: Path,
    arm: str,
    model: str,
    timeout: int,
    headroom_url: str = HEADROOM_ANTHROPIC_URL,
) -> dict[str, Any]:
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("claude executable not found")
    headroom = arm == "full-headroom"
    session_id = str(uuid.uuid4())
    turns: list[Turn] = []
    with tempfile.TemporaryDirectory(prefix=f"claude-canary-{arm}-") as raw_tmp:
        tmp = Path(raw_tmp)
        settings = tmp / "settings.json"
        budget_command = f"{sys.executable} -m headroom.cli.claude_tool_budget"
        _settings(
            settings,
            headroom=headroom,
            budget_command=budget_command,
            headroom_url=headroom_url,
        )
        state_dir = tmp / "hook-state"
        for index, prompt in enumerate(PROMPTS):
            command = [
                claude,
                "-p",
                "--verbose",
                "--model",
                model,
                "--output-format",
                "stream-json",
                "--include-hook-events",
                "--permission-mode",
                "bypassPermissions",
                "--dangerously-skip-permissions",
                "--settings",
                str(settings),
                "--setting-sources",
                "user",
                "--disallowedTools",
                "Edit,Write,NotebookEdit,WebSearch,WebFetch",
            ]
            if index == 0:
                command.extend(("--session-id", session_id))
            else:
                command.extend(("--resume", session_id))
            command.append(prompt)
            env = os.environ.copy()
            env["HEADROOM_CLAUDE_HOOK_STATE_DIR"] = str(state_dir)
            env["ANTHROPIC_BASE_URL"] = headroom_url if headroom else "https://api.anthropic.com"
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
                raise RuntimeError(f"claude exited {completed.returncode}: {message[-2000:]}")
            turn = _parse_stream(completed.stdout)
            turn.latency_seconds = elapsed
            turns.append(turn)

    score, score_max = _score(turns[-1].result)
    total_usage = Usage()
    for turn in turns:
        total_usage.uncached_input += turn.usage.uncached_input
        total_usage.cache_write_5m += turn.usage.cache_write_5m
        total_usage.cache_write_1h += turn.usage.cache_write_1h
        total_usage.cache_read += turn.usage.cache_read
        total_usage.output += turn.usage.output
    return {
        "arm": arm,
        "model": model,
        "score": score,
        "score_max": score_max,
        "accuracy_pct": 100.0 * score / score_max if score_max else 0.0,
        "latency_seconds": sum(turn.latency_seconds for turn in turns),
        "provider_cost_usd": sum(turn.provider_cost_usd for turn in turns),
        "api_equivalent_usd": total_usage.api_equivalent_usd(),
        "usage": asdict(total_usage),
        "tool_calls": sum(len(turn.tool_names) for turn in turns),
        "tool_names": [name for turn in turns for name in turn.tool_names],
        "turns": [asdict(turn) for turn in turns],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--headroom-url", default=HEADROOM_ANTHROPIC_URL)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    arms = args.arm or list(ARMS)
    report = {
        "pricing_source": PRICING_SOURCE,
        "pricing_usd_per_mtok": SONNET5_INTRO_PRICES,
        "runs": [
            _run_arm(
                args.project.resolve(),
                arm,
                args.model,
                args.timeout,
                args.headroom_url,
            )
            for arm in arms
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
