"""Bounded live canary for text and optical compression accuracy.

The CLI relaunches itself with an explicit, clean Headroom environment so shell
configuration cannot contaminate mode comparisons. Live calls are opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from headroom.tokenizers import get_tokenizer
from headroom.transforms.text_crusher import TextCrusher
from headroom.transforms.text_optical import OpticalCompressionResult, TextOpticalCompressor

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}
MODES = ("raw", "text", "optical", "text_optical")
CONTROLLED_HEADROOM_ENV = {
    "HEADROOM_MODE": "token",
    "HEADROOM_OPTICAL_TEXT": "0",
    "HEADROOM_OUTPUT_SHAPER": "0",
    "HEADROOM_NET_COST_POLICY": "0",
    "HEADROOM_ADAPTIVE_TTL": "0",
    "HEADROOM_OUTPUT_HOLDOUT": "0",
}
AUDITED_ENV_KEYS = frozenset(
    {
        *CONTROLLED_HEADROOM_ENV,
        "HEADROOM_OPTICAL_CACHE_DIR",
        "HEADROOM_CACHE_TTL",
    }
)


@dataclass(frozen=True, slots=True)
class CanaryCase:
    name: str
    question: str
    expected: dict[str, str]


@dataclass(frozen=True, slots=True)
class ModeResult:
    mode: str
    available: bool
    estimated_input_tokens: int
    estimated_savings_pct: float
    actual_input_tokens: int | None
    actual_savings_pct: float | None
    correct_fields: int | None
    total_fields: int
    accuracy_pct: float | None
    baseline_retention_pct: float | None
    completed_cases: int
    mean_latency_seconds: float | None
    answers: tuple[str, ...]
    errors: tuple[str, ...]


def build_canary_corpus(paragraphs: int = 240) -> str:
    """Build deterministic prose with summary facts and sparse exact markers."""
    if paragraphs < 240:
        raise ValueError("the accuracy corpus needs at least 240 paragraphs")

    lines = [
        "Authoritative incident brief. Incident codename: Alder. Region: eu-west-3. "
        "Owner: Mira Chen. Deadline: 2026-08-17 14:30 UTC. Final decision: keep stable prefix.",
        "Only the authoritative incident brief defines the summary fields.",
    ]
    markers = {17: "CITRINE-4821", 121: "LANTERN-7734", 233: "ORCHID-9056"}
    states = {40: "amber", 100: "blue", 200: "green"}
    for index in range(paragraphs):
        marker = markers.get(index, f"routine-{index:04d}")
        state = ""
        if index in states:
            state = f" Deployment state transition {len([i for i in states if i <= index])}: {states[index]}."
        lines.append(
            f"Observation {index:04d}. The verification marker is {marker}. "
            "The service stayed healthy during the measured window. Cache reads avoided "
            "repeated prefix writes while the current observation remained mutable. "
            "Operators preserve the stable prefix and inspect the marker only when the "
            f"corresponding observation is requested.{state}"
        )
    return "\n\n".join(lines)


def build_cases() -> tuple[CanaryCase, ...]:
    return (
        CanaryCase(
            name="summary",
            question=(
                "Summarize the authoritative incident brief as JSON with exactly these keys: "
                "incident, region, owner, deadline, decision. Use source wording for values."
            ),
            expected={
                "incident": "Alder",
                "region": "eu-west-3",
                "owner": "Mira Chen",
                "deadline": "2026-08-17 14:30 UTC",
                "decision": "keep stable prefix",
            },
        ),
        CanaryCase(
            name="lookup_early",
            question=(
                "Return JSON with one key named marker. Its value must be the exact verification "
                "marker for observation 0017."
            ),
            expected={"marker": "CITRINE-4821"},
        ),
        CanaryCase(
            name="lookup_middle",
            question=(
                "Return JSON with one key named marker. Its value must be the exact verification "
                "marker for observation 0121."
            ),
            expected={"marker": "LANTERN-7734"},
        ),
        CanaryCase(
            name="lookup_late",
            question=(
                "Return JSON with one key named marker. Its value must be the exact verification "
                "marker for observation 0233."
            ),
            expected={"marker": "ORCHID-9056"},
        ),
        CanaryCase(
            name="state_tracking",
            question=(
                "Return JSON with keys first, final, transitions for the deployment state "
                "changes. Keep transitions as a JSON string containing the decimal count."
            ),
            expected={"first": "amber", "final": "green", "transitions": "3"},
        ),
        CanaryCase(
            name="never_stated",
            question=(
                "Observation 9999 is not in the source. Return JSON with one key named status "
                "and value not_found. Do not invent a marker."
            ),
            expected={"status": "not_found"},
        ),
    )


def _extract_json(answer: str) -> dict[str, Any]:
    text = answer.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def score_answer(answer: str, expected: dict[str, str]) -> int:
    parsed = _extract_json(answer)
    return sum(
        str(parsed.get(key, "")).strip().casefold() == value.strip().casefold()
        for key, value in expected.items()
    )


def _text_prompt(source: str, question: str) -> str:
    return (
        "Read the source and answer from it only. Return JSON only.\n\n"
        f"<source>\n{source}\n</source>\n\nQuestion: {question}"
    )


def _question_part(provider: str, question: str) -> dict[str, Any]:
    text = f"Answer from the rendered source only. Return JSON only.\nQuestion: {question}"
    if provider == "openai":
        return {"type": "input_text", "text": text}
    return {"type": "text", "text": text}


def _parts_for_mode(
    provider: str,
    mode: str,
    source: str,
    question: str,
    optical: OpticalCompressionResult | None,
) -> list[dict[str, Any]] | None:
    if mode in {"raw", "text"}:
        part_type = "input_text" if provider == "openai" else "text"
        return [{"type": part_type, "text": _text_prompt(source, question)}]
    if optical is None:
        return None
    parts = optical.openai_parts() if provider == "openai" else optical.anthropic_blocks()
    return [*parts, _question_part(provider, question)]


def _call_openai(model: str, parts: list[dict[str, Any]], timeout: float) -> tuple[str, int | None]:
    from openai import OpenAI

    client = OpenAI(timeout=timeout)
    response = client.responses.create(
        model=model,
        input=cast(Any, [{"role": "user", "content": parts}]),
        max_output_tokens=250,
    )
    usage = getattr(response, "usage", None)
    return response.output_text, getattr(usage, "input_tokens", None)


def _call_anthropic(
    model: str, parts: list[dict[str, Any]], timeout: float
) -> tuple[str, int | None]:
    from anthropic import Anthropic

    client = Anthropic(timeout=timeout)
    response = client.messages.create(
        model=model,
        max_tokens=250,
        temperature=0,
        messages=cast(Any, [{"role": "user", "content": parts}]),
    )
    answer = "\n".join(block.text for block in response.content if block.type == "text")
    return answer, getattr(response.usage, "input_tokens", None)


def _call_model(
    provider: str, model: str, parts: list[dict[str, Any]], timeout: float
) -> tuple[str, int | None]:
    if provider == "openai":
        return _call_openai(model, parts, timeout)
    return _call_anthropic(model, parts, timeout)


def run_accuracy_canary(
    *,
    provider: str,
    model: str,
    live: bool = False,
    paragraphs: int = 240,
    max_cases: int = 6,
    timeout: float = 45.0,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"unsupported provider: {provider}")
    credential = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if live and not os.environ.get(credential):
        raise RuntimeError(f"live {provider} canary requires {credential} in the environment")

    source = build_canary_corpus(paragraphs)
    cases = build_cases()[:max_cases]
    tokenizer = get_tokenizer(model)
    count = tokenizer.count_text
    context = " ".join(case.question for case in cases)
    crushed = TextCrusher().compress(source, context=context, target_ratio=0.5).compressed

    owned_cache = None
    if cache_dir is None:
        owned_cache = tempfile.TemporaryDirectory(prefix="headroom-accuracy-canary-")
        cache_dir = owned_cache.name
    compressor = TextOpticalCompressor(cache_dir, count_tokens=count)
    optical = compressor.compress(
        source,
        provider=provider,
        model=model,
        immutable=True,
        age_turns=3,
        detail="low" if provider == "openai" else "high",
    )
    text_optical = compressor.compress(
        crushed,
        provider=provider,
        model=model,
        immutable=True,
        age_turns=3,
        detail="low" if provider == "openai" else "high",
        generation=1,
    )
    sources = {"raw": source, "text": crushed, "optical": source, "text_optical": crushed}
    optical_results = {"raw": None, "text": None, "optical": optical, "text_optical": text_optical}
    raw_estimate = sum(count(_text_prompt(source, case.question)) for case in cases)
    rows: dict[str, ModeResult] = {}
    raw_correct: int | None = None
    raw_actual_tokens: int | None = None

    for mode in MODES:
        transformed = optical_results[mode]
        available = mode in {"raw", "text"} or transformed is not None
        answers: list[str] = []
        errors: list[str] = []
        latencies: list[float] = []
        actual_tokens: list[int] = []
        correct: int | None = 0 if live and available else None
        total_fields = sum(len(case.expected) for case in cases)
        if transformed is None:
            estimated = sum(count(_text_prompt(sources[mode], case.question)) for case in cases)
        else:
            estimated = sum(
                transformed.image_tokens
                + transformed.manifest_tokens
                + count(_question_part(provider, case.question)["text"])
                for case in cases
            )

        if live and available:
            for case in cases:
                parts = _parts_for_mode(provider, mode, sources[mode], case.question, transformed)
                assert parts is not None
                started = time.perf_counter()
                try:
                    answer, billed_input = _call_model(provider, model, parts, timeout)
                except Exception as exc:  # Live canaries report protocol failures per mode.
                    latencies.append(time.perf_counter() - started)
                    errors.append(f"{type(exc).__name__}: {exc}"[:300])
                    answers.append("")
                    continue
                latencies.append(time.perf_counter() - started)
                answers.append(answer)
                if billed_input is not None:
                    actual_tokens.append(billed_input)
                assert correct is not None
                correct += score_answer(answer, case.expected)

        if mode == "raw":
            raw_correct = correct
            raw_actual_tokens = sum(actual_tokens) if len(actual_tokens) == len(cases) else None
        accuracy = correct / total_fields * 100.0 if correct is not None else None
        retention = None
        if correct is not None and raw_correct:
            retention = correct / raw_correct * 100.0
        actual_input = sum(actual_tokens) if len(actual_tokens) == len(cases) else None
        actual_savings = None
        if actual_input is not None and raw_actual_tokens:
            actual_savings = (raw_actual_tokens - actual_input) / raw_actual_tokens * 100.0
        rows[mode] = ModeResult(
            mode=mode,
            available=available,
            estimated_input_tokens=estimated,
            estimated_savings_pct=(raw_estimate - estimated) / max(1, raw_estimate) * 100.0,
            actual_input_tokens=actual_input,
            actual_savings_pct=actual_savings,
            correct_fields=correct,
            total_fields=total_fields,
            accuracy_pct=accuracy,
            baseline_retention_pct=retention,
            completed_cases=len(answers) - len(errors),
            mean_latency_seconds=(sum(latencies) / len(latencies) if latencies else None),
            answers=tuple(answers),
            errors=tuple(errors),
        )

    if owned_cache is not None:
        owned_cache.cleanup()
    return {
        "provider": provider,
        "model": model,
        "live": live,
        "cases": [case.name for case in cases],
        "effective_headroom_env": {
            key: os.environ.get(key) for key in sorted(AUDITED_ENV_KEYS) if key in os.environ
        },
        "results": {mode: asdict(row) for mode, row in rows.items()},
    }


def clean_worker_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if source is None else source)
    for key in tuple(env):
        if key.startswith("HEADROOM_"):
            env.pop(key)
    env.update(CONTROLLED_HEADROOM_ENV)
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=tuple(DEFAULT_MODELS), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--live", action="store_true", help="make bounded paid API calls")
    parser.add_argument("--paragraphs", type=int, default=240)
    parser.add_argument("--max-cases", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--timeout", type=float, default=45.0, help="per-request timeout")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    if args.worker:
        result = run_accuracy_canary(
            provider=args.provider,
            model=model,
            live=args.live,
            paragraphs=args.paragraphs,
            max_cases=args.max_cases,
            timeout=args.timeout,
        )
        print(json.dumps(result))
        return

    inherited = {key: os.environ.get(key) for key in sorted(AUDITED_ENV_KEYS) if key in os.environ}
    command = [
        sys.executable,
        "-m",
        "benchmarks.optical_accuracy_canary",
        "--worker",
        "--provider",
        args.provider,
        "--model",
        model,
        "--paragraphs",
        str(args.paragraphs),
        "--max-cases",
        str(args.max_cases),
        "--timeout",
        str(args.timeout),
    ]
    if args.live:
        command.append("--live")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=clean_worker_environment(),
        timeout=max(60.0, args.timeout * args.max_cases * len(MODES) + 30.0),
    )
    result = json.loads(completed.stdout)
    result["inherited_headroom_env"] = inherited
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
