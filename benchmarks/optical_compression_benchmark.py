#!/usr/bin/env python3
"""Synthetic cost benchmark for text, optical, and hybrid compression."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict, dataclass

from headroom.tokenizers import get_tokenizer
from headroom.transforms.text_crusher import TextCrusher
from headroom.transforms.text_optical import TextOpticalCompressor


@dataclass(frozen=True, slots=True)
class OpticalBenchmarkRow:
    mode: str
    billed_input_tokens: int
    tokens_saved: int
    savings_pct: float
    pages: int = 0
    deterministic: bool = True


def build_prose_corpus(paragraphs: int = 240) -> str:
    rows = []
    for index in range(paragraphs):
        rows.append(
            f"Observation {index:04d}. The service stayed healthy during the measured window. "
            "Cache reads avoided repeated prefix writes while the current tool observation "
            "remained mutable. The operator should preserve the stable prefix and inspect "
            f"request marker R-{index:04d} only when diagnosing this interval."
        )
    return "\n\n".join(rows)


def run_optical_benchmark(
    *, model: str = "gpt-4o-mini", provider: str = "openai", paragraphs: int = 240
) -> dict[str, OpticalBenchmarkRow]:
    text = build_prose_corpus(paragraphs)
    tokenizer = get_tokenizer(model)
    count = tokenizer.count_text
    raw_tokens = count(text)
    crushed = TextCrusher().compress(text, context="cache stability", target_ratio=0.5).compressed
    text_tokens = count(crushed)

    with tempfile.TemporaryDirectory(prefix="headroom-optical-bench-") as cache:
        compressor = TextOpticalCompressor(cache, count_tokens=count)
        low = compressor.compress(
            text,
            provider=provider,
            model=model,
            immutable=True,
            age_turns=3,
            detail="low",
        )
        high = compressor.compress(
            text,
            provider=provider,
            model=model,
            immutable=True,
            age_turns=3,
            detail="high",
        )
        hybrid = compressor.compress(
            crushed,
            provider=provider,
            model=model,
            immutable=True,
            age_turns=3,
            detail="low",
            generation=1,
        )

    def row(mode: str, billed: int, pages: int = 0) -> OpticalBenchmarkRow:
        saved = max(0, raw_tokens - billed)
        return OpticalBenchmarkRow(
            mode=mode,
            billed_input_tokens=billed,
            tokens_saved=saved,
            savings_pct=saved / raw_tokens * 100.0 if raw_tokens else 0.0,
            pages=pages,
        )

    return {
        "raw": row("raw", raw_tokens),
        "text": row("text", text_tokens),
        "optical_low": row(
            "optical_low",
            low.image_tokens + low.manifest_tokens if low else raw_tokens,
            len(low.pages) if low else 0,
        ),
        "optical_high": row(
            "optical_high",
            high.image_tokens + high.manifest_tokens if high else raw_tokens,
            len(high.pages) if high else 0,
        ),
        "hybrid_text_optical": row(
            "hybrid_text_optical",
            hybrid.image_tokens + hybrid.manifest_tokens if hybrid else text_tokens,
            len(hybrid.pages) if hybrid else 0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--paragraphs", type=int, default=240)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = run_optical_benchmark(
        model=args.model, provider=args.provider, paragraphs=args.paragraphs
    )
    if args.json:
        import json

        print(json.dumps({key: asdict(value) for key, value in results.items()}, indent=2))
        return
    print("mode                 billed_tokens  saved_tokens  savings   pages")
    for result in results.values():
        print(
            f"{result.mode:<21} {result.billed_input_tokens:>13,} "
            f"{result.tokens_saved:>13,} {result.savings_pct:>7.1f}% {result.pages:>7}"
        )


if __name__ == "__main__":
    main()
