"""Experimental deterministic text-as-image compression.

The compressor is intentionally separate from ordinary text transforms. It
only accepts old immutable prose-heavy outputs and fails closed when rendering
would not save a conservative number of provider-billed input tokens.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FACT_PATTERNS: tuple[tuple[int, re.Pattern[str], int], ...] = (
    (0, re.compile(r"\b[A-Z][A-Z0-9_]{2,}=[^\s)\"'<>]+"), 0),
    (
        0,
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        0,
    ),
    (0, re.compile(r"\b(?=[0-9a-fA-F]{7,40}\b)(?=[0-9a-fA-F]*\d)[0-9a-fA-F]+\b"), 0),
    (0, re.compile(r"\b(?=[A-Z0-9-]{3,120}\b)(?=[A-Z0-9-]*\d)[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b"), 0),
    (0, re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b"), 0),
    (1, re.compile(r"\bhttps?://[^\s)\"'<>]+"), 0),
    (1, re.compile(r"(?:[\w@~+.-]+)?(?:/[\w.@+-]+)+\.[A-Za-z]\w{0,8}\b"), 0),
    (1, re.compile(r"/[\w.@+-]+(?:/[\w.@+-]+)+/?"), 0),
    (1, re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?\b"), 0),
    (1, re.compile(r"(?:^|[^\w-])(--?[A-Za-z][\w-]+)"), 1),
    (2, re.compile(r"\b\d[\d,_]{3,}\b"), 0),
    (2, re.compile(r"\b(?:[a-z]+|[A-Z][a-z0-9]+)(?:[A-Z][a-z0-9]*)+\b"), 0),
)


def extract_optical_facts(
    text: str, *, max_entries: int = 64, max_scan_chars: int = 262_144
) -> tuple[tuple[str, int], ...]:
    """Extract a deterministic text sidecar for precision-critical identifiers."""
    if max_entries <= 0 or max_scan_chars <= 0:
        return ()
    scan = text[:max_scan_chars]
    found: dict[str, tuple[int, int]] = {}
    for priority, pattern, group in _FACT_PATTERNS:
        for match in pattern.finditer(scan):
            token = match.group(group).rstrip(".,:;!?")
            if not 3 <= len(token) <= 120:
                continue
            previous = found.get(token)
            count = 1 if previous is None else previous[1] + 1
            found[token] = (min(priority, previous[0]) if previous else priority, count)

    ranked = sorted(found, key=lambda token: (found[token][0], -len(token), token))
    kept: list[str] = []
    for token in ranked:
        if any(existing != token and existing.find(token) >= 0 for existing in kept):
            continue
        kept.append(token)
        if len(kept) >= max_entries:
            break
    return tuple((token, found[token][1]) for token in kept)


def _factsheet_text(facts: tuple[tuple[str, int], ...]) -> str:
    if not facts:
        return ""
    values = " | ".join(f"{token} x{count}" if count > 1 else token for token, count in facts)
    return (
        " Exact identifiers from the rendered text follow. Quote these values instead of "
        f"transcribing them from pixels: {values}."
    )


def _anthropic_image_tokens(model: str, width: int, height: int) -> int:
    """Match Anthropic's published 28px patch and resolution-tier rules."""
    normalized = model.lower().replace(".", "-")
    high_resolution = any(
        family in normalized
        for family in ("fable-5", "mythos-5", "opus-4-8", "opus-4-7", "sonnet-5")
    )
    max_edge, max_tokens = (2_576, 4_784) if high_resolution else (1_568, 1_568)

    def count(w: int, h: int) -> int:
        return math.ceil(w / 28) * math.ceil(h / 28)

    def fits(w: int, h: int) -> bool:
        return (
            math.ceil(w / 28) * 28 <= max_edge
            and math.ceil(h / 28) * 28 <= max_edge
            and count(w, h) <= max_tokens
        )

    if fits(width, height):
        return count(width, height)
    if height > width:
        return _anthropic_image_tokens(model, height, width)

    aspect_ratio = width / height
    low, high = 1, width
    while low + 1 < high:
        middle = (low + high) // 2
        short_edge = max(round(middle / aspect_ratio), 1)
        if fits(middle, short_edge):
            low = middle
        else:
            high = middle
    return count(low, max(round(low / aspect_ratio), 1))


def _openai_patch_profile(model: str, detail: str) -> tuple[int | None, int | None, float] | None:
    """Return the published patch budget, edge cap, and billing multiplier."""
    normalized = model.lower()
    multiplier = 1.0
    if any(name in normalized for name in ("gpt-5.4-mini", "gpt-5-mini", "gpt-4.1-mini")):
        multiplier = 1.62
    elif any(name in normalized for name in ("gpt-5.4-nano", "gpt-5-nano", "gpt-4.1-nano")):
        multiplier = 2.46
    elif "o4-mini" in normalized:
        multiplier = 1.72

    if "gpt-5.6" in normalized:
        if detail in {"original", "auto"}:
            return None, None, multiplier
        return 2_500, 2_048, multiplier
    if "gpt-5.5" in normalized:
        if detail in {"original", "auto"}:
            return 10_000, 6_000, multiplier
        return 2_500, 2_048, multiplier
    if "gpt-5.4" in normalized and "mini" not in normalized and "nano" not in normalized:
        if detail == "original":
            return 10_000, 6_000, multiplier
        return 2_500, 2_048, multiplier

    patch_families = (
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.2",
        "gpt-5.3-codex",
        "gpt-5-codex-mini",
        "gpt-5.1-codex-mini",
        "gpt-5.2-codex",
        "gpt-5.2-chat-latest",
        "o4-mini",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    )
    if any(name in normalized for name in patch_families):
        return 1_536, 2_048, multiplier
    return None


def _openai_patch_tokens(
    width: int, height: int, patch_budget: int | None, max_edge: int | None, multiplier: float
) -> int:
    patches = math.ceil(width / 32) * math.ceil(height / 32)
    if patch_budget is None or (
        patches <= patch_budget and (max_edge is None or max(width, height) <= max_edge)
    ):
        return max(1, math.ceil(patches * multiplier))

    scale = min(
        math.sqrt(patch_budget * 32 * 32 / (width * height)),
        max_edge / max(width, height) if max_edge is not None else 1.0,
    )
    scaled_width = width * scale
    scaled_height = height * scale
    adjusted = scale * min(
        math.floor(scaled_width / 32) / max(scaled_width / 32, 1),
        math.floor(scaled_height / 32) / max(scaled_height / 32, 1),
    )
    resized_width = max(1, math.floor(width * adjusted))
    resized_height = max(1, math.floor(height * adjusted))
    patches = min(patch_budget, math.ceil(resized_width / 32) * math.ceil(resized_height / 32))
    return max(1, math.ceil(patches * multiplier))


@dataclass(frozen=True, slots=True)
class OpticalRenderConfig:
    width: int = 768
    height: int = 768
    font_size: int = 10
    margin: int = 16
    columns: int = 2
    column_gutter: int = 16
    line_spacing: int = 2
    min_source_chars: int = 8_000
    min_savings_fraction: float = 0.15
    max_factsheet_entries: int = 64
    max_factsheet_scan_chars: int = 262_144
    semantic_colors: bool = False
    format_version: int = 1


@dataclass(frozen=True, slots=True)
class OpticalPage:
    index: int
    sha256: str
    width: int
    height: int
    png: bytes


@dataclass(frozen=True, slots=True)
class OpticalCompressionResult:
    source_sha256: str
    settings_sha256: str
    pages: tuple[OpticalPage, ...]
    manifest: str
    source_tokens: int
    image_tokens: int
    manifest_tokens: int
    estimated_tokens_saved: int
    provider: str
    model: str
    detail: str
    generation: int

    def anthropic_blocks(self) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": self.manifest}]
        for page in self.pages:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(page.png).decode("ascii"),
                    },
                }
            )
        return blocks

    def openai_parts(self) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"type": "input_text", "text": self.manifest}]
        for page in self.pages:
            encoded = base64.b64encode(page.png).decode("ascii")
            parts.append(
                {
                    "type": "input_image",
                    "detail": self.detail,
                    "image_url": f"data:image/png;base64,{encoded}",
                }
            )
        return parts


class TextOpticalCompressor:
    """Render eligible immutable text to deterministic, cached PNG pages."""

    _CODE_LINE = re.compile(
        r"^\s*(?:diff --git|@@ |[+\-]{3} |(?:class|def|fn|func|interface|struct)\s|"
        r"(?:import|from|use|package)\s|[{}][,;]?$)"
    )
    _IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{24,}\b")
    _SEMANTIC_LINE_STYLES: tuple[
        tuple[
            re.Pattern[str], tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
        ],
        ...,
    ] = (
        (
            re.compile(r"(?i)\b(error|failed|failure|fatal|exception|traceback|panic)\b"),
            ((116, 18, 18), (255, 232, 232), (210, 45, 45)),
        ),
        (
            re.compile(r"(?i)\b(warn|warning|retry|degraded|timeout)\b"),
            ((102, 61, 0), (255, 244, 204), (220, 155, 0)),
        ),
        (
            re.compile(r"(?i)\b(ok|passed|success|healthy|completed|ready)\b"),
            ((0, 84, 43), (229, 249, 235), (35, 160, 82)),
        ),
        (
            re.compile(r"^(?:\$|>|#|==+|--+|\[[A-Z][A-Z0-9 _-]*\])"),
            ((40, 48, 110), (235, 239, 255), (78, 95, 190)),
        ),
        (
            re.compile(r"(?:^|\s)(?:[A-Za-z]:)?[/\\][\w.@+\\/-]+(?::\d+)?"),
            ((0, 72, 112), (231, 246, 255), (40, 140, 195)),
        ),
    )

    def __init__(
        self,
        cache_dir: str | Path,
        config: OpticalRenderConfig | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config or OpticalRenderConfig()
        self.count_tokens = count_tokens or (lambda text: max(1, math.ceil(len(text) / 4)))

    def eligible(
        self,
        text: str,
        *,
        immutable: bool,
        age_turns: int,
        content_kind: str = "prose",
    ) -> tuple[bool, str]:
        if not immutable or age_turns < 2:
            return False, "not_old_immutable_content"
        if content_kind in {"system", "patch", "code", "tool_arguments", "current_output"}:
            return False, "exact_text_required"
        if len(text) < self.config.min_source_chars:
            return False, "below_size_floor"
        if text.lstrip().startswith(("{", "[")):
            return False, "structured_content"
        lines = text.splitlines() or [text]
        code_lines = sum(bool(self._CODE_LINE.search(line)) for line in lines)
        if code_lines / len(lines) > 0.08:
            return False, "code_heavy"
        if len(self._IDENTIFIER.findall(text)) > max(4, len(lines) // 12):
            return False, "identifier_heavy"
        printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
        if printable / max(1, len(text)) < 0.98:
            return False, "binary_or_control_heavy"
        return True, "eligible"

    def compress(
        self,
        text: str,
        *,
        provider: str,
        model: str,
        immutable: bool,
        age_turns: int,
        content_kind: str = "prose",
        detail: str = "low",
        generation: int = 0,
        warm_prefix: bool = False,
    ) -> OpticalCompressionResult | None:
        allowed, _ = self.eligible(
            text, immutable=immutable, age_turns=age_turns, content_kind=content_kind
        )
        if not allowed:
            return None
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        settings = {
            "config": self.config.__dict__
            if hasattr(self.config, "__dict__")
            else {name: getattr(self.config, name) for name in self.config.__slots__},
            "provider": provider,
            "model": model,
            "detail": detail,
            "generation": generation,
        }
        settings_hash = hashlib.sha256(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_key = hashlib.sha256(f"{source_hash}:{settings_hash}".encode()).hexdigest()
        page_dir = self.cache_dir / cache_key
        if warm_prefix and not (page_dir / "manifest.json").exists():
            return None

        pages = self._load_pages(page_dir)
        if pages is None:
            pages = self._render_pages(text)
            self._store_pages(page_dir, source_hash, settings_hash, pages)

        facts = extract_optical_facts(
            text,
            max_entries=self.config.max_factsheet_entries,
            max_scan_chars=self.config.max_factsheet_scan_chars,
        )
        manifest = self._manifest(source_hash, settings_hash, pages, generation, facts)
        source_tokens = self.count_tokens(text)
        manifest_tokens = self.count_tokens(manifest)
        try:
            image_tokens = sum(
                self.estimate_image_tokens(
                    provider=provider,
                    model=model,
                    width=page.width,
                    height=page.height,
                    detail=detail,
                )
                for page in pages
            )
        except ValueError:
            return None
        saved = source_tokens - image_tokens - manifest_tokens
        if saved <= 0 or saved / max(1, source_tokens) < self.config.min_savings_fraction:
            return None
        return OpticalCompressionResult(
            source_sha256=source_hash,
            settings_sha256=settings_hash,
            pages=tuple(pages),
            manifest=manifest,
            source_tokens=source_tokens,
            image_tokens=image_tokens,
            manifest_tokens=manifest_tokens,
            estimated_tokens_saved=saved,
            provider=provider,
            model=model,
            detail=detail,
            generation=generation,
        )

    def compress_anthropic_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        generation: int,
        mutable_tail_messages: int = 4,
    ) -> tuple[list[dict[str, Any]], int]:
        """Optically encode old prose tool results during a cold build or rebase."""
        output = copy.deepcopy(messages)
        transformed = 0
        cutoff = max(0, len(output) - max(1, mutable_tail_messages))
        for message_index, message in enumerate(output[:cutoff]):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                text = block.get("content")
                if not isinstance(text, str):
                    continue
                result = self.compress(
                    text,
                    provider="anthropic",
                    model=model,
                    immutable=True,
                    age_turns=max(2, cutoff - message_index),
                    content_kind="prose",
                    detail="high",
                    generation=generation,
                    warm_prefix=False,
                )
                if result is not None:
                    # Anthropic charges images by pixel area. Keep the existing
                    # extractive text path when it is cheaper than optical.
                    from headroom.transforms.text_crusher import TextCrusher

                    text_candidate = (
                        TextCrusher().compress(text, context="", target_ratio=0.5).compressed
                    )
                    text_candidate_tokens = self.count_tokens(text_candidate)
                    optical_tokens = result.image_tokens + result.manifest_tokens
                    if optical_tokens >= text_candidate_tokens:
                        continue
                    block["content"] = result.anthropic_blocks()
                    transformed += 1
        return output, transformed

    @staticmethod
    def estimate_image_tokens(
        *, provider: str, model: str, width: int, height: int, detail: str
    ) -> int:
        if provider == "anthropic":
            return _anthropic_image_tokens(model, width, height)
        if provider != "openai":
            raise ValueError(f"unsupported optical cost provider: {provider!r}")

        normalized = model.lower()
        tile_costs: tuple[int, int] | None = None
        if "gpt-4o-mini" in normalized:
            tile_costs = (2_833, 5_667)
        elif any(name in normalized for name in ("gpt-4o", "gpt-4.1", "gpt-4.5")) and not any(
            name in normalized for name in ("gpt-4.1-mini", "gpt-4.1-nano")
        ):
            tile_costs = (85, 170)
        elif normalized in {"gpt-5", "gpt-5-chat-latest"}:
            tile_costs = (70, 140)
        elif normalized in {"o1", "o1-pro", "o3"}:
            tile_costs = (75, 150)
        elif "computer-use-preview" in normalized:
            tile_costs = (65, 129)

        if tile_costs is not None:
            base, per_tile = tile_costs
            if detail == "low":
                return base
            fit_scale = min(1.0, 2048 / max(width, height))
            resized_width = width * fit_scale
            resized_height = height * fit_scale
            detail_scale = 768 / min(resized_width, resized_height)
            resized_width *= detail_scale
            resized_height *= detail_scale
            tiles = math.ceil(resized_width / 512) * math.ceil(resized_height / 512)
            return base + tiles * per_tile

        profile = _openai_patch_profile(model, detail)
        if profile is None:
            raise ValueError(f"no published OpenAI image-token profile for {model!r}")
        patch_budget, max_edge, multiplier = profile
        return _openai_patch_tokens(width, height, patch_budget, max_edge, multiplier)

    def _render_pages(self, text: str) -> list[OpticalPage]:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise RuntimeError("Text optical compression requires Pillow") from exc

        cfg = self.config
        font = ImageFont.load_default(size=cfg.font_size)
        mode = "RGB" if cfg.semantic_colors else "L"
        background: int | tuple[int, int, int] = (255, 255, 255) if cfg.semantic_colors else 255
        probe = Image.new(mode, (cfg.width, cfg.height), background)
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), "M", font=font)
        char_width = max(1, math.ceil(bbox[2] - bbox[0]))
        line_height = max(1, math.ceil(bbox[3] - bbox[1])) + cfg.line_spacing
        column_count = max(1, cfg.columns)
        text_width = cfg.width - 2 * cfg.margin - (column_count - 1) * cfg.column_gutter
        column_width = max(char_width * 20, text_width // column_count)
        columns = max(20, column_width // char_width)
        rows = max(5, (cfg.height - 2 * cfg.margin) // line_height)
        wrapped = self._wrap(text, columns)
        styled = self._wrap_styled(text, columns) if cfg.semantic_colors else []
        page_rows = rows * column_count
        chunks = [wrapped[i : i + page_rows] for i in range(0, len(wrapped), page_rows)]
        pages: list[OpticalPage] = []
        for index, lines in enumerate(chunks):
            image = Image.new(mode, (cfg.width, cfg.height), background)
            page_draw = ImageDraw.Draw(image)
            for column_index in range(column_count):
                column_lines = lines[column_index * rows : (column_index + 1) * rows]
                if not column_lines:
                    break
                x = cfg.margin + column_index * (column_width + cfg.column_gutter)
                if cfg.semantic_colors:
                    page_offset = index * page_rows + column_index * rows
                    styled_lines = styled[page_offset : page_offset + len(column_lines)]
                    for row_index, (line, style) in enumerate(styled_lines):
                        y = cfg.margin + row_index * line_height
                        fill = (0, 0, 0)
                        if style is not None:
                            fill, line_background, accent = style
                            page_draw.rectangle(
                                (x, y, x + column_width - 1, y + line_height - 1),
                                fill=line_background,
                            )
                            page_draw.rectangle(
                                (max(0, x - 3), y, max(0, x - 1), y + line_height - 1),
                                fill=accent,
                            )
                        page_draw.text((x, y), line, fill=fill, font=font)
                else:
                    page_draw.multiline_text(
                        (x, cfg.margin),
                        "\n".join(column_lines),
                        fill=0,
                        font=font,
                        spacing=cfg.line_spacing,
                    )
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=False, compress_level=9)
            png = output.getvalue()
            pages.append(
                OpticalPage(
                    index=index,
                    sha256=hashlib.sha256(png).hexdigest(),
                    width=cfg.width,
                    height=cfg.height,
                    png=png,
                )
            )
        return pages

    @staticmethod
    def _wrap(text: str, columns: int) -> list[str]:
        output: list[str] = []
        for raw_line in text.expandtabs(4).splitlines():
            if not raw_line:
                output.append("")
                continue
            remaining = raw_line
            while len(remaining) > columns:
                split = remaining.rfind(" ", 0, columns + 1)
                if split <= 0:
                    split = columns
                output.append(remaining[:split])
                remaining = remaining[split:].lstrip(" ")
            output.append(remaining)
        return output or [""]

    @classmethod
    def _wrap_styled(
        cls, text: str, columns: int
    ) -> list[
        tuple[
            str,
            tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None,
        ]
    ]:
        output: list[
            tuple[
                str,
                tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None,
            ]
        ] = []
        for raw_line in text.expandtabs(4).splitlines():
            style = cls._semantic_line_style(raw_line)
            wrapped = cls._wrap(raw_line, columns)
            output.extend((line, style) for line in wrapped)
        return output or [("", None)]

    @classmethod
    def _semantic_line_style(
        cls, line: str
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None:
        for pattern, style in cls._SEMANTIC_LINE_STYLES:
            if pattern.search(line):
                return style
        return None

    @staticmethod
    def _manifest(
        source_hash: str,
        settings_hash: str,
        pages: list[OpticalPage],
        generation: int,
        facts: tuple[tuple[str, int], ...],
    ) -> str:
        page_hashes = ",".join(f"{page.index + 1}:{page.sha256[:12]}" for page in pages)
        header = (
            "[headroom-optical v1 "
            f"source={source_hash[:16]} settings={settings_hash[:16]} "
            f"generation={generation} pages={len(pages)} page_hashes={page_hashes}]"
        )
        return header + _factsheet_text(facts)

    @staticmethod
    def _load_pages(page_dir: Path) -> list[OpticalPage] | None:
        manifest_path = page_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            pages = []
            for item in metadata["pages"]:
                png = (page_dir / item["file"]).read_bytes()
                digest = hashlib.sha256(png).hexdigest()
                if digest != item["sha256"]:
                    return None
                pages.append(
                    OpticalPage(
                        index=item["index"],
                        sha256=digest,
                        width=item["width"],
                        height=item["height"],
                        png=png,
                    )
                )
            return pages
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _store_pages(
        page_dir: Path,
        source_hash: str,
        settings_hash: str,
        pages: list[OpticalPage],
    ) -> None:
        page_dir.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "source_sha256": source_hash,
            "settings_sha256": settings_hash,
            "pages": [],
        }
        for page in pages:
            filename = f"page-{page.index:04d}.png"
            (page_dir / filename).write_bytes(page.png)
            metadata["pages"].append(
                {
                    "index": page.index,
                    "file": filename,
                    "sha256": page.sha256,
                    "width": page.width,
                    "height": page.height,
                }
            )
        (page_dir / "manifest.json").write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
