"""Idempotently bootstrap TokenSave and Serena for Claude and Codex."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

IGNORED_PARTS = frozenset(
    {
        ".git",
        ".tokensave",
        ".venv",
        "build",
        "corpus",
        "corpora",
        "data",
        "dist",
        "kernel",
        "node_modules",
        "target",
        "vendor",
    }
)
LANGUAGE_MARKERS = (
    ("python", ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")),
    ("rust", ("Cargo.toml",)),
    ("typescript", ("package.json", "tsconfig.json")),
    ("bash", (".shellcheckrc",)),
    ("go", ("go.mod",)),
    ("java", ("pom.xml", "build.gradle", "build.gradle.kts")),
)


def detect_languages(root: Path, *, file_limit: int = 2_000) -> list[str]:
    """Detect project languages without traversing massive source trees."""
    found = {
        language
        for language, markers in LANGUAGE_MARKERS
        if any((root / marker).exists() for marker in markers)
    }
    extension_languages = {
        ".py": "python",
        ".rs": "rust",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".sh": "bash",
        ".bash": "bash",
        ".go": "go",
        ".java": "java",
    }
    seen = 0
    for path in root.rglob("*"):
        if any(part.lower() in IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        seen += 1
        language = extension_languages.get(path.suffix.lower())
        if language:
            found.add(language)
        if seen >= file_limit:
            break
    preferred = ["python", "rust", "typescript", "bash", "go", "java"]
    return [language for language in preferred if language in found]


def limited_tokensave_prefix() -> list[str]:
    limited = shutil.which("tokensave-limited")
    if limited:
        return [limited]
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        "MemoryMax=5G",
        "-p",
        "MemorySwapMax=1G",
        "tokensave",
    ]


def serena_create_command(root: Path, languages: Sequence[str]) -> list[str]:
    command = ["serena", "project", "create", str(root), "--name", root.name]
    for language in languages:
        command.extend(("--language", language))
    return command


def update_serena_languages(config_path: Path, project_name: str, languages: Sequence[str]) -> bool:
    """Update an existing project file without discarding user settings."""
    if not config_path.exists():
        return False
    source = config_path.read_text(encoding="utf-8")
    updated = re.sub(
        r"^project_name:\s*[^\n]+",
        f'project_name: "{project_name}"',
        source,
        count=1,
        flags=re.MULTILINE,
    )
    block = "languages:\n" + "".join(f"- {language}\n" for language in languages).rstrip("\n")
    updated = re.sub(
        r"^languages:\s*\n(?:- [^\n]+\n?)+",
        block + "\n",
        updated,
        count=1,
        flags=re.MULTILINE,
    )
    if updated == source:
        return False
    config_path.write_text(updated, encoding="utf-8")
    return True


def ensure_codex_tokensave_timeout(
    config_path: Path | None = None, *, timeout_seconds: float = 30.0
) -> bool:
    """Restore the TokenSave startup timeout after its installer rewrites TOML."""
    if config_path is None:
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        config_path = codex_home / "config.toml"
    if not config_path.exists():
        return False
    source = config_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(^\[mcp_servers\.tokensave\]\n)(?P<body>(?:(?!^\[).*(?:\n|$))*)",
        flags=re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None or "startup_timeout_sec" in match.group("body"):
        return False
    updated = (
        source[: match.end("body")]
        + f"startup_timeout_sec = {timeout_seconds:.1f}\n"
        + source[match.end("body") :]
    )
    config_path.write_text(updated, encoding="utf-8")
    return True


def build_plan(
    root: Path, agents: Iterable[str], *, install_missing: bool = True
) -> list[list[str]]:
    root = root.resolve()
    plan: list[list[str]] = []
    if not shutil.which("tokensave"):
        if not install_missing:
            raise RuntimeError("tokensave is missing")
        if not shutil.which("cargo"):
            raise RuntimeError("tokensave is missing and cargo is unavailable")
        plan.append(["cargo", "install", "tokensave"])
    if not shutil.which("serena"):
        if not install_missing:
            raise RuntimeError("serena is missing")
        if not shutil.which("uv"):
            raise RuntimeError("serena is missing and uv is unavailable")
        plan.append(
            [
                "uv",
                "tool",
                "install",
                "--from",
                "git+https://github.com/oraios/serena",
                "serena-agent",
            ]
        )
    for agent in agents:
        plan.append(["tokensave", "install", "--agent", agent, "--git-hook", "no"])
    tokensave_action = "sync" if (root / ".tokensave" / "tokensave.db").exists() else "init"
    plan.append([*limited_tokensave_prefix(), tokensave_action, str(root)])
    languages = detect_languages(root)
    if not languages:
        raise RuntimeError(f"no supported project languages detected in {root}")
    if not (root / ".serena" / "project.yml").exists():
        plan.append(serena_create_command(root, languages))
    plan.append(["serena", "project", "health-check", str(root)])
    return plan


def run_plan(plan: Sequence[Sequence[str]], *, dry_run: bool) -> None:
    for command in plan:
        print("+", " ".join(command))
        if not dry_run:
            subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", default=".")
    parser.add_argument("--agent", action="append", choices=("claude", "codex"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-install-missing", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.project).resolve()
    agents = args.agent or ["claude", "codex"]
    try:
        languages = detect_languages(root)
        config_path = root / ".serena" / "project.yml"
        if config_path.exists() and not args.dry_run:
            update_serena_languages(config_path, root.name, languages)
        plan = build_plan(root, agents, install_missing=not args.no_install_missing)
        run_plan(plan, dry_run=args.dry_run)
        if not args.dry_run and "codex" in agents:
            ensure_codex_tokensave_timeout()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
