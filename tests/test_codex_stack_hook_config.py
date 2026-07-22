from __future__ import annotations

from pathlib import Path

from benchmarks import codex_stack_canary


def test_full_home_links_hooks_and_enables_reviewed_hook(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "auth.json").write_text("{}\n", encoding="utf-8")
    (source / "hooks.json").write_text('{"hooks": {}}\n', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(source))

    home = codex_stack_canary._prepare_codex_home(tmp_path / "runs", "full")

    assert (home / "hooks.json").is_symlink()
    assert codex_stack_canary._arm_flags("full") == [
        "--dangerously-bypass-hook-trust"
    ]


def test_non_full_arms_do_not_bypass_hook_trust() -> None:
    assert "--dangerously-bypass-hook-trust" not in codex_stack_canary._arm_flags(
        "bare"
    )
    assert "--dangerously-bypass-hook-trust" not in codex_stack_canary._arm_flags(
        "headroom"
    )


def test_full_without_headroom_uses_direct_openai_provider() -> None:
    flags = codex_stack_canary._arm_flags("full-no-headroom")
    assert "--dangerously-bypass-hook-trust" in flags
    assert 'model_provider="openai"' in flags
