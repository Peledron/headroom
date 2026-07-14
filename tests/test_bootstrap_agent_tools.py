from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "headroom" / "cli" / "bootstrap_agent_tools.py"
    spec = importlib.util.spec_from_file_location("bootstrap_agent_tools", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_languages_is_bounded_and_ignores_large_trees(tmp_path) -> None:
    mod = _module()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "ignored.go").write_text("package ignored\n", encoding="utf-8")
    assert mod.detect_languages(tmp_path) == ["python", "rust", "typescript", "bash"]


def test_serena_command_repeats_language_option(tmp_path) -> None:
    mod = _module()
    command = mod.serena_create_command(tmp_path, ["python", "rust"])
    assert command.count("--language") == 2
    assert command[-1] == "rust"


def test_existing_serena_config_preserves_unrelated_settings(tmp_path) -> None:
    mod = _module()
    config = tmp_path / "project.yml"
    config.write_text(
        'project_name: "old"\nlanguages:\n- typescript\nread_only: true\n',
        encoding="utf-8",
    )
    assert mod.update_serena_languages(config, "new", ["python", "rust"]) is True
    updated = config.read_text(encoding="utf-8")
    assert 'project_name: "new"' in updated
    assert "- python\n- rust" in updated
    assert "read_only: true" in updated


def test_plan_uses_limited_sync_for_existing_index(tmp_path, monkeypatch) -> None:
    mod = _module()
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / ".tokensave").mkdir()
    (tmp_path / ".tokensave" / "tokensave.db").write_bytes(b"")
    (tmp_path / ".serena").mkdir()
    (tmp_path / ".serena" / "project.yml").write_text(
        'project_name: "x"\nlanguages:\n- python\n', encoding="utf-8"
    )
    monkeypatch.setattr(mod.shutil, "which", lambda name: f"/bin/{name}")
    plan = mod.build_plan(tmp_path, ["claude", "codex"])
    assert ["tokensave", "install", "--agent", "claude", "--git-hook", "no"] in plan
    assert any("tokensave-limited" in command[0] and "sync" in command for command in plan)
    assert plan[-1][:3] == ["serena", "project", "health-check"]


def test_codex_timeout_is_restored_after_tokensave_rewrite(tmp_path) -> None:
    mod = _module()
    config = tmp_path / "config.toml"
    config.write_text(
        '[mcp_servers.tokensave]\nargs = ["serve"]\ncommand = "tokensave"\n\n'
        '[mcp_servers.serena]\ncommand = "serena"\n',
        encoding="utf-8",
    )
    assert mod.ensure_codex_tokensave_timeout(config) is True
    assert "startup_timeout_sec = 30.0" in config.read_text(encoding="utf-8")
    assert mod.ensure_codex_tokensave_timeout(config) is False


def test_bounded_tokensave_policy_is_appended_once(tmp_path) -> None:
    mod = _module()
    codex = tmp_path / ".codex" / "AGENTS.md"
    claude = tmp_path / ".claude" / "CLAUDE.md"
    codex.parent.mkdir()
    claude.parent.mkdir()
    codex.write_text("# Codex guidance\n", encoding="utf-8")
    claude.write_text("# Claude guidance\n", encoding="utf-8")

    changed = mod.ensure_bounded_tokensave_policy(
        ["codex", "claude"], home=tmp_path
    )
    assert changed == [codex, claude]
    assert codex.read_text(encoding="utf-8").count(mod.TOKENSAVE_BUDGET_MARKER) == 1
    assert claude.read_text(encoding="utf-8").count(mod.TOKENSAVE_BUDGET_MARKER) == 1

    assert mod.ensure_bounded_tokensave_policy(
        ["codex", "claude"], home=tmp_path
    ) == []
