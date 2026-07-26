"""Contain CrewAI's import-time monkeypatch of ``warnings.warn``.

Importing ``crewai`` runs ``_suppress_pydantic_deprecation_warnings()``, which
replaces ``warnings.warn`` with a wrapper accepting only ``(message, category,
stacklevel, source)`` and never restores it. Python 3.13's ``_strptime`` calls
``warnings.warn(..., skip_file_prefixes=...)``, so once ``crewai`` has been
imported, every later test that parses a date through ``dateparser`` dies with a
TypeError. That is how a passing benchmark in ``tests/test_evals`` fails only
when the full suite runs.

The patch is a CrewAI bug, not ours, and it belongs upstream. Here the job is to
keep it from leaking. The import happens at collection time, at the top of
``test_agents.py``, which is after this conftest loads and before any fixture
runs, so the pristine function is captured at module scope and put back around
every test in this package.
"""

from __future__ import annotations

import warnings

import pytest

# Captured before ``test_agents.py`` is collected, so this is the stdlib
# function rather than CrewAI's replacement.
_PRISTINE_WARN = warnings.warn


@pytest.fixture(autouse=True)
def _restore_warnings_warn():
    if warnings.warn is not _PRISTINE_WARN:
        warnings.warn = _PRISTINE_WARN
    try:
        yield
    finally:
        if warnings.warn is not _PRISTINE_WARN:
            warnings.warn = _PRISTINE_WARN
