"""Breaker suite for two pure helpers landed in headroom/proxy/handlers/anthropic.py:

- ``_flatten_system_text`` / ``_system_looks_subagent``: the sub-agent
  detection signal used to pick a cheaper cache-write ttl tier.
- ``_token_prefix_mutation_worth_it``: the token-mode gate that decides
  whether an aggressive compression pass is allowed to bust a warm cache.

Offline, hermetic. No network, no proxy, imports the module directly.
"""

import pytest

from headroom.proxy.handlers.anthropic import (
    _flatten_system_text,
    _system_looks_subagent,
    _token_prefix_mutation_worth_it,
)

MODEL = "claude-opus-4-8"


# ── claim 1: no false positive on the main 1M session ─────────────────────────
# _system_looks_subagent(system, model_id) must be False whenever the
# "<model_id>[1m]" marker is present anywhere in the flattened system text,
# regardless of str-vs-block-list shape or how the marker is phrased around it.

class TestNoFalsePositiveMain1M:
    def test_single_string_system(self):
        system = f"You are Claude Code. The exact model ID is {MODEL}[1m]."
        assert _system_looks_subagent(system, MODEL) is False

    def test_single_block(self):
        system = [{"type": "text", "text": f"The exact model ID is {MODEL}[1m]."}]
        assert _system_looks_subagent(system, MODEL) is False

    def test_marker_in_a_different_block_than_intro(self):
        system = [
            {"type": "text", "text": "You are Claude Code, an agentic CLI tool."},
            {"type": "text", "text": f"The exact model ID is {MODEL}[1m]."},
            {"type": "text", "text": "Follow the user's instructions."},
        ]
        assert _system_looks_subagent(system, MODEL) is False

    def test_marker_in_first_of_several_blocks(self):
        system = [
            {"type": "text", "text": f"model: {MODEL}[1m]"},
            {"type": "text", "text": "Tool descriptions follow."},
            {"type": "text", "text": "Bash(command: str) -> str"},
        ]
        assert _system_looks_subagent(system, MODEL) is False

    def test_bare_mention_elsewhere_plus_real_marker_still_false(self):
        # The model id is name-dropped once WITHOUT [1m] (e.g. in an unrelated
        # tool-choice explanation) but the real "<id>[1m]" line is also present
        # somewhere in the head. A naive "first occurrence only" scan could
        # trip on the bare mention; the real function scans the whole text.
        system = [
            {"type": "text", "text": f"Prefer {MODEL} for planning."},
            {"type": "text", "text": f"The exact model ID is {MODEL}[1m]."},
        ]
        assert _system_looks_subagent(system, MODEL) is False

    @pytest.mark.parametrize(
        "text",
        [
            f"The exact model ID is {MODEL}[1m].",
            f"model: {MODEL}[1m]",
            f"MODEL_ID={MODEL}[1m]",
            f"You are running {MODEL}[1m], a large language model.",
            f"<system>id={MODEL}[1m]</system>",
            f"{MODEL}[1m]",
            f"...{MODEL}[1m]...{MODEL}[1m]...",  # marker repeated
        ],
    )
    def test_phrasing_variants_of_the_marker_line(self, text):
        assert _system_looks_subagent(text, MODEL) is False
        assert _system_looks_subagent([{"type": "text", "text": text}], MODEL) is False


# ── claim 2: no false positive on non-Claude-Code traffic ─────────────────────

class TestNoFalsePositiveNonClaudeCode:
    def test_generic_system_prompt(self):
        system = "You are a helpful assistant."
        assert _system_looks_subagent(system, MODEL) is False

    def test_generic_block_list(self):
        system = [{"type": "text", "text": "You are a helpful assistant."}]
        assert _system_looks_subagent(system, MODEL) is False

    def test_model_id_absent_entirely(self):
        system = [{"type": "text", "text": "Some unrelated system prompt about tools."}]
        assert _system_looks_subagent(system, MODEL) is False

    def test_none_system(self):
        assert _system_looks_subagent(None, MODEL) is False

    def test_empty_list_system(self):
        assert _system_looks_subagent([], MODEL) is False

    def test_empty_string_system(self):
        assert _system_looks_subagent("", MODEL) is False


# ── claim 3: substring-prefix collision, fixed ─────────────────────────────────
#
# The implementation now anchors both checks with word-boundary lookaround
# (`(?<![a-z0-9\-])<id>(?![a-z0-9\-])`), not plain substring tests. Anthropic's
# own model ids are literal prefixes of each other by construction (see
# headroom/providers/anthropic.py ANTHROPIC_CONTEXT_LIMITS: "claude-opus-4" is
# a prefix of "claude-opus-4-5-20251101", "claude-opus-4-6", "claude-opus-4-7",
# "claude-opus-4-8"), so this is not a contrived edge case: the boundary
# anchoring is what keeps a short family id from matching inside a longer
# rendered id's token.

class TestSubstringCollision:
    def test_short_model_id_prefix_not_misclassified(self):
        # model_id = "claude-opus-4" (a strict prefix of the id actually
        # printed in the head, "claude-opus-4-8"). The head carries a genuine
        # "<full-id>[1m]" main-session marker, but not "<model_id>[1m]"
        # literally. Word-boundary matching means the bare prefix "claude-
        # opus-4" does not match as a complete token inside "claude-opus-4-8",
        # so the function correctly reports "not a sub-agent".
        short_id = "claude-opus-4"
        system = [{"type": "text", "text": f"The exact model ID is claude-opus-4-8[1m]."}]
        result = _system_looks_subagent(system, short_id)
        assert result is False, (
            "fixed: a 1M main-session head for claude-opus-4-8 is no longer "
            "misclassified as a sub-agent when queried with the shorter "
            "prefix id 'claude-opus-4' -- word-boundary matching rejects "
            "the partial-token match"
        )

    def test_model_id_prefix_of_variant_suffix_not_misclassified(self):
        # model_id = "claude-opus-4-8", head actually describes the
        # "-thinking" variant of the same family with its own [1m] marker.
        # "claude-opus-4-8" is not a complete token inside "claude-opus-4-8-
        # thinking" (it is immediately followed by "-thinking"), so the
        # boundary check rejects the match.
        system = [
            {"type": "text", "text": "The exact model ID is claude-opus-4-8-thinking[1m]."}
        ]
        result = _system_looks_subagent(system, MODEL)
        assert result is False, (
            "fixed: claude-opus-4-8-thinking[1m] head is no longer "
            "misclassified as a sub-agent when queried with the prefix id "
            "'claude-opus-4-8' -- word-boundary matching rejects the "
            "partial-token match"
        )

    def test_short_model_id_prefix_not_misclassified_via_plain_string_system(self):
        short_id = "claude-opus-4"
        system = f"The exact model ID is claude-opus-4-8[1m]."
        assert _system_looks_subagent(system, short_id) is False


# ── claim 4: type hardening, no crashes ────────────────────────────────────────

class TestFlattenSystemTextTypeHardening:
    @pytest.mark.parametrize(
        "system,expected",
        [
            (None, ""),
            ([], ""),
            ("", ""),
            ("plain string", "plain string"),
            ([None], ""),
            ([5], ""),
            (["not-a-dict"], ""),
            ([{"type": "text"}], ""),  # missing 'text'
            ([{"text": None}], ""),
            ([{"text": 5}], ""),
            ([{"text": ["nested", "list"]}], ""),
            # Separator-free join: a marker split across blocks stays intact.
            ([{"text": "a"}, None, {"text": "b"}], "ab"),
            (42, ""),
            (3.14, ""),
            ({"not": "a list or str"}, ""),
            (True, ""),
        ],
    )
    def test_no_crash_and_expected_flatten(self, system, expected):
        assert _flatten_system_text(system) == expected

    def test_block_missing_type_key_but_has_text_still_joins(self):
        # 'text' presence is what's checked, not 'type'.
        assert _flatten_system_text([{"text": "a"}]) == "a"


class TestSystemLooksSubagentTypeHardening:
    @pytest.mark.parametrize("model_id", [None, "", 0, 5, 3.14, [], {}, True, False])
    def test_bad_model_id_never_crashes_returns_false(self, model_id):
        system = [{"type": "text", "text": f"The exact model ID is {MODEL}[1m]."}]
        assert _system_looks_subagent(system, model_id) is False

    @pytest.mark.parametrize(
        "system",
        [None, [], "", 42, 3.14, {"not": "list-or-str"}, [None, 5, "junk"], True],
    )
    def test_bad_system_never_crashes(self, system):
        # Whatever the shape, an absent/garbage system head must never look
        # like a subagent match (no model id text can be found in it).
        assert _system_looks_subagent(system, MODEL) is False


# ── claim 5: token-mode prefix-mutation gate boundaries ────────────────────────

class TestTokenGateBoundaries:
    def test_warm_roomy_never_true(self):
        # The whole point of the gate: never bust a warm, low-pressure cache.
        for p_alive in (1.0, 0.9, 0.7, 0.5, 0.26, 0.2500001):
            for pressure in (0.0, 0.2, 0.5, 0.8, 0.8499, 0.84999999):
                assert (
                    _token_prefix_mutation_worth_it(
                        context_pressure=pressure, p_alive=p_alive
                    )
                    is False
                ), f"false True at p_alive={p_alive} pressure={pressure}"

    def test_p_alive_floor_exact_boundary_true(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.0, p_alive=0.25) is True
        )

    def test_p_alive_just_above_floor_false(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.0, p_alive=0.2500001)
            is False
        )

    def test_pressure_threshold_exact_boundary_true(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.85, p_alive=1.0) is True
        )

    def test_pressure_just_below_threshold_false(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.8499, p_alive=1.0)
            is False
        )

    def test_both_true_region(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.90, p_alive=0.10)
            is True
        )

    def test_both_false_region(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.10, p_alive=0.90)
            is False
        )

    def test_p_alive_zero_true(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.0, p_alive=0.0) is True
        )

    def test_pressure_at_one_true(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=1.0, p_alive=1.0) is True
        )

    def test_negative_p_alive_treated_as_near_lapse(self):
        # Documents current behaviour: a negative p_alive (should not occur
        # from the real caller, which clamps via max(0.0, ...)) still routes
        # through the <= floor branch and returns True.
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.0, p_alive=-1.0)
            is True
        )

    def test_nan_p_alive_and_pressure_fail_closed(self):
        # NaN compares False against everything, so both branches evaluate
        # False and the gate defaults to "do not mutate". Fails closed, not
        # a crash, but worth pinning: a NaN p_alive is NOT treated as
        # "near lapse" even though it usually signals a broken upstream
        # computation.
        assert (
            _token_prefix_mutation_worth_it(
                context_pressure=float("nan"), p_alive=float("nan")
            )
            is False
        )

    def test_nan_pressure_alone_fails_closed(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=float("nan"), p_alive=1.0)
            is False
        )

    def test_nan_p_alive_alone_falls_through_to_pressure_check(self):
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.0, p_alive=float("nan"))
            is False
        )
        assert (
            _token_prefix_mutation_worth_it(context_pressure=0.9, p_alive=float("nan"))
            is True
        )

    def test_positive_infinity_pressure_true(self):
        assert (
            _token_prefix_mutation_worth_it(
                context_pressure=float("inf"), p_alive=1.0
            )
            is True
        )

    def test_negative_infinity_p_alive_true(self):
        assert (
            _token_prefix_mutation_worth_it(
                context_pressure=0.0, p_alive=float("-inf")
            )
            is True
        )

    def test_negative_infinity_pressure_false(self):
        assert (
            _token_prefix_mutation_worth_it(
                context_pressure=float("-inf"), p_alive=1.0
            )
            is False
        )

    def test_custom_thresholds_respected(self):
        assert (
            _token_prefix_mutation_worth_it(
                context_pressure=0.5,
                p_alive=1.0,
                pressure_threshold=0.4,
            )
            is True
        )
        assert (
            _token_prefix_mutation_worth_it(
                context_pressure=1.0,
                p_alive=0.5,
                p_alive_floor=0.6,
            )
            is True
        )

    def test_exhaustive_grid_only_true_in_documented_regions(self):
        # Brute-force cross-check against the documented contract over a
        # fine grid: True iff p_alive <= 0.25 or context_pressure >= 0.85.
        for i in range(0, 21):
            p_alive = i / 20.0
            for j in range(0, 21):
                pressure = j / 20.0
                expected = p_alive <= 0.25 or pressure >= 0.85
                actual = _token_prefix_mutation_worth_it(
                    context_pressure=pressure, p_alive=p_alive
                )
                assert actual == expected, (
                    f"p_alive={p_alive} pressure={pressure} "
                    f"expected={expected} actual={actual}"
                )
