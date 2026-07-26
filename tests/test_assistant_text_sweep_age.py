"""The assistant-text sweep floor guards model-authored content.

This started life asserting the floor should be *lowered* to make retroactive
masking land shallower in the prefix. That was wrong on both counts. The two
sites the constant gates are both `sweep_assistant_text` paths, so lowering it
bought nothing on tool results, which is where the append actually lives, and
it pulled marker text closer to the tail of model-authored messages, the exact
shape of the 2026-07-17 mimicry incident.

tests/test_breaker3_sweep_chaos.py caught it. These tests pin the reasoning so
the next person does not repeat the change.
"""

from __future__ import annotations

from headroom.transforms import observation_masking


def test_the_floor_stays_conservative_for_model_authored_text():
    """Marker text near the tail of assistant prose is imitable.

    The module docstring documents tool results as safe to mask because they
    are environment-authored. Assistant text is not in that class, so its
    sweep floor is a safety parameter and not a cost knob.
    """
    assert observation_masking.ASSISTANT_TEXT_SWEEP_AGE >= 3, (
        "Lowering this moves CCR marker text into the window the model imitates "
        "from, which is how the 2026-07-17 fabricated-marker incident happened."
    )


def test_lowering_it_would_not_buy_much_anyway():
    """Assistant prose is 8.4 percent of appended content, Read is 51.7.

    Measured 2026-07-25 by scripts/measure_append_by_tool.py over 74 sessions.
    The constant is documented against those figures so the trade stays visible.
    """
    source = observation_masking.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "8.4 percent" in text, (
        "The comment on ASSISTANT_TEXT_SWEEP_AGE must keep the measured share "
        "that justifies leaving it alone, otherwise it reads as an arbitrary 3."
    )


def test_tool_results_are_priced_not_delayed():
    """Tool-result masking has no turn-age gate, and should not get one.

    A position-based delay guesses at cost. masking_gate_gain compares the
    rewrite against the read saving directly, which is the thing that matters.
    """
    assert hasattr(observation_masking, "masking_gate_gain")
    assert not hasattr(observation_masking, "RECENT_TURN_PROTECTION"), (
        "The old name implied it protected recent tool results. It never did."
    )
