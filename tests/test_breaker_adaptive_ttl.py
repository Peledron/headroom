"""Adversarial coverage for PrefixCacheTracker.record_turn_gap / recommended_ttl.

Chaos-engineering pass over the adaptive cache-TTL tier selection added in
headroom/cache/prefix_tracker.py. Each test tries to falsify one contract
claim rather than confirm the happy path. Written entirely offline against
the fixtures below: no proxy, no network, no ANTHROPIC_BASE_URL.

Contract under test (from the docstrings and the call site in
headroom/proxy/handlers/anthropic.py around line 1669):
  - recommended_ttl() returns "5m", "1h", or None.
  - Decision is on max(recent up-to-8 gaps): <=240 -> "5m", >300 -> "1h",
    (240, 300] ambiguous -> hold the previous recommendation (hysteresis).
  - <2 samples -> None.
  - record_turn_gap ignores None, non-finite, and negative values without
    raising and without corrupting the ring buffer.
"""

import math

import pytest

from headroom.cache.prefix_tracker import PrefixCacheTracker


def _tracker_with_gaps(gaps):
    t = PrefixCacheTracker("anthropic")
    for g in gaps:
        t.record_turn_gap(g)
    return t


# ── Claim 1: safety. Must never return "5m" while a >300s gap is still in
# the 8-sample window (a 5m cache set right after that would already have
# lapsed, forcing a cold rewrite on the very next request). ──────────────────


def test_breach_forces_1h_immediately_even_with_only_two_samples():
    t = _tracker_with_gaps([20.0, 400.0])
    assert t.recommended_ttl() == "1h"


def test_breach_stays_1h_for_every_call_while_still_in_the_8_sample_window():
    # A breach followed by up to 7 more fast turns: the breach is still one
    # of the 8 remembered samples at every step, so recent_max stays > 300
    # and the unconditional 1h branch must win every single time. This is
    # the direct attack on the safety claim: try to catch it returning "5m"
    # anywhere before the breach has actually aged out.
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(20.0)
    t.record_turn_gap(305.0)  # breach
    seen = [t.recommended_ttl()]
    for _ in range(7):
        t.record_turn_gap(1.0)  # near-zero gaps: as fast as it gets
        seen.append(t.recommended_ttl())
    assert all(v == "1h" for v in seen), (
        f"tier leaked to something other than 1h while breach was still "
        f"buffered: {seen}"
    )


def test_breach_evicts_on_the_8th_new_append_not_before():
    # deque maxlen=8. The breach needs exactly 8 fresh appends after it to
    # be pushed out (it occupies one of the 8 slots until then).
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(600.0)
    for _ in range(7):
        t.record_turn_gap(30.0)
    assert len(t._turn_gaps) == 8
    assert 600.0 in t._turn_gaps
    assert t.recommended_ttl() == "1h", "breach must still be safety-active with 7 new appends"
    t.record_turn_gap(30.0)  # 8th new append
    assert 600.0 not in t._turn_gaps
    assert t.recommended_ttl() == "5m"


def test_safety_claim_survives():
    # No counterexample found: max() over the buffer plus an unconditional
    # (non-hysteresis-gated) > tier_boundary branch makes it structurally
    # impossible to return "5m" while any >300s sample remains buffered.
    pass


# ── Related but distinct finding: the margin is a one-time commit gate,
# not a maintained floor. Once "5m" is committed, hysteresis lets the
# recent max drift all the way up to the 300s boundary itself with zero
# residual margin, even though margin_seconds=60 is documented as "keeps a
# safety buffer below the boundary". A session whose cadence degrades
# gradually (200 -> 250 -> 280 -> 299) gets no warning before the next gap
# can tip over 300 and force the exact cold rewrite the margin exists to
# avoid. ──────────────────────────────────────────────────────────────────


def test_margin_erodes_to_zero_once_5m_is_already_committed():
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(20.0)
    t.record_turn_gap(45.0)
    assert t.recommended_ttl() == "5m"
    # Ride the ambiguous band right up against the real 300s boundary. Every
    # one of these calls is well past the documented 240s safe line, yet
    # each still returns "5m" because hysteresis just holds the prior
    # commitment instead of re-checking the promised margin.
    for gap in (100.0, 200.0, 250.0, 280.0, 299.0):
        t.record_turn_gap(gap)
        assert t.recommended_ttl() == "5m"
    # One more second and the *next* real request could easily land past
    # the actual cache lifetime with no advance signal at all.


# ── Claim 2: hysteresis must not wedge a tier indefinitely against the
# cadence that's actually happening. ──────────────────────────────────────


def test_1h_recovers_to_5m_once_cadence_is_genuinely_fast_for_8_turns():
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(20.0)
    t.record_turn_gap(400.0)
    assert t.recommended_ttl() == "1h"
    for _ in range(8):
        t.record_turn_gap(50.0)
    assert t.recommended_ttl() == "5m"


def test_5m_flips_to_1h_on_a_single_breach_not_stuck():
    t = _tracker_with_gaps([20.0, 30.0])
    assert t.recommended_ttl() == "5m"
    t.record_turn_gap(310.0)
    assert t.recommended_ttl() == "1h"


def test_hysteresis_wedges_1h_forever_on_a_sustained_ambiguous_cadence():
    # REFUTED counterexample for claim 2. A session settles permanently at
    # 250s per turn: comfortably under the real 300s tier boundary (the
    # cache never actually lapses), but inside the (240, 300] ambiguous
    # band by the code's own margin. Once "1h" is set by an earlier breach,
    # every single one of the next 50 turns re-confirms "1h" even though no
    # sample in the entire rolling window ever again exceeds 300, because
    # the ambiguous branch only ever holds, never re-derives from the
    # current window's actual max.
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(20.0)
    t.record_turn_gap(400.0)  # breach: commits "1h"
    assert t.recommended_ttl() == "1h"
    results = []
    for _ in range(50):
        t.record_turn_gap(250.0)
        results.append(t.recommended_ttl())
    assert all(v == "1h" for v in results)
    # The buffer no longer contains anything remotely close to a breach...
    assert max(t._turn_gaps) == 250.0
    assert 400.0 not in t._turn_gaps
    # ...yet the tier never budges. A cadence that never again threatens
    # the 5-minute boundary pays the 2x write premium forever.


def test_ambiguous_cadence_that_dips_to_240_does_eventually_recover():
    # Sanity check on the flip side: if the cadence actually reaches the
    # code's own <=240 threshold for 8 consecutive turns, recovery does
    # happen. This confirms the wedge above is specific to cadences that
    # stay strictly inside (240, 300] and never touch the safe zone.
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(20.0)
    t.record_turn_gap(400.0)
    assert t.recommended_ttl() == "1h"
    for _ in range(8):
        t.record_turn_gap(240.0)
    assert t.recommended_ttl() == "5m"


# ── Claim 3: ring aging, both directions. ─────────────────────────────────


def test_stale_long_gap_ages_out_after_exactly_eight_newer_short_gaps():
    t = _tracker_with_gaps([600.0] + [30.0] * 8)
    assert list(t._turn_gaps) == [30.0] * 8
    assert t.recommended_ttl() == "5m"


def test_stale_short_gaps_do_not_mask_a_genuine_new_breach():
    # Buffer full of 8 fast gaps (an established "5m" session), then a
    # single new breach arrives. The seven still-buffered short gaps must
    # not delay or soften the flip: recent_max includes the new breach
    # immediately and the unconditional >300 branch does not consult
    # hysteresis, so this must be "1h" on the very next call.
    t = _tracker_with_gaps([30.0] * 8)
    assert t.recommended_ttl() == "5m"
    t.record_turn_gap(310.0)  # evicts the oldest 30.0, buffer now 7x30 + 1 breach
    assert len(t._turn_gaps) == 8
    assert t.recommended_ttl() == "1h"


# ── Claim 4: boundary exactness at 240s and 300s. ─────────────────────────


@pytest.mark.parametrize(
    "recent_gap,expected",
    [
        (239.999, "5m"),  # strictly inside the margin-adjusted safe zone
        (240.0, "5m"),  # <=240 is inclusive per the code (`<=`)
        (240.001, None),  # just past the safe line: ambiguous, no prior state
        (299.999, None),  # still ambiguous, still no prior state
        (300.0, None),  # `> 300` is strict: exactly 300 is NOT a breach
        (300.001, "1h"),  # first value that is strictly > 300
    ],
)
def test_first_decision_boundary_exactness(recent_gap, expected):
    t = _tracker_with_gaps([1.0, recent_gap])
    assert t.recommended_ttl() == expected


def test_300_exact_holds_prior_5m_rather_than_promoting_to_1h():
    # With a prior "5m" commitment, a gap of exactly 300.0s (not > 300) is
    # ambiguous and must hold, not flip. Confirms the boundary is
    # symmetric: 300.0 never counts as a breach on its own, regardless of
    # hysteresis state.
    t = _tracker_with_gaps([20.0, 30.0])
    assert t.recommended_ttl() == "5m"
    t.record_turn_gap(300.0)
    assert t.recommended_ttl() == "5m"


def test_300_exact_holds_prior_1h_rather_than_demoting_to_5m():
    t = _tracker_with_gaps([20.0, 400.0])
    assert t.recommended_ttl() == "1h"
    t.record_turn_gap(300.0)
    assert t.recommended_ttl() == "1h"


# ── Claim 5: record_turn_gap input hardening. ─────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        -5.0,
        -0.0001,
        "not a number",
        object(),
        [1, 2, 3],
        {"gap": 5},
    ],
)
def test_record_turn_gap_rejects_bad_values_without_crashing(bad):
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(bad)  # must not raise
    assert len(t._turn_gaps) == 0


def test_record_turn_gap_accepts_zero():
    # A gap of exactly 0.0 (two requests in the same instant) is a
    # legitimate, if extreme, fast-cadence sample and must be kept.
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(0.0)
    t.record_turn_gap(0.0)
    assert list(t._turn_gaps) == [0.0, 0.0]
    assert t.recommended_ttl() == "5m"


def test_record_turn_gap_accepts_huge_but_finite_float():
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(20.0)
    t.record_turn_gap(1e300)
    assert t.recommended_ttl() == "1h"


def test_record_turn_gap_huge_python_int_does_not_crash_and_is_ignored():
    # FIXED: record_turn_gap's try/except around float(gap_seconds) now also
    # catches OverflowError, so a Python int too large to represent as a C
    # double (which float() would previously raise on, uncaught) is silently
    # ignored instead of propagating -- consistent with the function's
    # documented contract that any value it cannot use as a finite,
    # non-negative duration is dropped without corrupting the ring buffer.
    t = PrefixCacheTracker("anthropic")
    huge_int = 10**400
    t.record_turn_gap(huge_int)  # must not raise
    assert len(t._turn_gaps) == 0


def test_record_turn_gap_bool_input_silently_poisons_the_buffer():
    # REFUTED (hardening gap). bool is a subclass of int, so float(True)
    # == 1.0 and float(False) == 0.0 succeed silently: there is no
    # isinstance(gap_seconds, bool) guard. A caller bug that passes a flag
    # (e.g. "has_gap" or "is_first_turn") instead of a computed duration
    # would silently record a fake 0.0/1.0 second gap and skew the
    # recommendation toward "5m" with no error anywhere. This assertion
    # encodes the intended hardened contract (reject non-numeric-duration
    # types including bool) and is expected to fail against the current
    # implementation.
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(True)
    t.record_turn_gap(False)
    assert len(t._turn_gaps) == 0, (
        f"bool inputs were coerced into the ring buffer as real gaps: "
        f"{list(t._turn_gaps)}"
    )


def test_record_turn_gap_numeric_string_is_accepted_by_design():
    # SURVIVES as intentional duck-typing, not a bug: the docstring frames
    # rejection purely in terms of non-finite/negative *values* after a
    # coercion attempt, so a coercible string is treated the same as a
    # coercible float on purpose.
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap("45.0")
    assert list(t._turn_gaps) == [45.0]


def test_record_turn_gap_non_coercible_string_is_ignored_not_crashed():
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap("not-a-number")
    assert len(t._turn_gaps) == 0


def test_record_turn_gap_nan_never_reaches_max():
    # Defense in depth: even if a NaN slipped past the isfinite() guard,
    # max() with a NaN present is order-dependent and can silently produce
    # a NaN or a wrong value. Confirm the guard actually keeps NaN out.
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(20.0)
    t.record_turn_gap(float("nan"))
    t.record_turn_gap(45.0)
    assert not any(math.isnan(g) for g in t._turn_gaps)
    assert t.recommended_ttl() == "5m"


# ── Claim 6: interaction with maxlen and sample-count edges. ──────────────


def test_zero_samples_returns_none():
    assert PrefixCacheTracker("anthropic").recommended_ttl() is None


def test_exactly_one_sample_returns_none_regardless_of_magnitude():
    assert _tracker_with_gaps([30.0]).recommended_ttl() is None
    assert _tracker_with_gaps([9999.0]).recommended_ttl() is None


def test_exactly_two_samples_is_the_minimum_that_can_decide():
    assert _tracker_with_gaps([30.0, 30.0]).recommended_ttl() == "5m"
    assert _tracker_with_gaps([30.0, 9999.0]).recommended_ttl() == "1h"


def test_invalid_records_do_not_count_toward_the_two_sample_minimum():
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(30.0)
    t.record_turn_gap(None)
    t.record_turn_gap(float("nan"))
    t.record_turn_gap(-1.0)
    assert t.recommended_ttl() is None  # only one real sample recorded
    t.record_turn_gap(30.0)
    assert t.recommended_ttl() == "5m"  # now two real samples


def test_hysteresis_can_return_a_tier_no_current_sample_individually_justifies():
    # REFUTED counterexample for claim 6's second half. Commit "5m" from a
    # genuinely fast pair, then push the buffer's max up into the
    # ambiguous band. The returned "5m" is now justified by nothing in the
    # *current* window: the current recent_max (299.0) does not satisfy
    # the code's own <=240 rule for "5m", it only survives because
    # hysteresis is quoting a decision made by samples that are still
    # technically present but no longer the reason a fresh evaluation
    # would pick "5m".
    t = PrefixCacheTracker("anthropic")
    t.record_turn_gap(20.0)
    t.record_turn_gap(45.0)
    assert t.recommended_ttl() == "5m"
    t.record_turn_gap(299.0)
    recent_max = max(t._turn_gaps)
    result = t.recommended_ttl()
    assert result == "5m"
    assert not (recent_max <= 240.0), (
        "expected the justifying-sample premise of this counterexample to hold"
    )


def test_ring_buffer_never_exceeds_maxlen_eight():
    t = _tracker_with_gaps([float(i) for i in range(20)])
    assert len(t._turn_gaps) == 8
    assert list(t._turn_gaps) == [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
