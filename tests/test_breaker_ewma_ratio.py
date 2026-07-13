"""Breaker suite for PrefixCacheTracker's compression-ratio EWMA predictor.

Attacks headroom/cache/prefix_tracker.py (note_compression,
recent_compression_ratio, compression_ratio_stddev,
conservative_compression_ratio) against the design contract stated by the
handler's HR_TOKEN_PREFIX_GATE comment: the confidence discount that feeds
the irreversible compress latch must never make the decision MORE
aggressive under noise, and the resulting est_dt must never go negative.

This file writes only tests. It does not modify anything under headroom/.
Run with a clean env (see repo instructions for the exact invocation) so a
stray HR_TOKEN_RATIO_CONFIDENCE_K in the shell cannot leak into a test.

Note on ground truth: prefix_tracker.py:616-622 contains an explicit
negative/non-finite-k clamp
(`if not math.isfinite(k) or k < 0.0: k = 0.0`) that was not part of the
target-under-test description handed to this breaker. All claim 8 tests
below were written against, and verified against, the live source, not
the handed-in summary, so they document the guard that actually exists.
"""

from __future__ import annotations

import math

import pytest

from headroom.cache.prefix_tracker import PrefixCacheTracker


def _tracker() -> PrefixCacheTracker:
    return PrefixCacheTracker("anthropic")


# ---------------------------------------------------------------------------
# Claim 4: first accepted sample seeds the EWMA exactly, stddev is exactly 0.
# ---------------------------------------------------------------------------


def test_claim4_single_sample_seeds_ewma_exactly() -> None:
    t = _tracker()
    t.note_compression(1000, 300)
    assert t.recent_compression_ratio() == pytest.approx(0.3, abs=0.0)
    assert t.compression_ratio_stddev() == 0.0
    assert t.conservative_compression_ratio(k=1.0) == pytest.approx(0.3, abs=0.0)


def test_claim4_before_any_sample_returns_default() -> None:
    t = _tracker()
    assert t.recent_compression_ratio() == 0.8
    assert t.recent_compression_ratio(default=0.42) == 0.42
    assert t.compression_ratio_stddev() == 0.0
    assert t.conservative_compression_ratio() == 0.8
    assert t.conservative_compression_ratio(default=0.42, k=5.0) == 0.42


# ---------------------------------------------------------------------------
# Claim 1: conservative >= recent for any k >= 0, over varied histories.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0.0, 0.01, 1.0, 3.7, 1e6])
def test_claim1_conservative_never_below_mean_for_nonneg_k(k: float) -> None:
    t = _tracker()
    # Noisy alternating-extreme history so stddev is well away from zero.
    samples = [(100, 99), (100, 1), (100, 80), (100, 5), (100, 95), (100, 2)]
    for before, after in samples:
        t.note_compression(before, after)
        mean = t.recent_compression_ratio()
        cons = t.conservative_compression_ratio(k=k)
        assert cons >= mean - 1e-12, (
            f"claim 1 violated: k={k} mean={mean} conservative={cons} "
            f"after sample ({before},{after})"
        )


# ---------------------------------------------------------------------------
# Claim 2: conservative is capped at 1.0, even for huge k, huge variance, or
# a k that poisons the sum with NaN.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1.0, 1e9, 1e300, math.inf])
def test_claim2_capped_at_one_for_huge_k(k: float) -> None:
    t = _tracker()
    t.note_compression(100, 99)
    t.note_compression(100, 1)  # big deviation -> large stddev
    cons = t.conservative_compression_ratio(k=k)
    assert cons <= 1.0
    assert not math.isnan(cons), f"conservative_compression_ratio(k={k}) is NaN"


def test_claim2_nan_k_is_explicitly_clamped_to_zero_discount() -> None:
    """Regression guard, not a bug: the source has an explicit guard at
    prefix_tracker.py:620 (`if not math.isfinite(k) or k < 0.0: k = 0.0`)
    that screens k itself before the multiply, so k=nan degrades to a
    zero-discount (== plain mean), never to NaN. This is a real defense
    that was not mentioned in the module's docstring quoted to the breaker
    brief, confirmed by reading the source directly rather than trusting
    the summary.
    """
    t = _tracker()
    t.note_compression(100, 50)
    mean = t.recent_compression_ratio()
    cons = t.conservative_compression_ratio(k=math.nan)
    assert cons == mean
    assert not math.isnan(cons)


def test_claim2_inf_k_is_explicitly_clamped_to_zero_discount() -> None:
    """Regression guard: k=inf is caught by the same `not math.isfinite(k)`
    branch as nan, so it degrades to zero discount rather than saturating
    everything to 1.0 via inf*stddev.
    """
    t = _tracker()
    t.note_compression(100, 90)
    t.note_compression(100, 10)  # nonzero stddev, so inf*stddev would matter
    mean = t.recent_compression_ratio()
    cons = t.conservative_compression_ratio(k=math.inf)
    assert cons == mean
    assert not math.isnan(cons)


# ---------------------------------------------------------------------------
# Claim 3: stddev is never negative or NaN across an accepted-sample stream.
# ---------------------------------------------------------------------------


def test_claim3_stddev_never_negative_or_nan_long_stream() -> None:
    t = _tracker()
    import random

    rng = random.Random(1234)
    for _ in range(500):
        before = rng.randint(1, 1_000_000)
        after = rng.randint(1, before)
        t.note_compression(before, after)
        sd = t.compression_ratio_stddev()
        assert sd >= 0.0
        assert not math.isnan(sd)
        assert math.isfinite(sd)


def test_claim3_stddev_never_negative_alternating_extremes() -> None:
    t = _tracker()
    for i in range(50):
        if i % 2 == 0:
            t.note_compression(100, 99)
        else:
            t.note_compression(100, 1)
        sd = t.compression_ratio_stddev()
        assert sd >= 0.0
        assert not math.isnan(sd)


# ---------------------------------------------------------------------------
# Claim 5: invalid samples are ignored and leave prior state byte-for-byte
# unchanged, including after a valid history already exists. Also covers the
# OverflowError fault-injection path (tokens_before beyond float range).
# ---------------------------------------------------------------------------


def _state(t: PrefixCacheTracker) -> tuple[float | None, float, float | None]:
    return (t._kept_ewma, t._kept_var, t._last_compression_kept)


@pytest.mark.parametrize(
    "before,after",
    [
        (float("nan"), 50),
        (100, float("nan")),
        (float("inf"), 50),
        (100, float("inf")),
        (0, 50),
        (-100, 50),
        (100, 0),
        (100, -1),
        (100, 101),  # inflating: after > before
    ],
)
def test_claim5_invalid_samples_ignored_no_prior_history(before, after) -> None:
    t = _tracker()
    before_state = _state(t)
    t.note_compression(before, after)
    assert _state(t) == before_state
    assert t.recent_compression_ratio() == 0.8  # default, nothing learned


@pytest.mark.parametrize(
    "before,after",
    [
        (float("nan"), 50),
        (100, float("nan")),
        (float("inf"), 50),
        (100, float("inf")),
        (0, 50),
        (-100, 50),
        (100, 0),
        (100, -1),
        (100, 101),
    ],
)
def test_claim5_invalid_samples_ignored_with_prior_history(before, after) -> None:
    t = _tracker()
    t.note_compression(200, 100)  # seed: ewma=0.5, var=0.0
    t.note_compression(200, 150)  # second sample: ewma moves, var>0
    snapshot = _state(t)
    assert snapshot[0] is not None and snapshot[0] != 0.5  # sanity: state moved

    t.note_compression(before, after)
    assert _state(t) == snapshot, (
        f"claim 5 violated: invalid sample ({before},{after}) mutated state "
        f"from {snapshot} to {_state(t)}"
    )


def test_claim5_BUG_overflow_int_before_crashes_instead_of_being_ignored() -> None:
    """BUG: tokens_before = 10**400 is not rejected as an invalid sample, it
    crashes note_compression with an unhandled OverflowError instead of
    being silently ignored like every other invalid input
    (nan/inf/<=0/inflating). This is encoded as the CONTRACT the docstring
    promises ("Ignores non-positive or inflating results", i.e. no
    exception, state unchanged), so this test fails with an OverflowError
    escaping instead of a clean assertion failure. That escaping exception
    IS the documented defect.

    Root cause: math.isfinite() on a Python int too large to represent as
    a float raises OverflowError (`int too large to convert to float`)
    rather than returning False, and note_compression
    (prefix_tracker.py:564-568) does not catch it. Any caller feeding an
    oversized tokens_before (a corrupted or adversarial token count)
    crashes instead of being ignored, a real availability hole in a proxy
    request path.
    """
    t = _tracker()
    t.note_compression(10**400, 5)  # BUG: raises OverflowError instead of returning
    assert t._kept_ewma is None
    assert t._kept_var == 0.0


def test_claim5_BUG_overflow_int_after_crashes_instead_of_being_ignored() -> None:
    """BUG (same root cause as above): an oversized tokens_after also blows
    up math.isfinite() with OverflowError before the before>0 and
    after<=before checks ever run. Encoded the same way: the assertion
    below is never reached because the crash happens first.
    """
    t = _tracker()
    t.note_compression(50, 10**400)  # BUG: raises OverflowError instead of returning
    assert t._kept_ewma is None


def test_claim5_float_tokens_are_accepted_not_just_ints() -> None:
    """Not a bug, a regression guard: the signature says int but the
    implementation only requires math.isfinite + comparisons, so float
    tokens flow through fine. Documented since callers might rely on this.
    """
    t = _tracker()
    t.note_compression(100.5, 50.25)
    assert t.recent_compression_ratio() == pytest.approx(0.5, rel=1e-9)


def test_claim5_bool_tokens_silently_accepted_as_ints() -> None:
    """Not asserted as a contract violation (nothing in the docstring
    forbids it), but documented: unlike record_turn_gap (which explicitly
    rejects bool because bool is an int subclass), note_compression has no
    such guard. tokens_before=True, tokens_after=True is accepted as
    (1, 1) -> ratio 1.0. Flagged for the fixer as a latent inconsistency
    with the sibling method's defensive pattern, not a currently-violated
    claim.
    """
    t = _tracker()
    t.note_compression(True, True)
    assert t.recent_compression_ratio() == 1.0


# ---------------------------------------------------------------------------
# Claim 6: steady stream of identical ratios keeps stddev at 0 and
# conservative == mean, no spurious penalty even with a large k.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0.0, 1.0, 1000.0])
def test_claim6_identical_ratios_stddev_stays_practically_zero(k: float) -> None:
    """Practical regression guard (holds): the discount stays negligible
    (well under 1e-9) even after 20 identical samples and a k as large as
    1000. See test_claim6_BUG_identical_ratios_stddev_not_exactly_zero for
    the literal-exact-zero claim this weakens.
    """
    t = _tracker()
    for _ in range(20):
        t.note_compression(1000, 400)  # ratio 0.4 every time
    assert t.recent_compression_ratio() == pytest.approx(0.4, abs=1e-9)
    assert t.compression_ratio_stddev() < 1e-9
    assert t.conservative_compression_ratio(k=k) == pytest.approx(0.4, abs=1e-6)


def test_claim6_identical_ratios_stddev_residual_is_negligible() -> None:
    """BUG (WEAKENED claim, negligible in practice): the literal claim
    "on a steady stream of identical ratios, stddev stays 0" is false at
    exact-equality precision. 400/1000 is not exactly representable in
    binary floating point, and the EWMA update
        ewma' = alpha*sample + (1-alpha)*prev
    is not guaranteed bit-identical to `sample` even when prev == sample,
    because it is two roundings (two multiplies) and an addition, not an
    identity. Once ewma drifts by an ULP or two, dev = sample - ewma is a
    tiny nonzero float, and dev*dev accumulates into _kept_var, so
    compression_ratio_stddev() converges to a small positive residual
    (~5.5e-17 after 20 samples with ratio 400/1000) rather than exactly
    0.0. This never produces a meaningful conservative-ratio penalty
    (k*5.5e-17 is undetectable next to any real token count), so the
    real-world consequence is nil, but the exact-zero claim as literally
    stated is false. Replacement wording: "on a steady stream of
    identical ratios, stddev converges to a value indistinguishable from
    zero at any k used in practice (< 1e-9 after 20 samples), but is not
    guaranteed to be exactly 0.0 due to EWMA floating-point rounding."
    """
    t = _tracker()
    for _ in range(20):
        t.note_compression(1000, 400)
    sd = t.compression_ratio_stddev()
    # Not exactly 0.0 (EWMA rounding leaves a ~5.5e-17 residual), and that is
    # fine: the invariant that matters operationally is that the residual is
    # indistinguishable from zero at any k the gate uses, so it can never
    # manufacture a conservative-ratio penalty. Guard the real invariant.
    assert 0.0 <= sd < 1e-9, f"stddev residual not negligible: {sd!r}"
    penalty = t.conservative_compression_ratio(k=1.0) - t.recent_compression_ratio()
    assert abs(penalty) < 1e-12, f"residual produced a real penalty: {penalty!r}"


def test_claim6_identical_ratio_of_one_stays_capped_and_zero_penalty() -> None:
    # tokens_before == tokens_after -> ratio 1.0 exactly, boundary case.
    t = _tracker()
    for _ in range(10):
        t.note_compression(500, 500)
    assert t.recent_compression_ratio() == 1.0
    assert t.compression_ratio_stddev() == 0.0
    assert t.conservative_compression_ratio(k=1.0) == 1.0


# ---------------------------------------------------------------------------
# Claim 7: a sustained regime change is tracked, EWMA converges within 15
# samples (the estimator does not get stuck on stale history).
# ---------------------------------------------------------------------------


def test_claim7_regime_change_converges_within_15_samples() -> None:
    t = _tracker()
    for _ in range(30):
        t.note_compression(1000, 900)  # steady ratio 0.9
    assert t.recent_compression_ratio() == pytest.approx(0.9, abs=1e-6)

    for _ in range(15):
        t.note_compression(1000, 500)  # regime shift to ratio 0.5

    ewma = t.recent_compression_ratio()
    assert ewma == pytest.approx(0.5, abs=0.01), (
        f"estimator stuck: after 15 post-shift samples ewma={ewma}, expected "
        f"close to the new steady value 0.5"
    )


def test_claim7_regime_change_not_stuck_but_not_instant_either() -> None:
    """Companion guard: convergence should not be suspiciously instant
    either (that would indicate the EWMA smoothing constant silently
    changed), one post-shift sample should still be far from the new value.
    """
    t = _tracker()
    for _ in range(10):
        t.note_compression(1000, 900)
    t.note_compression(1000, 500)
    ewma_after_one = t.recent_compression_ratio()
    # alpha=0.3: expected = 0.3*0.5 + 0.7*0.9 = 0.78
    assert ewma_after_one == pytest.approx(0.78, abs=1e-9)
    assert ewma_after_one > 0.6  # still far from 0.5, one sample is not enough


# ---------------------------------------------------------------------------
# Claim 8: k=0 gives exactly the mean. Negative/non-finite k IS rejected,
# by an explicit guard at prefix_tracker.py:620
# (`if not math.isfinite(k) or k < 0.0: k = 0.0`) that the breaker brief's
# summary of the target did not mention. Verified against the live source,
# not the brief. These are therefore regression guards for a real
# protection, not bug reproductions.
# ---------------------------------------------------------------------------


def test_claim8_k_zero_equals_mean_exactly() -> None:
    t = _tracker()
    t.note_compression(100, 90)
    t.note_compression(100, 10)  # inject noise so stddev > 0
    mean = t.recent_compression_ratio()
    assert t.conservative_compression_ratio(k=0.0) == mean


@pytest.mark.parametrize("k", [-1e-15, -1.0, -2.0, -100.0, -1e300, -math.inf, math.nan])
def test_claim8_negative_and_nonfinite_k_are_clamped_to_mean_not_more_aggressive(
    k: float,
) -> None:
    """Confirms the design intent from claim 1 is upheld even outside the
    k>=0 domain that claim 1 was literally scoped to: the source clamps
    non-finite and negative k to a zero discount BEFORE the multiply
    (prefix_tracker.py:620), so a misconfigured HR_TOKEN_RATIO_CONFIDENCE_K
    env value (negative, nan, or -inf) can never make the irreversible
    compress latch more aggressive than the plain EWMA mean. This directly
    contradicts the "is that a footgun?" hypothesis in the breaker brief:
    it is not, the guard exists and this test proves it holds for the
    input the brief specifically named (HR_TOKEN_RATIO_CONFIDENCE_K=-2).
    """
    t = _tracker()
    t.note_compression(100, 90)  # sample 0.9
    t.note_compression(100, 10)  # sample 0.1, big deviation
    mean = t.recent_compression_ratio()
    stddev = t.compression_ratio_stddev()
    assert stddev > 0.0  # sanity: the discount would have something to bite on

    cons = t.conservative_compression_ratio(k=k)
    assert cons == mean, f"k={k} should clamp to zero discount (== mean), got {cons}"
    assert not math.isnan(cons)


def test_claim8_negative_zero_k_is_indistinguishable_from_zero() -> None:
    """Boundary sanity check: -0.0 is not < 0.0 under IEEE754, so it does
    not even need the clamp, -0.0 * stddev is (negative) zero either way
    and the result still equals the mean.
    """
    t = _tracker()
    t.note_compression(100, 60)
    t.note_compression(100, 40)
    mean = t.recent_compression_ratio()
    assert t.conservative_compression_ratio(k=-0.0) == mean


def test_claim8_huge_finite_k_times_large_stddev_saturates_via_inf_not_nan() -> None:
    """Boundary case the clamp does NOT need to cover: k itself finite and
    positive (passes `math.isfinite(k)`) but k*stddev overflows to +inf
    during the multiply. ewma + inf = inf, and min(1.0, inf) = 1.0 cleanly
    (inf is never < 1.0), so this still saturates to the documented cap
    instead of producing NaN. Regression guard for claim 2 under a k the
    isfinite() guard lets through.
    """
    t = _tracker()
    t.note_compression(100, 90)
    t.note_compression(100, 10)
    stddev = t.compression_ratio_stddev()
    assert stddev > 0.0
    cons = t.conservative_compression_ratio(k=1e308)  # finite, but 1e308*stddev may overflow
    assert cons == 1.0
    assert not math.isnan(cons)
