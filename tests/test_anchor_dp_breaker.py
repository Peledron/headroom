"""Adversarial tests for headroom/cache/anchor_dp.py.

Chaos engineering against the DP's exactness claim, the weighted-objective
addition, the CANDIDATE_STEP/QUANTUM split, and the stabilize hysteresis
gate. This file does not modify headroom/cache/anchor_dp.py or any existing
test. Some tests below are expected to fail: they encode the contract the
writer claimed and let the real behaviour disagree with it, per the breaker
mandate. Each such test says so in its docstring.
"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from headroom.cache.anchor_dp import (
    EXPECTED_REMAINING_BUSTS,
    MIN_DEPTH,
    QUANTUM,
    TAIL_GUARD,
    _candidate_positions,
    _mean_rewrite_depth,
    _move_write_cost,
    _RewriteScale,
    optimal_anchor_depths,
    stabilize_anchor_depths,
)

# ---------------------------------------------------------------------------
# Section 1: randomized differential test against exhaustive enumeration,
# including the weighted objective. The ground truth below is written from
# scratch, it does not call segment_loss, bisect, or _RewriteScale, so it
# cannot share a bug with the code under test.
# ---------------------------------------------------------------------------


def _independent_loss(
    n: int, fractions: list[float], weights: list[float] | None, anchors: list[int]
) -> float:
    """Ground truth loss, computed independently of anchor_dp internals."""
    samples = sorted(min(max(f, 0.0), 1.0) * n for f in fractions)
    cum: list[float] | None
    if weights is not None and len(weights) >= n and n > 0:
        cum = [0.0] * (n + 1)
        for i in range(n):
            w = float(weights[i])
            if w != w:  # nan guard, mirrors max(0.0, nan) landing at 0.0
                w = 0.0
            cum[i + 1] = cum[i] + max(0.0, w)
        if cum[n] <= 0:
            cum = None
    else:
        cum = None

    def pos(depth: float) -> float:
        # Mirrors _RewriteScale.at: a sample landing between two message
        # boundaries is charged the partial message pro rata. Flooring here
        # instead would score the brute-force search on a different objective
        # than the DP optimizes, so a disagreement would say nothing about
        # whether the DP found its own optimum.
        if cum is None:
            return float(depth)
        if depth <= 0:
            return 0.0
        if depth >= n:
            return cum[n]
        idx = int(depth)
        remainder = depth - idx
        base = cum[idx]
        if remainder <= 0.0:
            return base
        return base + remainder * (cum[idx + 1] - base)

    total = 0.0
    for c in samples:
        below = 0
        for a in anchors:
            if a <= c and a > below:
                below = a
        total += max(0.0, pos(c) - pos(below))
    return total


def _brute_best(n: int, fractions: list[float], weights: list[float] | None, k: int) -> float:
    candidates = _candidate_positions(n)
    best = _independent_loss(n, fractions, weights, [])
    for size in range(1, min(k, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, size):
            loss = _independent_loss(n, fractions, weights, list(combo))
            if loss < best:
                best = loss
    return best


def test_dp_matches_exhaustive_enumeration_unweighted_and_weighted_fuzz() -> None:
    """Randomized differential test: DP loss must equal the best loss found
    by exhaustive enumeration of candidate subsets, under the same objective,
    for both the unweighted and the weighted path.

    Covers: samples on candidate boundaries, duplicate samples, samples at
    0.0 and 1.0, out-of-range fractions, n smaller than the weights list,
    weights shorter than n, all-zero weights, NaN weights, negative weights,
    and k larger than the candidate count. Candidate counts are kept small
    (<= 9) so brute force stays tractable within the trial budget.
    """
    random.seed(20260726)
    trials = 0
    failures: list[tuple] = []
    for _ in range(2500):
        n = random.choice([65, 96, 100, 128, 150, 176, 200])
        candidates = _candidate_positions(n)
        if len(candidates) > 9:
            continue
        trials += 1

        num_samples = random.randint(1, 8)
        fractions: list[float] = []
        for _ in range(num_samples):
            roll = random.random()
            if roll < 0.3 and candidates:
                fractions.append(random.choice(candidates) / n)  # exact boundary
            elif roll < 0.4:
                fractions.append(0.0)
            elif roll < 0.5:
                fractions.append(1.0)
            elif roll < 0.55:
                fractions.append(-0.3)  # out of [0, 1]
            elif roll < 0.6:
                fractions.append(1.7)  # out of [0, 1]
            else:
                fractions.append(round(random.random(), 4))
        if fractions and random.random() < 0.5:
            fractions = fractions + [fractions[0]]  # duplicate sample

        k = random.randint(1, 5)  # can exceed len(candidates)
        mode = random.choice(
            ["none", "uniform", "random", "zero", "short", "long", "nan", "negative"]
        )
        weights: list[float] | None
        if mode == "none":
            weights = None
        elif mode == "uniform":
            weights = [7.0] * n
        elif mode == "random":
            weights = [random.choice([0.0, 1.0, 5.0, 1000.0, 999999.0]) for _ in range(n)]
        elif mode == "zero":
            weights = [0.0] * n
        elif mode == "short":
            weights = [1.0] * max(0, n - 5)
        elif mode == "long":
            weights = [1.0] * (n + 10)
        elif mode == "nan":
            weights = [float("nan") if i % 7 == 0 else 1.0 for i in range(n)]
        else:
            weights = [-5.0 if i % 3 == 0 else 2.0 for i in range(n)]

        depths = optimal_anchor_depths(n, fractions, k, weights)
        dp_loss = _independent_loss(n, fractions, weights, depths)
        best = _brute_best(n, fractions, weights, k)
        if abs(dp_loss - best) > 1e-6:
            failures.append((n, fractions, k, mode, depths, dp_loss, best))

    assert trials > 500, f"only {trials} trials ran, fuzz budget too small"
    assert not failures, f"{len(failures)}/{trials} DP results disagree with brute force: {failures[:5]}"


def test_dp_matches_exhaustive_enumeration_extreme_magnitude() -> None:
    """The 1e-9 float tolerance in the path-recovery re-walk is a suspected
    weak point once weights push costs into the token-magnitude range
    (1e6 to 1e9). If recovery ever mismatches, it silently returns a partial
    anchor path whose achieved loss is worse than the table claims, which
    this differential check would catch as a loss mismatch.
    """
    random.seed(4242)
    failures: list[tuple] = []
    for _ in range(150):
        n = random.choice([300, 400, 500])
        candidates = _candidate_positions(n)
        if len(candidates) > 9:
            continue
        num_samples = random.randint(3, 8)
        fractions = [round(random.random(), 5) for _ in range(num_samples)]
        k = random.randint(1, 4)
        weights = [random.choice([1.0, 1e6, 1e9, 3.7e8]) for _ in range(n)]
        depths = optimal_anchor_depths(n, fractions, k, weights)
        dp_loss = _independent_loss(n, fractions, weights, depths)
        best = _brute_best(n, fractions, weights, k)
        if abs(dp_loss - best) > 1.0:  # absolute tolerance loosened for 1e9-scale sums
            failures.append((n, fractions, k, depths, dp_loss, best))
    assert not failures, f"extreme-magnitude mismatches: {failures[:5]}"


# ---------------------------------------------------------------------------
# Section 2: the "uniform weights equivalent to no weights" claim.
# ---------------------------------------------------------------------------


def test_fractional_sample_position_survives_weighting() -> None:
    """Root cause isolation for the floor the weighted path used to apply.

    A uniform weight is a pure positive rescaling of the same message-distance
    objective, so it must not change which fractional part of a sample position
    is kept. Truncating to int(depth) on the weighted path only, as at() once
    did, silently moved a sample down to the message boundary below it and made
    the two paths disagree on near-ties for no reason related to placement.
    Checked at a scale of 1.0, where the rescaling is the identity and any
    difference is the floor itself.
    """
    unweighted = _RewriteScale(100, None)
    uniform = _RewriteScale(100, [1.0] * 100)
    assert unweighted.at(50.7) == uniform.at(50.7), (
        f"unweighted.at(50.7)={unweighted.at(50.7)} but "
        f"uniform(w=1).at(50.7)={uniform.at(50.7)}: weighting floors "
        f"the sample position"
    )
    # Interpolation must stay inside the message it splits, and reproduce the
    # boundaries exactly at both ends.
    assert uniform.at(50.0) == 50.0
    assert uniform.at(51.0) == 51.0
    assert 50.0 < uniform.at(50.5) < 51.0


def test_uniform_weights_reproduce_the_unweighted_argmin() -> None:
    """Minimal reproducer for the claim tested by
    TestTokenWeightedObjective.test_absent_weights_reproduce_message_distance
    in tests/test_anchor_dp.py, which only exercises one hand-picked case
    where every churn sample happens to land on an exact integer message
    index (n=400, fractions=[0.3, 0.55, 0.9] -> samples 120.0, 220.0, 360.0).
    That masks the floor introduced by weighting: it only bites when a
    sample's scaled depth has a nonzero fractional part.

    n=96, fractions=[0.83, 0.44], k=1: unweighted picks depth 64, uniform
    weighting (any positive constant) picks depth 32. A uniform weight is
    a pure positive rescaling and must not move the argmin.
    """
    n = 96
    fractions = [0.83, 0.44]
    unweighted = optimal_anchor_depths(n, fractions, 1)
    uniform_1 = optimal_anchor_depths(n, fractions, 1, [1.0] * n)
    uniform_7 = optimal_anchor_depths(n, fractions, 1, [7.0] * n)
    assert unweighted == uniform_1 == uniform_7, (
        f"uniform weighting moved the argmin: unweighted={unweighted}, "
        f"uniform(1.0)={uniform_1}, uniform(7.0)={uniform_7}"
    )


def _objective(n: int, fractions: list[float], anchors: list[int]) -> float:
    """Message-distance loss, computed independently of the DP internals."""
    total = 0.0
    for f in fractions:
        c = min(max(f, 0.0), 1.0) * n
        total += c - max([0] + [a for a in anchors if a <= c])
    return total


def test_uniform_weight_equivalence_fuzz() -> None:
    """A uniform weight is a positive rescaling, so it must not change what the
    anchor set *costs*.

    Equal cost is the claim worth testing, not equal depths. The objective has
    exact ties: at n=128 a single anchor at 96 covering 2 samples and one at 48
    covering 4 both shed 192 message-units, so the argmin is a set and either
    element is a correct answer. Asserting on depths would fail on which tie the
    float arithmetic happened to break, which is not a property of the placement.
    The tolerance is for summation order, since the weighted path reaches the
    same value through a cumulative sum.
    """
    random.seed(7)
    mismatches = []
    for _ in range(500):
        n = random.choice([65, 100, 128, 200, 300, 400, 640])
        num_samples = random.randint(1, 10)
        fractions = [round(random.random(), 3) for _ in range(num_samples)]
        k = random.randint(1, 4)
        c = random.choice([0.001, 1.0, 7.0, 200.0, 1e6])
        unweighted = optimal_anchor_depths(n, fractions, k)
        uniform = optimal_anchor_depths(n, fractions, k, [c] * n)
        loss_unweighted = _objective(n, fractions, unweighted)
        loss_uniform = _objective(n, fractions, uniform)
        if abs(loss_unweighted - loss_uniform) > 1e-9 * max(loss_unweighted, 1.0):
            mismatches.append(
                (n, fractions, k, c, unweighted, uniform, loss_unweighted, loss_uniform)
            )
    rate = len(mismatches) / 500
    assert not mismatches, (
        f"uniform weighting changed the achieved loss in {len(mismatches)}/500 "
        f"trials ({rate:.1%}); examples: {mismatches[:5]}"
    )


# ---------------------------------------------------------------------------
# Section 3: deterministic edge cases named in the brief.
# ---------------------------------------------------------------------------


def test_k_larger_than_candidate_count_is_capped_not_erroring() -> None:
    n = 70  # only two candidates: 32, 48
    assert len(_candidate_positions(n)) == 2
    depths = optimal_anchor_depths(n, [0.5], 10)
    assert len(depths) <= 2
    assert all(d in (32, 48) for d in depths)


def test_all_zero_weights_fall_back_to_message_distance() -> None:
    n = 300
    fractions = [0.4, 0.8]
    baseline = optimal_anchor_depths(n, fractions, 2)
    assert optimal_anchor_depths(n, fractions, 2, [0.0] * n) == baseline


def test_weights_shorter_than_n_are_silently_ignored() -> None:
    """len(weights) < n falls all the way back to the unweighted objective,
    not a partial weighting. This is a real behavior worth flagging even
    though it matches the len(weights) >= n gate literally: a caller that
    passes a slightly-too-short weights array (e.g. an off-by-one against
    n_messages) gets no error and no signal, just silent unweighted mode.
    """
    n = 300
    fractions = [0.4, 0.8]
    baseline = optimal_anchor_depths(n, fractions, 2)
    short_weights = [1.0] * (n - 1)
    heavily_skewed_but_too_short = [1.0] * (n - 1)
    heavily_skewed_but_too_short[10] = 1e9  # would move the argmin if honoured
    assert optimal_anchor_depths(n, fractions, 2, short_weights) == baseline
    assert optimal_anchor_depths(n, fractions, 2, heavily_skewed_but_too_short) == baseline


def test_nan_weights_do_not_crash_and_degrade_to_unweighted() -> None:
    """max(0.0, nan) evaluates to 0.0 in CPython, so an all-NaN weights array
    makes every cumulative weight zero, which trips the cum[n] <= 0 guard and
    falls back to the unweighted objective. Documented here so a future
    change to the max() ordering does not silently start propagating NaN.
    """
    n = 200
    fractions = [0.3, 0.6]
    assert optimal_anchor_depths(n, fractions, 2, [float("nan")] * n) == optimal_anchor_depths(
        n, fractions, 2
    )


def test_samples_exactly_on_candidate_boundary_are_covered_by_that_anchor() -> None:
    """A churn sample landing exactly on a candidate value must be treated
    as covered by an anchor placed there (cost 0), matching _anchor_below's
    anchor <= position semantics.
    """
    n = 200
    candidate = 96  # a valid CANDIDATE_STEP-aligned depth
    fractions = [candidate / n]
    depths = optimal_anchor_depths(n, fractions, 1)
    assert depths == [candidate]


def test_duplicate_samples_do_not_change_the_optimum() -> None:
    n = 300
    fractions = [0.4, 0.4, 0.4, 0.4, 0.8]
    depths = optimal_anchor_depths(n, fractions, 2)
    dp_loss = _independent_loss(n, fractions, None, depths)
    best = _brute_best(n, fractions, None, 2)
    assert abs(dp_loss - best) < 1e-9


# ---------------------------------------------------------------------------
# Section 4: hysteresis and stability under the token-weighted objective.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known open item, deliberately not fixed here. The QUANTUM gate filters "
        "in messages while the objective now scores in tokens, so token mass "
        "inside the window is invisible to it. Closing the gap means letting an "
        "anchor move sub-QUANTUM when tokens justify it, which is the treadmill "
        "that cost 1.59M write tokens in the corpus. Gate the change on measured "
        "traffic, not on this constructed case. Remove the marker when fixed."
    ),
)
def test_near_neighbour_gate_can_permanently_pin_an_arbitrarily_costly_anchor() -> None:
    """The QUANTUM near-neighbour refusal in stabilize_anchor_depths is
    stated and tested (tests/test_anchor_stability.py,
    test_near_neighbour_relocation_is_refused_however_well_it_prices) purely
    in message-distance terms: the reasoning in the docstring is "shedding
    sixteen messages of expected depth ... covers writing the sixteen-
    message gap". That reasoning assumes cost scales with message count.

    With message_weights, a message-distance gate does not bound token cost:
    a single oversized message inside the blocked window (< QUANTUM messages
    away) can make the refused move worth millions of tokens, forever,
    because the anchor can never cross that gap regardless of how the price
    gate scores it.

    n=500, held anchor at 200, a 50,000,000-weight message sits at index 210
    (10 messages away, inside the QUANTUM=64 window), and every observed
    bust lands right after it. Moving the anchor 10 messages forward would
    eliminate ~50,000,010 tokens of rewrite per bust for a one-time cost of
    ~50,000,010 tokens, a 4x return over EXPECTED_REMAINING_BUSTS. The
    function still refuses the move, every round, forever.

    This test encodes the economic contract stabilize_anchor_depths claims
    to enforce (a move clears when it prices well) and is expected to FAIL
    against the current QUANTUM-distance gate, which prices moves it then
    vetoes on distance alone.
    """
    n = 500
    weights = [1.0] * n
    weights[210] = 50_000_000.0
    fractions = [211 / n] * 40
    samples = sorted(f * n for f in fractions)
    scale = _RewriteScale(n, weights)

    held = [200]
    proposed = [211]
    for _ in range(5):
        held = stabilize_anchor_depths(held, proposed, n, fractions, 1, weights)

    stuck_loss = _mean_rewrite_depth(held, samples, scale)
    optimal_loss = _mean_rewrite_depth(proposed, samples, scale)
    gain = (stuck_loss - optimal_loss) * EXPECTED_REMAINING_BUSTS
    cost = _move_write_cost(200, 211, scale)

    assert gain > cost, "test setup is not even economically justified, fix the fixture"
    assert held == [211], (
        f"anchor permanently stuck at {held} despite a {gain:.0f}-token justified "
        f"move (cost {cost:.0f}) because the target is only "
        f"{abs(211 - 200)} messages away, under QUANTUM={QUANTUM}"
    )


def test_price_gate_still_refuses_a_bad_move_within_quantum() -> None:
    """Sanity control for the test above: when the near move is NOT
    economically justified, refusing it is correct behaviour, not a bug.
    This one is expected to pass.
    """
    n = 300
    weights = [1.0] * n
    fractions = [0.55] * 20  # churn just past the live anchor, matches the
    # existing test_useful_anchor_is_not_pulled_backward fixture shape
    held = stabilize_anchor_depths([256], [260], n, fractions, 1, weights)
    assert held == [256]


def test_perf_test_is_load_bearing_against_a_naive_linear_scan() -> None:
    """Cross-check for TestTokenWeightedObjective's sibling perf assertion
    (test_refined_grid_stays_cheap_on_the_hot_path in tests/test_anchor_dp.py)
    which asserts elapsed < 0.05s per call. Reimplements the pre-optimization
    segment_loss as a linear scan (no bisect, no prefix sums) against the
    same parameters and confirms it blows through that ceiling, so the 50ms
    budget is not so loose that any implementation would pass it.
    """

    def optimal_anchor_depths_naive(
        n_messages: int,
        churn_fractions: list[float],
        k_anchors: int,
        message_weights: list[float] | None = None,
    ) -> float:
        if k_anchors <= 0 or n_messages <= MIN_DEPTH + TAIL_GUARD:
            return 0.0
        candidates = _candidate_positions(n_messages)
        if not candidates or not churn_fractions:
            return 0.0
        samples = sorted(min(max(f, 0.0), 1.0) * n_messages for f in churn_fractions)
        k = min(k_anchors, len(candidates))
        scale = _RewriteScale(n_messages, message_weights)

        def segment_loss_naive(anchor: float, bound: float) -> float:
            total = 0.0
            for c in samples:
                if anchor <= c < bound:
                    total += scale.span(anchor, c)
            return total

        far = float(n_messages) + 1.0
        m = len(candidates)
        inf = float("inf")
        f = [[inf] * m for _ in range(k + 1)]
        for i, pos in enumerate(candidates):
            f[1][i] = segment_loss_naive(0.0, pos)
        for j in range(2, k + 1):
            for i, pos in enumerate(candidates):
                for h in range(i):
                    prev = f[j - 1][h]
                    if prev >= inf:
                        continue
                    cost = prev + segment_loss_naive(candidates[h], pos)
                    if cost < f[j][i]:
                        f[j][i] = cost
        best_loss = inf
        for j in range(1, k + 1):
            for i, pos in enumerate(candidates):
                if f[j][i] >= inf:
                    continue
                total = f[j][i] + segment_loss_naive(pos, far)
                if total < best_loss:
                    best_loss = total
        return best_loss

    fractions = [i / 64.0 for i in range(64)]
    weights = [200.0] * 2000

    start = time.perf_counter()
    for _ in range(20):
        optimal_anchor_depths(2000, fractions, 4, weights)
    real_elapsed = (time.perf_counter() - start) / 20

    start = time.perf_counter()
    for _ in range(5):
        optimal_anchor_depths_naive(2000, fractions, 4, weights)
    naive_elapsed = (time.perf_counter() - start) / 5

    assert real_elapsed < 0.05, f"optimized DP took {real_elapsed * 1000:.1f}ms, expected < 50ms"
    assert naive_elapsed > 0.05, (
        f"naive linear-scan reimplementation took only {naive_elapsed * 1000:.1f}ms, "
        f"under the 50ms gate; the perf test would not have caught this regression"
    )
