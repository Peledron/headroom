"""Exactness and edge cases for DP cache-breakpoint placement."""

from itertools import combinations

from headroom.cache.anchor_dp import (
    MIN_DEPTH,
    QUANTUM,
    TAIL_GUARD,
    AbstractVsKeepWarmArmStats,
    _candidate_positions,
    decide_abstract_vs_keep_warm,
    fallback_anchor_depth,
    optimal_anchor_depths,
)


def brute_force_loss(n: int, fractions: list[float], anchors: list[int]) -> float:
    samples = [min(max(f, 0.0), 1.0) * n for f in fractions]
    points = [0] + sorted(anchors)
    total = 0.0
    for c in samples:
        below = max(p for p in points if p <= c)
        total += c - below
    return total


def brute_force_best(n: int, fractions: list[float], k: int) -> float:
    candidates = _candidate_positions(n)
    best = brute_force_loss(n, fractions, [])
    for size in range(1, min(k, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            best = min(best, brute_force_loss(n, fractions, list(combo)))
    return best


class TestEdgeCases:
    def test_zero_budget_and_tiny_history(self) -> None:
        assert optimal_anchor_depths(500, [0.5], 0) == []
        assert optimal_anchor_depths(MIN_DEPTH + TAIL_GUARD, [0.5], 3) == []

    def test_no_samples_falls_back_to_mid_anchor(self) -> None:
        n = 200
        assert optimal_anchor_depths(n, [], 3) == [fallback_anchor_depth(n)]

    def test_positions_quantized_and_tail_guarded(self) -> None:
        depths = optimal_anchor_depths(300, [0.1, 0.5, 0.9], 3)
        for d in depths:
            assert d % QUANTUM == 0
            assert MIN_DEPTH <= d <= 300 - TAIL_GUARD

    def test_returns_sorted_unique(self) -> None:
        depths = optimal_anchor_depths(400, [0.2, 0.4, 0.6, 0.8] * 4, 3)
        assert depths == sorted(set(depths))


class TestOptimality:
    def test_matches_brute_force_enumeration(self) -> None:
        # Exactness on assorted churn shapes: DP loss must equal the loss of
        # the best subset found by exhaustive enumeration.
        cases = [
            (300, [0.9, 0.85, 0.95], 2),
            (300, [0.1, 0.9], 2),
            (500, [0.3, 0.31, 0.29, 0.9], 3),
            (500, [0.5] * 10, 1),
            (1000, [0.2, 0.4, 0.6, 0.8], 3),
        ]
        for n, fractions, k in cases:
            depths = optimal_anchor_depths(n, fractions, k)
            dp_loss = brute_force_loss(n, fractions, depths)
            best_loss = brute_force_best(n, fractions, k)
            assert abs(dp_loss - best_loss) < 1e-9, (n, fractions, k, depths)

    def test_clustered_churn_gets_local_anchor(self) -> None:
        # All churn near 90% depth of a 640-message history: one anchor jammed
        # right below the cluster beats a mid-depth anchor by construction.
        n = 640
        depths = optimal_anchor_depths(n, [0.9, 0.88, 0.92], 1)
        assert len(depths) == 1
        # Deepest grid point at or below the shallowest sample (0.88 * 640).
        assert depths[0] == int(0.88 * n) // QUANTUM * QUANTUM

    def test_bimodal_churn_splits_anchors(self) -> None:
        # Churn at two distinct depths: with budget 2 the DP must cover both
        # clusters instead of doubling up on one.
        n = 640
        depths = optimal_anchor_depths(n, [0.3, 0.3, 0.3, 0.8, 0.8, 0.8], 2)
        assert len(depths) == 2
        assert depths[0] <= 0.3 * n < depths[1] <= 0.8 * n


class TestAbstractVsKeepWarmArm:
    """Log-only pricing arm: keep-warm cost per turn vs one-time bust cost."""

    def test_keep_warm_wins_for_short_remaining_conversation(self) -> None:
        # A single expected future turn barely accrues keep-warm cost, so
        # busting the suffix to save it is not worth it.
        decision = decide_abstract_vs_keep_warm(
            history_tokens=1000,
            summary_tokens=200,
            suffix_tokens=500,
            expected_remaining_turns=1.0,
        )
        assert decision.would_abstract is False
        assert decision.keep_warm_cost < decision.abstract_cost

    def test_abstract_wins_for_long_remaining_conversation(self) -> None:
        # Many expected future turns make keeping a large history span warm
        # expensive enough that a one-time bust to summarize it pays off.
        decision = decide_abstract_vs_keep_warm(
            history_tokens=5000,
            summary_tokens=200,
            suffix_tokens=500,
            expected_remaining_turns=50.0,
        )
        assert decision.would_abstract is True
        assert decision.keep_warm_cost > decision.abstract_cost

    def test_negative_inputs_are_clamped_not_rejected(self) -> None:
        decision = decide_abstract_vs_keep_warm(
            history_tokens=-10,
            summary_tokens=-5,
            suffix_tokens=-5,
            expected_remaining_turns=-1.0,
        )
        assert decision.history_tokens == 0
        assert decision.summary_tokens == 0
        assert decision.suffix_tokens == 0
        assert decision.expected_remaining_turns == 0.0
        assert decision.would_abstract is False

    def test_stats_tally_agreement_with_current_always_keep_warm(self) -> None:
        # The production DP arm never abstracts today, so recording against
        # current_would_abstract=False must count every abstain as agreement
        # and every would_abstract=True as a disagreement, purely for the
        # log-only counters, without changing production behavior.
        stats = AbstractVsKeepWarmArmStats()
        keep_warm_decision = decide_abstract_vs_keep_warm(1000, 200, 500, 1.0)
        abstract_decision = decide_abstract_vs_keep_warm(5000, 200, 500, 50.0)

        stats.record(keep_warm_decision, production_would_abstract=False)
        stats.record(abstract_decision, production_would_abstract=False)

        snapshot = stats.snapshot()
        assert snapshot["would_keep_warm"] == 1
        assert snapshot["would_abstract"] == 1
        assert snapshot["agrees_with_production"] == 1
        assert snapshot["disagrees_with_production"] == 1
