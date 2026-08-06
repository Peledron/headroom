"""A live cache anchor must stay put unless moving it pays for itself.

The DP in :mod:`headroom.cache.anchor_dp` rescales its churn samples by the
current message count and re-solves over a ring buffer that turns over every
turn, so its argmin walks even when nothing about the conversation changed.
Quantization hides the walk until the drift crosses a grid boundary, then the
anchor jumps a whole quantum. Measured against the forwarded-request corpus
that cost 1.59M cache-write tokens across 75 anchor moves in 987 turns, the
single largest remaining write item after the overlay and TTL fixes.
"""

from __future__ import annotations

import itertools

from headroom.cache.anchor_dp import (
    QUANTUM,
    optimal_anchor_depths,
    stabilize_anchor_depths,
)


def test_anchor_never_walks_backward_while_history_grows():
    """Churn held at a constant *fraction* sits at a forward-moving absolute
    message, so chasing it is a treadmill. Advancing is legitimate and cheap:
    the cached entry still matches up to the old anchor, so only the gap is
    written. Retreating is not, because no cached entry ends at the shallower
    boundary and the whole prefix is re-written at the 1h premium. The
    invariant is direction, not stillness."""
    depths = [128]
    for n in range(300, 360, 2):
        proposed = optimal_anchor_depths(n, [0.9] * 20, 2)
        after = stabilize_anchor_depths(depths, proposed, n, [0.9] * 20, 2)
        for previous in depths:
            if previous in after:
                continue
            # Only a *newly placed* shallower anchor is a retreat. An anchor
            # that was already live below the one being replaced has not moved
            # and has cost nothing, so counting it here reported a retreat
            # every time a deeper anchor legitimately advanced.
            retreated = [d for d in after if d < previous and d not in depths]
            assert not retreated, f"anchor {previous} retreated to {retreated} at n={n}"
        depths = after


def test_oscillating_proposal_never_moves_the_anchor():
    """The DP flip-flopping between two optima must not re-bill the prefix."""
    depths = [192]
    moves = 0
    for proposed in itertools.islice(itertools.cycle([[128], [192]]), 24):
        after = stabilize_anchor_depths(depths, proposed, 400, [0.5] * 20, 1)
        if after != depths:
            moves += 1
        depths = after
    assert moves == 0
    assert depths == [192]


def test_genuinely_better_anchor_still_wins():
    """Stickiness must not become a lock. Churn that consistently lands deep
    has to be able to pull a stale shallow anchor forward."""
    assert stabilize_anchor_depths([64], [320], 400, [0.85] * 20, 1) == [320]


def test_useful_anchor_is_not_pulled_backward():
    """An anchor that already sits just below the churn is doing its job.
    Retreating from it re-writes the whole prefix at the 1h premium and buys
    a *worse* rewrite depth, so the price gate must refuse it. This is the
    direction that cost 1.59M tokens in the corpus."""
    # Churn lands at 0.55 * 512 = 281.6, just past the live anchor at 256.
    assert stabilize_anchor_depths([256], [128], 512, [0.55] * 20, 1) == [256]


def test_forward_move_clears_when_the_anchor_has_gone_stale():
    """The mirror case: churn has moved well past the anchor, so advancing
    sheds real rewrite depth and only writes the gap."""
    assert stabilize_anchor_depths([128], [256], 512, [0.55] * 20, 1) == [256]


def test_unplaced_session_takes_the_dp_proposal_verbatim():
    """With nothing to preserve there is no move to price."""
    proposed = optimal_anchor_depths(400, [0.7] * 20, 2)
    assert stabilize_anchor_depths([], proposed, 400, [0.7] * 20, 2) == proposed


def test_anchor_past_the_tail_is_released():
    """A recorded anchor that history has since truncated past must not pin
    a position that no longer exists."""
    depths = stabilize_anchor_depths([900], [128], 200, [0.6] * 20, 1)
    assert 900 not in depths


def test_near_neighbour_relocation_is_refused_however_well_it_prices():
    """A live anchor must not chase the argmin in small steps.

    Churn pinned to a fraction of a growing history marches forward in absolute
    terms, so every turn offers a slightly better position just ahead. Each such
    step prices well alone, and following them all is a treadmill that pays a
    rewrite per turn. The coarse candidate grid used to make these proposals
    impossible; with the grid refined for placement accuracy the hysteresis has
    to hold explicitly.
    """
    # Churn at 0.9 * 304 = 273.6, sixteen messages past the live anchor.
    assert stabilize_anchor_depths([256], [272], 304, [0.9] * 20, 2) == [256]


def test_growth_does_not_march_the_anchor_forward_every_turn():
    """The end-to-end version: over a run of appended turns under fixed-fraction
    churn, anchor moves must be bounded by the hysteresis, not by the turn count.

    An anchor is allowed to advance, and under fixed-fraction churn it should:
    the target really is moving. What it must not do is take a step per turn.
    QUANTUM is what separates those, so the ceiling is how many whole QUANTUM
    steps the history grew by, per anchor, and it does not rise if the same
    growth is delivered over more and smaller turns.
    """
    k_anchors = 2
    start, stop, step = 300, 420, 2
    depths = [128]
    moves = 0
    for n in range(start, stop, step):
        proposed = optimal_anchor_depths(n, [0.9] * 20, k_anchors)
        after = stabilize_anchor_depths(depths, proposed, n, [0.9] * 20, k_anchors)
        if sorted(after) != sorted(depths) and depths:
            moves += 1
        depths = after
    ceiling = -(-(stop - start) // QUANTUM) * k_anchors
    assert moves <= ceiling, f"anchor moved {moves} times, ceiling {ceiling}"
