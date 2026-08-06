"""Per-auth-mode compression policy — Phase F PR-F2.1, extended in F2.2 (Python parity).

Hand-mirrored port of `headroom_core::compression_policy::CompressionPolicy`
(Rust). The Rust crate is the source of truth; this module exists so
the Python proxy's `TransformPipeline` (which still runs `CacheAligner`
and other detector-only transforms) can read the same per-mode flags
the Rust dispatcher does.

A parity test (`tests/test_compression_policy.py`) instantiates one of
each variant and asserts the field map matches what the Rust unit
tests assert. F2.2 should consider exposing the Rust struct via PyO3
to retire this hand-mirror — that's deliberately out of scope here so
F2.1/F2.2 can ship.

See `crates/headroom-core/src/compression_policy.rs` for the canonical
docstring (per-mode rationale, why-a-struct, etc.).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from headroom.proxy.auth_mode import AuthMode

# ── F2.2 per-mode default values (CONSERVATIVE pending bake telemetry) ──
# Mirrors the Rust ``pub(crate) const`` block in
# ``crates/headroom-core/src/compression_policy.rs``. Centralised here so
# a follow-up tune lands in one place per language. The Rust tests assert
# the values directly; the Python parity tests assert the *fields* match
# (not the values — that would double-pin against drift in a way that
# masks real divergence).
#
# Per the realignment build constraints (project memory
# ``feedback_realignment_build_constraints.md``): "configurable / no
# hardcoded values". The configuration *is* the per-mode default — we
# deliberately do NOT add a separate env var per field. Operators tune
# by editing these constants and shipping a new build, mirroring the
# Rust pattern exactly.

#: PAYG: aggressive — let volatile content noise up to ~128 tokens slip
#: before flagging. Higher than Subscription because PAYG users opt in
#: to aggressive compression.
_VOLATILE_TOKEN_THRESHOLD_PAYG: int = 128

#: Subscription: conservative — flag volatile content earlier (32
#: tokens) so cache prefixes stay stable.
_VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION: int = 32

#: PAYG: cap lossy compression at 45% of original tokens. Aggressive
#: but bounded — F2.1 had no cap (effectively ``1.0``), F2.2 introduces one.
_MAX_LOSSY_RATIO_PAYG: float = 0.45

#: Subscription: conservative cap at 25%. Cache stability over savings.
_MAX_LOSSY_RATIO_SUBSCRIPTION: float = 0.25

#: Anthropic prompt-cache write multiplier: a ``cache_creation`` token
#: costs 1.25x a plain input token (5-minute TTL tier). Input to the
#: net-cost mutation formula (#856). Mirrors the Rust ``pub const``.
#: ponytail: hardcoded to the 5m tier. A client on Anthropic's 1h cache
#: (ENABLE_PROMPT_CACHING_1H / cache_control.ttl="1h", which Headroom
#: preserves) writes at 2.0x, so its mutations are gated with a ~40%
#: under-stated write penalty. Harmless while the net-cost gate stays
#: default-off (HEADROOM_NET_COST_POLICY); thread the TTL from
#: cold_prefix.anthropic_cache_ttl_seconds through ContentRouter ->
#: net_mutation_gain if that gate is ever turned on.
CACHE_WRITE_MULTIPLIER: float = 1.25

#: Anthropic prompt-cache write multiplier for the 1-hour TTL tier: a
#: ``cache_creation`` token in the extended-cache-ttl breakpoint costs 2x a
#: plain input token. A mutation that busts a suffix cached at 1h therefore
#: wastes 2x, not 1.25x, so the break-even needs ~65% more remaining reads
#: than the 5m tier. Selected per call via ``write_multiplier_for_ttl``.
CACHE_WRITE_MULTIPLIER_1H: float = 2.0

#: Anthropic prompt-cache read multiplier: a ``cache_read`` token costs
#: 0.1x a plain input token. Input to the net-cost mutation formula
#: (#856). Mirrors the Rust ``pub const``.
CACHE_READ_MULTIPLIER: float = 0.1


def _clamp_write_multiplier(write_multiplier: float | None) -> float:
    """Resolve the write multiplier, clamping absurd inputs to a real tier.

    ``None`` selects the 5-minute default. Otherwise the value is clamped to the
    two real Anthropic tiers ``[1.25, 2.0]``: a value at or below zero would
    invert the sign of the ``(w - r)`` penalty term and force a busting admit, so
    it must never reach the formula. NaN is passed through untouched so the
    downstream ``isnan`` guard fails the decision closed (no admit).
    """
    if write_multiplier is None:
        return CACHE_WRITE_MULTIPLIER
    if math.isnan(write_multiplier):
        return write_multiplier
    return min(max(write_multiplier, CACHE_WRITE_MULTIPLIER), CACHE_WRITE_MULTIPLIER_1H)


def write_multiplier_for_ttl(ttl: str | None) -> float:
    """Cache-write multiplier for a breakpoint's TTL tier.

    Anthropic prices the two ephemeral tiers differently: the 1-hour
    (extended) tier writes at 2x a plain input token, the 5-minute (default)
    tier at 1.25x. The net-cost gate must charge the tier of the suffix it
    would invalidate, or it under-counts the cost of busting a 1h cache by
    ~60%. Anything other than the literal ``"1h"`` (including ``"5m"`` and
    ``None``) uses the default 5-minute multiplier.
    """
    return CACHE_WRITE_MULTIPLIER_1H if ttl == "1h" else CACHE_WRITE_MULTIPLIER


@dataclass(frozen=True, slots=True)
class CompressionPolicy:
    """Per-auth-mode policy that downstream compression stages consult.

    Five fields after F2.2:

    - ``live_zone_only`` (F2.1) — gate for the Python TransformPipeline
      to skip pre-cache-marker mutation.
    - ``cache_aligner_enabled`` (F2.1) — gate for the Python
      ``CacheAligner`` transform.
    - ``volatile_token_threshold`` (F2.2) — per-mode token threshold
      below which content is treated as cache-stable. Plumbed through
      the struct; no current detector consumes it (intentional — the
      detector refactor is a follow-up).
    - ``max_lossy_ratio`` (F2.2) — per-mode upper bound on lossy
      compression aggressiveness (fraction in ``[0.0, 1.0]``). Plumbed
      through the struct; no current compressor consumes it.
    - ``toin_read_only`` (F2.2) — TOIN learning gate. ``True`` =
      serve cached patterns but never write new observations from
      this request (Subscription).
    """

    live_zone_only: bool
    """When True, transforms MUST NOT modify bytes outside the post-
    cache-marker live zone. The Rust live-zone dispatcher is already
    live-zone-only by construction, so this flag is effectively a
    no-op on the Rust path; it exists for the Python TransformPipeline
    where transforms like CacheAligner inspect the cached prefix."""

    cache_aligner_enabled: bool
    """When False, the CacheAligner transform's `should_apply` MUST
    return False — that's the load-bearing F2.1 gate for the cache-
    instability complaints in #327 / #388."""

    volatile_token_threshold: int
    """F2.2: per-mode token-count threshold below which content is
    treated as cache-stable. Subscription is conservative
    (low → flag aggressively → keep prompts stable); PAYG aggressive
    (high → tolerate more volatile noise). Plumbed but unconsumed in
    F2.2 — the volatile detector in ``cache_aligner.py`` is shape-
    based, not token-count-based; wiring it is a follow-up."""

    max_lossy_ratio: float
    """F2.2: per-mode upper bound on lossy compression aggressiveness,
    expressed as the fraction of original tokens that may be dropped
    (``0.0`` = no lossy, ``1.0`` = unlimited). Subscription ``0.25``;
    PAYG ``0.45``. Plumbed but unconsumed in F2.2 — distinct from the
    caller-driven ``target_ratio`` kwarg in the Python ContentRouter."""

    toin_read_only: bool
    """F2.2: when True, TOIN serves cached recommendations but
    never writes new pattern observations from this request.
    Subscription True (consistency over learning); PAYG/OAuth False
    (network effect keeps growing). The gate is read by
    ``smart_crusher.py`` and ``content_router.py`` at the
    ``record_compression`` call sites."""

    def net_mutation_gain(
        self,
        delta_t: int,
        suffix_tokens: int,
        expected_reads: float,
        p_alive: float,
        write_multiplier: float | None = None,
    ) -> float:
        """Net gain (in plain-input-token cost units) of a mutation that
        removes ``delta_t`` tokens from a message whose cached suffix is
        ``suffix_tokens`` long (#856).

        ``write_multiplier`` overrides ``w`` with the actual TTL tier of the
        cache being written/invalidated (``write_multiplier_for_ttl``): 2.0 for
        the 1-hour tier, 1.25 (the default when ``None``) for 5-minute. Charging
        the true tier is load-bearing, a mutation that busts a 1h suffix costs
        2x, so its break-even sits ~65% deeper than the 5m formula assumed.

        Mirrors ``CompressionPolicy::net_mutation_gain`` in the Rust
        crate (source of truth — see its docstring for the derivation)::

            gain = dT * (w + r*(R - 1)) - P_alive * (w - r) * (S + dT)

        The warm-case penalty covers ``S + dT``: with a live cache the
        ``dT`` tokens are already cache-written, so keeping them costs
        only reads — a mutation avoids at most ``dT*r*R``, not a fresh
        write.

        Inputs are clamped: ``delta_t``/``suffix_tokens`` to ``>= 0``
        (the Rust signature takes ``u32``), ``expected_reads`` to
        ``>= 0`` (NaN → 0), ``p_alive`` to ``[0, 1]`` (NaN → 1, the
        conservative full-penalty assumption — same as Rust).
        """
        w = _clamp_write_multiplier(write_multiplier)
        r = CACHE_READ_MULTIPLIER
        dt = max(0, delta_t)
        suffix = max(0, suffix_tokens)
        # Python max()/min() propagate NaN from the first argument, unlike
        # f32::max in the Rust source of truth — guard explicitly.
        reads = 0.0 if math.isnan(expected_reads) else max(expected_reads, 0.0)
        alive = 1.0 if math.isnan(p_alive) else min(max(p_alive, 0.0), 1.0)
        return float(dt) * (w + r * (reads - 1.0)) - alive * (w - r) * float(suffix + dt)

    def should_mutate_deep(
        self,
        delta_t: int,
        suffix_tokens: int,
        expected_reads: float,
        p_alive: float,
        write_multiplier: float | None = None,
    ) -> bool:
        """Decision form of :meth:`net_mutation_gain`: mutate iff the
        gain is strictly positive."""
        return (
            self.net_mutation_gain(
                delta_t, suffix_tokens, expected_reads, p_alive, write_multiplier
            )
            > 0.0
        )

    def break_even_reads(
        self,
        delta_t: int,
        suffix_tokens: int,
        write_multiplier: float | None = None,
    ) -> float:
        """Remaining-read count at which a warm-cache (``p_alive=1``)
        mutation breaks even::

            R = ((w - r) / r) * (S/dT)   = 11.5 * S/dT  (Anthropic 5-min)
                                        = 19.0 * S/dT  (Anthropic 1-hour)

        With the corrected penalty this reproduces the #856 anchors
        exactly: 2K/50K -> 287.5, 50K/10K -> 2.3 (at the 5m tier).
        ``write_multiplier`` selects the tier (1h roughly ``19 * S/dT``).

        Returns 0 when ``delta_t`` is ``<= 0`` (no savings — callers
        gate on ``delta_t > 0``; the Rust signature takes ``u32``).
        Mirrors the Rust method.
        """
        if delta_t <= 0:
            return 0.0
        w = _clamp_write_multiplier(write_multiplier)
        r = CACHE_READ_MULTIPLIER
        return ((w - r) / r) * (float(max(0, suffix_tokens)) / float(delta_t))


def policy_for_mode(mode: AuthMode) -> CompressionPolicy:
    """Resolve the F2.1+F2.2 policy for an auth mode.

    PAYG and OAuth are identical (aggressive: live-zone-not-only,
    cache-aligner on, relaxed thresholds, TOIN write-enabled).
    Subscription is the user-visible win: live-zone-only with cache
    aligner disabled, tighter thresholds, TOIN read-only.

    F2.2-followup may diverge OAuth from PAYG once telemetry is
    collected.
    """
    if mode == AuthMode.PAYG:
        return CompressionPolicy(
            live_zone_only=False,
            cache_aligner_enabled=True,
            volatile_token_threshold=_VOLATILE_TOKEN_THRESHOLD_PAYG,
            max_lossy_ratio=_MAX_LOSSY_RATIO_PAYG,
            toin_read_only=False,
        )
    if mode == AuthMode.OAUTH:
        # Identical to PAYG in F2.1/F2.2. The parity test in
        # ``tests/test_compression_policy.py`` is the canary that
        # catches a future divergence and forces a deliberate update
        # there + in the Rust crate.
        return CompressionPolicy(
            live_zone_only=False,
            cache_aligner_enabled=True,
            volatile_token_threshold=_VOLATILE_TOKEN_THRESHOLD_PAYG,
            max_lossy_ratio=_MAX_LOSSY_RATIO_PAYG,
            toin_read_only=False,
        )
    if mode == AuthMode.SUBSCRIPTION:
        return CompressionPolicy(
            live_zone_only=True,
            cache_aligner_enabled=False,
            volatile_token_threshold=_VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION,
            max_lossy_ratio=_MAX_LOSSY_RATIO_SUBSCRIPTION,
            toin_read_only=True,
        )
    raise ValueError(f"Unhandled AuthMode variant: {mode!r}")


def policy_default_payg() -> CompressionPolicy:
    """The PAYG-equivalent policy used when the
    ``HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT`` flag is disabled
    (default in F2.1 c1-c4; flipped to enabled in c5/5).

    Centralised so the proxy handlers do not duplicate the constant,
    and so a future change to PAYG semantics propagates to both the
    enforcement-on and enforcement-off paths.
    """
    return policy_for_mode(AuthMode.PAYG)


_ENFORCEMENT_ENV = "HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT"


def is_enforcement_enabled() -> bool:
    """Read the enforcement flag from the environment.

    Same env var the Rust proxy reads (``Config::auth_mode_policy_enforcement``)
    so the two paths stay in lockstep with one operator switch.
    Default (when unset): ``True`` from F2.1 c5/5 onward, matching
    the Rust default after the c5 flip.

    NOT cached — read every call so an operator can flip the env var
    in a hot-reload scenario. The cost is one ``dict.get`` per call,
    well below noise.
    """
    val = os.environ.get(_ENFORCEMENT_ENV, "enabled").strip().lower()
    # Same set of off-values the telemetry beacon honours
    # (`headroom/telemetry/beacon.py::_OFF_VALUES`) so operators don't
    # have to remember a different vocabulary per flag.
    return val not in ("disabled", "off", "false", "0", "no")


def resolve_policy(auth_mode: AuthMode | None) -> CompressionPolicy:
    """Resolve the effective ``CompressionPolicy`` for a request.

    - If the enforcement flag is off, returns PAYG-equivalent
      regardless of the classified auth mode.
    - If the enforcement flag is on and ``auth_mode`` is ``None``,
      returns PAYG-equivalent (defensive default for the unclassified
      / batch-row path).
    - Otherwise returns the per-mode policy.

    This is the single public entry point handlers should call when
    deriving the policy from a request's classification result.
    """
    if auth_mode is None or not is_enforcement_enabled():
        return policy_default_payg()
    return policy_for_mode(auth_mode)
