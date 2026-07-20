# Upstream integration plan, 2026-07-20

Upstream is `chopratejas/headroom` (origin). Our consolidated branch
`codex-claude-stack-optimization-20260714` last shared history with it on
2026-07-14. Upstream has moved about 130 commits since then. This plan
sequences the integration after tonight's restart, not before it. The
bleeding-money fixes (effort pin, model-cap, closed-loop accounting,
protect_recent, rebase-flush) land on our branch and go live first. The
rebase is a separate, deliberate pass.

## Conflict surface (measured)

Our tonight files: `handlers/anthropic.py`, `hybrid_mode.py`, `streaming.py`,
`server.py`, `output_shaper.py`, `output_effort_policy.py`,
`operational_audit.py`, plus new modules `cache_reconciliation.py`,
`replay_capture.py`, `structural_ledger.py`, `touch_registry.py`,
`anchor_dp.py` additions, and `evals/novelty_routing_eval.py`.

The hotspot is `handlers/anthropic.py`. Two open PRs edit it directly
(#2365, #2444) and the compressor-registry series changes its call sites.
Secondary collisions: `cache/prefix_tracker.py` (#2382 versus our churn
observation stash) and `server.py` plus `cost.py` (#2437 versus our /stats
and pricing). Our new modules are additive and graft with no conflict.

## Tier 0, security and dependencies (take verbatim)

These are already merged upstream, so a rebase brings them for free. Called
out because they are non-negotiable and must not be dropped in a selective
cherry-pick.

- #2342 exclude the compromised ast-grep-cli 0.44.1 (supply-chain trojan).
- #2348 bump mcp to clear three high-severity CVEs.
- #2349 refresh the stale uv.lock. Windows-wheel and lock follow-ups ride
  along.

## Tier 1, accounting and cache fixes that reconcile with tonight's work

The careful tier. Each one touches code we changed or a measurement we rely on.

- #2408 (open) subscription usage dedup by message id. Confirmed present as a
  live bug on our branch: `compute_window_tokens` in
  `headroom/subscription/session_tracking.py` sums `message.usage` per
  transcript line, and Claude Code splits one response across many lines that
  each carry the same request-level usage. Window totals and every savings
  rate are inflated by content-block count. Fix is isolated to one function,
  zero conflict with tonight's files, folded into tonight's commit batch
  rather than deferred to the rebase.
- #2437 (open) account CCR retrieval drawback as net savings. This is
  upstream's version of workstream H closed-loop accounting. Adopt its
  `cost.py` pricing model to replace our hardcoded 1.25x/0.1x/2x multipliers in
  `cache_reconciliation.py` and the anchor DP arm.
- #2439 (open) move the savings-ledger append off the event loop. Apply the
  same pattern to our reconciliation append in the streaming finalize path,
  which is currently synchronous inside a try/except.
- #2382 (merged) preserve client cache_control ttl when consolidating
  breakpoints. Reconcile against our `_hr_last_churn_observation` stash on the
  prefix tracker and against the effort-pin gate, both read cache_control.
- #2365 (open) compress cache-mode cold starts and tag prefix-mismatch
  passthrough. The key overlap. This is upstream independently building
  compress-on-bust, the same idea as tonight's hybrid rebase-flush. Read it
  first, then decide adopt-theirs, keep-ours, or merge-concepts. Do not apply
  blind.
- #2444 (open) redeclare the CCR tool from sessionless Anthropic history.
  Reconcile in `handlers/anthropic.py` alongside #2365.

## Tier 2, compressor registry migration (structural)

The `feat(transforms)` pluggable compressor registry series (#2370, #2371,
#2373, #2388, #2391, #2399, #2400, #2404, #2411) refactored the transform
dispatch into a registry. Our compression-path wiring sits on the old dispatch
shape. Migrate as one coherent unit. Our new modules are unaffected. This is
the bulk of the mechanical conflict work, but it is mechanical once the
pattern is understood. #2433 (open) adds a lossless-compaction provider seam
on top, take it with the series.

## Tier 3, aligned but optional

- #2418 four-tier cost calculator. Could underpin the Tier 1 pricing swap.
  Evaluate together with #2437.
- #2395 model-router shadow mode (log-only). Philosophically aligned with our
  log-only method, but the standing stance is that the harness owns model
  choice and we ship only the subagent cap. Note and defer.
- #2425 and #2413 Serena symbol-first guidance and code-memory defaults.
  Relevant to the local environment, low risk.
- The token-count None-guard series (#2434, #2431, #2347, #2324, and kin) is
  routine hardening that arrives with the rebase.

## Integration mechanics

A straight `git rebase origin/main` of our branch is the wrong tool. Our branch
carries 212 commits of prior divergence, much of the hybrid and optical work is
probably already represented upstream in a different packaging, so a blind
rebase would fight duplicate content and the registry refactor at once.

Recommended sequence:

1. Audit our 212 commits against upstream. Identify the genuinely unique
   contributions (the ones not superseded by upstream's own hybrid work). Only
   those get re-applied.
2. Branch a fresh integration branch from origin/main. Tier 0 and the routine
   None-guards are already in it.
3. Re-apply our additive modules first. They cherry-pick clean:
   `cache_reconciliation.py`, `replay_capture.py`, `structural_ledger.py`,
   `touch_registry.py`, the anchor DP arm, and the novelty eval.
4. Hand-reconcile the `handlers/anthropic.py` and `hybrid_mode.py` wiring
   against #2365, #2444, and the registry call sites. This is the real work.
5. Fold the Tier 1 accounting fixes, adopting upstream's cost model over our
   hardcoded multipliers.

## Verification

Workstream E (the replay corpus) is the regression harness for this exact
migration. Capture live traffic before the rebase, replay the same requests
against the integrated branch, and diff token outcomes request by request. A
clean replay diff is the graduation gate for the rebase, same bar the
log-only workstreams must clear before they mutate live traffic.
