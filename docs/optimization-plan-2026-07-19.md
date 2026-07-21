# Optimization plan, 2026-07-19

## Verified research baselines (figures that survived verification)

- AgentDiet (arXiv 2509.23586): 39.9 to 59.7 percent input reduction, 21.1 to
  35.9 percent cost reduction. Uses LLM-driven reflection, so for this project
  it is the benchmark to beat with deterministic rules, not a recipe.
- CROP (arXiv 2604.14214): 80.6 percent token reduction via regularized prompt
  optimization. Basis for the phase 2 output-budget direction.
- Agentic Plan Caching (arXiv 2506.14852): 50.31 percent cost, 27.28 percent
  latency reduction with cached plan templates. Basis for phase 3.
- Multi-turn degradation (arXiv 2505.06120): 39 percent average quality drop
  with unreliable history. The reason trajectory reduction is a quality lever,
  not only a cost lever.
- Anthropic cache economics: writes 1.25x (5m) or 2x (1h), reads 0.1x.
  Official break-even is 1 hit for the 5m tier and 2 hits for the 1h tier.
- Measured locally 2026-07-19: changing output_config.effort busts the prompt
  cache (undocumented). See workstream A.
- SLM difficulty routing: the claimed 70 to 80 percent figure never verified,
  and the harness owns model choice anyway. The shipped variant here is the
  subagent model cap. Not planned further.

Standing plan for the next round of headroom work. Ordered by expected payoff
per unit of risk. Everything lands log-only first unless marked otherwise.
Companion docs: local-change-ledger-2026-07-19.md (what already shipped),
session-investigation-2026-07-19.md, and
upstream-integration-plan-2026-07-20.md (the tiered plan for merging the 130
upstream commits and open PRs into this fork).

## Measured finding driving workstream A

Changing `output_config.effort` between requests invalidates the Anthropic
prompt cache, even though the caching docs only document `thinking` changes as
invalidating and the effort docs recommend "dynamic effort" with no cache
warning. Measured in a live session on 2026-07-19:

| time (UTC) | event | cache_read | cache_creation |
|---|---|---|---|
| 15:09:49 | effort high | 71,240 | 239 |
| 15:09:56 | shaper flips to low | 0 | 70,806 |
| 15:10:08 | revert to high | 0 | 71,924 |

One flip caused two full-prefix cache writes, about 143k tokens at 1.25x write
price instead of 0.1x read price, roughly $1.64 wasted at Fable pricing. The
output savings from one low-effort mechanical continuation are at most a few
hundred output tokens, a few cents. The flip loses by two orders of magnitude
at this prefix size.

## Workstream A, fix effort routing (active change, first)

1. In `route_effort`, never mutate `output_config.effort` when the request
   carries any `cache_control` breakpoint (tools, system, or message blocks).
   A cached conversation makes the flip guaranteed net-negative. Uncached
   requests keep the existing lowering since there is no cache to bust.
2. Delete the legacy `thinking.budget_tokens` clamp. That mutation is a
   documented messages-tier cache bust on every model that still accepts a
   budget.
3. Count both outcomes in `operational_audit` (effort lowered vs effort pinned
   by cache) so the fix is verified by the same measurement that found the bug.

The verbosity tail injection stays. It sits after the last system
`cache_control` breakpoint and does not bust anything.

Deployment note: the live proxy on port 8787 keeps running the old code until
the owner restarts it. The fix lands in the repo only.

## Workstream B, novelty-routing experiment (offline, zero API cost)

Pre-registered design, target `headroom/evals/novelty_routing_eval.py`.

0. Corpus survey over `~/.claude/projects/*/*.jsonl`. Gate: at least 30
   sessions and 1000 tool outputs, otherwise everything downstream is a
   labeled smoke test.
1. Corpus loader: ordered event stream per session, normalized tool targets,
   explicit handling of subagent sidechains and compaction boundaries.
2. Deterministic labeler. "Needed later" means any of: same target re-fetched
   later, a later `headroom_retrieve` recovers it, or rare-token overlap with
   a later assistant message. Each label records which rule fired.
3. Label audit: hand inspection of 20 positives and 20 negatives, no single
   rule above about 90 percent of positives, per-session positive rate neither
   near 0 nor near 1.
4. Baselines: mask-nothing, mask-by-age (threshold sweep), seeded
   random-at-matched-rate, and current production rules imported from
   `cross_turn_dedup` and `read_lifecycle`.
5. Candidate 1: max cosine similarity of each new tool output against recent
   context embeddings via `relevance/embedding.py`, with an embedding
   non-degeneracy check.
6. Candidate 2: query-aware variant, target built from the last user message
   plus the most recent unresolved error.
7. Comparison at matched masking rates. Metric: needed-but-masked events per
   1000 outputs. Uncertainty: cluster bootstrap over sessions. Decision rule:
   a candidate wins only with at least a 30 percent cut versus the age
   baseline and a CI excluding zero. Two clean losses kill the line.
8. Verdict report. Candidate 3 (trained JEPA predictor loss as novelty score,
   Clin-JEPA EMA target encoder plus Sub-JEPA regularizer recipes from the
   comment-lint corpus) is planned only if candidate 1 or 2 beats the age
   baseline.

## Workstream C, abstract-vs-keep-warm DP arm (log-only)

Add a priced arm to `headroom/cache/anchor_dp.py`: keeping H history tokens
warm costs 0.1H per turn, replacing a span with summary S busts the suffix T
at 1.25(S + T). Abstraction pays only when 0.1H > 1.25(S + T). Emit what the
arm would decide next to what current logic does. No behavior change until the
canary compares them.

## Workstream D, structural state ledger at bust time (log-only)

Deterministic parser over tool_use and tool_result blocks producing a file
state table, command outcomes, unresolved errors, and the verbatim task line.
Fused into the structural-bust path where the suffix rewrite is already paid.
First stage writes the rendered ledger to the audit log per bust event and
injects nothing.

## Workstream E, phase 0 replay corpus

Capture request and response pairs around `operational_audit.py`, feed
`benchmarks/claude_stack_canary.py`. This is the graduation gate for B, C,
and D, and the precondition for wiring `semantic_compression_decision.py`
from admission-only to actual mutation.

## Workstream F, touch registry correction

Switch `/admin/touch` pre-warm from the max_tokens 1 replay to the sanctioned
max_tokens 0 shape, and use the official break-even counts (1 hit for the 5m
tier, 2 for the 1h tier) in the ski-rental math.

## Workstream G, main-thread model cap misclassification (urgent)

The subagent model cap keys on the absence of the `<model_id>[1m]` marker in
the system head (`_system_lacks_1m_marker`, `handlers/anthropic.py`). The
2026-07-19 session investigation confirmed five consecutive main-conversation
requests silently rewritten from claude-fable-5 to claude-sonnet-5 around
13:57 because the marker was absent from a genuine main-thread request. A
model rewrite moves the request to a different cache namespace, so every
misfire also forces a full-prefix cache write on both sides of the flap.
Fix per findings 5 and 6 of docs/session-investigation-2026-07-19.md: classify
sub-agents by positive signal, log every rewrite with the reason, and surface
rewrites in /stats so a misfire is visible the day it happens.

## Workstream H, closed-loop cache accounting (urgent)

The proxy currently optimizes open-loop: the savings ledger books transform
deltas while billed cache writes from unplanned busts never enter any
calculation. Measured 2026-07-19: cache_read pinned at ~23k (tools plus system
tier) while 73k to 103k of message history re-wrote at 1.25x on every request
in the busted stretches, roughly one dollar per request, invisible in /stats.
Three bust sources bypass the DP cost model entirely: output_config.effort
changes (cache key), model cap rewrites (cache namespace), and content churn
inside the warm prefix with protect_recent=0. Build, log-only first:

1. Per request, join predicted cache read against billed usage (already logged
   at handlers/streaming.py:295 and outcome.py:455) into one reconciliation
   record: request id, predicted read, billed read, billed write,
   alive_fraction, first diverged message index, and which transforms ran.
2. Extend the churn observation at handlers/anthropic.py:1491 to report the
   first diverged message index and attribute it: client-originated content
   change versus a headroom transform whose decision shifted between requests.
3. An unplanned-bust counter: billed read materially below predicted for a
   request the DP expected warm. Alert threshold, not just a log line.
4. After one session of data names the bust owners: route every cache-key-
   affecting mutation (effort, model, content) through one priced gate instead
   of three independent code paths, and revisit protect_recent=0, since the
   recent window is exactly where transform decisions still change as content
   ages.

## Workstream I, tokenizer-aware user-message normalization (low priority, log-only first)

Deterministically normalize the newest user message for token efficiency:
whitespace cleanup, dictionary-word spell correction, and a fixed
phrase-to-short-form substitution table. No generative rephrasing, no SLM in
the request path, and no mutation of anything inside backticks, quotes, code
fences, paths, or identifier-shaped tokens (reuse the Kompress must-keep
class). Typos in identifiers are meaning, not noise, is_tru must reach the
model exactly as typed.

The binding constraint the usual framing misses: the rewritten message
becomes history on the next request, and the client re-sends the original
text. The rewrite must therefore be re-applied byte-identically on every
later request, keyed by content hash in a persistent store, or the
normalizer itself becomes a per-session cache buster. Determinism across
proxy restarts and versions is part of the contract.

Expected yield is small (user prose is tens of tokens inside contexts
re-read at 0.1x), so this ships log-only first: record would-be token delta
per message and per session before any mutation is enabled. Build only after
A through H have landed and the reconciliation data is clean.

## Workstream J, bust-time flush and tier-aware reconciliation (active, 2026-07-20)

Two connected defects found while watching live traffic after the workstream
A through H deploy.

J1, flush deferred injection into a forced write. The structural-bust path
(`_structural_bust_requires_fresh_5m`, handlers/anthropic.py) forces a fresh
5m write when the client prefix has diverged past threshold, but the CCR
injection deferral 300 lines later still defers "to preserve cache", holding
back the retrieve tool and system instructions to protect a cache the same
request already decided to discard. Fix: set a bust-forced flag at the force
point, and at both deferral sites flush (inject now) when the flag is set,
since the suffix is re-billed regardless. The flag is set only inside the
bust branch, which already gates on alive_fraction below threshold, so the
flush never fires speculatively. This is the flush the session's rebase work
was meant to deliver on the external-bust path, distinct from the hybrid
economic rebase (which fires on the mode's own deferred queue).

J2, tier-aware reconciliation TTL. cache_reconciliation uses a flat
CACHE_TTL_SECONDS of 300, correct for a forced 5m write but wrong for any
session adaptive_ttl placed on the 1h tier, where a cold read at 400 seconds
is a real bust inside the 1h window yet gets misclassified as scheduled
expiry and hidden from the alarm. Fix: thread the actual per-request TTL
(from `_force_ttl` or the breakpoint ttl, stashed on the prefix tracker like
the churn observation) into `record()`, and classify expiry against the real
tier rather than a flat 300s.

## Later phases, carried from the session plan

- Output regularization beyond the current shaper: per-request-shape output
  budgets, with an output-token-per-request-shape counter in the audit as the
  baseline and rollback signal.
- Plan caching, log-only: record what a template cache would have injected for
  recurring task signatures and diff it against the plans the model actually
  produced.
- Supersession-rule tightening in `read_lifecycle`, the detection half that
  feeds workstream D.
- Wiring semantic compression to actual mutation, blocked on workstream E.

## Execution order

A, G, and H first, they are actively losing money, then F, then B, with C and
D built log-only behind it, E whenever the graduation gate is wanted. Nothing mutates live behavior without the canary
reporting on both token savings and the audit's rewrite and repeat counters.
