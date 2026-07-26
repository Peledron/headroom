# Research intake, 2026-07-26

Two research summaries arrived from an outside conversation. This document
examines every load-bearing claim in them against what this repo already
measures and already ships, then states what is worth building.

The short version: most of it is already here, one central number is wrong in a
way that matters, one item is genuinely new and worth doing, and two items
cannot be built at this layer at all.

## 1. The 12.5 inequality is a misread of our own price ratio

The summary states the constraint as:

    read cost = 0.1 x H
    rewrite cost = 1.25 x (S + T)
    compaction pays off when 0.1 H > 1.25 (S + T), so H > 12.5 (S + T)

This is not the break-even condition. It is a price ratio wearing a break-even
costume. `docs/cache-economics.md:17` says "a cache write costs 12.5 to 20 times
what the same token costs to read", which is just 1.25/0.1 and 2.0/0.1. That
ratio has been restated as a condition on history length, which it never was.

The real condition is already recorded at `docs/handover-2026-07-21.md:131` and
implemented in `headroom/transforms/compression_policy.py:162`:

    gain = dT (w + r (R - 1)) - P_alive (w - r) (S + dT)

With `P_alive = 1`, `w = 1.25`, `r = 0.1`, gain > 0 reduces to:

    R x dT > 11.5 x S

Three differences, each of which changes decisions:

1. **The horizon `R` is a multiplier, and the summary has no horizon at all.**
   A rewrite is paid once and repaid on every later read. Dropping `R` prices a
   20 turn session as if it were a 1 turn session, which refuses mutations that
   pay for themselves several times over.
2. **The numerator is `dT`, what you delete, not `H`, how long the history is.**
   Saving is proportional to removed tokens. A 200k history where you can only
   compress 3k is a bad trade no matter how large `H` is.
3. **The coefficient is 11.5, not 12.5.** Tokens already in the warm prefix cost
   `w - r` to move, not `w`. Keeping them costs reads, not a fresh write. The
   1 token difference is 8 percent of the threshold.

**Action:** none in code, the correct form already ships. This section exists so
the 12.5 figure does not get re-derived from the price ratio a third time.

## 2. Status audit of existing plans

Verified against the tree, not against memory.

| Doc | Item | Status |
|---|---|---|
| `optimization-plan-2026-07-19.md` | A, effort routing priced against bust cost | **Shipped** (`headroom/proxy/effort_pricing.py`, sticky per lineage) |
| `optimization-plan-2026-07-19.md` | D, structural state ledger at bust time | **Shipped** (`headroom/proxy/structural_ledger.py`) |
| `optimization-plan-2026-07-19.md` | E, offline replay corpus | **Shipped**, it is what the prefix-waste measurements ran on |
| `optimization-plan-2026-07-19.md` | H, closed-loop cache accounting | **Shipped** (`headroom/proxy/cache_reconciliation.py`) |
| `optimization-plan-2026-07-19.md` | I, tokenizer-aware normalization | **Open, log-only by design.** See section 3A |
| `diff-only-rereads-design.md` | header says "not yet implemented" | **Stale.** Shipped at `headroom/transforms/read_lifecycle.py:556` with tests |
| `prefix-waste-2026-07-25.md` | tools array as top divergence source | **Partly shipped.** Canonical serialization landed, tail relocation did not |

Doc headers updated in this pass: `diff-only-rereads-design.md`.

## 3. Claim by claim

### A. BPE glue-word stripping is net-negative

**Verdict: right conclusion, wrong primary reason, already our policy.**

The claim is that deleting words like "a" and "the" fragments BPE merges and can
raise the token count. Direction is right, magnitude is oversold: deleting text
almost always still reduces token count, just sublinearly. Selling this on
tokenizer economics invites someone to measure the token delta, find it
negative, and ship it.

The reason that actually holds is cache, not tokenization. Editing prose that
has already been sent changes prefix bytes and busts the cache. A 3 percent
token saving on a message that is already warm costs `(w - r)` on everything
after it. That is the argument, and it does not depend on any tokenizer detail.

`optimization-plan-2026-07-19.md` Workstream I already scopes this as log-only
for exactly this reason. No change needed. Do not promote it out of log-only.

### B. Meta-tools and lazy schema loading

**Verdict: cannot be built at this layer for Claude Code. Partly shipped for
what can.**

The proxy cannot remove tools from the array. Claude Code decides its own tool
set, and a `tool_use` block naming a tool the proxy stripped is an error, not a
saving. Meta-tool indirection is a *client* pattern, not a proxy pattern.

What the proxy can do, and `prefix-waste-2026-07-25.md` already proposes, is
relocate late-arriving schemas so they append at the tail instead of splicing
into a warm prefix. That is the version of this idea that survives contact with
the constraint. `headroom/proxy/tool_schema_compaction.py` and the server tool
search deferral path already cover part of it. The tail relocation itself is
still open and is the single highest-value unshipped item on the list, because
the tools array was measured at 14 percent of prefix waste.

The cited "tool selection accuracy falls from 74 percent to 49 percent" figure
has no source attached. Do not put it in a design doc until it does.

The consumer-side version of this claim is real and is not a headroom change: my
own MCP server set is large, and trimming it is a config edit, not code.

### C. Rejection of file-based memory

**Verdict: agreed, and already the design.**

`self-optimizing-pruning-design-2026-07-21.md` already puts the world model in
proxy-managed ephemeral state, and `structural_ledger.py` is the deterministic
parse of tool blocks rather than a generative summary. Nothing to change.

The critique overreaches in one place: it treats CLAUDE.md-style memory as a
mistake to be corrected, but that file is client-side and outside this proxy's
control. It is not competing with the ledger, it is a different mechanism at a
different layer.

### D. Output regularization

**Verdict: the one genuinely new item, and worth building. Not in any doc.**

Output tokens cost about 5 times input at Anthropic list price. Every mechanism
in this repo works on the input side. That is the gap.

A suffix-injected brevity directive is cheap in exactly the way that matters: it
lands at the tail, so it does not touch the warm prefix, and the shared formula
prices it at `S = 0` for the injection itself.

Two risks that have to be measured before it ships, not after:

1. **Terser narration can mean more turns.** If suppressing explanation makes
   the model act without stating a plan, and the plan was wrong, the correction
   costs a full turn. One turn at 150k prefix is 15k read tokens, which buys a
   great deal of narration. The metric is cost per completed task, not output
   tokens per turn.
2. **It collides with the operator's own instructions.** My CLAUDE.md asks for
   specific prose behavior. A proxy-injected directive that contradicts it
   produces worse answers and is hard to attribute.

**Build order:** log-only first. Record output tokens per request, bucketed by
whether the turn was a tool continuation. That number does not exist yet, and
without it the 5x premium is an argument about list prices rather than about
this workload. Only then consider injection, gated and off by default, with the
same visible-notice discipline the model router got.

**Status, 2026-07-26: the log-only half is built.**
`headroom/proxy/output_accounting.py` holds the ledger and the classifier, and
both response paths in `handle_anthropic_messages` call
`_observe_output_tokens`. Each response emits an `OUTPUT_DIAG` line and the
distribution is served at `/stats` under `output`. It changes no request and no
response, and it holds no prompt text. Injection stays unbuilt until the live
split says which shape is expensive.

### E. Phase 0 replay corpus and audit counters

**Verdict: already done, listed as if pending.**

The replay corpus exists and the prefix-waste and read:write measurements ran on
it. Audit counters for prefix busts ship as `PREFIX_DIVERGE`, `LINEAGE_REJECT`,
`STRUCTURAL-CHURN`, and `CACHE-MISS-ATTRIBUTION`.

The one counter the summary asks for that genuinely did not exist is output
tokens per request. See 3D.

## 4. What to build, in order

1. ~~**Output-token audit counter, log-only.**~~ Section 3D. **Shipped
   2026-07-26.** Needs live traffic now, not code.
2. **Tail relocation of late-arriving tool schemas.** Section 3B. Already
   designed in `prefix-waste-2026-07-25.md`, measured at 14 percent of prefix
   waste, not yet built. This is now the top unshipped item.
3. **Answer the difficulty-gate question from the existing transcripts.**
   Superseded plan: shadow logging was going to write live question text to
   disk to build a population. That corpus already exists. Roughly 2 GB of
   Claude Code and Codex transcripts are on disk, so replaying
   `estimate_difficulty` over them answers "how many turns a week would route
   cheap" today instead of after a week of collection, and writes nothing new.
4. **Read the mask-gate retrieval rate.** The counter shipped 2026-07-25. It
   needs live traffic, not code. Served at `/stats` under
   `compression.mask_gates`.

## 5. What not to build

- **Lexical pruning of prose.** Section 3A. Stays log-only.
- **Proxy-side tool removal or meta-tool rewriting.** Section 3B. Breaks
  `tool_use` resolution.
- **Generative state summarization.** Already rejected in the pruning design,
  and the marker-mimicry incident is the standing reason model-authored text
  does not get to define proxy state.

## 6. Open measurements

None of these have numbers yet. Each blocks a decision above.

| Measurement | Blocks |
|---|---|
| Output tokens per request, split by tool continuation | Output regularization |
| Retrieval rate, episode gate vs age gate | Whether the episode boundary is too eager |
| Turns per week admitted by the difficulty gate | Whether cheap-model routing is worth an A/B |
| Prefix waste attributable to tool schemas after canonicalization | Whether tail relocation is still worth 14 percent |
