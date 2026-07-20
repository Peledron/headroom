# Local change ledger through 2026-07-19

This ledger records work created locally since this clone, including local-only
commits, active worktrees and branches, stashes, and the current uncommitted
work. Vendor updates reachable from a configured remote are not presented as
local inventions, even when they entered this branch through a local merge.

## Repository layout and ownership

The active checkout is on `codex-claude-stack-optimization-20260714` at
`f4e76b61`. No commit or merge was made during the 2026-07-19 implementation
session. The checkout was already dirty when that session started, so the
pre-existing worktree changes are identified separately below.

Other local worktrees and branches found in the shared Git repository:

- `codex-array-tool-output-fix-20260712` at `c533d0d4`, checked out in
  `headroom-codex-array-fix`
- `codex-cache-audit-20260711` at `a617455f`, checked out in
  `vendor-headroom-codex-audit`
- `codex-hybrid-optical-complete-20260713` at `1ede2d3f`, checked out in
  `headroom-hybrid-optical-audit-20260713`
- `headroom-hybrid-complete-public` at `c48d21d8`, checked out under `/tmp`
- `local-cache-bust-fixes` at `1ede2d3f`
- local `main` at `a617455f`

Two stashes remain:

- `stash@{0}`: `codex-pre-upstream-main-20260714`
- `stash@{1}`: local `.serena/project.yml` before the hybrid fast-forward merge

## Local-only commit history

### Cache-bust and prefix-cost work, 2026-07-12

Commits `0b36c1d6` through `63088269` introduced the local cache-bust fork.
The main techniques were:

- Freeze cached message prefixes so byte changes do not turn cheap cache reads
  into full writes.
- Use message-level net-cost and break-even gates before mutating warm content.
- Detect subagent-shaped traffic and select shorter cache TTLs for short-lived
  sessions.
- Estimate compression payoff with an exponentially weighted compression ratio
  and confidence bounds instead of assuming a fixed gain.
- Catch oversized compression-accounting values instead of allowing an
  `OverflowError` to break the request path.
- Keep test doubles aligned with the prefix tracker's telemetry contract.

The reason for this group was economic correctness. A token reduction can cost
more money when it invalidates an Anthropic prefix that would otherwise be read
at the cache discount.

### Codex array tool-output fix, 2026-07-13

Commits `c533d0d4` and `d5be594c` taught the OpenAI/Codex path to compress array
shapes used by tool outputs. The technique was shape-aware routing rather than
treating every tool result as plain text. This fixed a gap where large arrays
passed through because the compressor only recognized scalar text forms.

### Hybrid and optical work, 2026-07-13 to 2026-07-14

Commits `ed6239a2`, `a0e46f69`, and `1ede2d3f` added hybrid context handling,
optical text compression controls, corrected Codex context limits, and a
full-stack optimization pass. The techniques were:

- Split the request into a stable cached prefix and a compressible live tail.
- Preserve model-specific context limits when making pressure decisions.
- Use deterministic text transforms and canary comparisons to measure quality,
  latency, token savings, and cache behavior together.
- Add tool budgets and stack-level canaries so a lower token count is not
  accepted when it causes extra tool calls or task failure.

The reason was to combine cache stability with useful context reduction instead
of choosing a blanket cache mode or blanket token mode.

### Claude cache and observation-masking stack, 2026-07-16 to 2026-07-18

Commits `926b32ef` through `f4e76b61` form the current local feature series.
They include:

- Native hybrid-mode cache policies instead of scattered environment-only
  branches.
- Dynamic-programming anchor placement, structural-churn measurement, empirical
  cache-survival estimates, and expected-read forecasts.
- Observation masking for old tool results using recoverable content markers.
- Separate handling for block-list tool results and selected large tool input
  fields.
- Transcript-derived masking thresholds and explicit logs when the gain gate
  declines a candidate.
- Accounting that attributes masked tokens on the actual forwarded path.
- Request-path discovery on every relevant branch, with referenced tools kept
  resident.
- An opt-in rollback for tool input masking after a marker-mimicry incident.
- Rebase-time history sweeps, per-block store-failure isolation, beta guards,
  missing-tool stubs, and model-emitted recovery-marker guards.
- Breaker tests that intentionally attack prefix replay, marker recovery,
  reference tracking, structural churn, and cache-cost assumptions.

The main design pattern is reversible indirection. Large historical content is
replaced only when the original can be retrieved by a stable marker. Breaker
tests then target the cases where that indirection could become ambiguous or
unavailable.

### Where dynamic programming is deployed

The deployed dynamic programming is the cache-anchor planner in
`headroom/cache/anchor_dp.py`. It is not the larger touch, compress, clear Markov
decision process proposed in the investigation note. That general MDP remains a
research plan.

The deployed planner works as follows:

1. `PrefixCacheTracker.observe_client_churn()` compares the current client
   history with the prior history and stores the depth where their common prefix
   stopped. The tracker retains a bounded ring of churn-depth samples.
2. Late in the Anthropic request path, the cache-control placement code checks
   whether hybrid mode has DP anchors enabled. `HybridModeConfig.dp_anchors`
   defaults to true and `HR_DP_ANCHORS=0` disables it.
3. The planner runs only once a conversation has at least 80 messages and only
   when Anthropic's maximum of four cache-control breakpoints leaves spare
   capacity after system, tools, and existing message markers are counted.
4. `optimal_anchor_depths()` builds candidate message depths on a 64-message
   grid. It excludes the first 32 messages and the newest 16-message live tail.
   Quantization keeps an anchor at the same byte position while a conversation
   grows between grid boundaries.
5. The dynamic-programming table minimizes expected rewrite distance over the
   observed churn depths for the available number of anchors. In effect, it
   chooses segment boundaries so a history rewrite loses the smallest expected
   cached suffix instead of always using one fixed midpoint.
6. The Anthropic handler attaches 1 hour ephemeral cache markers at the selected
   message depths using copy-on-write updates. It never edits the tracker's
   stored message objects in place.

When there are too few churn samples, the planner uses a deterministic
quantized fallback depth. This gives stable placement before the session has
enough evidence for an empirical optimum.

## Worktree state present before the 2026-07-19 implementation

The checkout already contained edits to `.github/copilot-instructions.md`, the
Anthropic and streaming handlers, and the proxy server. It also contained the
untracked investigation note, SSE marker guard, touch registry, and their
tests. Those changes added:

- A streaming SSE guard that holds only in-flight Anthropic `tool_use` blocks
  long enough to repair recovery-marker mimicry.
- A wire-level premium-model cap based on prompt-text subagent detection.
- A process-memory registry and loopback-only `/admin/touch` route for replaying
  a cached prefix with `max_tokens=1`.
- Tests for the SSE guard, model cap, and touch registry.

The 2026-07-19 session preserved this work and changed the unsafe parts rather
than discarding it.

## Changes made in the 2026-07-19 implementation session

### Native subagent model selection

`headroom wrap claude` now sets Claude Code's own
`CLAUDE_CODE_SUBAGENT_MODEL`, defaulting to `claude-sonnet-5`. An explicit user
value wins, and `HR_SUBAGENT_MODEL_CAP=0` disables the default. The installed
Claude Code binary was checked for this exact environment variable before the
change was made.

The proxy's prompt-text model rewrite is now disabled by default and requires
`HR_SUBAGENT_MODEL_CAP_WIRE_FALLBACK=1`. This removes the observed failure mode
where a 122-message main thread lost its rendered `[1m]` marker and was silently
downgraded. Prompt text is kept only as an opt-in compatibility fallback.

"Opt-in" applies only to that unsafe proxy fallback. The preferred behavior is
automatic when Claude is launched through `headroom wrap claude`: Claude Code
itself receives `CLAUDE_CODE_SUBAGENT_MODEL=claude-sonnet-5` and selects Sonnet
when it creates a subagent. No proxy inference or model rewrite is involved.
Operators who cannot launch through the wrapper can explicitly enable the old
wire fallback, accepting that it identifies subagents from rendered system
text. Direct `claude` launches that do not set `CLAUDE_CODE_SUBAGENT_MODEL` and
do not enable the fallback receive no Headroom subagent cap.

### Pre-send structural-bust handling

The Anthropic handler now treats a warm prefix with less than 50 percent of its
messages still alive as an unavoidable fresh write. It forces the cheaper 5
minute cache tier before forwarding, instead of discovering after the response
that an accidental 1 hour write premium was paid. The threshold and behavior
can be adjusted with `HEADROOM_STRUCTURAL_BUST_ALIVE_THRESHOLD` and
`HEADROOM_STRUCTURAL_BUST_TTL_5M`.

### Model-substitution and tool-call visibility

The new in-process operational audit counts every proxy model rewrite by source,
target, and reason. It also hashes tool name plus canonical input to report
repeated tool calls without storing tool payloads. `/stats` now exposes these
counters under `operational_audit`.

This is an audit, not an automatic deletion feature. A repeated read can be
intentional after state changed, so the proxy reports it rather than suppressing
it.

### Adaptive cache touching

The touch registry now deep-copies stored wire requests, detects their 5 minute
or 1 hour TTL, and uses a ski-rental break-even threshold. The decision compares
one discounted cache read with the avoided cache-write multiplier. A normal
`/admin/touch` call replays only entries near expiry. A forced administrative
call still replays immediately. `/stats` exposes entry age, TTL, and break-even
age.

### Semantic-compression admission contract

A new decision module encodes the investigation's gain formula and hard safety
rules. It rejects `thinking`, `tool_use`, structured payloads, non-informational
tool results, and content inside a still-live cached prefix. An allowlisted
informational tool result is admitted only when immediate and expected
downstream token savings exceed compute cost, latency cost, and cache-risk cost.

The module does not yet mutate content. This is intentional. Shipping an
extractive compressor without a replay quality corpus would violate the stated
requirement that task success be measured alongside token savings.

Here, "content mutation" means changing bytes in the API request's conversation
content before it is sent to Anthropic. Examples include replacing a long
`tool_result` with selected sentences, deleting low-salience log lines, or
substituting a summary and retrieval marker for the original text. It does not
mean changing source files in the repository.

Changing request content has two independent risks. First, the model may need a
fact or exact string that the compressor removed. Second, changing any byte
inside an otherwise live Anthropic cache prefix invalidates the downstream
prefix match and can turn a discounted cache read into a full cache write. The
new decision module only calculates whether a candidate is safe and has positive
estimated economic gain. It returns an allow or deny decision and a reason. It
does not currently rewrite the request body. Observation masking remains the
deployed mutation mechanism because it stores the original content and provides
a retrieval path.

### Existing cross-turn deduplication

The repository already contains cross-turn deduplication in
`headroom/transforms/cross_turn_dedup.py` and ContentRouter wiring, plus a
Responses-path integration. It is not a universal Anthropic tool-result dedup
pass. Anthropic's cached-prefix rules and the requirement to preserve structured
or later-reparsed tool output make automatic blanket dedup unsafe. Observation
masking is the current recoverable mechanism for eligible historical results.

### Canary and bypass workflow

`benchmarks/claude_stack_canary.py` already provides direct Anthropic and
Headroom arms by switching `ANTHROPIC_BASE_URL`. No new bypass endpoint was
needed. Direct traffic uses `https://api.anthropic.com`, while Headroom traffic
uses the selected loopback proxy.

One isolated canary was run through a temporary stateless proxy on port 8791.
The active user proxy on port 8787 was not restarted or changed. The request
used Sonnet, returned `OK`, and `/stats` reported:

- 1 request
- 7,536 input tokens and 4 output tokens
- 0 model substitutions
- `claude-sonnet-5` as the dispatched model

Claude Code's built-in system context made even this one-word canary expensive,
so the direct no-Headroom arm was not repeated. The reported Claude-side cost
for the single request was $0.217698.

## Verification status

Passing local checks:

- Python compilation for all changed implementation and test files
- Ruff on all files touched by the 2026-07-19 implementation
- Operational-audit unit tests
- Structural-churn and fresh-write decision tests
- Touch-registry unit tests except the application-lifespan integration case
- Semantic-compression decision tests
- Claude wrapper model-selection helper tests
- Explicit model-cap decision tests proving a Fable main model is unchanged,
  while the compatibility fallback rewrites only an identified Fable subagent
  when the fallback is enabled
- Seven existing cross-turn dedup tests before tokenizer initialization
- Isolated live proxy health, Claude request, and `/stats` inspection

Known verification limits:

- The existing FastAPI integration helper hangs during application lifespan
  startup for the model-cap and `/admin/touch` route tests. The pure logic tests
  pass, and the separate proxy process starts and serves requests correctly.
- The remaining cross-turn dedup tests require downloading the `o200k_base`
  tiktoken data. The sandboxed attempt failed DNS resolution, and the approved
  retry stalled. No product assertion failed before that dependency boundary.
- No second live Claude request was run because the first minimal request showed
  that another arm would consume thousands of input tokens.

## Fable single-turn internal-iteration incident, 2026-07-19

Claude transcript `8d7e8a7a-68d8-4259-8762-c19dada111a4.jsonl` records one
outer Fable request that expanded into seven Anthropic internal inference
iterations. The provider usage totals were:

- 1,408,085 uncached input tokens
- 290,560 cache-read input tokens
- 276,687 cache-write input tokens
- 3,634 output tokens

The corresponding Headroom request was `hr_1784465518_000131`. It entered the
proxy with 207,225 tokens and changed the cached prefix, then wrote 276,687
tokens. Headroom had injected Anthropic Tool Search and deferred eight tool
definitions. The turn loaded `objective-analyst`, `claude-api`, and
`engineering-baseline`, then issued six searches for an unavailable
`WebSearch` tool. Each failed search caused another provider-side inference
iteration over roughly 268,000 to 298,000 input tokens. Anthropic only exposed
the aggregate internal-iteration accounting in the final usage event, so a
response-side rate detector could report the incident but could not stop its
first iteration fan-out.

Safeguards added after the incident:

- Proxy-owned Tool Search is never injected into Claude Code traffic. The
  injection remains available to direct non-Claude clients when explicitly
  enabled.
- At 100,000 input tokens or more, Claude Code requests that already contain
  both deferred tools and a Tool Search definition are rewritten before the
  upstream call. Deferred definitions are materialized and the search tool is
  removed. `HEADROOM_LARGE_TOOL_SEARCH_GUARD_TOKENS` changes the threshold.
- `WebSearch` is included in the resident-tool allowlist so it is not deferred
  when Tool Search is used by an eligible client.
- Anthropic `usage.iterations` is parsed and recorded. Multiple internal
  iterations emit `ANTHROPIC-ITERATION-FANOUT`, with iteration and input-token
  totals exposed through the operational audit.
- A project `PreToolUse` hook denies `Skill(claude-api)` on the Claude main
  thread. It permits the skill only when Claude identifies a subagent through
  a subagent transcript path or an `agent_id`. A new Claude session is needed
  to load the project hook.

The 100,000-token protection is deliberately scoped to Tool Search fan-out.
A blanket 100,000-token context ban would reject ordinary long-context Claude
sessions. A true user-approval gate would require client cooperation: Headroom
could return HTTP 428 with a short-lived approval identifier, but Claude Code
or its wrapper would have to display the approval prompt and retry once with a
matching token. A permanent bypass header is not equivalent to per-request
approval. Cache-write and internal-iteration totals arrive too late to stop the
first bad request, though they can quarantine later requests.

Focused unit tests, Ruff, and compilation passed before activation. The checkout
was then installed as `vendor-headroom[all]` into the `uv` tool environment.
Before the live restart, the checkout started successfully on isolated port
8791 and returned a ready health response. The isolated process was stopped,
then `headroom-init-user.service` was restarted once. Its dependency preflight
passed, port 8787 returned healthy on PID 444270, existing Codex WebSocket
sessions reconnected, and `/stats` exposed the new Anthropic fan-out counters.
No paid Claude canary was used for activation.

### Tool Search history-compatibility correction

The first activation exposed an existing-session compatibility failure. Claude
history still contained a `server_tool_use` reference to
`tool_search_tool_regex`, but the new Claude Code injection gate stopped adding
that server-tool definition. Anthropic rejected the continuation with HTTP 400
before inference because the referenced tool was absent.

The correction distinguishes tool availability from tool deferral. For Claude
Code histories that reference Tool Search, Headroom restores the exact
`tool_search_tool_regex_20251119` definition and materializes every normal tool.
The large-context guard also retains the search definition while removing all
`defer_loading` flags. This preserves history validity without recreating the
unavailable deferred-tool loop.

The generic historical-reference detector now covers both `tool_use` and
`server_tool_use`. Ordinary missing custom tools continue to receive non-callable
historical stubs. Headroom's `headroom_retrieve` and memory tools already use
session-sticky replay. Native server tools can have the same validation class if
a client removes their definitions mid-session, but Headroom does not remove
those definitions and cannot reconstruct an arbitrary server-tool version from
the historical name alone.

Focused Tool Search and breaker tests, Ruff, the installed-runtime assertion,
and `git diff --check` passed. The corrected checkout was installed and
`headroom-init-user.service` was restarted once. A subsequent explicit health
query was refused by the execution approval reviewer because its usage allowance
was exhausted. No bypass was attempted. The continuing Codex connection through
Headroom confirmed that the proxy path recovered after the restart.

## Current uncommitted inventory

Tracked modifications:

- `.github/copilot-instructions.md`
- `headroom/cli/wrap.py`
- `headroom/proxy/handlers/anthropic.py`
- `headroom/proxy/handlers/streaming.py`
- `headroom/proxy/helpers.py`
- `headroom/proxy/server.py`
- `tests/test_cli/test_wrap_helpers.py`
- `tests/test_structural_churn.py`

Untracked files:

- `docs/session-investigation-2026-07-19.md`
- `docs/local-change-ledger-2026-07-19.md`
- `headroom/proxy/operational_audit.py`
- `headroom/proxy/semantic_compression_decision.py`
- `headroom/proxy/sse_marker_guard.py`
- `headroom/proxy/touch_registry.py`
- `tests/test_operational_audit.py`
- `tests/test_anthropic_iteration_fanout.py`
- `tests/test_anthropic_tool_search_client_gate.py`
- `tests/test_claude_skill_main_thread_gate.py`
- `tests/test_semantic_compression_decision.py`
- `tests/test_sse_marker_guard.py`
- `tests/test_subagent_model_cap.py`
- `tests/test_subagent_model_cap_decision.py`
- `tests/test_touch_registry.py`

This inventory should be refreshed immediately before any commit because it is
the boundary between pre-existing user work and the final reviewed patch.
