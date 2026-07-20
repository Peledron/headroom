# Session investigation — 2026-07-19

Findings from a live investigation into cache-cost behavior, run against the
running proxy (`http://127.0.0.1:8787`), `savings_events.jsonl`, and
`headroom/proxy/logs/proxy.log`. Carries forward one open item from
2026-07-18. Written to survive a `/clear` — pick up from "Open work" below.

## Confirmed findings (verified against live data, not inferred)

1. **Stale-figure bug in this transcript, not in headroom.** Early in this
   session a "$4.37 saved, 35%" figure was quoted as current when it was
   actually a prior day's lifetime/history figure. Live `/stats` for the
   actual current session showed 11.7% savings on ~$0.74 baseline. **Lesson:
   always re-pull `/stats` fresh, never trust a number carried in
   conversation context across a gap.**

2. **`prefix_frozen` is a deliberate mode trade-off, not a limitation.**
   `headroom/proxy/cost.py:~628`. In default `PROXY_MODE_CACHE`, headroom
   declines to compress a message whenever doing so would alter bytes inside
   Anthropic's already-cached prefix, because that forces a cold cache-write
   rewrite (~20x the cache-read rate) for everything downstream. Headroom
   self-reports the alternative: `HEADROOM_MODE=token` compresses frozen
   messages too, trading the cache-read discount for ~25–35% more effective
   session length before hitting context limits. Not yet toggled — flagged as
   a lever, not applied.

3. **Two confirmed real cache busts, 2026-07-18 20:10–20:11 UTC+2, unfixed.**
   `savings_events.jsonl`: `reason=prefix_change`, `ttl_exceeded=False`,
   preceded by `STRUCTURAL-CHURN` events with `alive_fraction` collapsing to
   0.00–0.33 (i.e. under a third of messages still matched the server's
   cached prefix). Root cause: a compaction-shaped structural change to
   message history broke Anthropic's byte-identical prefix match, forcing a
   full cold rewrite (98,011 tokens at full cache-write price).
   **Proposed fix (not yet implemented, verified absent from `cost.py` as of
   this session):** a pre-send check in the compression pipeline that detects
   `alive_fraction` collapse *before* forwarding, and short-circuits to an
   intentional fresh 1h-TTL cache write (cheaper, planned) instead of letting
   Anthropic reject an accidental prefix mismatch mid-flight.

4. **Touch-registry mechanism (`headroom/proxy/touch_registry.py`) verified
   working end-to-end against live Anthropic servers.** `/admin/touch`
   replay on a real session returned `cache_read_input_tokens=49588,
   cache_creation_input_tokens=0` — confirmed full-prefix cache hit at 0.1x
   rate. This mechanism prevents **TTL-expiry** busts (idle gap exceeds the
   cache window) — it is not the same failure mode as #3 above
   (`prefix_change`/structural churn), and does not run continuously; it's
   only triggered by `cache-touch-listener.py` on desktop lock/sleep/resume
   D-Bus signals. It would not have prevented #3, and did not need to for the
   161k-token growth investigated in this session (see finding #7 — that
   growth was not a bust at all).

5. **Subagent model cap exists and is firing continuously — confirmed 58
   times in today's session alone.** `headroom/proxy/handlers/anthropic.py`,
   env var `HR_SUBAGENT_MODEL_CAP` (default `claude-sonnet-5`). When a
   request's rendered system prompt is classified as a subagent call (see
   #6), headroom silently rewrites the model from whatever was
   requested (Fable 5 / Opus) down to the cap. Confirmed via log line
   `SUBAGENT_MODEL_CAP: hr_... rewriting subagent model claude-fable-5 ->
   claude-sonnet-5`, 58 occurrences between session start and 13:5x today.
   **This is why "5-hour usage wasn't climbing as fast as expected" —
   substantial fractions of session traffic were actually billed at Sonnet
   rates, not Fable, invisibly.**

6. **Subagent-vs-main-session detection is a text heuristic, not a real
   identity signal — confirmed flimsy by the code's own history.**
   Two functions: `_system_looks_subagent` (token-bounded substring match on
   model ID, no `[1m]` suffix) and `_system_lacks_1m_marker` (stricter
   variant used for TTL/freeze decisions). Code comments confirm a prior bug
   already found and fixed here: "a breaker found the bare-substring form let
   such sub-agents escape the freeze and bust." The authors' own risk
   assessment: a false positive here "only mis-prices a cache write, never
   changes a response" — accepted as bounded-risk by design, not robust.
   **User has asked this be hardened — see Open work.**

7. **The 49,281→208,960 "before" token jump (2026-07-19 11:34:44) was NOT a
   cache bust.** Live counter check: `headroom_cache_bust_total{provider=
   anthropic}` unaffected at time of event; the structural-churn log line for
   this exact request read `alive_fraction=0.98` with bust penalty
   explicitly **scaled down, not applied**. The jump was ordinary content
   growth: a `Skill(claude-api)` invocation loaded a large multi-thousand-
   line reference document into the conversation for the first time. Traced
   via full log for request `hr_1784460853_000042`:
   - `SUBAGENT_MODEL_CAP`: rewrote claude-fable-5 → claude-sonnet-5 (this
     specific call was already correctly billed at Sonnet, per #5/#6)
   - `STRUCTURAL-CHURN`: minor, scaled down
   - `MASKING_GATE: declined (gain=-46...)` — compression evaluated and
     explicitly declined because compressing this content would have net
     *added* cost (dense reference markdown has low redundancy to exploit)
   - `CCR: skipping request-side compression` — confirms the decline was
     acted on, not a missed evaluation
   - Result: content passed through uncompressed by design, cached normally
     going forward (`adaptive_ttl_5m` in stage timings), no rewrite penalty

## Corrections made mid-session (record honestly — don't re-litigate)

- Initially conflated the TTL-lapse mechanism (#4/#3's *cause*) with the
  161k-token growth event (#7), presenting them as the same event. They are
  unrelated: #7 was ordinary growth, correctly left uncompressed; the TTL/
  structural-churn bust pattern happened only on 2026-07-18, not at 11:34
  today.
- Initially claimed thinking-block "irreducibility" as settled fact before
  testing; corrected to: reads are not uniformly irreducible, but thinking-
  block replay specifically **is** hard-blocked by an API-level signature
  check (not a policy choice) — editing/summarizing a thinking block before
  replay breaks the signature and 400s on the same model. The only sanctioned
  pruning mechanism is `clear_thinking_20251015` (context editing, beta
  `context-management-2025-06-27`) — all-or-nothing per block, not
  summarization, and itself forces a fresh cache-write for what follows.

1. **[URGENT — actively firing wrong on THIS session's main thread, confirmed
   live] Harden subagent detection (#6).** User request, explicit: "such a
   skill must only be fireable by a sonnet sub-agent" and "that main session
   detection is very flimsy." **This is no longer theoretical — confirmed
   live in this session at 13:57–14:02 today**: five consecutive requests on
   the actual 122-message main conversation thread
   (`hr_1784462261_000115` through `hr_1784462525_000119`, `msgs=122`,
   `tok_before=236496`) were logged `SUBAGENT_MODEL_CAP: rewriting subagent
   model claude-fable-5 -> claude-sonnet-5`, despite `~/.claude/settings.json`
   correctly set to `fable[1m]` throughout. This is the primary session, not
   a subagent dispatch — the heuristic is misclassifying the main thread
   itself, not just an edge case. **Also confirmed compounding**: the same
   requests carry `output_shaper:effort:high->low` in `mutation_reasons` —
   effort is being silently forced down alongside the model, doubling the
   quality impact of the misdetection.
   Concrete next step: read `_system_looks_subagent` /
   `_system_lacks_1m_marker` in full (`headroom/proxy/handlers/anthropic.py`,
   ~line 183–235) and design a more robust signal than a text-pattern match
   on the rendered system prompt. Consider: an explicit header/param the
   harness sets when dispatching a subagent (rather than inferring from
   prompt text), since the current approach is inherently fragile against
   anything that alters how the system prompt renders (quoted `[1m]` in doc
   text, a 429 retry that drops the marker, a long-running session's system
   prompt growing/shifting in a way that trips the pattern, etc). **First
   diagnostic step before redesigning**: reproduce why *this specific*
   session's `[1m]` marker stopped matching — check whether something
   earlier in this session (skill injection, compaction, a mid-session
   system message) altered the rendered system-prompt text in a way that
   broke the token-boundary match `_system_lacks_1m_marker` relies on. That's
   probably a faster root-cause path than the full redesign.

2. **[not started] Implement the #3 pre-send bust-detection fix.** Confirmed
   still absent from `cost.py` as of this session. Add a pre-send check using
   the already-logged `alive_fraction` structural-churn signal to
   short-circuit to an intentional fresh TTL write before Anthropic rejects
   the mismatched prefix, rather than after.

3. **[not started, lower priority] `/stats` transparency for model
   substitution.** The subagent cap has been silently rewriting ~significant
   fractions of this session's requests with zero visibility in `/stats` or
   `/context` (which reads configured default, not wire-level dispatched
   model). Add a substitution-count line item.

4. **[idea, not scoped] Semantic/attention-based compression strategy.**
   User asked whether an attention-mechanism approach (keep only
   semantically load-bearing tokens, strip filler) could work for content
   like the declined `claude-api` skill dump. Answer given: real technique
   (LLMLingua-style extractive compression), but the specific case that
   prompted the question (#7) is a poor fit — dense, already-tight reference
   markdown has little redundancy to exploit, which is exactly why
   `MASKING_GATE` declined it (gain=-46...). Where this would help is verbose
   content (tool output, prose), not information-dense docs. Would need to be
   a new, distinct compression strategy gated by the same gain-check logic
   already in the pipeline, evaluated per-content-type rather than applied
   uniformly. Not scoped into an implementation plan yet.

## Semantic compression — design (planned, not implemented)

Motivating case: `MASKING_GATE` correctly declined to compress the
`claude-api` skill dump (finding #7) because dense reference markdown has
low redundancy for byte-level methods. An attention/extractive approach
targets the opposite content profile — verbose tool output, prose, repeated
log lines — where meaning survives dropping filler tokens even though bytes
don't repeat.

**Hard constraints (non-negotiable, confirmed this session):**

- Never touch `thinking` blocks. This is an API-level signature check, not a
  policy choice — editing breaks replay and 400s on the same model. No
  compression strategy can apply here; only `clear_thinking_20251015`
  (all-or-nothing deletion via context editing) is available at all.
- Never touch `tool_use` input blocks. These are structured args the agent
  or a downstream parser may re-read verbatim; summarizing breaks
  executability, not just fidelity.
- Only apply to content about to be freshly written to cache anyway (the
  uncached tail, or content behind a deliberate fresh-write boundary).
  Compressing bytes inside an already-cached, still-valid prefix trades a
  small savings for a full cache-bust rewrite downstream — the same failure
  mode diagnosed in finding #3. The gain check must account for this, not
  just raw byte reduction.

**Where it plugs into the pipeline:** as a second-stage check after
`MASKING_GATE`'s byte-level gain estimate returns negative/low, not as a
replacement for it. Byte-level dedup stays the first, cheaper pass.

**Gain formula (extends the existing `MASKING_GATE` estimate):**

```
gain = (tokens_saved_this_request × price_factor)
     + (expected_downstream_cache_read_savings, if compression doesn't
        break the alive prefix match)
     - (compressor_overhead: compute cost + added latency cost)
     - (cache_invalidation_risk_cost, if content sits in a region that
        might still be read from cache by a near-future request)
apply only if gain > 0, same decision boundary style as MASKING_GATE
```

**Candidate algorithms, cheapest first:**

1. **Structural/syntactic heuristics per known content type** — collapse
   repeated stack-trace frames, truncate repetitive log/tool output beyond N
   unique lines, strip boilerplate already covered elsewhere in context. No
   model call, near-zero latency. Default tier.
2. **Local extractive scoring** (TF-IDF/BM25 salience against the recent
   turn's task context) to rank and keep only high-scoring sentences/chunks
   of a long tool-output or prose block. No LLM call needed. Second tier,
   for content structural heuristics don't clearly resolve.
3. **Cheap-model extractive summarization** (Haiku call) — highest quality,
   adds real latency and its own token cost, only justified above a size
   threshold where the savings clearly dominate the extra round trip. Opt-in
   tier, gated by content size.

**Content-type targeting (default allow-list, not a blanket policy):**
`tool_result` blocks flagged as read-only/informational (bash stdout, search
results, doc dumps the agent won't diff against). Explicitly excluded by
default: anything the agent will programmatically re-parse, JSON blobs
consumed verbatim downstream, code blocks referenced later in the session.

**Evaluation plan:** this needs a breaker-style eval before it ships, not
just a token-count before/after. Measure downstream task success (does the
agent still successfully reference facts that were in the compressed block)
against token savings, using this project's own `savings_events.jsonl` /
`proxy.log` history as a replay corpus — real sessions, not synthetic ones.
A savings-only metric would hide the failure mode this whole feature exists
to avoid (content loss forcing more expensive rediscovery later, which is
the same complaint that motivated this design in the first place: "clearing
or compacting is not ideal since it wastes tokens for the model to
rediscover context").

## DP-grounded decision engine for touch / compress / clear (planned)

Two tiers, cleanest-first:

**Tier 1 — touch-vs-bust scheduling is literally the ski-rental problem.**
Pay a small recurring cost (a touch) to keep the cache warm, or let it lapse
and pay one large cost (a bust/rewrite) once. This has a known, provably
optimal deterministic policy: keep touching (renting) until cumulative touch
spend reaches the bust cost (the price of buying), then stop — guaranteed
never worse than 2x the offline-optimal cost, with no need to predict future
session length. Maps directly onto `cache-touch-listener.py`'s scheduling
logic (`TOUCH_DELAY_S`, the 03:00–08:00 exclusion window). Concrete next
step: replace the current fixed 55-minute delay with a threshold computed
from measured touch cost (near-zero, one API round trip at `max_tokens: 1`)
versus the actual measured bust cost for this session's prefix size — the
threshold naturally adapts instead of being a fixed constant.

**Tier 2 — the general freeze/compress/semantic-compress/clear decision is
a Markov Decision Process, solved by value iteration.** State: (cache-alive
fraction, accumulated prefix size, time since last write, model, TTL
remaining). Action: one of {freeze, byte-compress, semantic-compress,
clear/compact}. Cost: immediate $ of the action, transitioning to the next
state. Objective: minimize total expected cost over a bounded lookahead
horizon (a Bellman backup, memoized — the DP the user referenced). Headroom
already logs enough (`savings_events.jsonl`, `alive_fraction` per request,
historical bust frequency) to fit transition costs empirically rather than
guess them by hand. This tier is real research work, not a quick add —
scope it as a follow-on once Tier 1 (ski-rental touch scheduling) is
shipped and its savings are measured.

## Other open items found on re-reading the transcript (not yet acted on)

8. **Sonnet canary run — requested twice, never actually completed.** User
   asked for "a canary run of sonnet, giving the same prompt and letting it
   do a breaker run or something" to directly compare headroom-on vs.
   headroom-off `usage.cache_read_input_tokens`. Substituted with a live
   `/admin/touch` replay instead (finding #4), which validated the touch
   mechanism but is not the same experiment — it doesn't isolate the
   headroom-on-vs-off delta the user actually asked for. Blocked on: doing
   this properly needs a bypass path around the proxy (`ANTHROPIC_BASE_URL`
   points at headroom everywhere in this environment), which means either
   extracting the live upstream API key (declined without explicit
   permission — flagged as a live-credential move, correctly) or adding a
   proxy `?bypass=1`-style debug passthrough that still logs but doesn't
   transform. The passthrough route is the safer implementation to build if
   this experiment is wanted.

9. **"How many commands were redundant or did not provide interesting
   information for the goal?" — asked, never answered.** No count was ever
   computed. Feature idea worth pairing with the semantic-compressor work:
   a **tool-call redundancy audit** — detect duplicate or no-op tool calls
   within a session (same file read twice with no edit between, a grep
   re-run with an identical pattern, a stat/status check repeated with no
   state change) and surface them as a session-end report line, similar to
   how `/context` breaks down token share by category. This is a distinct,
   smaller feature from the semantic compressor (it's about *call-level*
   redundancy, not *content-level* compression) but shares the same
   motivating complaint.

10. **Tool-result 27% share after "dedup and everything was done" — the
    premise was never actually verified.** User's question assumed
    deduplication was already happening on tool results; this session never
    confirmed headroom actually deduplicates tool-result content at all, as
    opposed to just running the byte-level `MASKING_GATE` compression pass.
    Those are different things — dedup would mean recognizing *repeated*
    content across turns (e.g. the same file re-read verbatim) and
    referencing it rather than re-sending it; `MASKING_GATE` is a
    single-request compression gain check. **Needs verification**: grep
    `headroom/proxy/` for any dedup-specific logic distinct from the masking
    pipeline before assuming the 27% figure reflects a dedup pass that may
    not exist yet.

## Verification methods used (for repeatability)

- `curl -s http://127.0.0.1:8787/stats` and `/metrics` for live proxy state
- `/home/pengolodh/.headroom/savings_events.jsonl` — per-request before/after
  token counts, `client` field (`claude-code` / `codex` / `openai` / `proxy`)
- `/home/pengolodh/.headroom/logs/proxy.log` — full per-request pipeline
  trace (`STRUCTURAL-CHURN`, `MASKING_GATE`, `CCR`, `SUBAGENT_MODEL_CAP`,
  `PERF`, `STAGE_TIMINGS`), searchable by `request_id`
- `POST /admin/touch` — live, real replay against Anthropic to get a
  ground-truth `cache_read_input_tokens` figure, not proxy self-report
- Direct grep of `headroom/proxy/handlers/anthropic.py` and `cost.py` source
  for mechanism verification, always before stating a mechanism as fact
