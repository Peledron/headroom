# What was built, how it works, and why it was chosen

Written 2026-07-25, covering the work on this fork from 2026-07-21 onward. It
is meant to be read in order: each layer only makes sense once the one below it
is in place.

The short version. Headroom's job is to make a long agent session cost less
without making it worse. Every mechanism here follows from one fact about how
the provider bills a request, and one measurement of where this proxy was
losing money against that fact.

## 1. What a token costs

Anthropic prices a request in four buckets. In units of the base input token:

| bucket | multiplier |
| --- | --- |
| uncached input | 1.0 |
| cache read | 0.1 |
| cache write | 1.25 |
| output | about 5 |

Two consequences drive everything else.

Reading a warm prefix is nearly free. A 150k-token prefix that hits cache bills
like 15k tokens.

Rewriting that prefix is not. It bills 1.25x, so 187.5k. Against the 0.1x it
would have cost to read, the marginal price of busting a live prefix is about
1.15x per token. One bust of a 150k prefix costs about as much as 172k fresh
input tokens, which is more than most turns produce in a day of output.

So in a long session the dominant cost term is not how big the context is. It
is how often the context gets rewritten. A proxy that compresses aggressively
but rewrites the prefix while doing it loses, and loses by a factor, not by a
margin.

The scoreboard for this is the read:write ratio. A plain Claude Code session
with no proxy runs near 100 reads per write. This proxy was running far below
that, which meant headroom itself was paying the write premium over and over.
`docs/cache-economics.md` has the baseline numbers and the argument for why
100:1 is not the right target here.

## 2. The mechanism everything rests on

The provider matches the longest common byte prefix of the request against what
it has cached, bounded by up to 4 `cache_control` breakpoints.

Five corollaries, all of which turn into code later:

1. A bust is a depth, not a boolean. Diverging in the last message leaves 99
   percent of the prefix readable. Diverging in the system prompt leaves none.
   Any mechanism that reports "cache miss" without reporting depth is useless.
2. Byte identity, not semantic identity. Key order, whitespace, and the order
   of the tools array all count. Two requests that mean the same thing and
   serialize differently share no prefix past the first differing byte.
3. The head is expensive and the tail is cheap. Anything that grows during a
   session and sits near the head (the tools array, the system prompt) is a
   standing tax.
4. Caches are keyed per model. Switching model mid-session throws the whole
   cached prefix away and pays to build a second one.
5. With at most 4 breakpoints, a bust at depth `c` costs everything between `c`
   and the nearest breakpoint at or above it. Breakpoint placement is therefore
   an optimization problem, not a convention.

## 3. Layer one: measure before building

Nothing below was chosen because it sounded plausible. Each one was named by
the ledger first.

`TOKEN_DIAG` and the reconciliation ledger attribute each turn's write tokens
to a cause. `STRUCTURAL-CHURN` and the structural ledger separate writes caused
by the client's own growth from writes caused by headroom's transforms.
`PREFIX_DIVERGE` and `CACHE-MISS-ATTRIBUTION` report where the first divergent
byte landed, which turns corollary 1 into a number per turn.

That is how the 92 large-write requests holding 58 percent of all write tokens
were found, and how the tools array was identified as the single largest
divergence source at about 14 percent of the waste
(`docs/prefix-waste-2026-07-25.md`).

The rule this establishes: a mechanism that cannot be attributed does not get
built, and a fix whose effect cannot be measured does not get called a fix.

## 4. Layer two: stop breaking the prefix by accident

These are pure losses. Writes with no compression benefit at all. Fixing them
was worth more than any amount of cleverness further up.

**Non-canonical serialization.** The client's JSON key order is not guaranteed
stable across turns, and transforms re-serialize. Canonicalizing the request
before any transform runs makes turn N and turn N+1 comparable byte for byte.

**Tools array growth.** MCP servers add tools during a session, and the tools
array sits ahead of the messages. One added tool rewrote the entire prefix.
Fixed by sorting tools deterministically and holding the forwarded set stable,
so a tool that appears later does not reorder what was already cached.

**Lineage matching that was too strict.** The tracker compared the forwarded
prefix against the previous turn and, on any mismatch, discarded the whole
lineage and started over. In practice most mismatches were a shallow tail
difference. `LINEAGE_REJECT` made the rejections visible, and relaxing the
match to a strict prefix comparison stopped a mostly-matching history from
being thrown away. `LINEAGE_RESTORE` covers the recovery path.

**Restart amnesia.** Lineages lived only in memory, so restarting the proxy
made every live session rewrite its whole history, measured at roughly 195k
write tokens per restart. `headroom/cache/lineage_persistence.py` snapshots the
store atomically and off the request path. Two properties matter more than
speed: the write is atomic, because a truncated snapshot would restore a
history that was never forwarded, and nothing in the module may raise into the
caller, because a persistence failure must degrade to the old behavior rather
than fail a turn.

**Freeze semantics.** Compression that ran once and then never again mid
lineage was not a bug, it was the freeze doing its job. Writing that down
stopped a real fix from being aimed at the wrong place.

## 5. Layer three: spend a rewrite on purpose

Once accidental busts are gone, every remaining rewrite is a decision. Each one
has the same shape:

```
cost      = S * (write_multiplier - read_multiplier)     about 1.15 * S
gain/turn = whatever the mutation saves per later turn
break_even_turns = cost / gain_per_turn
```

and the decision is whether the session will run that many more turns. Three
things follow.

**Every gate needs a horizon.** `headroom/proxy/turn_horizon.py` supplies it.
The obvious estimator is Lindy (a session that has run N turns will run about N
more), which says the same thing about turn 40 of a typo fix and turn 40 of a
thirty-file refactor. Since the error is asymmetric in cost (an overestimate
spends a rewrite that never repays, an underestimate only misses a saving), the
horizon is estimated from observable difficulty instead, from signals that are
free at request time.

**Free cases are worth more than marginal ones.** A cold prefix has nothing to
rewrite. A turn whose prefix is busting anyway pays the rewrite either way, so
any structural change rides along at no extra cost. Both cases skip the horizon
question entirely, and they are where most of the real saving comes from.

**Decisions must be sticky per lineage.** Difficulty alternates turn to turn as
tool runs and user asks interleave. A per-turn decision pays a full rewrite on
every alternation and loses on all of them. Once a lineage commits to a
setting it holds until the lineage itself breaks.

The gates built on this:

- The masking gate, which spends a rewrite to shrink history.
- The effort router (`headroom/proxy/effort_pricing.py`, `EFFORT_PRICE`), which
  spends one to lower reasoning effort. Effort is a body mutation ahead of the
  cached suffix, so on Anthropic it busts the prefix. The old rule was "never
  switch when caching is on", which is safe and wrong whenever the stretch is
  long enough to repay, or the prefix is being rewritten anyway.
- The compaction advisor (`headroom/proxy/compaction_advisor.py`,
  `COMPACT_ADVISE`, `DEFER_COMPACT`), which prices compaction instead of
  waiting for the context window to fill. Claude Code's own trigger is a safety
  rule, not an economic one, and it fires at the worst moment: deep in a task,
  on a prefix that has been read cheaply for hundreds of turns. Priced, the
  same operation can be moved to a boundary where it costs least, and deferred
  when a strong compression bust is about to happen anyway.
- Anchor placement (`headroom/cache/anchor_dp.py`), which uses the observed
  distribution of bust depths to place the 4 breakpoints so that expected loss
  is minimized rather than guessed.

## 6. Layer four: send fewer bytes, at the right depth

These shrink what is forwarded. All of them mutate history, so none of them run
without a gate from layer three saying the rewrite repays.

**Observation masking with an age gate.** Old tool results are replaced by a
marker plus a hash, retrievable on demand. The age gate exists because the
recent tail is where the model is actually working.

**Read lifecycle.** A Read becomes stale when its file is later edited, and
superseded when the same file is read again. Both are provably safe to replace
because the content in context is either wrong or duplicated. Measured on real
traffic, about 67 percent of Read bytes are stale and 12 percent superseded.

**Diff-only re-reads.** A re-read of a file that moved by a few lines used to
append the whole file again. Unmasked tool results at or above 500 tokens are
36.2 percent of all genuinely appended content, and the appended tail is what
sets the read-to-write ceiling, so sending the delta turns an L-byte append
into a d-byte one. The base read is deliberately left alone by the
stale/superseded pass, since a diff against a removed base means nothing.
(`docs/diff-only-rereads-design.md` predates the implementation and still says
"not yet implemented".)

**Region scoping.** A masked file that gets re-read is restored only around the
region in play, not in full.

**Closed-episode micro-compaction.** A finished thread of conversation is
compacted preemptively at a cheap moment rather than left for the emergency
compaction to sweep up at an expensive one.

## 7. Layer five: routing the model itself

This is the smallest lever and the one with the most ways to go wrong, so it is
also the most constrained. `headroom/proxy/model_pricing.py` holds the
arithmetic and `_maybe_route_model` in the Anthropic handler holds the policy.

Caches are keyed per model, so a switch abandons the warm prefix and rewrites
it at the new model's price, and coming back rewrites it again at the old one:

```
switch_cost = S * write_multiplier * r
return_cost = S * write_multiplier
per_turn_gain = (1 - r) * (S * read_multiplier + new_input + out * out_mult)
```

At a 150k prefix, a price ratio of 0.2, and a turn producing 1,500 output
tokens, the switch costs about 37k and the return about 188k against a gain
near 18k per turn. That is 12 to 13 easy turns before it repays. A single easy
question routed to a cheap model is a clear loss, and a per-request rule engine
would route exactly that.

Four constraints, each one a gate the switch has to pass:

1. **The price ratio is asserted, never inferred.** Per-model prices are not
   published in a form the proxy can read, so the ratio comes from the operator
   through `HEADROOM_MODEL_ROUTE_PRICES` (for example
   `claude-sonnet-5=0.2`). With no entry for the target model the gate stands
   aside. Values outside `(0, 1]` are dropped, since a target priced at or
   above the current model has nothing to offer.
2. **Only a genuinely easy turn qualifies.** The difficulty estimate starts a
   turn at 0.5 (unknown work is not easy work) and only one signal pulls it
   down: a fresh, short, plainly phrased question. Everything else adds. An
   earlier version scored from zero and treated a tool continuation as evidence
   of easy work, which had it backwards, since a tool continuation is the
   middle of a task someone is depending on. Being wrong here costs a wrong
   answer on real work, so the estimate has to earn "easy" rather than fall
   into it.
3. **The decision is sticky per lineage**, for the reason in layer three.
4. **No switch without notice.** A routed response carries a text block naming
   the model that answered and how to turn the behavior off, on every routed
   turn rather than only the first. A path that cannot carry that notice
   (direct SSE, where the early events have already left) refuses to route at
   all.

What is deliberately not decided here is whether the cheap model is good enough
for the work. That is a quality question, it needs a live A/B against real
traffic, and no amount of arithmetic substitutes for it. Until that measurement
exists the gate stays off by default and admits one shape of turn.

## 8. Layer six: safety rails

**The marker guards.** Masked content is replaced by a marker carrying a hash.
On 2026-07-17 a model reproduced marker text into a Write call and the marker
went to disk as file content. The guards close that: any tool_use input
containing marker text is expanded when its hash resolves in the compression
store, and refused when it does not. Tool results are environment-authored and
safe to mask. Tool inputs are model-authored and never are, which is why the
age gate has never been relaxed for inputs. `headroom/proxy/handlers/anthropic.py`
holds the buffered guard and `headroom/proxy/sse_marker_guard.py` the streaming
one, which holds back a tool_use block until its `content_block_stop` arrives
so it still has something to rewrite.

**Recovery, added 2026-07-25.** The refusal used to say "re-emit the tool call
with the real content written out in full", which is advice the caller cannot
follow, because the content it is missing is the content that was masked away.
`headroom/proxy/marker_recovery.py` replaces it. A hash absent from the store
is almost never expired (capacity is a whole session, the TTL is a day), it is
a hash that was retyped rather than copied. Since every stored key is the same
24 hex characters, a one-character slip shows up as a Hamming distance of one
against exactly one stored key and a dropped tail shows up as a prefix. The
message names the near miss and gives two routes that work: retrieve by the
exact hash, or re-read the file, since disk is the source of truth. The
suggestion is never applied silently, because expanding the wrong entry would
put the wrong bytes into a file edit, which is the failure the guard exists to
prevent.

**The refusal does not stop the turn, also 2026-07-25.** A refusal that ends
the turn parks the client's agent loop until a human types, which turns a
mistyped hash into an interruption. Now the clean tool calls in the turn
survive, so the loop keeps going on their results. When the blocked call was
the only one, the response has to be shaped as `end_turn` (there is no valid
shape with no tool_use block and a `tool_use` stop reason), so instead the
proxy replays the refusal as an assistant turn, adds a short retry prompt as a
user turn, and spends one upstream call. The client receives whatever the model
does next, which is a live turn. Exactly one retry: a model that repeats the
mistake will not be talked out of it by a second copy of the same message. The
streaming path cannot do this (its earlier events are already gone), so it
keeps the surviving-call rule and falls back to ending the turn.

**The native detector breaker.** The Rust content detector deadlocked on first
use under some conditions, and the breaker that guards it is process-wide and
one-way by design. That is right in production and wrong in a test process,
where one test tripping the breaker silently changed the behavior of every
later test. Fixed with an autouse reset fixture in the affected file. The
underlying deadlock is still open (task 24).

## 9. One request, end to end

1. Read the body, canonicalize it, decide bypass.
2. Route the model if and only if all four gates in layer seven pass.
3. Resolve the session lineage from the tracker store, restoring from the
   on-disk snapshot if this is the first turn after a restart.
4. Compare the forwarded prefix against the previous turn, log divergence depth
   and attribution.
5. Ask the horizon how many more turns will read this prefix.
6. Offer each history mutation (masking, read lifecycle, diff-only re-reads,
   micro-compaction) to its gate. Free cases pass immediately. Marginal ones
   have to beat break-even against the horizon.
7. Place cache breakpoints against the observed churn depth distribution.
8. Forward. Record what was actually sent, so step 4 has something to compare
   against next turn.
9. On the way back, guard the response for marker mimicry, add the routing
   notice if the model was changed, then reconcile usage against what was
   predicted and write the ledger line.

## 10. What is not settled

- **Quality of the cheap model is unmeasured.** The routing gate prices the
  switch correctly and says nothing about whether the answer is as good. This
  needs a live A/B the user has to authorize.
- **The native detector deadlock** is still open (task 24). The breaker
  contains it, it does not fix it.
- **13 test failures inherited from the upstream integration** are still open
  (task 25). They reproduce at the integration HEAD and are drift in test
  doubles and workflows, not in the mechanisms above.
- **The post-fix measurement window opens 2026-07-25 16:12.** Ratio numbers
  taken from before that mix in the pre-fix behavior and should not be compared
  directly.
