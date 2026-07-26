# Cache economics: where the write tokens go

Single overview of the prompt-cache cost model, the measured attribution against
a live 126-hour corpus, and the decision rules now in the code. Written
2026-07-25, replacing the scattered analysis in the earlier handovers.

## Why write tokens are the thing to watch

Anthropic bills three input tiers. Relative to the base input price:

| Tier | Multiplier |
| --- | --- |
| Cache read | 0.1x |
| Cache write, 5m tier | 1.25x |
| Cache write, 1h tier | 2.0x |

A cache write costs 12.5 to 20 times what the same token costs to read. So the
read:write ratio, not the absolute read volume, is what separates a cheap
session from an expensive one. A healthy Claude Code session appends 1 to 3k of
new content per turn against a 150k-plus prefix, which lands near 100:1. Every
point below that is prefix content being paid for a second time.

## Measured baseline

Two independent sources, same window:

- `~/.headroom/logs/cache_reconciliation.jsonl`, 1820 requests over 126 hours.
  Billed totals: read 183,098,400, write 21,142,176. **Ratio 8.66.**
- `~/.headroom/replay/`, the bodies actually forwarded upstream, 948 turns
  across 11 conversations, paired turn to turn.

The replay corpus is the authority for cause, because it records what the
provider saw rather than what the client sent.

## Attribution

Pairing each turn against its predecessor in the same conversation and diffing
the forwarded bodies with `cache_control` markers stripped:

| Class | Turns | Write | Read | Ratio |
| --- | --- | --- | --- | --- |
| Steady state, gap under 5m | 865 | 6.21M | 108.6M | 17.5 |
| Idle past 5m | 32 | 4.18M | 3.75M | 0.9 |

Two findings carry nearly everything:

1. **32 turns, 3.4% of traffic, produced 40% of all write tokens.** They average
   130k of write against almost no read. The signature in the reconciliation log
   is unmistakable: read stalls at roughly 23k, which is exactly the system plus
   tools block, while the whole message body is written fresh.
2. **102 of the 865 steady-state turns rewrote already-forwarded history**, and
   those 102 turns account for 3.41M of the 6.21M steady-state write. Twelve
   percent of turns caused fifty-five percent of the steady-state cost.

The remainder, roughly 2.8M, is genuine appended content. That is the floor.

## Defect 1: the message tier was on 5m while system was on 1h

Across 948 turns the system and tools breakpoint carried a 1h TTL on 942, while
the message breakpoint carried 5m on 830. Any pause past five minutes therefore
lapsed the message history but not the system block, which is precisely the
"read stalls at 23k" signature.

The TTL rule was reactive and size-blind. It looked only at the last eight turn
gaps and flipped to 1h after a breach, then fell back to 5m as soon as the
cadence sped up again, so the next pause paid full price all over again.

The economics it was missing: staying on 5m risks rewriting the whole prefix P
at 1.25x, while choosing 1h costs a 0.75x premium on the per-turn delta d only.
Break-even is therefore

    T* = (1.25 * P) / (0.75 * d)

At the measured P of 100k to 300k and d of about 7.2k, T* lands near 46 turns.
The corpus shows a breach roughly every 30 turns. At these sizes 1h is the
cheaper tier, and the old rule could not see that because it never looked at P.

**Fix:** `PrefixCacheTracker.prefers_long_ttl` weighs the prefix at risk against
the 1h premium and the session's own observed breach count. Sessions with a
small prefix, or with no breach on record, stay on 5m. The breach counter is
session-lifetime, so it does not age out of the eight-slot gap window.

## Defect 2: the cached-prefix replay bailed out permanently

`overlay_cached_prefix` replays the exact bytes the provider already billed for
the unchanged leading run of messages, which is what keeps a compressed prefix
byte-stable across turns. It required the recorded forwarded and original
message lists to have equal length, and returned the freshly-optimized messages
untouched whenever they differed.

Once compression injected or merged a single message, that guard tripped on
every subsequent turn of the session. From then on each turn re-derived the
prefix from scratch, and any non-idempotent normalization rewrote history.

Four such mutations show up in the corpus:

- the `caller` field being dropped from `tool_use` blocks, exactly 30 characters
- a tool result masked on a later turn after being forwarded in full earlier
- message content normalized from a one-element list into a bare string
- tab and newline normalization applied on some turns and not others

Each one changes a message the provider has already cached, which forces a
rewrite of everything after it.

**Fix:** alignment by tool id. Tool call ids are provider-assigned and survive
masking, compression and annotation churn, so they pin a forwarded message to
the original it came from even when the text between them was rewritten. The
count mismatch no longer disables the replay. Safety rules kept deliberately
tight:

- a message with no tool id is paired by position only while no drift exists
- when the forwarded list is shorter than the original, a message was dropped
  and position proves nothing, so unanchored messages get no positional
  fallback at all
- the per-index canonical equality check against the client's own messages is
  unchanged, so wrong bytes are still never forwarded

## Defect 3: a structural bust forced the 5m tier

On a detected bust the handler forced a fresh 5m write. A bust rewrites the
prefix whichever tier it lands on, so forcing the short tier guarantees paying
for that same rewrite again soon. The bust path now honours `prefers_long_ttl`,
which means the expensive write is made to last when the prefix is large.

## Defect 4: compression markers expired mid-session

`DEFAULT_CCR_TTL_SECONDS` was 1800. A compressed tool result stays in the
transcript for the whole session, so its marker stays referenceable for the
whole session. After thirty minutes the store dropped the entry, the marker
guard could no longer expand it, and it blocked the tool call outright with an
unknown-hash error. The client lost work that was never at risk.

Raised to 86400. Capacity, `max_entries` with LRU eviction, is the real bound.
The TTL only reaps stale sessions.

## What this does not fix

The append floor is real. Roughly 2.8M of the 10.4M write in the replay window
is new content that has to be written once. Cold starts add more: 142 requests
in the reconciliation log read nothing at all, and each new session or subagent
legitimately starts cold.

## The ceiling argument was wrong

An earlier draft of this section argued 100:1 was unreachable because the measured
append of 2.26k tokens per turn was a floor, and that a compressing proxy must
therefore show a lower ratio than an unproxied session. Both halves are wrong.

An unproxied Claude Code request on a comparable context reports r:117.3k w:538,
a per-request ratio of 217. Headroom on the same class of workload averages
100.6k read and 11.6k write per request, a ratio of 8.66. The read scale matches,
so the gap is entirely on the write side: headroom writes roughly 21x more per
request than the unproxied client. That is waste, not a floor.

The consequence for the 2.26k "append floor" number above: it cannot be the
genuine cost of new content, because the unproxied client appends the same
conversation for about 538. Most of the difference is headroom reformatting or
re-emitting history in a way the diff counts as new content. Treat 2.26k as an
upper bound contaminated by mutation, not as a floor, and re-derive it by
diffing against the unproxied client's own forwarded bodies rather than against
headroom's previous output.

Removing the two waste classes takes the projected write from 21.1M to somewhere
near 5 to 6M on the same traffic, which is a ratio in the low 30s.

## Why 100:1 is not reachable here, and why that is fine

The 100:1 seen in a plain Claude Code session is not a quality bar this proxy can
hit, and chasing it would make the bill worse. The argument is arithmetic, not an
estimate.

Every token read from cache was written to cache exactly once. In steady state,
with a context of C tokens and d tokens appended per turn, a turn reads C and
must write at least d. So the ratio is bounded:

    ratio = C / d

Measured on this corpus, C is 129k read per turn and the append floor d is 2.26k
per turn (824 turns whose history was byte-identical, so their write is pure
append). That gives a hard ceiling of 129 / 2.26, which is 57. Hitting 100 needs
either C at 226k or d at 1.29k. Nothing in cache placement moves either one: d is
the size of the new assistant message plus the new tool result, which is set by
the workload.

That is exactly why an unproxied session shows 100:1. It is not placing
breakpoints better, it is carrying a bigger C because nothing compressed it. Same
d, larger numerator, higher ratio, and a strictly higher bill. The ratio rises as
compression gets worse. It is a diagnostic, not an objective.

The objective is cost. Priced in plain input-token units (read 0.1x, 5m write
1.25x, 1h write 2.0x):

    before   read 183.1M x 0.1 = 18.3M     write 21.1M x ~1.25 = 26.4M   total ~44.7M
    after    read ~183M  x 0.1 = 18.3M     write ~4.2M x ~1.25 =  5.3M   total ~23.6M

Write is the larger line item in both, which is why it was the right thing to
attack. The fixes cut total spend roughly in half while the ratio only moves from
8.7 to the low 30s. Track the cost line above. If the ratio is wanted as a health
signal, compare it against the C/d ceiling for the session in hand, not against
100.

## How to re-measure

The reconciliation log is the scoreboard:

    python3 - <<'PY'
    import json
    rows = [json.loads(l) for l in open('/home/pengolodh/.headroom/logs/cache_reconciliation.jsonl') if l.strip()]
    r = sum(x['billed_cache_read'] for x in rows)
    w = sum(x['billed_cache_creation'] for x in rows)
    print(f'read={r:,} write={w:,} ratio={r/max(w,1):.2f}')
    PY

Compare like for like. The log accumulates across sessions, so snapshot the line
count before a change and slice from there rather than comparing whole-file
totals against the baseline above.
