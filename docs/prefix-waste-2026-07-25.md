# Where the cache writes actually go, measured 2026-07-25

First measurement taken against the *forwarded* replay corpus with billed figures
read from the same records, so the byte diff and the cost come from one source.
Corpus: `~/.headroom/replay/replay-0000.jsonl`, 1185 records, 69 sessions,
1075 consecutive-request pairs.

Earlier attribution in this project diffed headroom output against headroom's own
previous output at *message* granularity. Any re-serialization of an unchanged
message counted as new content, which is how the bogus "2.26k per turn append
floor" and the retracted "100:1 is impossible" ceiling were produced. This
measurement compares raw bytes in true cache-prefix order (system, tools,
messages) and is not subject to that error.

## Headline

| quantity | tokens | ratio against read |
| --- | --- | --- |
| billed read | 131.97M | |
| billed write | 11.73M | 11.3x |
| forced write (byte-diff floor) | 7.74M | 17.0x |
| genuinely new content | 2.61M | 50.6x |

Three separate gaps, and they want three different fixes.

1. **11.73M billed against a 7.74M byte floor.** Headroom is billed 1.5x more
   than its own forwarded bytes require. This is anchor and breakpoint placement,
   which is what the anchor-stability work addresses.
2. **7.74M forced against 2.61M genuinely new.** A 2.97x rewrite multiple.
   Content above the divergence point gets rewritten even though it did not
   change. This is the reserialization and dedup work.
3. The 50.6x floor is not 100x, but it is not a wall either. It is the floor for
   *this* corpus with today's transform set, not a property of the design.

## Top divergence source: the tools array

Ranked by wasted write, the largest single row is `system_or_tools`: 1.63M
tokens, 14% of all billed write, across only 18 pairs. Roughly 90k per event.

The system block is part of the session grouping key, so these cannot be system
divergences. They are `tools` divergences inside a single session with a stable
system prompt. Confirmed directly: 11 of 69 sessions mutate their tools array,
16 transitions total, and the tool count only ever rises within a session:

    16, 16, ... 16, 21, 21, ... 21, 22, 22, ...

That is deferred tool loading. `ToolSearch` resolves a deferred tool and the
client appends its schema to `tools`. Because `tools` sits ahead of every message
in the cache prefix, appending one schema invalidates the entire transcript.
A 2k schema addition bills a 90k rewrite.

Two related instabilities showed up in the same diffs and are worth separating:

- A `defer_loading: true` key toggling on and off the placeholder tool entry.
- A 39-line block appearing in one direction and disappearing in the other.

Non-monotone churn like that is pure waste with no upside and is fixable inside
headroom regardless of what the client does.

### The fix worth building

Forward a stable minimal `tools` array and inject newly-loaded tool schemas as a
message at the tail of the transcript instead of letting them land in `tools`.
Growth then appends where appending is cheap, rather than invalidating the head.
This converts a ~90k full-prefix bust into a ~2k append, and it is exactly the
class of rewrite a proxy is positioned to do.

Lesser version if the above proves unsafe: pin per-session tool ordering and
never allow the array to shrink or a key to toggle. That does not recover the
first-append bust but removes the oscillation.

## Note on interleaved clients

The corpus contains at least two concurrent Claude Code clients sharing the
proxy, distinguishable by `cc_version`, working directory, and model. Grouping on
`user_id` alone merges them, because `metadata` carries only `user_id`. Any
analysis or controller state keyed on `user_id` will see one imaginary session
churning wildly where there are really two stable ones. The measurement script
groups on the system block plus the opening exchange to avoid this. Whether
headroom's own session identity has the same defect is the next thing to check,
and it would explain the `observe_client_churn` 0.33 STRUCTURAL-BUST reading
recorded on 2026-07-24.

## Reproducing

    .venv/bin/python scripts/measure_prefix_waste.py
