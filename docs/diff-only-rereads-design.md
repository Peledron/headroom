# Diff-only re-reads: design, shipped

Task #5. Written 2026-07-25 after measuring where cache write actually goes
(see rewrite-mechanisms-2026-07-25.md).

**Status, 2026-07-26: implemented.** `ReadLifecycleManager._plan_reread_diffs`
at `headroom/transforms/read_lifecycle.py:556`, covered by
`tests/test_diff_only_rereads.py`. The rest of this document is the design as
written, kept because it records why the earlier full-content policy was wrong.

## The current policy costs more than it saves

When a file is read, edited, then read again, ReadLifecycleManager classifies
the earlier read as STALE and swaps its content for a short recovery marker.
The newer read is forwarded in full.

That swap edits a message the provider has already cached. Anthropic matches
on a linear byte prefix, so rewriting a message at depth k re-bills every byte
from k to the furthest breakpoint. Measured over 1175 recorded turns, rewrites
of already-forwarded messages came to 4.00M tokens against 2.55M tokens of
genuine append. Retroactive edits cost more than the new content does.

## The inversion

Leave the earlier read alone. It is already in the cached prefix, so keeping it
is free. Send the newer read as a unified diff against it.

    today  : mask old (rewrite at depth k) + full new (L tokens appended)
    instead: keep old (no rewrite)         + diff  (d tokens appended)

Read rises by roughly d, since one full copy is carried either way. Write drops
by L plus the rewrite amplification. Write bills at 1.25x on the 5m tier and
2.0x on the 1h tier against read at 0.1x, so the trade is strongly favourable
whenever d is small relative to L.

The two policies are mutually exclusive per re-read pair. A diff is only
applicable if its base is still visible, and masking the earlier read removes
exactly that base.

## Why it is not implemented yet

read_lifecycle.py rewrites content the model reads and then generates against.
On 2026-07-17 a masked rebase in this area put a fabricated marker on disk as a
file's contents, and the invented hash was unrecoverable. The module carries
that history in its comments.

A change of this shape needs its own session with full context, not the tail of
an exhausted one. It should also land behind a config flag defaulting off, so
it can be validated against live traffic before it decides what the model sees.

## Implementation sketch

1. Config flag, default off.
2. In ReadLifecycleManager, detect re-read pairs where the earlier content is
   still resident and unmasked.
3. Compute a unified diff, newer against earlier.
4. Only proceed if the diff is materially smaller than the full content. A
   rewritten file diffs to roughly its own size, so fall through to today's
   behaviour there.
5. Suppress STALE/SUPERSEDED masking for that pair, since the base must stay.
6. Replace the newer read's content with the diff, labelled so the model knows
   it is a delta and against which earlier read.
7. Store the full newer content in CCR so it stays retrievable.

## Related

Task #6 (scope masked-file re-reads to the relevant region) shares step 2 and
the same risk profile. Both are read-path content rewrites and should ship
together, behind the same flag.
