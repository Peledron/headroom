# Where the cache write goes, measured 2026-07-25

Source: the forwarded replay corpus, 1175 consecutive request pairs across 71
sessions, grouped by (system, first message) so concurrent clients stay apart.
Billed figures come from the same records as the byte diff, so cost and cause
share one source.

## Headline

    billed read                   131.97M
    billed write                   11.73M     ratio 11.3x
    forced write (byte floor)       7.74M     ratio 17.0x
    genuinely new content           2.61M     ratio 50.6x

## What each turn actually sends

    genuinely appended messages     2.55M     avg 2170 per turn
    old messages rewritten          4.00M     avg 3406 per turn

Every turn rewrites something. 61 percent of what is forwarded as "new" is
re-sent old content rather than genuine append.

## Rewrite waste by depth

    depth 1-2 messages    n=987   1.52M   37.7 percent
    depth 3-10            n= 88   0.32M    8.1 percent
    depth 11-50           n= 82   0.64M   15.8 percent
    depth 50+             n= 35   1.55M   38.5 percent

Read the 50+ row carefully. 35 events carry 38.5 percent of the total, about
44k each. Those are full-history sweeps, and a sweep that runs when the prefix
is already dead costs nothing. So 4.00M is an upper bound on waste, not waste.
Separating dead-prefix sweeps from live-prefix ones needs the bust flag to be
correct, which is the head-fingerprint fix below.

## Mechanisms

1. Anchor cache_control churn. Attaching or detaching a breakpoint changes the
   bytes of the message it sits on, so a walking anchor is a walking rewrite.
   Simulating old against new placement over the same 1150 turns gives 27 moves
   and 818,785 tokens against 26 moves and 788,239 tokens, a 3.7 percent
   reduction. Real but small: the 1h anchor is not the dominant cost, which
   corrects the earlier reading of this file.

2. Retroactive masking. RETRACTED. The first version of this section claimed a
   tool result rode verbatim for three assistant turns and that lowering the
   floor to 1 would land the edit 1 to 2 messages deep instead of 11 to 50.
   Both halves were wrong. The constant gated two `sweep_assistant_text` call
   sites, so it never touched tool results at all, and assistant text is
   model-authored: pulling its floor toward the tail moves CCR marker text into
   the window the model imitates from, which is the shape of the 2026-07-17
   fabricated-marker incident. tests/test_breaker3_sweep_chaos.py caught it.

   The constant is now ASSISTANT_TEXT_SWEEP_AGE, back at 3, documented against
   the measured 8.4 percent share of assistant prose so the trade is visible.
   Tool-result masking has no turn-age gate and should not get one: it is priced
   by masking_gate_gain, which compares rewrite cost against read saving rather
   than guessing from position. So this mechanism contributes nothing, and the
   11-to-50-deep rewrites in the depth table have another cause, still open.

3. Tools-array growth. ToolSearch appends schemas to `tools`, which sits ahead
   of every message in the prefix, so a 2k schema addition invalidates the whole
   transcript. 16 transitions across 11 sessions, about 90k each, 1.63M total.
   Tool counts only ever climb within a session: 16, then 21, then 22.

   observe_client_churn compared only `messages`, so these read as 1.0, prefix
   intact, when the prefix was entirely dead. That is why a sudden large write
   did not trigger compression: the turn was a total bust that headroom could
   not see, so free_rebase_at_bust never fired on the one turn where rewriting
   is free. It now hashes system plus tools into a head fingerprint and reports
   0.0 when that head moves.

## Ceiling

The first pass at this split was wrong. It bucketed by message role and put
50.6 percent in "assistant turns and other", calling that irreducible.
Attributing every appended block to the tool or role that produced it, deduped
by tool_use_id so a block is counted on the turn it arrived
(scripts/measure_append_by_tool.py, 74 sessions):

    Read results, unmasked, >= 500 tok   1,219,677   51.7%   n=214  avg 5,699
    tool_use inputs (the calls)            395,289   16.8%
    assistant prose                        198,131    8.4%
    small unmasked results                 177,271    7.5%
    already masked or compressed           147,687    6.3%
    Bash results, >= 500 tok               140,564    6.0%
    other large results                     32,755    1.4%
    user messages                           25,054    1.1%
    thinking blocks                              0

That totals 2.34M against the 2.55M the diff-based pass measured, so the two
independent methods agree to within 8 percent.

Read alone is 51.7 percent of everything appended, and the truly irreducible
slice (assistant prose, thinking, user messages) is 9.5 percent, not 50.6.
Within tool_use inputs the largest entries are Bash 205k, Edit 74k, Write 44k.

This relocates the lever. Masking a Read result saves the re-read stream but
never the one-time append: the full text is forwarded once, unconditionally,
because the model asked for it. 1.22M of write is that first forward. Cutting
it means sending less than the model asked for on the first pass, which is a
quality trade and needs a pilot, tracked as task #11.

Diffing re-reads, the obvious way to cut Read append, does not help.
scripts/measure_reread_diffs.py finds 108 tokens of same-window re-reads across
the same 74 sessions. The model reads many different files and windows, it
almost never re-reads one. Implemented anyway behind a default-off flag, and
measured at zero.

100x is therefore not reachable by removing waste alone at this context size.
It is partly a property of context depth: a 217x observation on vanilla was
117.3k read against a 538 token tail append. Headroom can beat vanilla on total
tokens while showing a worse ratio, because compression trades write against
read by design.

## The other candidate lever, also rejected

AST code compression on Read results (content_router.py enable_code_aware,
default off with the comment "use code graph MCP tools instead"). Half that bet
landed: the code-graph tools are used heavily. The other half did not: the model
still reads files directly. Running the real compressor over the real reads
(scripts/measure_code_aware_on_reads.py) gives 231,333 tokens on 1,193,645, 19.4
percent, with the compressor self-rejecting its own output as invalid syntax on
a dozen Python files and emitting *more* than it consumed on three C++ files
(worst ratio 1.29). That measurement supports leaving the flag off.

## Where that leaves 100x

132M read against 1.32M write is 100x. The append floor is 2.36M today, and
2.13M with code-aware enabled, so even with every byte of rewrite waste removed
the ceiling is about 62x. Closing the last 1M means not forwarding large Read
results verbatim on the first pass, which changes what the model sees rather
than how it is cached. That is a quality trade and belongs behind a piloted
flag, not a default.

The ratio is also partly antagonistic to the cost goal. Compression moves tokens
out of the read stream, which shrinks the numerator. A worse ratio at a lower
bill is the better outcome, and the write churn, which is the part that was
straightforwardly broken, is what the four fixes address.

## The actual root cause, found after the above

Everything above measures symptoms. Splitting the divergence cost by whether the
prefix was still alive when the edit landed (scripts/measure_waste_vs_free.py)
separates three cases:

    pure append (correct behaviour)                 1034 turns   3.28M   38.5%
    head moved (prefix already dead, edit free)       11 turns   0.28M    3.3%
    prefix was live (waste headroom caused)          246 turns   4.93M   57.8%

Only the third is actionable, and its causes are all one thing: headroom's
forwarded representation of an old message is not stable across turns.

    compression arrived late (new marker in old message)  2.01M  23.5%  n=99
    unmarked content resized                              1.02M  12.0%  n=41
    client serialization changed                          1.01M  11.8%  n=66
    marker text changed                                   0.78M   9.2%  n=30

Headroom already has the mechanism that would prevent all of it.
overlay_cached_prefix replays the previously forwarded bytes for every leading
message that still agrees with the client, and stops at the first divergence. It
is correct. It just almost never runs. From the proxy's own TOKEN_DIAG lines,
overlay applied on 36 of 952 turns, and on all 72 turns carrying 20 or more
messages the tracker had no previous forwarded prefix at all.

The reason is SessionTrackerStore.resolve_tracker. It reused a tracker only when
a recorded chain was a strict prefix of the incoming history, on the stated
assumption that client histories are append-only. They are not. Claude Code
strips ephemeral `<system-reminder>` blocks out of old user messages, and the
canonicalizer cannot absorb that because it is a change to content text rather
than to a transport key. One such edit at message 168 of 175 failed the test,
started a fresh lineage, and discarded the forwarded bytes for the 167 messages
that were fine.

That also explains the length dependence. The chance that some old message has
churned grows with history length, so the longest conversations, which carry all
the cost, lost their tracker every single time.

Measured on the corpus (scripts/measure_lineage_breaks.py): histories of 20 or
more messages break lineage on 18.1 percent of turns, at mean mismatch depth
167.4, discarding 24.1M tokens of valid prefix.

The fix falls back to the longest common prefix and reuses the tracker when that
prefix clears two guards, tuned together against the corpus:
lineage_rematch_min_messages (default 8) and lineage_rematch_fraction (default
0.5). That retains 191 of 222 broken lineages and recovers 24.3M of the 25.4M
discarded, 95.8 percent.

Both guards are needed and they catch different things. A first attempt used the
fraction alone at 0.9, which recovered only 79.7 percent and, on a replay of a
40 turn trace, failed to fire at all: an edit twelve messages from the tail of an
80 message history scores 0.85 and was rejected. The absolute floor is what keeps
sibling subagents apart, since two of them under one session id share the system
prompt and opening message but diverge within one or two more, well under eight.
The fraction is what keeps client compaction out, since that overlaps near 0.01.

Reuse is safe because overlay_cached_prefix re-verifies message by message and
stops at the first real divergence, so a partial match can never replay past it.

Measured effect, A/B over the real resolve_tracker and overlay_cached_prefix on a
synthetic 40 turn trace whose churn parameters come from the corpus
(scripts/project_lineage_fix.py): write span 137,510 before against 59,019 after,
a 57.1 percent reduction and a 2.33x ratio multiplier. That is the fix in
isolation on synthetic traffic, not a live ratio. The corpus cannot give a live
projection because it records only forwarded bodies, and this fix operates on the
relationship between the client's originals and what was forwarded. An earlier
attempt to project from the corpus alone produced 3.2 percent and was discarded
as an artifact of comparing post-transform bytes against post-transform bytes.

## Status

Implemented and tested: anchor stabilization, head-fingerprint bust detection,
marker eviction bound, diff-only re-reads (default off, measured at zero),
partial lineage rematch. ASSISTANT_TEXT_SWEEP_AGE reverted to 3 after the
breaker caught the change.

Not yet active. The live proxy runs from an installed copy under
~/.local/share/uv/tools/headroom-ai/, not this tree, so none of it is in effect
and the replay corpus was produced entirely by the old code. Reinstalling and
restarting the proxy is the only way to get a post-fix number, and it ends any
session currently running through it.
