# Codex stack canary, 2026-07-13

This canary compares three subscription-authenticated Codex configurations on
the same three-turn repository investigation:

- `bare`: direct OpenAI, with user config, rules, hooks, MCP servers, skills,
  global instructions, and Headroom disabled.
- `headroom`: Headroom hybrid proxy and output shaping, with the other user
  tooling disabled.
- `full`: the installed Codex configuration with Headroom, TokenSave, Serena,
  Context7, hooks, skills, global instructions, and RTK policy available.

The task asks the model to inspect the OpenAI context-limit selector, inspect
the hybrid-mode state machine, and return nine exact facts as JSON. Each run is
a fresh thread. Arm order is randomized. Codex reports cumulative thread usage
on resumed turns, so totals use the final cumulative counters rather than the
sum of each reported turn.

## Sources and cost method

API-equivalent costs use OpenAI standard-processing prices current on
2026-07-13:

| Model | Input / 1M | Cached input / 1M | Output / 1M |
|---|---:|---:|---:|
| GPT-5.6 Sol | $5.00 | $0.50 | $30.00 |
| GPT-5.6 Terra | $2.50 | $0.25 | $15.00 |
| GPT-5.6 Luna | $1.00 | $0.10 | $6.00 |

Source: [OpenAI API pricing](https://platform.openai.com/docs/pricing).

Subscription utilization is reported separately. OpenAI documents rolling
message ranges and shared limits for Plus, not a fixed token or credit pool.
Source: [Codex pricing](https://learn.chatgpt.com/docs/pricing).

## Luna screening

Two balanced repeats, six threads and eighteen messages total:

| Arm | Accuracy | Mean input | Mean cached input | Mean output | Mean latency | Mean API equivalent |
|---|---:|---:|---:|---:|---:|---:|
| Bare | 83.3% | 248,425 | 201,088 | 2,228 | 57.6 s | $0.080811 |
| Headroom | 94.4% | 279,853 | 176,896 | 2,404 | 71.7 s | $0.135071 |
| Full | 88.9% | 296,398 | 232,192 | 2,696 | 99.2 s | $0.103598 |

Compared with bare, Headroom-only used 12.7% more input and 67.1% more
API-equivalent cost. Full used 19.3% more input and 28.2% more cost. The sample
is small and Headroom-only had high cache variance across its two repeats.

Weekly subscription utilization moved from 76% to 79% over the whole Luna
matrix. The service cannot attribute that shared window movement to individual
arms.

## Sol confirmation

One balanced repeat, three threads and nine messages total:

| Arm | Accuracy | Input | Cached input | Output | Latency | API equivalent |
|---|---:|---:|---:|---:|---:|---:|
| Bare | 100% | 177,956 | 129,280 | 1,974 | 54.4 s | $0.367240 |
| Headroom | 100% | 177,897 | 93,184 | 1,634 | 52.7 s | $0.519177 |
| Full | 88.9% | 425,012 | 315,392 | 3,889 | 138.7 s | $0.822466 |

Headroom-only was token-neutral and 3.2% faster than bare, but a lower cache
share made its API-equivalent cost 41.4% higher. Full used 138.8% more input,
124.0% more API-equivalent cost, and 155.0% more latency.

The Headroom project ledger recorded 27,570 proxy tokens removed from the
Headroom-only arm and 46,827 from the full arm. Those proxy savings did not
offset the full arm's extra model sampling steps.

Weekly subscription utilization moved from 80% to 82% over the whole Sol
matrix.

## Tool trace and cause

Sol bare used four shell calls. Headroom-only used five shell calls. Full used
four shell calls plus eleven TokenSave MCP calls:

| Full-stack TokenSave tool | Calls |
|---|---:|
| `tokensave_status` | 2 |
| `tokensave_context` | 3 |
| `tokensave_read` | 4 |
| `tokensave_body` | 2 |

Serena and Context7 were available but not called. Full also queried the
TokenSave SQLite schema directly before making another context call.

The full arm's serialized tool events were smaller than bare's, but it made
roughly three times as many tool calls. Each tool call creates another model
sampling step that resends accumulated context. Per-turn input deltas show the
effect:

| Turn | Bare | Headroom | Full | Bare tools | Headroom tools | Full tools |
|---|---:|---:|---:|---:|---:|---:|
| Context-limit investigation | 58,924 | 68,460 | 142,353 | 2 | 3 | 6 |
| Hybrid-mode investigation | 85,966 | 80,252 | 250,172 | 2 | 2 | 9 |
| JSON answer, no tools | 33,066 | 29,185 | 32,487 | 0 | 0 | 0 |

The no-tool turn is nearly identical across the three arms. Tool-loop
amplification, not a fixed Serena startup prompt, caused most of the full-stack
overhead.

## Policy implication

TokenSave remains useful for open-ended semantic discovery, but repeated status,
context, read, body, SQLite, and shell passes are counterproductive on a small,
known-file question. The client policy should:

1. Check repository status once per session or index change.
2. Use at most one semantic context query per investigation phase.
3. Use at most two exact TokenSave source fetches before handing off.
4. Avoid `read` and `body` calls for the same symbol or file.
5. Use Serena for exact definition and reference confirmation.
6. Use direct reads immediately when the prompt names a small set of files.
7. Query SQLite only for structural questions unsupported by MCP tools.

## Optimized full-stack screen

After installing the measured TokenSave policy and a Codex `PreToolUse`
exploration budget of two calls per turn, one clean-checkout Luna screen gave:

| Arm | Accuracy | Input | Cached input | Output | Latency | API equivalent |
|---|---:|---:|---:|---:|---:|---:|
| Optimized full | 100% | 141,285 | 77,312 | 2,254 | 80.3 s | $0.085228 |

This is 49.5% fewer input tokens and 36.9% lower API-equivalent cost than the
earlier Headroom-only Luna mean. Against bare Luna, it is 43.1% fewer input
tokens with 16.7 percentage points higher accuracy. Raw API-equivalent cost is
5.5% above bare because the optimized run had a smaller cached share.
Accuracy-adjusted cost is about 12% lower than bare.

The six tool calls were exactly two per turn. The first turn used Serena
activation and instructions. The other turns each used two bounded shell
reads. Headroom removed another 149,433 proxy tokens during the screen. Weekly
subscription utilization reached 90%, so no broad rerun was made.

The canary now links the real `hooks.json` into its full-stack temporary home
and passes `--dangerously-bypass-hook-trust` only for this reviewed automation.
Production hooks keep Codex's normal trust behavior.

Raw local reports were written to `/tmp/codex-stack-luna.json` and
`/tmp/codex-stack-sol.json`. They are not committed because they contain local
absolute paths and verbose tool traces.
