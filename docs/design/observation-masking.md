# Observation masking design
## Scope
Observation masking replaces old, large Anthropic `tool_result` content with a
small recovery marker. It is a hybrid-mode policy, enabled by default through
`HybridModeConfig.observation_masking`. This note covers design only.
Defaults are `mask_after_turns=3` assistant turns and `mask_min_tokens=400`.
Both conditions must hold. The environment override follows the existing
hybrid policy pattern. `HEADROOM_OBSERVATION_MASKING=0` disables it. There is
no standalone `HR_*` primary feature flag.

Extension 2026-07-17: discovery also walks assistant `tool_use` blocks and
masks a fixed allowlist of bulky input keys (`content`, `file_text`,
`new_string`, `old_string`) under identical age, size, economics, and CCR
rules, with marker prefix `[Tool input masked:`. Motivation: a 15-session
transcript breakdown measured tool_use args at 35% of resent tokens, the
largest single source, previously untouched by any layer. The original scope
followed the masking paper, which only studied tool outputs; inputs were never
deliberately excluded. Defaults were also lowered to `mask_after_turns=2` and
`mask_min_tokens=150` from the same breakdown (size floor is the binding
lever, age is not), overridable via `HR_MASK_AFTER_TURNS` and
`HR_MASK_MIN_TOKENS`.
## Existing machinery and overlap
- `read_lifecycle.py` already maps Anthropic `tool_use` IDs to tool results,
  replaces stale or superseded Reads, stores originals in CCR, and emits the
  load-bearing `Retrieve original: hash=` text. Reuse its metadata and
  best-effort storage patterns, but do not extend its state model. Masking is
  based on age and size for every tool, not file lifecycle evidence.
- `read_maturation.py` holds Reads outside the cache until they mature. Its
  markers and any `Retrieve original: hash=` result are already compact and
  must be treated as already masked. Maturation remains a Read-specific cache
  placement policy.
- `cross_turn_dedup.py` replaces repeated spans with in-context pointers. It is
  prefix-monotonic and content-based. Masking targets a whole old observation,
  so it should run after dedup and measure the remaining content.
- `lossless_compaction.py` only performs format-native reversible rewrites. It
  does not provide durable recovery and should stay independent.
- `compression_store.py` is the existing CCR persistence layer. Use
  `get_compression_store()` and store the original with the tool name, tool use
  ID, an explicit content hash, and strategy `observation_masking`.
- `headroom/ccr/` already expands hash recovery pointers. No new retrieval
protocol is needed.
## Selection and placeholder
Count age by assistant turns after the assistant turn containing the matching
`tool_use`. A result becomes eligible once at least `mask_after_turns`
subsequent assistant turns exist. Derive the tool name from `tool_use_id` when
possible and use `unknown` otherwise.
Use the configured tokenizer for the size threshold. Record both token and
UTF-8 byte counts for accounting and diagnostics. Normalize the first
non-empty content line to one line, escape control characters, and cap the
excerpt to a small fixed character limit.
Proposed marker:
```text
[Tool result masked: tool=Read, tokens=812, bytes=3248, head="first line...". Retrieve original: hash=0123456789abcdef01234567]
```
The marker must be deterministic for the same original block. Preserve all
other block fields. Skip non-string content in the first version. Skip content
that already contains a CCR marker, a read lifecycle marker, a maturation
marker, or the exact `Tool result masked:` prefix. A second pass must return
byte-identical messages.
## Cache gate integration
Masking is a historical mutation. It must not run as an ordinary pipeline
transform with only `frozen_message_count` filtering.
In `headroom/proxy/handlers/anthropic.py`, discover candidates and estimate
their exact post-marker token saving before the `TOKEN_PREFIX_GATE` decision
near the current `net_mutation_gain` call. Add masking `dT` to the mutation
saving considered by that decision. Use the same cached suffix `S`, cadence
forecast `R`, hazard-based `survival_p_alive`, structural-churn alive fraction,
and TTL write multiplier.
Factor the gate inputs and decision into a shared helper rather than copying
the formula. The helper must call `CompressionPolicy.net_mutation_gain`.
Masking may proceed only when one of these facts is true:

1. Client structural churn has already killed every cache byte that the
   candidate mutation would invalidate.
2. The same request is performing an admitted historical compression or
   hybrid rebase that already invalidates that range, with masking `dT`
   included in the admitted gain.
3. A masking-only decision has strictly positive net mutation gain for the
   exact candidate batch.

The pressure override may admit masking only when it admits the same
historical compression or rebase. It must not create a masking-only override.
`compress_latched` is not blanket permission for new masking on later turns.
Each newly eligible frozen result needs a fresh piggyback or gain-positive
decision.

Apply the admitted mask batch to the working message copy after cached
substitutions are resolved and before the main compression pipeline result is
finalized. Update optimized token counts and `transforms_applied`. When the
gate is closed, use the original message objects and bytes for every candidate.
Do not write CCR entries for rejected candidates.

## Files to touch in phase 2

- Add `headroom/transforms/observation_masking.py` for pure discovery,
  estimation, marker creation, and application.
- Update `headroom/proxy/hybrid_mode.py` with the default-on policy field and
  environment override.
- Update `headroom/proxy/handlers/anthropic.py` to share the mutation gate,
  include masking `dT`, and apply only admitted batches.
- Reuse `headroom/transforms/compression_policy.py` without changing its gain
  formula unless extracting a typed gate input removes duplication.
- Add focused transform tests and extend
  `tests/test_proxy_anthropic_cache_stability.py` for gate behavior.

## Test plan

- Unit test age below, at, and above three assistant turns.
- Unit test content below, at, and above 400 tokens.
- Verify tool-name lookup, byte and token sizes, excerpt normalization, CCR
  hash recovery, string-only handling, and storage failure fallback.
- Verify an already masked or existing recovery marker is byte-identical and
  causes no second store write.
- Verify gate closed leaves the full request byte-identical and stores nothing.
- Verify structural-death piggyback and gain-positive masking apply, while
  gain-zero and gain-negative batches do not.
- Verify masking `dT` changes the shared gain result and transform accounting.
- Verify a prior compression latch alone cannot admit a later frozen mutation.
- Keep the full Anthropic cache-stability suite green, including frozen-prefix
  and repeated-request tests.

## Risks

- Wrong assistant-turn attribution can mask a fresh observation.
- Token estimates that differ from the forwarded marker can admit a losing
  mutation. Compute the candidate marker before the gate.
- Applying after the wrong cache substitution step can silently restore old
  bytes or invalidate a warm prefix.
- Unbounded excerpts can erase savings or expose multiline control content.
- CCR store failure removes guaranteed recovery. Fail open by retaining the
  original content when durable storage is required but unavailable.
