# Experimental optical text compression

Headroom can render eligible immutable prose tool output as deterministic PNG pages. This
path is experimental, disabled by default, and currently wired only for Anthropic traffic in
hybrid mode.

```bash
HEADROOM_MODE=hybrid HEADROOM_OPTICAL_TEXT=1 headroom proxy
```

`HEADROOM_OPTICAL_CACHE_DIR` changes the content-addressed page cache. A warm provider prefix
never renders a missing page. Rendering occurs only during a cold prefix build or an approved
hybrid rebase. A rejected provider request is retried with the original pre-optical body.

## Safety policy

The renderer rejects system content, code, patches, tool arguments, current output, structured
JSON, identifier-heavy data, small inputs, and other content whose exact structure matters.
It also compares estimated image cost with ordinary extractive text compression and keeps text
when text is cheaper.

Images are not byte-safe. Precision-critical identifiers are therefore extracted into a
deterministic adjacent text sidecar. The sidecar is included in token-cost decisions. Recent
content remains native text.

Cost gates use current upstream formulas by exact provider family. Anthropic uses 28 by 28
visual patches plus its standard and high-resolution resize tiers. OpenAI uses its published
tile bases for GPT-4o, GPT-4.1, GPT-4o mini, GPT-5, o-series, and computer-use models, plus
published patch budgets and multipliers for GPT-5.x mini, nano, Codex, and GPT-4.1 mini and
nano families. Unknown OpenAI models fail closed and remain text.

OpenAI adapters exist, but the Codex path remains disabled until a live Responses protocol
canary proves that the required image parts are accepted. OpenAI server-side cached text can
already cover nearly the full prompt, leaving little uncached input for optical compression to
improve.

## Accuracy canary

The canary compares raw text, extractive text, optical text, and sequential extractive-plus-
optical text. The last condition is not the proxy hybrid state machine. It is labeled
`text_optical` to prevent those concepts from being conflated.

Dry run, with no paid calls:

```bash
python -m benchmarks.optical_accuracy_canary --provider openai --model gpt-4o-mini
python -m benchmarks.optical_accuracy_canary --provider anthropic --model claude-haiku-4-5
```

Live run:

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" \
  python -m benchmarks.optical_accuracy_canary --provider openai --model gpt-4o-mini --live
```

Use an environment variable or secret manager for credentials. Never put a key in a command
history entry or repository file.

The CLI relaunches its worker with inherited `HEADROOM_*` variables removed and these explicit
controls: token mode, optical disabled at the proxy level, adaptive TTL disabled, output shaping
disabled, net-cost policy disabled, and zero output holdout. It prints both inherited and worker
environments with each result. It reports deterministic field accuracy, baseline retention,
latency, protocol errors, and estimated input savings. The default six-case suite covers
structured summarization, early, middle, and late exact lookup, state transitions, and a
never-stated probe for silent confabulation. Use `--max-cases` only for a cheaper smoke run.
Live runs also sum the input tokens returned by the upstream provider and report actual savings
against the raw arm. Published formulas remain the preflight gate. Provider usage is the final
billing evidence.

## Evidence and limitations

The design was informed by:

- OpenAI image-input token calculation guidance
- George Mandis's article on time-based OpenAI image processing costs
- Sean Goedecke's text-as-image-token analysis
- ThePrimeagen's video experiment and transcript: <https://www.youtube.com/watch?v=Bbt8cEyzsTk>
- pxpipe's renderer and published evaluations: <https://github.com/teamchong/pxpipe>

The video experiment reported similar aggregate game outcomes but roughly twice the time per
game and only 23 completed image-context games versus 50 text-context games. That result does
not establish parity.

pxpipe reports strong results on its default Fable model, but also reports silent exact-string
errors on other models. Its published dense 12-character hex recall is 13/15 for Fable and 0/15
for Opus, Sol, and Grok in the cited runs. It also reports that warm Codex Responses prompts can
be about 98 percent cached, producing only about 1 percent static-slab savings when history does
not collapse. These are upstream results, not Headroom results.

Headroom must therefore keep optical conversion off by default until its own live canary shows
acceptable correctness, exact lookup, confabulation, latency, and cost for each exact model and
render profile.

With the current Pillow renderer and corrected July 2026 upstream accounting, `gpt-4o-mini`
optical conversion does not pass the profitability gate. Its published low-detail base is 2,833
tokens per image. Any earlier Headroom result that used an 85-token base for that model is
invalid. The dry canary currently accepts raw optical for `gpt-5.4-mini` at about 16 percent
estimated input savings on its synthetic corpus, while ordinary text compression saves about
81 percent. Accuracy remains unmeasured until the live canary runs.
