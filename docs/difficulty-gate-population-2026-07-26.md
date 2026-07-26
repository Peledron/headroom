# Difficulty gate population, measured 2026-07-26

Status: answered, and the gate was rebuilt on the result. The cheap-model
route pays on cold prefixes only, and no tuning of the difficulty weights
changes that. Separately, the signals themselves were wrong: measured against
the work that actually followed each turn, the easy label was anti-correlated
with effort. That part is fixed.

## The question

`estimate_difficulty` in `headroom/proxy/model_pricing.py` scores a turn from
local signals, and `price_model_switch` decides whether routing that turn to a
cheaper model repays the prefix rewrites the switch forces. The weights were
written as a starting point and never measured, so the operator question stayed
open: on real traffic, how many turns per week actually route cheap?

The original plan was a shadow logger, accumulating live question text over a
week. That was the wrong instrument. It writes new prompt text to disk to build
a population that already exists on disk, in local Claude Code and Codex
transcripts. `scripts/replay_difficulty.py` replays the estimator over those
instead and writes nothing.

## Corpus

2346 transcript files, 26127 turns, spanning 4.3 weeks, about 6100 turns per
week. Context size is read from each provider's own usage block rather than a
local tokenizer, so the result is independent of how headroom counts tokens.

## What the gate admits

Numbers in this section and the two after it describe the gate **as it stood
before the rebuild**, since they are what motivated it. The section "The
signals were measuring the wrong thing" below carries the post-rebuild figures.

| | turns | share |
|---|---|---|
| all turns | 26127 | |
| tool continuations | 22567 | 86.4% |
| fresh asks | 3560 | 13.6% |
| easy at threshold 0.35 | 362 | 1.39% of all, 10.2% of fresh |

85 easy turns per week. Every one is a fresh ask, and none is a continuation,
which follows from the arithmetic: a continuation adds 0.3 to a 0.5 base, and
0.8 cannot come back under 0.35.

`quick_question` fired exactly 362 times, matching the easy count exactly. That
is not a coincidence, it is the only path to easy. The base is 0.5 and the sole
negative signal is `quick_question` at −0.35. Every other signal adds. So the
gate reduces to one regex, and `UNKNOWN_TURN_SCORE` plus `DEFAULT_EASY_THRESHOLD`
decide the rest before any weight is consulted.

## Why it cannot pay on a warm prefix

Easy turns do not cluster. Measured inside a session, in order:

| run length | occurrences | turns |
|---|---|---|
| 1 | 299 | 299 |
| 2 | 24 | 48 |
| 3 | 5 | 15 |

91.2% of runs are a single isolated turn. The longest run in 4.3 weeks is 3, and
the mean is 1.10. So the router is asked to switch models and switch back to buy
one cheap turn, paying two prefix rewrites for it.

Feeding the measured prefix sizes and run lengths through `price_model_switch`,
with a generous 200 new input tokens and 500 output tokens per turn:

| cheap price ratio | would switch | reason | break-even turns (median / p90) |
|---|---|---|---|
| 0.2 | 78 of 362 | all `free_cold_prefix` | 16.7 / 17.8 |
| 0.08 | 78 of 362 | all `free_cold_prefix` | 13.1 / 13.9 |

The 284 warm-prefix turns are refused as `horizon_too_short` at both ratios.
Break-even needs 13 to 18 consecutive easy turns and the traffic never supplies
more than 3, so a twelve-fold price gap changes the decision on exactly zero
turns. Every switch the router does make is one where `prefix_tokens` is zero
and the switch is therefore free.

The router is already behaving correctly. The limit is the shape of the traffic,
not the scoring.

## Conclusions

Operator decision, 2026-07-26: **leave the cheap-model route off.** It is off
by default and off in this deployment, and it stays that way. The measured
upside is 78 free cold-prefix switches, about 18 per week, against a residual
tail where a turn admitted as easy still ran 147 follow-on assistant turns.
That trade is not worth taking while the tail is unbounded, so the route is
not enabled and the sections below describe what it would do if it were.

Both locks are independent and both are currently open circuits:
`HEADROOM_MODEL_ROUTER_ENABLED` is unset, so `ModelRouterConfig.enabled` stays
False and no rule fires, and `HEADROOM_MODEL_ROUTE_PRICES` is unset, so
`_price_model_route` returns None before reading anything. Enabling the router
without prices self-disables with a warning.

The 78 cold-prefix switches, about 18 per week, would be free by construction
if the route were turned on.

Do not tune the difficulty weights for savings, and do not A/B the warm-prefix
route. The gap between a 3-turn horizon and a 13-turn break-even is too large
for weight changes to close. Anything that widens the easy population without
changing the clustering makes the router refuse more turns, not fewer.

Do not spend on a better estimator to save money either. A gate that admitted
every fresh ask, ten times the current population, would still face the same
break-even and the same isolated runs.

The weights were rebuilt anyway, for correctness rather than savings. See
below: they were admitting the most expensive traffic in the corpus, which is
a live hazard the moment anything else makes the warm-prefix route pay.

## One fragility worth pinning

204 of the 362 easy turns, 56%, score **exactly** 0.35, which equals
`DEFAULT_EASY_THRESHOLD`, and are admitted only because `is_easy` compares with
`<=`. They are quick questions in a large context: 0.5 − 0.35 + 0.2. The other
158 sit at 0.15.

So the population is bimodal with nothing in between, and more than half of it
balances on an exact tie. Lowering the threshold by any amount, or raising the
`large_context` weight by any amount, silently removes 56% of the easy turns.
Given the conclusion above this changes no decision today, but it should be
pinned by a test so the tie is a choice on record rather than an accident.

## The signals were measuring the wrong thing

Everything above prices the route. It says nothing about whether the turns the
gate calls easy are easy, and they are not.

Each fresh ask can be scored against what followed it, using the transcript's
own record: output tokens on the reply, and how many further assistant turns
ran before the next typed prompt. Over 2713 fresh asks that produced work, the
median is 8 follow-on assistant turns.

Against that, easy turns run a **median 4462 output tokens to a hard turn's
1418**. The label is inverted. What the gate admits are short conversational
steering messages sent in the middle of deep work, and those are the single
most expensive kind of turn to route to a cheap model.

### The hand-picked words did not survive contact with the corpus

`_HARD_WORDS` held twelve words chosen by hand. Ranked by the median follow-on
work of the turns containing them:

| word | n | lift | verdict |
|---|---|---|---|
| deadlock | 18 | 9.25x | earns its weight |
| prove | 24 | 1.62x | earns its weight |
| why, design, refactor, debug, derive, root cause | 6 words | 0.88x to 1.25x | no signal |
| architect, race, trade-off, explain | 4 words | 0.25x to 0.62x | fires on cheap turns |

Ten of twelve did not predict effort and four predicted the opposite.
`_QUICK_QUESTION`, the sole path to easy, separated 6 median turns from 8. A
0.75x lift was carrying a −0.35 swing.

### Deriving a replacement, and why it is short

Ranking all vocabulary by follow-on work, then splitting the corpus in half
and requiring a word to hold its lift on the half it was not derived from: 44
words cleared 1.8x on the derive half and **16 replicated**. Two-thirds of
word-level lift at these sample sizes is chance. Any list built without the
replication step would have been two-thirds noise.

What replicated is not reasoning vocabulary. It is the language of opening an
unattended run: `acknowledge`, `condition`, `reaches`, `untill`, `stop`,
`wait`, `breakers`, `recovery`, `measure`, plus bare assent, `yes` and `sure`.

### The strongest signal, and the confound underneath it

| feature | n | lift |
|---|---|---|
| assent opener (`yes`, `well`, `ok`, `good`, `sure`) | 361 | **2.00x** |
| run-control words | 387 | 1.38x |
| code fence | 127 | 1.12x |
| old `_QUICK_QUESTION` | 456 | 0.75x |

Assent openers lead everything, and shortness adds nothing to them: the lift
is 2.00x with or without a 200-character filter.

The obvious objection is that words like `continue` appear when a model has
stalled in an already-long conversation, so they may be measuring context size
rather than difficulty. Stratifying by prefix size splits the signal in two
and shows the objection is right about half of it:

| context band | `continue`-type | `yes`-type | everything else |
|---|---|---|---|
| under 50k | **1** (n=43) | **40** (n=54) | 6 (n=705) |
| 50k to 100k | 32 (n=21) | 15 (n=25) | 6 (n=310) |
| over 100k | 19 (n=63) | 14 (n=157) | 8 (n=1335) |

Resumption words are cheap in a small context, the cheapest shape in the
corpus at a median of 1 follow-on turn, and only expensive where
`large_context` already charges for them. So they are **not** scored as
difficulty, since that would count the same evidence twice.

Assent is not confounded. It leads in every band and is strongest where
`large_context` never fires, at 40 median turns against 6. That is the shape of
approving a plan and launching its whole implementation.

### What shipped

`_HARD_WORDS` cut to the two measured words. `_STEERING_ASSENT` added at +0.4,
which also vetoes `quick_question` so no routing log can call a steering
message a quick question. `_RUN_CONTROL` added at +0.15, a small weight for a
1.38x lift. Resumption words deliberately absent. Each weight is pinned by a
test naming its measured lift.

Replaying the rebuilt gate over the same corpus:

| | n | p50 turns | p90 turns | p99 turns | p90 output |
|---|---|---|---|---|---|
| easy, old gate | 357 | 5 | 30 | 140 | 37734 |
| easy, new gate | 366 | 5 | 29 | 113 | 31570 |
| dropped by the rebuild | 32 | 13 | 85 | 158 | 114134 |
| newly admitted | 41 | 6 | 37 | 101 | 38514 |

The population is the same size, 86 easy turns per week against 85. The trade
is favourable: the 32 turns dropped carry more than twice the median effort of
the 41 admitted, and three times their p90 output.

### What this does not fix

The tail survives. p99 follow-on turns fell from 140 to 113 and the worst
admitted turn still ran 147 turns of work:

```
'what do you mean draining? also what about the other data sourcces...'
```

That is a genuine question, with no assent opener and no hard word, that
happened to open an enormous piece of work. No signal available locally sees
it coming, because the cost lives in the reply rather than the ask. Widening
the word lists will not reach it, and the replication result says most words
that look like they would are noise.

The practical consequence is bounded, though. Nothing above changes a routing
decision today: break-even still needs 13 to 18 consecutive easy turns against
a longest measured run of 3, so all switches remain `free_cold_prefix`. This
was worth doing as de-risking rather than saving. The gate's easy population
was precisely the most expensive traffic, and anything that later makes the
warm-prefix route viable would have routed the worst possible turns cheap.

## Reproducing

```
.venv/bin/python scripts/replay_difficulty.py
.venv/bin/python scripts/replay_difficulty.py --threshold 0.5 --include-sidechains
```

Note that `tool_count` cannot be recovered from a transcript. The estimator
leaves it unscored on purpose, so the replay passes zero and no score is
affected.
