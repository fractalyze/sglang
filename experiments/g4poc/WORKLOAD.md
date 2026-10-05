# g4poc workload, decision metrics and gate

Study `g4poc`: lowest cost per 1M tokens (then E2E latency) for serving
`google/gemma-4-26B-A4B-it` (FP8 weights) with SGLang on one RTX 5090, for non-streaming
multi-turn multilingual role-play chat. This page covers the workload generator
(`workload/`), the decision metrics and the adapted gate (`gate/`), the quality guards and
the prefill/decode split harness. The FP8 server itself is PA's (`BASELINE-FP8.md`).

## 1. Workload

### Schema (the customer-replaceable contract)

One session per JSONL line (`workload/schema.py`); text, not token ids, so the server-side
chat template sees what a real client sends and the file serves any tokenizer:

```json
{"session_id": "s000123", "language": "ko", "persona_id": "ko-barista",
 "system": "<persona card + rules>",
 "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "turns": [{"user": "...", "reply": "...", "think_s": 0.0, "max_new_tokens": 300}],
 "source": "allenai/WildChat-1M@7d6490e"}
```

`history` is the conversation before the first timed turn; `turns[k].reply` is the scripted
reply (it enters the history after turn k and sets how many tokens turn k decodes in scripted
mode); `think_s` is the user's delay after the previous reply. The customer's anonymized
samples need only a converter to this schema: system prompt, alternating messages, and the
timestamps of each user message (think time = user send time minus previous reply time).
`schema.validate` refuses anything the replayer cannot use.

### Generator (`python -m workload generate`)

- **Text source:** WildChat-1M (`allenai/WildChat-1M` at `7d6490e`, ODC-BY): real users'
  chats with a per-conversation language label. `build-pool` keeps (user, assistant) pairs of
  non-toxic conversations, chat-shaped only (no code fences, user turns <= 1200 chars), and
  flags role-play conversations by a multilingual keyword match. All 14 shards give 50K pairs
  each for en/zh/es/fr/ru (cap), 10.5K ko, 6.5K ja, 11.8K de; role-play-flagged pairs: 2.0K
  en, 600 zh, 575 ru, 540 ko, 441 es, 162 ja, 100 fr, 65 de.
- **Persona cards:** 16 cards written for this study (`workload/personas.py`), 2 per language
  in en, ko, ja, zh, es, fr, de, ru, each with the language's reply rules. They are the system
  prompt of every session.
- **Session:** language by weight (default en 25%, ko 20%, ja 15%, zh 15%, es/fr/de/ru 6.25%
  each), a card of that language, a first-prompt length target (lognormal, median 4.5K,
  sigma 0.35, clipped 1.5K..9K tokens), and a turn count (1 + exponential, mean 6, max 16).
  History is filled with whole same-language conversations, pair by pair in their order
  (role-play conversations drawn with probability 0.7), until the first prompt reaches the
  target; timed turns continue until the next prompt would pass 10,240 tokens. Scripted replies
  are the WildChat assistant replies cut to <= 300 tokens at a sentence end. Think time:
  lognormal, median 15 s, sigma 0.6, clipped 2..120 s. Token counts use the Gemma-4 tokenizer
  and template overhead measured from it (5 tokens per message, 8 per prompt). Every draw is
  seeded: (pool, config, seed) names the file.

### Measured shape of the default file (2,000 sessions, seed `g4poc-v1`, exact Gemma-4 template)

File sha256 `78b7a04cd50db17b...` (pool `3cdf7944e4b81b4c...`); regenerate with the commands in section 5.

| quantity | p10 | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|---|
| prompt tokens, all turns (n = 10,304) | 3,587 | 5,535 | 8,211 | 9,799 | 10,235 | 5,705 |
| prompt tokens, first turn | 2,995 | 4,602 | 7,157 | 9,165 | 9,791 | 4,865 |
| output tokens (scripted) | 32 | 190 | 294 | 300 | 300 | 177.6 |
| turns per session | 1 | 4 | 12 | 16 | 16 | 5.15 |
| think time, s | 7.0 | 15.1 | 32.3 | 59.8 | 120 | 17.9 |

Languages: en 507, ko 428, zh 316, ja 255, de 133, fr 122, ru 124, es 115.

**Limitations.** The history and user turns are real multilingual chat, but mostly not
role-play: the role-play framing comes from the persona card, and only a minority of pairs
come from role-play conversations (few exist in WildChat for ko/ja/de/fr). Content affects MoE
routing and output lengths, so the customer's samples should replace this file before any
number is quoted as the customer's. Scripted reply lengths follow WildChat's assistant
replies (p50 190, 16% within 10 tokens of the cap, cut at a sentence end), not the deployed
character's.

### Replay (`gate/loadgen.py`)

Non-streaming `/generate` with token ids from the chat template (the Gemma-4 template, applied
client-side, exactly what `/v1/chat/completions` would apply). Two arrival modes:

- **poisson (session layer, fixed offered load):** at t=0, `concurrency` sessions are already
  part-way through (random turn, first sends spread over one think time); then sessions arrive
  as a Poisson process at `concurrency / expected_session_s`. Each turn is sent think time
  after the previous reply. The plan (who starts when, at which turn, with which nonce) is
  drawn from the pair's seed, so both legs of a pair replay the same offered load.
- **slots (in-flight layer):** `concurrency` slots each replay a seeded session sequence back
  to back, a turn sent `think_s x think_scale` after the previous reply (0 = always in flight).

**Modes.** `scripted` (default): the history carries the scripted replies and every request
decodes exactly the scripted reply's length (`ignore_eos`), so both arms of a pair send
identical prompts and do identical work. `closed`: the model's own reply enters the history,
natural EOS, 300-token cap (for realism checks; not for paired timing).

**Nonces are per session**, in the system prompt: turns of a session share its cached prefix
as multi-turn chat does, no two sessions or pairs do (`--nonce-at after_system` lets sessions
of one persona share the card's prefix, as a real fleet would). The prefix cache stays on.

**Template fact that matters for caching.** Gemma-4's generation prompt is
`<|turn>model\n<|channel>thought\n<channel|>`, but a past model turn renders as
`<|turn>model\n<reply><turn|>`. So the next turn's prompt matches the cached sequence only up
to `<|turn>model\n`: the previous reply's KV (<= 300 tokens) and the empty-thought tokens are
recomputed every turn, by any client that sends chat messages. This is the real behaviour;
scripted mode reproduces it exactly (unit-tested with a template of the same shape).

*Caveat, checked with the served tokenizer (2026-10-05):* a client that resends the model's own
reply matches no further than one that sends the scripted reply. The next prompt leaves the
cached sequence at the same token, right after `<|turn>model\n`, because the empty
`<|channel>thought\n<channel|>` of the generation prompt never appears in a rendered past turn.
So scripted mode does not understate the hit rate of real chat clients. A lever that frees the
KV just before that boundary (e.g. the last sliding window of the previous prompt) loses those
hits for real clients too. Only a client that resends raw token ids, generation-prompt tokens
included, could match through its previous reply (<= 300 tokens more per turn).

## 2. Decision metrics (`gate/metrics.py`)

Fixed with the user on 2026-10-04 (coordinator message "Decision metric fixed"):

- **Primary:** the highest in-flight request concurrency per GPU whose E2E p90 meets the SLO,
  and the goodput at that point. The customer has no SLO yet, so every sweep reports the
  capacity point at E2E p90 <= **6, 10 and 15 s** (`config.SLOS_E2E_P90_S`; default 10 s).
- **Two layers, always separated:** (a) in-flight requests: time-average number of requests
  between send and reply in the window (`inflight_mean`, plus the peak); (b) sessions:
  time-average live sessions, idle think time included (`sessions_active_mean`). Where idle
  sessions' history lives shows up as the prefix-cache hit rate.
- **E2E latency:** send to full reply, p50/p90/p99 over requests sent in the window; implied
  per-request decode rate = output tokens / E2E (p50, p10).
- **Goodput:** output and total (prompt + output) tokens of requests finished in the window,
  per second, per GPU, at the capacity point; a point with any failed request never qualifies.
- **Cost, both bases:** $/1M output tokens and $/1M total tokens =
  GPU $/hr / (tok/s x 3600) x 1e6, at **$0.40, $0.70, $1.00, $1.50 per GPU-hour**
  (`config.GPU_PRICES_USD_PER_HR`). No single price is presented as fact.
- **Prefix-cache hit rate:** cached / prompt tokens of the window's requests (per-request
  `cached_tokens`).
- **KV retractions:** `sglang:num_retracted_requests_total` over the window (every gate server
  runs with `--enable-metrics`), else the scheduler's "Retract requests" log lines. Also
  sampled every 2 s: running/queued requests, full and SWA token usage.

## 3. Adapted gate (`gate/`)

Copied from `experiments/gemma4-nvfp4-5090/gate` (unchanged: `server.py`, `hostwatch.py`,
`gpu.py`, `treehash.py`, `fidelity.py`, `quality.py`, `client.py`);
rewritten for this workload: `config.py`, `loadgen.py`, `metrics.py`, `stats.py`,
`runner.py`, `checkpoint.py`, `rpquality.py`, `pd.py`, `__main__.py`.

| aspect | gemma4nv gate | g4poc gate |
|---|---|---|
| prefix cache | flushed before every rep | **on**; flushed only after warm-up, so each leg starts cold; hit rate reported per leg and pooled |
| nonces | fresh prompts per pair/rep | **per session** (turns share their session's prefix) |
| timed unit | fixed batches (W8/W1/W32) | **fixed offered load**: the pair's seeded session plan (`L64`: 64 concurrent sessions, 120 s warm-up, 480 s window) |
| ABBA | yes, >= 4 pairs, A/A noise bar | same; bar = max(3 sigma_AA, 1%) |
| deciding metric | W8 composite | `e2e_p90_gain` (control p90 / candidate p90); guards `e2e_p50_gain`, `output_tput_gain`; `e2e_p99_gain` reported |
| eager decode step | refuses the leg | reported (the running batch legitimately exceeds the captured graph sizes) |
| new hard checks | - | no failed or abandoned request; client lag p99 <= 1 s (the client, not the server, would have shaped the load); same plan within each pair |
| timed output agreement | per stream | per (session, turn); hard only for scripted A/A or `numerics_unchanged` |
| checkpoint | bytes == HF LFS hashes | bytes == the pin written by `gate pin-checkpoint` (the served FP8 files are derived, so no Hub hash exists) |
| host safety, quiescence, telemetry, weights checksum, server-arg diff, fidelity (KL vs pinned reference) | | unchanged; same host.lock as gemma4nv |

The gated load must sit near the base's capacity knee to see capacity levers (a KV strategy
that fits more requests changes latency only when the base is close to its limit): re-pin
`L64` from the base's sweep before the first A/A (a gate change).

**Quality guards.**
- GSM8K + tool-call JSON (`gate quality`, `quality-compare`): gemma4nv's sets and rules.
- Multilingual role-play (`gate/rpquality.py`): 80 items = first turns of 10 sessions per
  language (persona + ~5K history + user turn), so the guard sees the long contexts KV changes
  touch. `rp-quality --set-baseline` fixes the items and the base's greedy replies.
  *Reference consistency:* the candidate's teacher-forced NLL per token on the base's replies
  may rise at most 0.02 nats/token, and its own replies must stay in the persona's language at
  least as often (script + stopword detector). *Pairwise judge* (`rp-judge`): an LLM judge
  (any OpenAI-compatible endpoint, or `--judge-ref` to launch one) compares base and candidate
  in both orders; an item counts only when both orders agree; the candidate fails when the
  95% CI lower bound of its net loss rate exceeds 5 points. Both thresholds are set before
  any trial and are arbitrary.

## 4. Prefill/decode split (`gate/pd.py`)

- `gate pd-measure`: prefill-only-like points (uncached prompts of 5,120 and 10,240 tokens,
  `max_new_tokens=1`, 1/2/4/8 in flight: prefill tok/s and TTFT) and decode-only-like points
  (a batch prefilled first, then 300 tokens each on the cached prefixes, batch 8..64: decode
  tok/s and TPOT; best = highest throughput with TPOT p90 <= 30 ms and the prefix still cached).
- `gate pd-model`: per request, prefill GPU-s = P(1-h)/R_p and decode GPU-s = O/R_d (P, h, O
  from the colocated sweep's capacity point at the chosen SLO); P:D GPU ratio = their quotient;
  disaggregated output tok/s per GPU vs the colocated capacity point; $/1M output tokens at
  every price; fleet size for 2,200 concurrent sessions. KV transfer per request = all full-
  layer KV (10,240 B/token, FP8) + the sliding window's last 1,024 tokens (102,400 B/token):
  ~156 MB for a 5K prompt. Its link cost is a parameter (`--link-gbps`) until the bs2<->bs3 link
  is measured; it is reported as latency and a per-link request ceiling, not charged GPU time.
  The model assumes the prefill side keeps the cross-turn prefix cache (hit rate h).

## 5. How to run

```
# local or any host with the tokenizer: build the session file
python -m workload fetch --out-dir <dir>            # 14 WildChat shards, 3 GB
python -m workload build-pool --parquet <dir>/*.parquet --out pool.jsonl
python -m workload generate --pool pool.jsonl --tokenizer <gemma-4 tokenizer dir> --out sessions.jsonl
python -m workload stats --sessions sessions.jsonl --tokenizer <dir>

# on bs2 (deploy to its own dir; deploy.sh rsyncs with --delete)
gate/deploy.sh build-server-2 /data/jooman/g4poc/harness-pb
G4POC_MODEL_DIR=<FP8 text model dir> gate/run.sh gate pin-checkpoint --source "<converter commit>"
gate/run.sh gate smoke --ref base
gate/run.sh gate sweep --ref base --load inflight --concurrency 4,8,16,24,32,48
gate/run.sh gate sweep --ref base --load L64 --concurrency 32,64,128,256
gate/run.sh gate run --control base --candidate base --pairs 4    # A/A, then set-noise
gate/run.sh gate pd-measure --ref base; gate/run.sh gate pd-model --pd .. --sweep ..
```

Tests: `python gate/tests/test_g4poc.py` (70, no GPU: a fake `/generate` server with a prefix
cache checks the replay end to end, including the cross-turn hit and per-session nonces).

## 6. Smoke runs

**Harness smoke (NVFP4, smoke-only, not a result).** Coordinator-approved while the FP8 base
was not up: bs2, ref `nvfp4-smoke` (gemma4nv's NVFP4 base), `gate smoke` (L8-smoke: 8
concurrent sessions, think time x0.2, 20 s warm-up, 60 s window), run
`smoke-nvfp4-smoke-20261004-145317-build-server-2-b5f5cc`. The whole path worked: capped
launch under host.lock, warm-up replay, flush, timed replay, Prometheus sampling, report.
31 window requests, 0 failed, client lag p99 14 ms, mean prompt 5,102 tokens, mean output 210,
hit rate 0.49 (short window, many sessions' first turn), 0 retractions, 0 eager decode steps,
`sglang:evicted_tokens_total` +68K tokens in the window at full-pool usage <= 0.49 (to explain
on the FP8 base: likely the SWA pool's eviction). Host: peak tree RSS 14.0 GB at weight load,
6.6 GB serving, MemAvailable >= 49.7 GB, load1 < 1. It also showed SGLang emits the labelled
retraction counter only after its first increment; an absent counter now reads as 0.

**FP8 base smoke:** pending PA's pinned ref (`BASELINE-FP8.md`: model dir, flags, commit).
