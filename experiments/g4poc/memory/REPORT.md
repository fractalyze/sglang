# g4poc memory levers: more concurrent sessions per RTX 5090 (PC, build-server-3)

Scope: memory levers L4-L11 from `BASELINE-FP8.md` section 6, applied to the FP8 baseline of
`google/gemma-4-26B-A4B-it` on build-server-3's RTX 5090. Each lever was preregistered in
`memory/PREREG.md` (and as a vault trial `g4poc-l*` / `g4poc-s1`) before it ran. Code:
`memory/capacity.py` (step-0 tool), `memory/refs.json` (one ref per lever). Raw records:
`memory/runs/<run>/` (`capacity.json` with every burst, plus the server's memory log lines).

Unit note: SGLang logs "GB" for bytes / 2^30; this file writes GiB for those. MiB figures are
nvidia-smi's.

## 0. Summary

**Final config: `final-hc`** (one RTX 5090, FP8 checkpoint; SGLang tree `a0491db764`, branch
`jumanzii/g4poc-final-hicache` = `53752c62aa` + PB's C2-A tile commits + PC3's two HiCache fixes):

```
SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_MOE_CONFIG_DIR=<C1 tuned fused-MoE config dir> SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES=1 \
SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW=1 SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH=1 \
python -m sglang.launch_server --model-path <gemma-4-26B-A4B-it FP8 text checkpoint> \
  --kv-cache-dtype fp8_e4m3 --context-length 16384 --disable-prefill-cuda-graph \
  --json-model-override-args '{"max_position_embeddings": 16384}' \
  --max-running-requests 64 --mem-fraction-static 0.955 --swa-full-tokens-ratio 0.268 \
  --cuda-graph-max-bs-decode 48 \
  --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through \
  --hicache-io-backend kernel --hicache-mem-layout page_first
```

Host RAM: the server pins a 12 GB host pool and peaks at ~18 GB RSS. SGLang's start check
(`mem_cache/pool_host/base.py`) also wants 10 GiB of cgroup headroom beyond each pool it pins, and counts the
cgroup's whole usage at that moment, including file cache charged to it. That usage varied from 4.7 to 10 GB
between identical launches with the weights already cached, and reached ~24 GB when another job's memory pressure
had evicted the weights, so the server re-read all ~25 GB of shards inside its own cgroup (headroom 4.0 GiB). In a
28 GB cgroup (this study's host-safety ceiling) 3 of 9 final-config starts failed with "Not enough host memory
available", and the passing ones cleared it by 0.3-3.6 GB. In deployment: read the weight files outside the
server's cgroup just before launch (page cache stays charged to whoever read it first), give the cgroup >= 32 GB,
and retry the start on that error (the study's queues do all three that the host allows).

| | mem-base | mem-final (memory levers) | + compute levers (`final-mem-c1-c2a`) | + fixed HiCache (**`final-hc`**) |
|---|---|---|---|---|
| burst capacity (5K in / 300 out, no shared prefix) | 17 | 29 | - | - |
| multi-turn in-flight capacity at E2E p90 <= 10 s (240 s sweep points) | 12 | 20 | 24 | 32 (p90 9.55 s) |
| **operating point, sustained 30 min** | - | - | - | **C28: p90 8.92 s, 0 failures** |
| output tok/s at the operating point | 558 | 677 | 833 | **919** (+65%) |
| $ per 1M output tokens at $0.70/GPU-h | 0.348 | 0.287 (-18%) | 0.234 (-33%) | **0.212 (-39%)** |
| multi-turn exactness at concurrency 1 (12 later turns) | - | - | 12/12 control | 12/12 identical to control |
| quality vs mem-base (GSM8K 1319, tool-JSON, role-play NLL + language) | 96.21 | 95.91, pass | 96.44, pass | 96.36; language: no regression shown (section 5) |

All on build-server-3; capacity points replicated on the same host. **final-hc headline: C28 sustained for 30
minutes at E2E p90 8.92 s, 919 output tok/s, $0.212 per 1M output tokens** (9,353 requests, 0 failed, no memory
creep). C32 meets the 10 s SLO only at the edge (30-minute p90 9.92 s, 899 tok/s), so C28 is the operating point.
final-hc's sweep points are PC3's runs (`hicache/PREREG.md` HC4); the soaks and its quality anchor are in section 5. Server GPU memory
stays under the 31,642 MiB rule (512 MiB below what CUDA can use) in every config; on a GPU shared with another
job, use `--mem-fraction-static 0.94` (section 3).

What each part does:
- **Memory levers (mem-final):** L9 sizes the RoPE tables to the served context, L10 shrinks the request table, the
  freed memory and the activation slack go to the KV pool (0.955), the pool split is refit (0.268), L8 stores the
  tied embedding / LM head once in FP8, expandable segments keep the allocator inside the memory rule.
- **Compute levers (PB, `COMPUTE.md`):** C1 tuned fused-MoE config, C2-A sm120 FP8-KV extend-attention tiles;
  they shorten every turn, so one more step of concurrency fits under the SLO.
- **HiCache with two SGLang fixes (PC3, `hicache/UPSTREAM.md`):** a 12 GB host-RAM copy of evicted prefixes keeps
  the hit rate at ~0.71 past the device pool's cliff. Stock HiCache crashed the scheduler at C28 (admission
  under-reserves sliding-window slots on a load-back) and loaded back stale KV (write-through raced the overlap
  scheduler's forward); both are fixed behind default-off switches, with regression tests.

Not kept: L5 (sliding-window release; it frees the very window the next turn resumes from, for chat clients too),
chunked prefill 2048 (-2 sessions), a running-request cap (no effect on the cache cliff). L6 and L11 were
analysed, not built.

The cliff past the capacity point is the prefix cache: once more sessions' histories compete for the pool than it
holds, every turn re-prefills ~5K tokens and goodput falls. **With realistic think time that is the normal regime
(section 3c, independent sessions): one GPU serves ~70 chat sessions at a 30 s mean think time and ~113-119 at
60 s, at a 10 s p90; at the T30 edge that is ~430 output tok/s, about $0.45 per 1M output tokens at $0.70/GPU-h.**
Use sessions per GPU, not the zero-think throughput ceiling, to size the fleet. For chat with think time >= 30 s and
~12 GB of host RAM per GPU, deploy device-only `final-mem-c1-c2a` (HiCache gains nothing there); HiCache and the two
in-flight flags pay off for in-flight-heavy traffic.

## 1. Result in one table

Control `mem-base` = PA's r03 (`--mem-fraction-static 0.93 --swa-full-tokens-ratio 0.3
--disable-prefill-cuda-graph`, FP8 weights and KV, Triton attention, context 16384).
Step-0 metric: the largest N simultaneous 5,000-token random prompts (no shared prefix) decoding
exactly 300 tokens each that all run in one decode batch with no retraction.

| ref | change vs mem-base | weights GiB | full / sliding pool (tokens) | free after graph capture GiB | burst peak MiB | max clean sessions | predicted |
|---|---|---|---|---|---|---|---|
| mem-base | - | 25.12 | 90,172 / 27,051 | 1.73 | 30,990 | **17** | - |
| L9 `mem-l9-rope16k` | `--json-model-override-args {"max_position_embeddings": 16384}` | **24.43** | 108,092 / 32,427 | 1.75 | 30,966 | **20** | 20 |
| L7 `mem-l7-f096` | `--mem-fraction-static 0.96` | 25.12 | 114,375 / 34,312 | 0.78 | 31,958 | **21** | 21 |
| L7 probe `mem-l7-f097` | `--mem-fraction-static 0.97` | 25.12 | 122,443 / 36,732 | 0.49 | OOM | **0** (OOM on the first 10K prefill) | fails the margin |
| L10 `mem-l10-c64` | `--max-running-requests 64` | 25.12 | 90,172 / 27,051 (unchanged) | **1.90** | 30,812 | **17** | 17 |
| L4 `mem-l4-r0276` | `--swa-full-tokens-ratio 0.276` | 25.12 | 95,928 / 26,476 | 1.78 | 30,940 | **18** | 18 |
| **stack1** `mem-stack1` | L9 + L10 + `--mem-fraction-static 0.955 --swa-full-tokens-ratio 0.268` | 24.43 | 139,415 / 37,363 | 1.16 | 31,588 | **25** | 26 (interval 25-27) |
| **stack2** `mem-stack2` | stack1 + L5 (`SGLANG_OPT_SWA_RELEASE_SLID_WINDOW=1`, `SGLANG_SWA_EVICTION_INTERVAL=32`) + ratio 0.226 | 24.43 | 157,376 / 35,566 | 1.18 | 31,574 | **29** | 29 (interval 27-30) |
| **stack3** `mem-stack3` | stack2 + L8 (`SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE=1`) + `--cuda-graph-max-bs-decode 48` | **23.74** | 179,427 / 40,550 | 1.10 | 31,652 | **33** | 33 (interval 32-34) |
| **final** `mem-final` | stack1 + L8 + `--cuda-graph-max-bs-decode 48` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (L5 off) | 23.66 | 160,395 / 42,985 | 1.17 | 31,560 | **29** | 29 (interval 28-30) |

Decode CUDA graphs to bs 48 (needed once more than 32 requests decode; above the captured sizes decode runs
eager) cost 0.19-0.20 GiB of graph memory against 0.13 GiB to bs 32; that ~0.07 GiB is inside every stack
that carries the flag (stack3, final) and was not gated on its own.

**Chunked prefill 2048 (`mem-final-c2048`, probe; the latency verdict is PB's C3).** 27 clean sessions vs 29 at
the default 4,096 (`cap-mem-final-c2048-20261005-144101-build-server-3-c75d05`), against a predicted +1-2.
The pools are identical (the static fraction sizes them, not the chunk), and sliding use per session rises:
peak sliding tokens at N = 27 were 42,526 vs 39,342 (~+118 per session), so the sliding pool fills at 28. The
smaller chunk lowers the GPU peak by ~240 MiB (31,310-31,318 vs 31,552-31,560); spent as pool (fraction
+~0.008) that buys back about one session, still <= 28.

stack1 holds **25 sessions vs 17 (+47%)** with server flags only and unchanged numerics (pool sizes
and a RoPE table length). Two code levers behind default-off switches add L5 (+4, numerics exact) and
L8 (+4, numerics change within the quality guard): **stack3 holds 33 sessions, +94% over mem-base.** L5 did not
survive the multi-turn workload (section 4), so the deployable stack is mem-final (stack1 + L8): 29 sessions.

## 2. What each lever does (mechanism, measured)

**Per-session KV use (burst, measured on every ref).** Full pool: 5,299 tokens per session (prompt +
output) at 10,240 B/token. Sliding pool: 1,322 N + 2,544 tokens for N sessions at 102,400 B/token
(exact at N = 14/16/17 on mem-base; at N >= 22 the per-session slope rises to ~1,400-1,480). So one
session costs ~54 MB of full KV and ~135 MB of sliding KV.

**L9, RoPE tables (flag, exact).** Gemma-4 builds two FP32 cos/sin tables for
`max_position_embeddings` = 262,144 positions: 262,144 x 256 x 4 B = 256 MiB (sliding layers, head 256)
and 262,144 x 512 x 4 B = 512 MiB (full layers: `Gemma4RotaryEmbedding` stores all 512 dims, the 384
non-rotated ones as identity). They are 0.75 of the 0.87 GiB "unattributed" residual in
BASELINE-FP8.md section 4. The frequencies do not depend on the table length, so overriding
`max_position_embeddings` to the served context (16,384; SGLang then extends the table to 16,768 for its
safety margin) changes no value. Weights line 25.12 -> 24.43 GiB; the freed 0.69 GiB goes to the pool:
+3 sessions.

**L7, static memory fraction (flag).** Each +0.01 moves 0.307 GiB (0.01 x 30.71 GiB free before load)
from the activation reserve to the pool. 0.96 gives 21 sessions. 0.97 OOMed on the first 10K-token
prefill (176 MiB requested, 172 MiB free, 0.49 GiB free after capture).
*Safety-margin finding:* the preregistered rule measured headroom against nvidia-smi's 32,607 MiB, but
torch reports the card's capacity as 31.40 GiB (32,154 MiB); about 453 MiB is never available to the
process. Against 32,154, mem-l7-f096's burst peak (31,958 MiB) left only ~196 MiB, so 0.96 is not a safe
deployment value. From stack1 on, the rule is: burst peak <= 31,642 MiB (>= 512 MiB under the CUDA-visible
capacity).

**L10, `--max-running-requests 64` (flag).** SGLang sizes `req_to_token` for max(2048, tokens /
context x 512) requests (2,818 x 16,384 int32 = 184.7 MB here). It is allocated *after* the KV pool is
sized, out of the activation reserve, so the cap alone gains no session; it frees 178 MiB of the
reserve (1.73 -> 1.90 GiB), which L7 then converts (~+0.0055 of fraction).

**L4, pool ratio (flag).** With the budget fixed, 18 sessions need the sliding/full ratio inside
~0.273-0.278; 0.276 gave 18 with both pools at 99%. The ratio is a re-fit, not a lever: it must be
re-solved after every change of budget or per-session use (stack1 uses 0.268; its step-0 ran out of
sliding pool first at 26, so ~0.275 would fit better).

**Why the sliding slope is 1,322 and not ~1,024 (L5 analysis).** With the radix cache on (the
default `UnifiedRadixCache`), the prompt is inserted into the tree when prefill ends; out-of-window
SWA slots are freed at that point (`SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS`, on by default),
and the prompt's last 1,024 SWA slots are owned by the tree and locked by the request. The request's
own decode eviction (`maybe_evict_swa`) only frees slots above its tree-protected prefix and only once
`seqlen - 1 - 1024 >= swa_evicted_seqlen + SGLANG_SWA_EVICTION_INTERVAL`, which a 300-token reply never
reaches. So each session holds 1,024 prompt-window slots + up to 300 decoded slots = 1,324, and
`SGLANG_SWA_EVICTION_INTERVAL` has no effect on this workload. The 300 slots that slide out of the
window during decode (~31 MB, 16% of a session) can only be freed by releasing part of a locked tree
node's SWA while the request runs: a radix-tree change, not a flag.

## 3. Real workload: in-flight capacity at the SLO (scripted multi-turn sessions)

`gate sweep --load inflight` on bs3: N slots each replay PB's scripted multi-turn sessions back to back
(no think time, prefix cache on, cache flushed between points), 60 s warm-up and a 240 s window per
point. Capacity at an SLO = the highest point whose E2E p90 meets it.

mem-base (`sweep-mem-base-20261005-115555-build-server-3-8949a5`):

| in flight | E2E p50 s | E2E p90 s | output tok/s | total tok/s | prefix-cache hit |
|---|---|---|---|---|---|
| 8 | 3.24 | 5.14 | 465 | 15,496 | 0.744 |
| 12 | 3.85 | 6.39 | 558 | 17,651 | 0.692 |
| 16 | 6.59 | 11.80 | 422 | 14,319 | 0.229 |
| 20 | 9.29 | 14.75 | 375 | 12,466 | 0.002 |
| 24 | 13.07 | 17.25 | 347 | 12,053 | 0.002 |

Capacity: 8 at p90 <= 6 s, **12 at 10 s**, 20 at 15 s. The knee is the prefix cache: past ~12 in flight the
pool can no longer keep a finished turn's prefix until the session's next turn arrives, the hit rate
collapses, every turn re-prefills ~5K tokens, and throughput falls while latency climbs.

stack1 (`sweep-mem-stack1-20261005-122247-build-server-3-3c99af`):

| in flight | E2E p50 s | E2E p90 s | output tok/s | total tok/s | prefix-cache hit |
|---|---|---|---|---|---|
| 8 | 3.26 | 5.11 | 468 | 15,626 | 0.753 |
| 12 | 3.75 | 6.20 | 575 | 18,185 | 0.726 |
| 16 | 4.69 | 7.57 | 635 | 20,990 | 0.691 |
| 20 | 5.72 | 8.93 | 666 | 22,187 | 0.656 |
| 24 | 11.72 | 16.97 | 402 | 13,815 | 0.021 |
| 28 | 12.00 | 18.45 | 425 | 13,996 | 0.002 |
| 32 | 15.21 | 19.26 | 458 | 13,551 | 0.002 |

Capacity: 8 at 6 s, **20 at 10 s (mem-base 12, +67%, as predicted)**, 20 at 15 s (unchanged: the collapse
moves from 16 to 24, past the 15 s point mem-base already reached at 20). At the 10 s capacity point
stack1 delivers 666 output tok/s against mem-base's 558 (+19%), so $/1M output tokens falls 16% at any
GPU price.

final at 0.955 (`sweep-mem-final-20261005-140358-build-server-3-5be695`; stack1 + L8 + decode graphs to 48 +
expandable segments, L5 off):

| in flight | E2E p50 s | E2E p90 s | output tok/s | total tok/s | prefix-cache hit |
|---|---|---|---|---|---|
| 8 | 3.16 | 4.98 | 479 | 15,974 | 0.757 |
| 12 | 3.74 | 6.07 | 587 | 18,661 | 0.723 |
| 16 | 4.70 | 7.24 | 653 | 21,499 | 0.704 |
| 20 | 5.57 | 8.77 | 677 | 22,590 | 0.659 |
| 24 | 6.50 | 10.95 | 681 | 23,731 | 0.608 |
| 28 | 10.63 | 18.05 | 468 | 15,483 | 0.114 |
| 32 | 14.86 | 19.23 | 476 | 14,091 | 0.002 |

No failed request; retractions 0 through C24 (5 at C28, 1 at C32). Prompts reached 10,243 tokens (189
prefills of >= 9K). Capacity: 8 at 6 s (C12 misses by 0.07 s), **20 at 10 s** (C24 misses by 0.95 s), **24 at
15 s** (mem-base and stack1 20). L8's pool moved the cache cliff from C24 (stack1) to C28.

Cost-optimal point per SLO (the max-goodput point that meets it), $/1M output tokens at $0.4 / 0.7 / 1.0 /
1.5 per GPU-hour:

| SLO (E2E p90) | point | output tok/s | $/1M output |
|---|---|---|---|
| 6 s | C8 | 479 | 0.232 / 0.406 / 0.580 / 0.870 |
| 10 s | C20 | 677 | 0.164 / **0.287** / 0.410 / 0.616 |
| 15 s | C24 | 681 | 0.163 / 0.286 / 0.408 / 0.612 |

At the 10 s SLO that is +21% output tok/s and -18% $/1M output against mem-base on the same host (C12, 558
tok/s, $0.348 at $0.70).

Replicate on the same host (`sweep-mem-final-20261005-151211-build-server-3-ceaa42`, the capacity point and its
neighbours, run 70 min after the first sweep):

| in flight | E2E p90 s (first) | output tok/s (first) | prefix-cache hit (first) | retractions |
|---|---|---|---|---|
| 16 | 7.23 (7.24) | 655 (653) | 0.704 (0.704) | 0 |
| 20 | 8.89 (8.77) | 674 (677) | 0.653 (0.659) | 0 |
| 24 | 10.94 (10.95) | 679 (681) | 0.604 (0.608) | 1 |

Run-to-run spread is within 0.5% on output tok/s and 0.12 s on p90; the 10 s capacity is C20 in both runs and
C24 misses 10 s by 0.94-0.95 s in both, so the capacity verdict does not depend on the run.

**Fleet rule (what sets the GPU count).** The cliff is set by how many sessions' histories compete for the device
pool, not by the running batch: capping running requests at 24 left the C28 collapse unchanged (section 5). Route
sessions sticky to a replica and keep the sessions whose history must stay cached per GPU (active and idle) at or
below the cache capacity: ~20 per RTX 5090 with mem-final for a 10 s p90 (24 for 15 s); add replicas rather than
queueing past it. The sweep has no think time, so there every cached session is also in flight; real role-play
sessions sit idle between turns while their histories still hold the pool, so cached-session capacity, not
in-flight capacity, sets the GPU count, unless idle histories are allowed to fall out and be recomputed on their
next turn. PB's fleet model (`gate pd-model`, WORKLOAD.md section 4; COMPUTE.md) quantifies that trade-off.

**Deployable: `mem-final` at 0.955.** The server's own GPU memory stayed flat at 31,556-31,570 MiB over the
whole sweep (2 s samples), under the rule's 31,642 MiB. Burst capacity 29 sessions (mem-base 17).

**Quality anchor, mem-final vs mem-base** (both arms at a 0.85 static fraction; role-play arms with 1,024-token
prefill chunks so the input-logprob requests fit; pool size does not change the arithmetic):

| guard | mem-base | mem-final | verdict |
|---|---|---|---|
| GSM8K, all 1,319 | 96.21 | 95.91 | pass: tolerance 1.0 pt; paired, 8 items right only in base and 4 only in final, exact McNemar p = 0.39 |
| tool-JSON | 40/40 | 40/40 | pass |
| role-play reference NLL | 0.18784 | 0.18985 nats/token (+0.0020) | pass: budget 0.02 |
| role-play language adherence | 68/80 | 67/80 | pass under the paired rule below (1 flip, inside the band) |

Runs: `quality-mem-qa-{base-20261005-144330,final-20261005-144912}`, `rp-quality-mem-qr-{base-20261005-145406,
final-20261005-145702}` (all `-build-server-3-*`, records in `runs/`).

*Language-adherence rule (changed for this study, 2026-10-05).* The guard used to fail on any drop in the count of
replies in the prompt's language. A paired per-item check replaces it: the candidate fails if more items flip from
adherent to non-adherent than a numerics-neutral config change flips. The reason: at a fixed config the role-play
run is token-identical run to run, but any config change re-rolls the batched outputs, numerics-neutral or not, so
the count moves with the config, not the arithmetic. Measured:

| pair | outputs token-identical | adherence flips |
|---|---|---|
| A/A, same config, run twice (mem-qr-base; mem-qr-final) | 80/80; 80/80 | 0; 0 |
| numerics-neutral change: mem-base vs mem-q-ctl (pool sizes, RoPE table length, L5 on, L8 off) | 4/80 | 1 |
| numerics-neutral change: mem-base at swa ratio 0.3 vs 0.25 (`mem-qr-base-r025`) | 3/80 | 1 |
| numerics-neutral change: mem-base vs final-mem-c1-c2a at swa ratio 0.25 (`final-c1c2a-qr-r025`) | 3/80 | 0 |
| mem-final vs mem-base | 4/80 | 1 |

All three non-A/A pairs flip the same item, `s000794/0`, a request to translate a Japanese line into Chinese. Every
arm answers in Chinese; replies that quote the Japanese words in a gloss are tagged `ja` (adherent), plain Chinese
translations `zh`. The band is 1/80, and mem-final sits inside it.

*Memory under the real workload, and a co-tenant on the GPU.* Single 2 s samples ~515 MiB above the plateau
(stack1 sweep 32,113 MiB at 12:33:09, mem-final step-0 32,087 at 14:03:10, mem-final sweep 32,079 at 14:23:09)
are not SGLang. build-server-3 also runs zorch-playground, whose watchdog submits a canary job to its GPU
executor container every 10 min at hh:m3:09 (gateway log `POST /api/run`); the job's Python process opens a
CUDA context for under a second. A 0.2 s per-process sample at 14:33:09.7 caught it (`/usr/local/bin/python`,
498 MiB) next to `sglang::scheduler` flat at 31,546 MiB. Outside those samples:
- stack1 (default allocator) sat at 31,590-31,768 MiB, over the rule by up to 126 MiB;
- mem-final (expandable segments) stayed at 31,556-31,570 MiB, so expandable segments is what makes 0.955
  meet the rule.

With the canary on top, the card reached 32,079 MiB, 75 MiB under the CUDA-visible capacity, and nothing
failed. On a host that shares the GPU with another job, leave that job's headroom: the fallback is
`mem-final-f094` (`--mem-fraction-static 0.94`, ~470 MiB less pool, not measured).

## 3b. The combined final on this host: memory + compute levers (`final-mem-c1-c2a`)

PB's combined final = mem-final + C1 (tuned Triton fused-MoE config for this checkpoint,
`SGLANG_MOE_CONFIG_DIR`) + C2-A (sm120 FP8-KV extend-attention tiles,
`SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES=1`), SGLang tree `1425761173` (= `53752c62aa` + the tile commits);
ref in `gate/refs.json`. Run on build-server-3 as the cross-host replicate of PB's build-server-2 numbers
(`sweep-final-mem-c1-c2a-20261005-165852-build-server-3-bea80b`; MoE config sha256 b3bcce12..., Triton 3.7.1 on
both hosts):

| in flight | E2E p50 s | E2E p90 s | output tok/s | total tok/s | prefix-cache hit | retractions | GPU MiB |
|---|---|---|---|---|---|---|---|
| 4 | 1.82 | 3.62 | 318 | 11,899 | 0.773 | 0 | 31,494 |
| 8 | 2.91 | 4.57 | 516 | 17,109 | 0.747 | 0 | 31,496 |
| 12 | 3.40 | 5.51 | 645 | 20,337 | 0.715 | 0 | 31,518 |
| 16 | 4.16 | 6.33 | 744 | 24,134 | 0.703 | 0 | 31,518 |
| 20 | 4.96 | 7.59 | 778 | 25,794 | 0.649 | 0 | 31,520 |
| 24 | 5.57 | **8.59** | **833** | 28,911 | 0.624 | 0 | 31,522 |
| 28 | 7.89 | 13.82 | 616 | 21,085 | 0.148 | 2 | 31,524 |
| 32 | 10.96 | 15.15 | 600 | 18,860 | 0.002 | 1 | 31,526 |

| SLO (E2E p90) | mem-base: capacity, output tok/s | mem-final | final-mem-c1-c2a | $/1M output at $0.4 / 0.7 / 1.0 / 1.5 per GPU-h |
|---|---|---|---|---|
| 6 s | C8, 465 tok/s | C8, 479 | **C12, 645** | 0.172 / 0.301 / 0.431 / 0.646 |
| 10 s | C12, 558 | C20, 677 | **C24, 833** | 0.133 / **0.234** / 0.334 / 0.500 |
| 15 s | C20 by p90; max goodput C12, 558 | C24, 681 | C28 by p90; max goodput C24, 833 | as 10 s |

At the 10 s SLO: +49% output tok/s and -33% $/1M output over mem-base (C12, $0.348 at $0.70), +23% and -18.5%
over mem-final. The compute levers shorten every turn (p90 at C20 8.77 -> 7.59 s), so C24 now fits under 10 s; the
cache cliff stays at C28, a memory property (hit rate at C24 0.624 vs mem-final 0.608). Server memory 31,494-31,526
MiB (rule <= 31,642); no failed request; prompts to 10,243 tokens; host MemAvailable >= 48.9 GB.

Quality anchor vs mem-base (same arms as section 3: 0.85 fraction; role-play with 1,024-token chunks):
GSM8K 1319 **96.44** vs 96.21 (paired 7 items right only here, 4 only in base, McNemar p = 0.55); tool-JSON 40/40;
role-play reference NLL +0.0030 nats/token (budget 0.02); language adherence 67 vs 68 of 80, one
adherent-to-non-adherent flip, inside the 1/80 band (`s001101/0`: a Russian user asks for an English level test;
both arms frame an English quiz in Russian, and this arm's longer quiz is tagged `en`). Passes.

Same-host replicate (`sweep-final-mem-c1-c2a-20261005-175047-build-server-3-f72a00`, run 75 min later):
C16/C20/C24 at E2E p90 6.32 / 7.52 / 8.58 s (first 6.33 / 7.59 / 8.59) and 743 / 789 / 832 output tok/s
(744 / 778 / 833); 10 s capacity C24 in both, $0.234 per 1M output at $0.70. Spread <= 1.4% on tok/s.

## 3c. Chat sessions with think time: sessions per GPU for sizing

Every number above comes from the in-flight layer with zero think time: each live session always has a request in
flight. Real role-play sessions sit idle between turns while their histories still compete for the cache. This
section replays live sessions with think time: load `think30` / `think60` (`gate/config.py`), each slot one live
session whose turns are sent `think_s x think_scale` after the previous reply (the session file's think time is
lognormal, median 15 s, mean 17.9 s, clipped 2-120 s; scales 1.676 and 3.352 give means of ~30 s and ~60 s), a new
session starting when one ends; 240 s warm-up, 480 s window; the first sends spread over one think time. The
point's concurrency is the number of concurrent sessions. Caveats: a session's first turn is sent with no think
time before it (1 turn in ~5), and the population is closed (fixed session count, not Poisson arrivals).

> [!warning] **These slots runs carry a load-generator bug** (fixed in `gate/loadgen.py` after they ran, with a
> regression test): a session whose next turn fell after the window end returned, and its slot at once started a
> fresh session with an uncached ~5K-token first turn. In each point's last think period every slot fired such a
> prefill (arrivals 325 vs ~150 prefills/min, queue spikes of 41-113 at the end of every point), so the tails below
> are pessimistic; the session capacities read from them are lower bounds. The poisson runs below are not affected
> (a cut session is not replaced).

**Mean think time 30 s** (no failed request at any point):

| config | sessions | requests in flight | turns/s | output tok/s | E2E p50 s | E2E p90 s | prefix-cache hit | retractions |
|---|---|---|---|---|---|---|---|---|
| final-hc | 48 | 6.9 | 1.85 | 334 | 3.56 | **7.07** | 0.286 | 1 |
| final-hc | 72 | 14.5 | 2.53 | 434 | 5.54 | 11.41 | 0.076 | 0 |
| final-hc | 96 | 29.1 | 2.97 | 536 | 10.35 | 16.55 | 0.013 | 3 |
| final-hc | 120 | 48.4 | 2.98 | 532 | 16.85 | 24.31 | 0.002 | 2 |
| final-mem-c1-c2a | 48 | 7.7 | 1.83 | 326 | 4.05 | **7.50** | 0.028 | 0 |
| final-mem-c1-c2a | 72 | 14.3 | 2.54 | 435 | 5.56 | 11.12 | 0.004 | 0 |
| final-mem-c1-c2a | 96 | 27.2 | 3.06 | 552 | 9.45 | 15.07 | 0.002 | 3 |

Runs: `sweep-final-hc-20261005-215152-build-server-3-d847de` (48-96), `sweep-final-hc-20261005-205259-build-server-3-efb672`
(120, the overload reference; a co-tenant CI job ran on the host during it), `sweep-final-mem-c1-c2a-20261005-211313-build-server-3-3fe265`.

- **Pessimistic bound (slots): ~60-65 sessions at a 10 s p90 with 30 s mean think time** (interpolated between 48 and 72),
  for both configs. At 48 sessions a GPU delivers ~330 output tok/s, **about $0.58 per 1M output tokens** at
  $0.70/GPU-h, roughly 2.7x the zero-think cost: idle histories do not stay cached (hit rate <= 0.29), so nearly
  every turn re-prefills ~5.8K tokens, and throughput saturates near 3 turns/s (~17K uncached prefill tok/s).
- **HiCache as configured gives no meaningful capacity gain here** (p90 7.07 vs 7.50 s at 48 sessions, 11.41 vs
  11.12 s at 72). A 12 GB host pool plus the device pool should hold roughly 80 sessions' histories, yet the hit
  rate at 48 sessions is 0.29; why is under investigation (SWA host-pool split, first-turn misses, churn).
- Two numbers to keep apart: the zero-think C28 result (919 output tok/s, $0.212 per 1M) is the **per-GPU
  throughput ceiling**; sessions per GPU at realistic think time is the number to **size the fleet** with.

**Mean think time 60 s** (no failed request; `sweep-final-hc-20261005-223037-*`, `sweep-final-mem-c1-c2a-20261005-230954-*`):

| config | sessions | requests in flight | turns/s | output tok/s | E2E p50 s | E2E p90 s* | prefix-cache hit |
|---|---|---|---|---|---|---|---|
| final-hc | 72 | 8.2 | 1.56 | 285 | 4.32 | 15.01 | 0.072 |
| final-hc | 108 | 16.1 | 2.16 | 389 | 6.01 | 20.98 | 0.017 |
| final-hc | 144 | 25.7 | 2.78 | 493 | 8.14 | 25.62 | 0.004 |
| final-mem-c1-c2a | 72 | 8.0 | 1.58 | 291 | 4.17 | 14.16 | 0.007 |
| final-mem-c1-c2a | 108 | 14.8 | 2.21 | 396 | 5.74 | 17.95 | 0.003 |
| final-mem-c1-c2a | 144 | 24.0 | 2.81 | 495 | 7.38 | 24.38 | 0.003 |

\*The T60 p90s are inflated by the load-generator bug above: each point ends in a 2-4 minute burst (queue
41-116, uncached prefill pinned at its ~16.5K tok/s ceiling) although the average prefill demand is ~9-10K tok/s. The means (turns/s, in-flight, hit rate) are usable;
the T60 session capacity at a 10 s p90 is not measured (by the means, likely ~100-120 sessions with independent
arrivals). The T30 points have no mid-window bursts at 48 sessions. What T60 does show: histories are evicted
almost entirely (hit <= 0.07 with HiCache, <= 0.01 without), throughput follows the turn rate, and HiCache again
gives no gain.

**Independent sessions (poisson arrivals; the numbers to size with).** Loads `pthink30` / `pthink60`: sessions arrive
as a Poisson process at `concurrency / expected_session_s` (150 s at T30, 280 s at T60), so no two are
phase-correlated; 240 s (T30) or 300 s (T60) warm-up, 480 s window. Each point's plan is one seeded draw of
heavy-tailed sessions (1-16 turns), so the target does not map exactly to the live count: **read capacity in measured
live sessions** (PB2 replayed the seeded plans on the CPU; the digests match these runs). No burst episodes (queue
> 20 for > 60 s) at any point; queue max <= 5.

| config | think | target | live sessions | turns/s | output tok/s | E2E p50 s | E2E p90 s | hit | failed |
|---|---|---|---|---|---|---|---|---|---|
| final-hc | 30 s | 48 | 41.3 | 1.54 | 265 | 2.71 | 5.15 | 0.462 | 0 |
| final-hc | 30 s | 64 | 63.4 | 2.28 | 394 | 4.25 | 8.05 | 0.116 | 0 |
| final-hc | 30 s | 80 | 64.4 | 2.36 | 427 | 5.17 | 9.02 | 0.107 | 0 |
| final-mem-c1-c2a | 30 s | 48 | 42.0 | 1.54 | 265 | 3.14 | 5.93 | 0.047 | 0 |
| final-mem-c1-c2a | 30 s | 64 | 63.6 | 2.28 | 395 | 4.35 | 8.19 | 0.005 | 0 |
| final-mem-c1-c2a | 30 s | 80 | 64.3 | 2.37 | 428 | 5.23 | 8.99 | 0.003 | 0 |
| final-hc | 30 s | 72 | 67.1 | 2.45 | 430 | 4.91 | 9.03 | 0.088 | 0 |
| final-hc | 30 s | 76 | 73.0 | 2.51 | 444 | 5.60 | 10.73 | 0.067 | 1 |
| final-mem-c1-c2a | 30 s | 72 | 67.3 | 2.45 | 430 | 4.99 | 8.97 | 0.003 | 0 |
| final-mem-c1-c2a | 30 s | 76 | 72.8 | 2.52 | 446 | 5.57 | 10.41 | 0.004 | 1 |
| final-hc-cp2048-lpm | 30 s | 64 | 64.4 | 2.26 | 390 | 4.89 | 9.40 | 0.034 | 1 |
| final-hc | 60 s | 96 | 82.3 | 1.68 | 305 | 3.84 | 6.70 | 0.037 | 1 |
| final-hc | 60 s | 120 | 113.0 | 2.31 | 420 | 5.34 | 9.73 | 0.013 | 0 |
| final-hc | 60 s | 144 | 119.0 | 2.43 | 441 | 5.67 | 10.43 | 0.008 | 0 |
| final-mem-c1-c2a | 60 s | 96 | 82.8 | 1.70 | 308 | 3.89 | 6.47 | 0.002 | 0 |
| final-mem-c1-c2a | 60 s | 120 | 112.6 | 2.30 | 418 | 5.11 | 8.65 | 0.002 | 0 |
| final-mem-c1-c2a | 60 s | 144 | 118.0 | 2.41 | 436 | 5.41 | 9.37 | 0.002 | 0 |

Runs (`runs/`): `sweep-final-hc-20261006-011528-*`, `-031205-*`, `-032720-*`; `sweep-final-mem-c1-c2a-20261006-015330-*`,
`-023118-*`, `-043524-*`, `-044916-*`; `sweep-final-hc-20261006-040929-*`; `sweep-final-hc-cp2048-lpm-20261006-035518-*`. The failed requests (one each at three points, <= 0.12%)
are a client artifact: the one recorded with detail is `ServerDisconnectedError` 0.5 ms after send with nothing
processed. SGLang closes an idle keep-alive connection after 5 s and the client's pool kept connections for 15 s, so
under think time it sometimes reused a socket the server had just closed. The gate now drops idle connections after
2 s (`gate/loadgen.py`); no server request failed.

*Deployment note (keep-alive).* SGLang's HTTP server closes idle keep-alive connections after 5 s
(`SGLANG_TIMEOUT_KEEP_ALIVE`). Chat clients idle between turns, so a pooled connection can be reused just as the server
closes it, and the request fails at once with a disconnect before any byte is processed. Set the client's keep-alive
below 5 s and retry once on a disconnect that arrives before any response bytes (the request never ran). Raising the
server's keep-alive is the alternative, at the cost of more idle sockets held open on the server.

**Capacity edges (poisson, measured live sessions, E2E p90 <= 10 s):** **T30 ~70 sessions per GPU** (both configs
meet 10 s at 67 live sessions, p90 8.97-9.03 s, and miss at 73, 10.41-10.73 s), **T60 ~113-119.** At the T30 edge
a GPU delivers ~430 output tok/s, about $0.45 per 1M output tokens at $0.70/GPU-h. Session capacity follows the turn rate: both edges sit near 2.3-2.4 turns/s per GPU, since nearly every turn
re-prefills its ~5.8K-token history once think time exceeds what the cache can hold; so sessions per GPU grow
roughly in proportion to think time. The slots runs above (phase-correlated, with the window-end bug) are the
pessimistic bound.

**Recommendation for chat with think time.**
- **With a 12 GB host pool, HiCache helps only for zero or short think time.** At T30 it is within noise of
  device-only (p90 8.05 vs 8.19 s at ~64 sessions); at T60 it is slightly worse (p90 +0.2 to +1.1 s, no hit gain).
- So for chat with think time >= 30 s on servers with ~12 GB of host RAM per GPU, deploy **device-only
  `final-mem-c1-c2a` with default chunking**. It is also simpler: a 24 GB memory scope, no HiCache start-check
  retries, and batched outputs reproducible run to run.
- **HiCache (`final-hc`) pays off** for in-flight-heavy traffic (10 s capacity C24 -> C32, +13% goodput at the
  operating point) and, per PB's retention model, for larger host pools (>= 48 GB per GPU) at T30 (model only, not
  measured here).
- The final in-flight config's two flags (chunked prefill 2048, LPM scheduling) are in-flight levers: under think
  time `final-hc-cp2048-lpm` loses host-tier hits (0.034 vs 0.116 at ~64 sessions) and its p90 is ~17% higher;
  PB's same-host control decides the cause.

## 4. Code levers

### L5: release the slid-out part of the tree-locked SWA window (implemented, exact)

Commit `8158a6fe68`, switch `SGLANG_OPT_SWA_RELEASE_SLID_WINDOW` (default off). In decode, every
`SGLANG_SWA_EVICTION_INTERVAL` tokens (32 in stack2), `ScheduleBatch.maybe_evict_swa` calls
`UnifiedRadixCache.release_swa_window_below(req, seqlen - 1 - window)` (the frontier `_evict_swa` already
uses). `SWAComponent.release_window_lock_below` splits the locked window node at the frontier (the split's
older half inherits the lock count and any segment-boundary uuid), drops this request's SWA lock on every
segment node below the split, and stamps the node at the split as the request's new segment boundary; the
request's receipt carries the new uuid, so its final release walks the shortened segment. Released nodes
become evictable (SWA LRU reclaims them under pressure); nothing the request still reads is released, and
the last window stays locked until finish, so the insert at finish still gives the next turn a full window.
Python unified tree core only (no SWA host pool, no EAGLE).

- Capacity: stack1 25 -> stack2 29 (sliding tokens held at 25 sessions: 36,698 -> 26,400, ~1,056/session).
- Exactness: `SGLANG_DEBUG_SWA_POISON_RELEASED_WINDOW` writes NaN into every released slot's KV. With it on,
  greedy outputs at concurrency 1 match stack1 (same tree, switch off) token for token on 6/6 random 5K
  prompts (`exact-mem-stack2-poison-*` vs `exact-mem-stack1b-*`).
- Tests: `TestSWASlidWindowRelease` in `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`
  (split and release accounting, a window shared by two holders, frontier before the segment start).
- Batched: at concurrency 16 on 16 role-play prompts, 7/16 outputs match stack1 token for token, against
  6/16 for two runs of stack1 itself (`exact-mem-stack1b-rp-*`), with mismatches from token 1 on; all 16 poisoned
  outputs are normal text. In this tool the batch composition varied between the two runs, so exactness is
  checkable only at concurrency 1 (the role-play quality run, by contrast, reproduced token for token at a
  fixed config; see the language-adherence rule in section 3).
- **Real workload: not kept.**
  `sweep-mem-stack2-20261005-132151-build-server-3-cbfe29`:

  | in flight | E2E p90 s | output tok/s | hit rate (stack1) |
  |---|---|---|---|
  | 8 | 5.11 | 468 | 0.755 (0.753) |
  | 12 | 6.22 | 574 | 0.722 (0.726) |
  | 16 | 10.16 | 475 | 0.362 (0.691) |
  | 20 | 12.31 | 484 | 0.252 (0.656) |
  | 24 | 16.27 | 441 | 0.130 (0.021) |

  10 s capacity 12 (stack1 20). In the scripted mode a session's next turn carries the scripted reply, not the
  generated tokens, so its prefix match ends at the previous **prompt's** end, and resuming there needs that
  prompt's last SWA window: the window L5 releases during decode. A released window stays matchable only until
  the SWA LRU needs room, which starts at C16. Real chat clients hit the same cut: Gemma-4's chat template
  ends the generation prompt with an empty thought channel after `<|turn>model\n`, and a past turn in the
  history never contains it, so the next turn's match also ends at the previous prompt's `<|turn>model\n` and
  needs the window L5 released. Only a token-level client that resends the exact generated tokens (thought
  channel included) would keep the match through the reply. Not kept; no live-reply retest.

### L8: one FP8 vocab table for the tied embedding and LM head (implemented, numerics change)

Commit `53752c62aa`, switch `SGLANG_OPT_GEMMA4_FP8_VOCAB_TABLE` (default off). After load,
`Gemma4ForCausalLM._use_fp8_vocab_table` quantizes the tied BF16 table (262,144 x 2,816) to E4M3 with one
fp32 scale per row and replaces both `model.embed_tokens` and `lm_head` with `_Fp8VocabTable`; the BF16
table is dropped and the cache emptied before the KV pool is sized. The lookup gathers FP8 rows and
dequantizes; the head runs the tuned Triton FP8 vocab-head kernel (`triton_small_m_fp8_vocab_head`) in
48-row chunks, so every batch width reads FP8.

- Capacity: stack2 29 -> stack3 33; weights 24.43 -> 23.74 GiB.
- Quality (same stack with the switch off as control, mem 0.85 and 1024-token chunks in both arms so the
  guards' input-logprob requests have headroom): GSM8K 200 97.0 vs 97.0; tool-JSON 40/40 vs 40/40;
  role-play reference NLL +0.00095 nats/token (budget 0.02); language adherence 67/80 vs 67/80.
- Tests: `TestGemma4Fp8VocabTable` in `test/registered/gemm/test_gemma4_fp8_lm_head.py` (head within the E4M3
  bound at batch widths across the 48-row chunk edge, lookup within the bound including the subnormal floor,
  the swap drops the BF16 table and refuses an untied head).

### Not implemented: L6 and L11

| # | lever | per-session effect | numerics | what it would take |
|---|---|---|---|---|
| L6 | full-layer V from K (`attention_k_eq_v`) | -5,120 B/token full = -27 MB/session (-14%) | changes (FP8 rounding of V) | not an alias: K = RoPE(k_norm(x) * w_k), V = v_norm(x) without scale, from the same projection. Storing K only needs the Triton decode and extend kernels to rebuild V by un-rotating each key by its position (128 rotated dims) and dividing by w_k |
| L11 | FP4 KV (`--kv-cache-dtype fp4_mx_block16`) | -44% KV bytes | changes (long-context risk) | accepted with Triton as "plain" access, but the pool then dequantizes the whole layer buffer to BF16 on every attention call (`_get_key_buffer`): ~0.8 GiB transient and many GB of traffic per decode step. A usable L11 needs an FP4-reading Triton kernel |

## 5. HiCache (host-RAM prefix cache): stock HiCache fails, fixed HiCache is the final config

Code reading of this tree (paths under `python/sglang/srt/`):
- `--enable-hierarchical-cache` builds two pinned host pools for a hybrid-SWA model, full and SWA
  (`mem_cache/hybrid_pool_assembler.py`); `--hicache-size` (GB) is split between them by device bytes,
  `--hicache-ratio` (default 2.0) sizes each as a multiple of its device pool. The pool is mmap'd with
  MAP_POPULATE and `cudaHostRegister`-pinned at start, and counts against the server's memory scope.
- No validation rejects HiCache with hybrid SWA, Triton attention, FP8 KV or page size 1; the transfer
  kernels need 128 B-aligned per-token rows (1,024 B full, 2,048 B SWA here). Use `--hicache-io-backend
  kernel --hicache-mem-layout page_first` (the `direct` paths copy token by token at page 1) and
  `write_through` (with `write_back` every eviction blocks on copies and internal-node SWA is never saved).
- Write-through copies a node to host when it is inserted; a device-evicted node with a host copy still
  matches, and admission loads back the full KV plus the last SWA window, overlapped layer by layer with the
  prefill forward.
- **Conflict with L5:** an SWA host pool sets `has_swa_host_pool`, and L5 is off by design then (its split
  would hit nodes with an in-flight host write; `release_window_lock_below` asserts on that). Combining the
  two is a small but delicate change (skip nodes with a pending write, then re-verify exactness).
- Host RAM at 3x stack3's device pool would be ~18 GB pinned (5.5 GB full + 12.5 GB SWA), above the 24G
  scope together with the server's ~7 GB RSS; the step-2 run uses `--hicache-size 12`.

**Step 2, stock HiCache (`mem-hc`): a large gain at C24, then a scheduler crash at C28, and load-back is not exact.**

> [!warning] The stock-HiCache numbers in this step (mem-hc C24, the running-cap probe, the $ per 1M output
> derived from them) are **numerics unverified: load-back is inexact at concurrency 1.** The fixed config
> (final-hc, below) is exact. In a greedy
> multi-turn check (4 sessions x 3 turns, later turns loading their prefix back from host, PC3), 7 of 8 later
> turns diverged within 0-15 tokens with identical cached-token counts, and the drift is semantic (a German
> character answers in English, steps out of character); first-turn fresh prefills are identical. Degraded
> continuations can change output lengths, so the goodput below may not be comparable to the device-only arm.
> Attribution: the write-through race below (bug 2), also present on the base commit with none of the study's flags.
`mem-hc` = mem-final + `--enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through
--hicache-io-backend kernel --hicache-mem-layout page_first` (`sweep-mem-hc-20261005-145932-build-server-3-803de2`).
Host pools: full 318,445 tokens (3.26 GB) + SWA 85,344 tokens (8.74 GB), ~2x each device pool.

| point | E2E p50 s | E2E p90 s | output tok/s | total tok/s | prefix-cache hit | retractions |
|---|---|---|---|---|---|---|
| mem-final C24 | 6.50 | 10.95 | 681 | 23,731 | 0.608 | 0 |
| mem-hc C24 | 5.90 | **9.08** | **801** | 27,754 | **0.744** | 0 |
| mem-final C28 | 10.63 | 18.05 | 468 | 15,483 | 0.114 | 5 |
| mem-hc C28 | - | - | - | - | - | scheduler crash; every request failed |

- At C24 the host pool keeps prefixes the device pool had to evict: hit rate 0.61 -> 0.74, +18% output tok/s,
  p90 under 10 s. The 10 s capacity would be 24 (mem-final 20), at $0.243 vs $0.287 per 1M output at $0.70/GPU-h
  (numerics unverified, see the warning above).
- At C28 the scheduler admitted a prefill the SWA pool could not hold: `alloc_token_slots` raised
  `Out of memory ... Try to allocate 2318 tokens. Available swa: 2270 (available_size=2270 +
  component_evictable_size_=0)`, with 7,949 full tokens free. Without HiCache the same overload retracts
  (mem-final C28: 5 retractions, no failure). The prefill admission check (`PrefillAdder._check_prefill_budget`
  with `swa_host_hit_length`, then `init_load_back`) under-reserves sliding slots when a host load-back or a
  chunked continuation is in the batch. That is an SGLang bug, not a configuration limit; HiCache is not
  deployable until it is fixed (open items).
- **With the running batch capped at 24** (`mem-hc-c24` = mem-hc + `--max-running-requests 24`,
  `sweep-mem-hc-c24-20261005-154401-build-server-3-dd6e45`) the scheduler did not crash and the overload collapse
  is gone; the excess requests wait in the queue:

  | in flight | E2E p50 s | E2E p90 s | output tok/s | prefix-cache hit | queue mean | retractions |
  |---|---|---|---|---|---|---|
  | 24 | 5.94 | 9.15 | 796 | 0.741 | 0.1 | 1 |
  | 28 | 6.42 | 10.16 | 782 | 0.714 | 4.0 | 0 |
  | 32 | 8.01 | 10.87 | 804 | 0.714 | 8.0 | 0 |

  Numerics unverified (warning above). Goodput stays at ~800 tok/s from C24 to C32 (mem-final: 468-476 at C28/C32 with p90 18-19 s). The cap does not
  remove the admission bug; it kept the scheduler under it for 3 x 5 min of overload, so it is a flag-only
  fallback, not a fix.

- **The hold comes from the host cache, not the cap.** Control `mem-final-c24` (mem-final +
  `--max-running-requests 24`, no HiCache, `sweep-mem-final-c24-20261005-162807-build-server-3-ac8796`):

  | in flight | E2E p90 s | output tok/s | prefix-cache hit | mem-final (uncapped): p90 / tok/s / hit |
  |---|---|---|---|---|
  | 20 | 8.80 | 674 | 0.652 | 8.77 / 677 / 0.659 |
  | 24 | 11.06 | 675 | 0.605 | 10.95 / 681 / 0.608 |
  | 28 | 18.15 | 447 | 0.052 | 18.05 / 468 / 0.114 |
  | 32 | 19.06 | 463 | 0.002 | 19.23 / 476 / 0.002 |

  Split: the cap alone changes nothing (C20/C24 within replicate spread, the C28 collapse unchanged); the host
  cache alone gives the C24 gain; with both, the C28/C32 hold is the host cache's, and the cap only keeps the
  scheduler below the admission bug. A request waiting in the server queue still belongs to a live session whose
  prefix competes for the device pool, so a running-batch cap does not protect the cache. Not kept.

- Host memory: SGLang keeps 10 GiB of the cgroup headroom free beyond a pinned host pool
  (`HICACHE_HOST_MEMORY_RESERVE_BYTES` in `mem_cache/pool_host/base.py`), so the 8.74 GB SWA host pool failed
  its check in the 24G scope (16.2 GiB headroom - 10 GiB < 8.74 GB) although the server needed less. The run
  used a 28G scope (`G4POC_SERVER_MEMORY_MAX=28G`, the protocol's ceiling); measured peak scope RSS 17.7 GB,
  host MemAvailable >= 38.6 GB, no swap growth.

**Step 3, fixed HiCache (PC3; preregistration `hicache/PREREG.md` HC2-HC4, upstream draft `hicache/UPSTREAM.md`).**

| bug | symptom | cause | fix (default-off switch) |
|---|---|---|---|
| 1 | scheduler crash at C28: SWA allocation fails | prefill admission pins only the device match (`last_node`) during the budget check; after `init_load_back` the request's sliding-window lock anchors at `best_match_node` and also locks device SWA the check had counted as evictable (350 tokens in the trace) | pin `best_match_node` during the check too (`SGLANG_OPT_HICACHE_PIN_LOAD_BACK_WINDOW`; test `TestHiCacheLoadBackSWAWindowPin`) |
| 2 | loaded-back turns diverge at concurrency 1 (7 of 8) | under the overlap scheduler the write-through D2H copy waits only on the scheduler stream, while the forward writing the finished request's last token is still queued on the forward stream: one half-written token per turn, inside the next turn's sliding window | `device_to_host_stream.wait_stream(forward_stream)` before each write (`SGLANG_OPT_HICACHE_FENCE_WRITE_THROUGH`; test `test_hicache_write_fence.py`) |

Bug 2 reproduces on the base commit with none of the study's flags (5/12 identical); the device-only A/A and
HiCache without load-backs are 12/12.

`final-hc` = final-mem-c1-c2a + HiCache (12 GB, write-through, kernel io, page_first) + both fixes, tree
`a0491db764`, 28G scope (`sweep-final-hc-20261005-174836-build-server-3-583abd`, cliff
`sweep-final-hc-20261005-183217-build-server-3-c4d30a`):

| in flight | E2E p50 s | E2E p90 s | output tok/s | total tok/s | prefix-cache hit | retractions | final-mem-c1-c2a: p90 / tok/s / hit |
|---|---|---|---|---|---|---|---|
| 20 | 4.86 | 7.37 | 805 | 26,514 | 0.712 | 0 | 7.59 / 778 / 0.649 |
| 24 | 5.27 | 7.98 | 905 | 31,098 | 0.742 | 0 | 8.59 / 833 / 0.624 |
| 28 | 5.87 | 8.91 | 919 | 30,936 | 0.710 | 3 | 13.82 / 616 / 0.148 |
| 32 | 6.87 | **9.55** | **942** | 29,995 | 0.720 | 6 | 15.15 / 600 / 0.002 |
| 40 | 7.73 | 11.63 | 890 | 30,841 | 0.702 | 3 | - |

- 10 s capacity C32 (final-mem-c1-c2a C24), 942 output tok/s, $0.206 per 1M output at $0.70/GPU-h. The hit rate
  holds at ~0.71 to C40, where the device-only arm collapses at C28. No failed request; the retractions (3-6 per
  point) are requeued, not failures. Server GPU memory <= 31,542 MiB.
- Multi-turn exactness (4 role-play sessions x 3 turns, greedy, concurrency 1, a 16K-token device pool so every
  later turn is a host load-back) vs final-mem-c1-c2a with device hits: **12/12 token-identical**, identical
  cached-token counts (`exactmt-final-hc-smallpool-20261005-183035-build-server-3-7c864b` vs
  `exactmt-final-mem-c1-c2a-20261005-181244-build-server-3-35c7a9`).

**30-minute soak at C32** (`sweep-final-hc-20261005-191020-build-server-3-d90dd0`, 1,800 s window after 60 s warm-up):
9,171 requests, **0 failed**, 31 retractions (requeued and completed); E2E p50 6.54 s, **p90 9.92 s**, p99 10.99 s;
899 output tok/s (30,456 total), prefix-cache hit 0.705, $0.216 per 1M output at $0.70/GPU-h. No creep over 30 min
(5-minute buckets in `soak-buckets.txt`): decode throughput 1,212-1,222 tok/s, server GPU memory flat at 31,490 MiB
(peak 31,536), server RSS 17.518 -> 17.523 GB, host MemAvailable 37.5-38.1 GB, no swap growth. Over 30 minutes C32
meets the 10 s SLO only at the edge (the 240 s sweep point read 9.55 s and 942 tok/s), so the headline operating
point is C28.

**30-minute soak at C28, the operating point** (`sweep-final-hc-20261005-201619-build-server-3-2eab84`): 9,353
requests, **0 failed**, 19 retractions (requeued and completed); E2E p50 5.70 s, **p90 8.92 s**, p99 9.88 s; **919
output tok/s** (30,470 total), prefix-cache hit 0.709, **$0.212 per 1M output** at $0.70/GPU-h. No creep: decode
throughput 1,220-1,246 tok/s per 5-minute bucket, server GPU memory flat at 31,490 MiB (peak 31,542), server RSS
17.531 -> 17.534 GB, host MemAvailable 38.4-38.8 GB. Over 30 minutes C28 beats C32 on throughput as well as
latency (C32: 899 tok/s, p90 9.92 s): past C28 the extra sessions add retractions and queueing, not goodput.

**Quality anchor vs mem-base** (0.85 fraction; role-play with 1,024-token chunks; 28G scope):
GSM8K 1319 **96.36** vs 96.21 (paired 9 items right only here, 7 only in base, McNemar p = 0.80); tool-JSON 40/40;
role-play reference NLL +0.0039 nats/token (budget 0.02). Language adherence 65 vs 68 of 80: three
adherent-to-non-adherent flips and none the other way (net 3, exact McNemar p = 0.25), against a numerics-neutral
band of 1 (net 1 in each neutral pair). Two of the three are the mixed-language items every config change flips
(`s000794/0`, `s001101/0`); the third, `s000135/0`, is a real one: a Chinese role-play opening ("I'm Kelly...")
answered in English, where mem-base and final-mem-c1-c2a answer in Chinese.

Language adherence across three final-hc runs: 65, 67, 68/80 (mem-base 68;
`rp-quality-final-hc-qr-20261005-185028`, `-194522`, and PC3's log-only-tree run `rp-quality-final-hc-qr-pfx-20261005-190542`,
same numerics). One run falls outside the neutral band (3 flips, p = 0.25), two sit inside it (1 each). The band rests on three
numerics-neutral pairs with net adherent-to-non-adherent changes of 1, 1 and 0 (the third:
final-mem-c1-c2a at swa ratio 0.25 vs mem-base, `rp-quality-final-c1c2a-qr-r025-20261005-204923`). HiCache
makes batched outputs vary run to run (A/A 66/80 token-identical), and the outlier flip did not reproduce
(`s000135/0` answers in Chinese in the other two runs). Per-request load-back logs show no flipped item was caused by
a load-back (the one flip in the logged run, `s000794/0`, was Chinese from its first token on a plain device-hit
prefill; none of the 48 load-backs restored a tombstone-recovered copy). No regression shown; not excluded at n = 80.

*Reproducibility with HiCache.* Batched outputs are not bit-reproducible run to run: which path serves a shared
prefix (device hit, host load-back, or recompute) depends on the timing of asynchronous host copies, and the paths
produce different FP8 bytes for the same prefix. At concurrency 1, load-back is exact (12/12 multi-turn turns
token-identical to device hits). Without HiCache, a fixed config reproduces token for token (80/80). Customers who
need reproducible outputs should know this.

## 6. Open items and harness caveats

- **Where a turn's prefix match ends.** The gate's scripted mode puts the session's scripted reply into the
  next turn's history, so the match ends at the previous prompt's end and every turn recomputes the previous
  reply. Real chat clients end at the same point: the generation prompt's empty thought channel after
  `<|turn>model\n` never appears in a past turn (section 4). The measured hit rates therefore hold for chat
  clients; only token-level clients that resend the exact generated tokens would match further.
- **Quality anchors never exercise cache reuse.** GSM8K, tool-JSON and the role-play guard are single-turn or
  fresh-prefill, so they cannot catch a lever that corrupts reused KV (HiCache load-back did, see section 5). Any
  cache or offload lever needs a multi-turn exactness check: greedy, concurrency 1, several sessions x turns,
  later turns served from the reused prefix, compared token for token against the same turns with the lever off.
  mem-final's own device prefix hits are assumed exact; PC3's mem-final A/A arm of that check confirms it.
- **When batched outputs reproduce.** At a fixed config the role-play quality run (80 prompts, up to 32 in
  flight) was token-identical across runs (80/80, twice). Two runs of the exactness tool at concurrency 16 on
  the same config matched 6/16: there the batch composition varied between runs, and so does it in the
  scheduling-dependent in-flight sweeps. Any config change, including numerics-neutral ones (pool sizes), re-rolls
  batched outputs (3-4/80 identical). So exactness is checked at concurrency 1, and batched quality is compared
  per item against a numerics-neutral config-change band, not against an A/A. With HiCache on, even a fixed config
  re-rolls between runs (66/80), because the serving path of a shared prefix depends on async copy timing.
- **HiCache upstream.** The two fixes are default-off switches on this fork; `hicache/UPSTREAM.md` is a draft
  issue/PR, not posted. One open item remains there: a node that adopts a later request's FULL slots on SWA
  tombstone recovery keeps its old FULL host copy (numerics-level, two computations of the same prefix), which
  matters only for bitwise reproducibility if that node is evicted and loaded back.
- **L5 variant.** Releasing only windows that slide out during decode past the prompt's end (keeping the
  prompt's last window, the next turn's resume point) might keep L5's burst gain without the hit-rate loss.
  Not implemented.
- **Co-tenant GPU job on build-server-3** (zorch-playground canary, ~500 MiB for < 1 s every 10 min); see
  section 3. Memory records from 14:43 on also log per-process use (`runs/gpuprocs-20261005.csv`; totals in `runs/gpumem-20261005.csv`).

## 7. Reproduce

```bash
# bs1 -> bs3; the harness is copied with the commit stamp
experiments/g4poc/gate/deploy.sh build-server-3 /data/jooman/g4poc/harness-pc
# on bs3 (cap.sh sources gate/env.sh and points G4POC_MODEL_DIR at the FP8 text checkpoint)
/data/jooman/g4poc/memlogs/cap.sh mem-final 26,27,28,29,30,31,32 --long 10000x6
/data/jooman/g4poc/memlogs/sweep.sh mem-final 8,12,16,20,24,28,32
/data/jooman/g4poc/memlogs/quality.sh quality mem-qa-base --gsm8k-n all --set-baseline
/data/jooman/g4poc/memlogs/quality.sh quality mem-qa-final --gsm8k-n all
/data/jooman/g4poc/memlogs/quality.sh rp-quality mem-qr-base --set-baseline
/data/jooman/g4poc/memlogs/quality.sh rp-quality mem-qr-final
# HiCache needs a 28G scope (SGLang keeps 10 GiB of headroom beyond the pinned host pool)
G4POC_SERVER_MEMORY_MAX=28G /data/jooman/g4poc/memlogs/sweep.sh mem-hc 24,28,32
# GPU memory: 2 s samples, total and per process (the per-process file separates a co-tenant)
while true; do echo "$(date +%T),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 2; done
```

Every launch holds the host lock and runs in the gate's memory-capped scope (`gate/server.py`).
Tests (CPU): `python memory/test_capacity.py`, `python gate/tests/test_g4poc.py`.
