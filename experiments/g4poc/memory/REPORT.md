# g4poc memory levers: more concurrent sessions per RTX 5090 (PC, build-server-3)

Scope: memory levers L4-L11 from `BASELINE-FP8.md` section 6, applied to the FP8 baseline of
`google/gemma-4-26B-A4B-it` on build-server-3's RTX 5090. Each lever was preregistered in
`memory/PREREG.md` (and as a vault trial `g4poc-l*` / `g4poc-s1`) before it ran. Code:
`memory/capacity.py` (step-0 tool), `memory/refs.json` (one ref per lever). Raw records:
`memory/runs/<run>/` (`capacity.json` with every burst, plus the server's memory log lines).

Unit note: SGLang logs "GB" for bytes / 2^30; this file writes GiB for those. MiB figures are
nvidia-smi's.

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

stack1 holds **25 sessions vs 17 (+47%)** with server flags only and unchanged numerics (pool sizes
and a RoPE table length). The burst peak stays 566 MiB below the CUDA-visible capacity.

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

*Memory under the real workload:* sampled every 2 s, the GPU peaked at 32,113 MiB (12:33:09, one sample in
the C12 window), 41 MiB under the CUDA-visible capacity; between such spikes it sat at 31,590-31,770 MiB.
Prefill batches never exceeded 4,096 new tokens (p50 2,587), so the spike is not a larger prefill batch;
allocator fragmentation is the suspect. The step-0 bursts (peak 31,588) did not show it, so 0.955 is not a
deployable value without a fix (allocator setting or a lower fraction).

## 4. Code levers: what is left and what each would take

| # | lever | per-session or one-time effect | numerics | what it takes |
|---|---|---|---|---|
| L5 | free the prompt-window SWA slots as decode slides (section 2) | -300 sliding tokens = -31 MB/session (-16%), ~+19% sessions | exact | UnifiedTreeCore: split the request's locked leaf at the eviction frontier and tombstone the left part's SWA during decode; insert-at-finish must accept the shorter SWA tail |
| L6 | full-layer V from K (`attention_k_eq_v`) | -5,120 B/token full = -27 MB/session (-14%), ~+16% sessions | changes (FP8 rounding of V) | not an alias: K = RoPE(k_norm(x) * w_k), V = v_norm(x) without scale, from the same projection. Storing K only needs the Triton decode and extend kernels to rebuild V by un-rotating each key by its position (128 rotated dims) and dividing by w_k |
| L8 | FP8 embedding + tied LM head | -0.69 GiB once, ~+3.5 sessions | changes (logits) | the tree's `SGLANG_OPT_GEMMA4_FP8_LM_HEAD` adds a 740 MB FP8 copy for speed (more memory, not less); L8 needs the BF16 table dropped: FP8 rows + per-row scales for both the lookup (gather + dequant) and the head GEMM at every batch size |
| L11 | FP4 KV (`--kv-cache-dtype fp4_mx_block16`) | -44% KV bytes | changes (long-context risk) | accepted with Triton as "plain" access, but the pool then dequantizes the whole layer buffer to BF16 on every attention call (`_get_key_buffer`): ~0.8 GiB transient and many GB of traffic per decode step. A usable L11 needs an FP4-reading Triton kernel |

## 5. Reproduce

```bash
# bs1 -> bs3; the harness is copied with the commit stamp
experiments/g4poc/gate/deploy.sh build-server-3 /data/jooman/g4poc/harness-pc
# on bs3 (cap.sh sources gate/env.sh and points G4POC_MODEL_DIR at the FP8 text checkpoint)
/data/jooman/g4poc/memlogs/cap.sh mem-stack1 22,24,25,26,27,28,30 --long 10000x6
/data/jooman/g4poc/memlogs/sweep.sh mem-stack1 8,12,16,20,24,28,32
```

Every launch holds the host lock and runs in the gate's memory-capped scope (`gate/server.py`).
Tests (CPU): `python memory/test_capacity.py`, `python gate/tests/test_g4poc.py`.
