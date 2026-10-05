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
L8 (+4, numerics change within the quality guard): **stack3 holds 33 sessions, +94% over mem-base.**

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

**Deployable: `mem-final` at 0.955.** The server's own GPU memory stayed flat at 31,556-31,570 MiB over the
whole sweep (2 s samples), under the rule's 31,642 MiB. Burst capacity 29 sessions (mem-base 17).

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
- Batched: at concurrency 16 on 16 role-play prompts, 7/16 outputs match stack1 token for token, against an
  A/A (stack1 vs stack1) of 6/16 with mismatches from token 1 on; all 16 poisoned outputs are normal text.
  This engine is not batch-invariant, so exactness is checkable only at concurrency 1.
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

## 5. HiCache (host-RAM prefix cache) feasibility

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

> [!gap] Step 2 (in-flight C24/C28/C32 with and without HiCache, L5 off in both) is queued (queue 8).

## 6. Open items and harness caveats

- **Where a turn's prefix match ends.** The gate's scripted mode puts the session's scripted reply into the
  next turn's history, so the match ends at the previous prompt's end and every turn recomputes the previous
  reply. Real chat clients end at the same point: the generation prompt's empty thought channel after
  `<|turn>model\n` never appears in a past turn (section 4). The measured hit rates therefore hold for chat
  clients; only token-level clients that resend the exact generated tokens would match further.
- **Batched outputs are not reproducible on this engine** (A/A 6/16 identical at concurrency 16): exactness
  checks must run at concurrency 1, and batched numerics need logprob/KL against an A/A bar.
- **L5 variant.** Releasing only windows that slide out during decode past the prompt's end (keeping the
  prompt's last window, the next turn's resume point) might keep L5's burst gain without the hit-rate loss.
  Not implemented.
- **Co-tenant GPU job on build-server-3** (zorch-playground canary, ~500 MiB for < 1 s every 10 min); see
  section 3. Memory records from 14:43 on also log per-process use (`memlogs/gpuprocs.csv`).

## 7. Reproduce

```bash
# bs1 -> bs3; the harness is copied with the commit stamp
experiments/g4poc/gate/deploy.sh build-server-3 /data/jooman/g4poc/harness-pc
# on bs3 (cap.sh sources gate/env.sh and points G4POC_MODEL_DIR at the FP8 text checkpoint)
/data/jooman/g4poc/memlogs/cap.sh mem-stack1 22,24,25,26,27,28,30 --long 10000x6
/data/jooman/g4poc/memlogs/sweep.sh mem-stack1 8,12,16,20,24,28,32
```

Every launch holds the host lock and runs in the gate's memory-capped scope (`gate/server.py`).
Tests (CPU): `python memory/test_capacity.py`, `python gate/tests/test_g4poc.py`.
