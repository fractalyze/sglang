# g4poc compute levers on one RTX 5090 (PB, build-server-2)

Draft; filled in as levers are gated. Workload: `WORKLOAD.md`. Base: PA's r03 (`BASELINE-FP8.md`),
gate ref `base`. All numbers here are bs2; do not compare them directly with bs3 (PC).

## 1. Baseline capacity curve (base, bs2)

Run: `runs/sweep-base-20261004-152459-build-server-2-c0acf7` (gate `sweep --load inflight`). N slots each
replay sessions back to back (no think time), so N requests are always in flight; greedy, scripted replies,
prefix cache on; 60 s warm-up + 240 s window per point, cache flushed between points. 0 failed requests.

| in flight | E2E p50 / p90 / p99 (s) | out tok/s | total tok/s | prefix hit | retracted req |
|---|---|---|---|---|---|
| 4 | 2.01 / 3.94 / 4.59 | 294 | 10,898 | 0.76 | 0 |
| 8 | 3.24 / 5.12 / 5.67 | 468 | 15,626 | 0.74 | 0 |
| 12 | 3.75 / 6.31 / 7.23 | 568 | 18,005 | 0.70 | 0 |
| 16 | 6.39 / 11.89 / 13.54 | 426 | 14,437 | 0.22 | 0 |
| 20 | 8.93 / 14.50 / 17.51 | 379 | 12,562 | 0.002 | 3 |
| 24 | 12.8 / 17.2 / 19.3 | 349 | 12,182 | 0.002 | 3 |
| 32 | 16.9 / 20.0 / 21.5 | 400 | 11,879 | 0.002 | 3 |

Per SLO: the largest in-flight count that meets it, and the cheapest point that meets it ($/1M tokens at
$0.40 / 0.70 / 1.00 / 1.50 per GPU-hour; illustrative prices, not a quote):

| E2E p90 SLO | max in flight | cheapest point | goodput out / total tok/s | $/1M output | $/1M total |
|---|---|---|---|---|---|
| 6 s | 8 | 8 | 468 / 15,626 | 0.237 / 0.415 / 0.593 / 0.889 | 0.0071 / 0.0124 / 0.0178 / 0.0267 |
| 10 s | 12 | 12 | 568 / 18,005 | 0.196 / 0.343 / 0.489 / 0.734 | 0.0062 / 0.0108 / 0.0154 / 0.0231 |
| 15 s | 20 | 12 | 568 / 18,005 (at 20: 379 / 12,562) | 0.196 / 0.343 / 0.489 / 0.734 (at 20: 0.293 / 0.513 / 0.732 / 1.098) | 0.0062 / 0.0108 / 0.0154 / 0.0231 |

### Why goodput falls past 12: the full-attention pool

From the server log of the same run (decode and prefill batch lines inside each timed window):

| in flight | full-pool usage mean / max | sliding-pool usage mean / max | running mean | queued mean | prefill new / cached tokens | hit |
|---|---|---|---|---|---|---|
| 8 | 0.52 / 0.66 | 0.41 / 0.58 | 8.0 | 0.0 | 943K / 2,693K | 0.74 |
| 12 | 0.73 / 0.89 | 0.62 / 0.82 | 11.9 | 0.0 | 1,270K / 2,902K | 0.70 |
| 16 | 0.95 / 1.00 | 0.75 / 0.99 | 14.9 | 1.1 | 2,605K / 755K | 0.22 |
| 20 | 0.95 / 1.00 | 0.74 / 1.00 | 15.3 | 4.6 | 2,935K / 6K | 0.00 |
| 24 | 0.95 / 1.00 | 0.66 / 0.90 | 13.6 | 10.3 | 2,850K / 6K | 0.00 |
| 32 | 0.96 / 1.00 | 0.72 / 0.97 | 14.9 | 17.1 | 2,771K / 6K | 0.00 |

The full-attention pool (89,571 tokens) is the binding pool; the sliding pool never is on average. A
running request holds its whole ~5.6K-token context in the full pool, so ~15-16 requests fill it. Below
that, the radix cache keeps each slot's history between its turns and a turn prefills only its ~1.7K new
tokens (hit 0.70). At 16 the running requests alone fill the pool, cached histories are evicted, and from
20 up nothing survives: every turn re-prefills its ~5.6K tokens, at most ~15 run at once and the rest queue,
so prefill work per reply roughly triples and goodput drops a third. The cliff sits at about
pool tokens / mean context; only more full-pool tokens per GB (PC's memory levers) or shorter contexts move
it. With real think time, idle sessions' histories also compete for the pool, so the cliff comes earlier
in sessions than in in-flight requests (host offload of idle histories, HiCache, is the lever for that).

## 2. Compute levers

Gate: `base` vs candidate, load `inflight-C12` (the goodput peak, below the cliff), 4 ABBA pairs, fresh
server per leg; deciding metric E2E p90 gain over a bar of max(3 sigma A/A noise at the same load, 1%);
fidelity on pair 0; no regression of output tok/s. Predictions: `compute/PREREG.md` and vault trials
`g4poc-c<N>`.

### C1 Tuned Triton fused_moe config — kept (+1.2% E2E p90 at 12 in flight)

`SGLANG_MOE_CONFIG_DIR` points at a fused_moe config tuned on the bs2 RTX 5090 (fp8_w8a8 per-channel, E=128,
N=704; `compute/moe_tune.py` runs SGLang's tuner without Ray, `compute/merge_moe_configs.py` merges the parts).
The full 1,920-config space took ~14 min per token count, so the tune was pruned to 648 configs per part: decode
M 1-32 (BLOCK_M 16-64), prefill M 256-4096 (BLOCK_M 64-256), and M 768/1536 after the first merge left 1536 on the
1024 entry (SGLang picks the nearest tuned M): 802 us there against 666 us for the default config, now 628 us.

| | E2E p90 | output tok/s | prediction (PREREG.md) |
|---|---|---|---|
| gate, 12 in flight (4 ABBA pairs) | **-1.2%** (gain 1.012, CI95 1.006-1.015, 4/4 pairs) | +1.1% | -6% .. -1% |
| A-B-B-A sweeps, 8 in flight | -1.0% (control drift 0.3%) | +0.5% | |

Kernel (fused_moe, us): M 1 31 -> 21, M 4-512 -2..-5%, M 1024 +3%, M 1536-4096 -6..-8% (`runs/c1-moe-config/`).
Fidelity and leg integrity pass. Why small: the default config is already near the HBM bound at decode sizes and
MoE is ~21% of GPU time, so E2E moves by about a fifth of the kernel gain. Records: `compute/runs/`, vault
`g4poc-c1` (kept).

### C2-A sm120 FP8-KV Triton extend tiles — kept (+11.5% E2E p90 at 12 in flight)

**Why this lever.** On SM120 with an FP8 KV cache, Triton is the only prefill backend that serves Gemma-4.
flashinfer and fa3/fa4 are blocked by the Gemma-4 backend assert (fa4 has no head_dim 512 kernel on SM120).
Two other options were rejected:
- **trtllm_mha.** Its SM120 prefill takes BF16 KV only, which halves the pool (cliff ~16 -> ~8 in flight).
- **XQA** (`--decode-attention-backend trtllm_mha`). It is decode-only: it cannot touch the prefill kernel,
  runs through the experimental hybrid wrapper, and turns off Gemma-4's fused KV store.

No existing flag or env reaches the extend kernel on SM120. Its tiles were sized for BF16 K/V: head_dim 512
uses 32x32 tiles with 8 warps and 1 stage, and head_dim 256 uses 64x64 with 8 warps, which runs at 255
registers per thread.

**Change.** `SGLANG_OPT_TRITON_EXTEND_SM120_FP8_KV_TILES=1` (default off; SGLang a10694ad32, `extend_attention.py`
`_SM120_FP8_KV_EXTEND_TILES`, unit test `compute/test_sm120_tiles.py`):
- head_dim 512: (BLOCK_M 32, BLOCK_N 32, BLOCK_N_PREFIX 64, 8 warps, 1 stage).
- head_dim 256: (32, 32, 32, 4 warps, 1 stage).
Both were picked by `compute/extend_attn_bench.py` (270 configs on the C12 prefill shapes, `runs/c2/bench.json`).

**Sizing (before the gate).** Base profile at C12 (`compute/profile_extend_share.py`, 10 s GPU window):

| kernel | share of wall time |
|---|---|
| fused_moe | 33.3% |
| CUTLASS FP8 GEMMs | 26.8% |
| Triton extend attention (head_dim 512 / 256) | 14.9% (8.8% / 6.1%) |
| Triton decode attention (stage 1 + 2) | 7.9% |

The microbench mixes run 2.93x (512) and 2.02x (256) faster, which predicts 0.088 x (1 - 1/2.93) +
0.061 x (1 - 1/2.02) = 8.8% less GPU time; with the slowest shape's speedups alone, 7.4%.

**Numerics.** The FP8-prefix path casts q to FP8 and quantizes the softmax weights to FP8 against each KV tile's
running max, so the tile width changes the rounding. Against fp32 attention, the new tiles' error equals the
default's on every served shape (`runs/c2/accuracy.json`, ratio 1.00). Long role-play KL check
(`compute/kl_check.py`, 8 first turns of ~5K tokens), teacher-forced, batched:

| | KL mean / p99 | worst top-1 |
|---|---|---|
| fast tiles | 0.032 / 0.37 | 0.911 |
| A/A (base batched vs serial) | 0.020 / 0.26 | 0.927 |
| limits | 0.041 / 0.52 | 0.907 |

- **Greedy, one prompt at a time.** The base repeats 8/8; the new tiles diverge on 7/8, mostly at near-ties.
- **Bit-exact control.** A tile variant that keeps the default KV tile widths is bit-identical one prompt at
  a time (8/8). Batched, it still shows one KL-14 position in the same zh reply where the fast tiles show three
  of 4-6. A few positions there jump under any perturbation, which is consistent with MoE routing near-ties.

**Gate** (`runs/c2-gate-20261005-154627-build-server-2-b1c285`, 4 ABBA pairs at inflight-C12, bar 1%):

| | base (pooled) | C2-A (pooled) | gain |
|---|---|---|---|
| E2E p50 / p90 / p99 (s) | 4.08 / 6.50 / 7.38 | 3.85 / 5.83 / 6.41 | 1.06 / **1.115** / 1.15 |
| output / total tok/s | 547 / 18,506 | 605 / 20,228 | 1.106 / 1.09 |

- **p90.** Gain 1.115 (CI95 1.114-1.122), with pair gains 1.116-1.121. The prediction was -12% .. -5%.
- **Fidelity.** Gate fidelity passes: forced KL 0.009, decode KL equal to the control's.
- **Mechanism.** The p99 gains most because the slowest replies carry the largest prefills.

A-B-B-A sweeps (`runs/c2-c8-c16-abba.json`):

| in flight | E2E p90 base -> C2-A | gain | output tok/s | control drift |
|---|---|---|---|---|
| 8 | 5.09 -> 4.69 s | 1.085 (-7.9%) | +6.7% | 0.15% |
| 16 (past the base cliff, hit 0.17) | 11.95 -> 9.54 s | 1.253 (-20.2%) | +22.0% | 0.4% |

Past the cliff every turn re-prefills most of its history, so faster prefill pays more. On the base, C2-A
brings 16 in flight under the 10 s SLO. Vault: `g4poc-c2` (kept), claim
`c-g4poc-sm120-triton-extend-tiles-fp8kv`.

## 3. Harness fixes found on the way (2026-10-05)

Both broke the gate's first use on this SGLang commit (91132098df) and are fixed before any gated number.

- **Input logprobs ran out of GPU memory** (calibrate, 12:33 KST). The teacher-forced fidelity pass puts ~15 short
  prompts in one prefill batch; their ~2,900 logprob rows were scored in SGLang's default 2,048-row chunks, and one
  chunk's fp32 logits over the 262,144-token vocab is 2 GiB on top of its bf16 copy, with 1.7 GB left outside the
  static pools at mem 0.93. Every gate server now gets `SGLANG_LOGPROB_CHUNK_SIZE=128` (`gate/config.py`
  `SERVER_ENV`; a ref may not set it). Only the input-logprob path reads it, so timed legs and the KV pool are
  unchanged, and the LM-head GEMM shapes it fixes are the same for every arm. Any stack at a higher mem fraction
  (PC's 0.955) needs this even more: the default chunk cannot fit there at all.
- **`/weights_checker` changed its body.** `per_engine_checksum` is now one sha256 string and the per-tensor
  checksums sit under `ranks`; the gate still read gemma4nv's older list form and raised after calibrate's fidelity
  passes, and would have ended every `gate run` leg (weights checked at load and at the end). The parser now reads
  the current body and reports `ok: false` without a digest, so two missing digests no longer compare equal.
