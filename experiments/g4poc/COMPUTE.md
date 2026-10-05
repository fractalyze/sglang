# g4poc compute levers on one RTX 5090 (PB, build-server-2)

Workload: `WORKLOAD.md`. Base: PA's r03 (`BASELINE-FP8.md`), gate ref `base`. The study's final config is
`final-hc-cp2048-lpm` (section 3) for traffic with requests always in flight; chat with think time is better served
by the device prefix cache alone (section 5). All numbers here are bs2 unless marked bs3. Where both hosts ran the same
point, they agree within ~1%.

**Headline.** With requests always in flight, one RTX 5090 serves Gemma-4-26B-A4B FP8 for multi-turn role-play at
**28 in flight, held for 30 min: p90 8.59 s, p99 11.1 s, 952 output tok/s, $0.204 per 1M output tokens at
$0.70/GPU-hour, -41% against the FP8 base ($0.343)**, quality unchanged (bs3 soak, PC2; bs2's same-load sweep
agrees: p90 8.57 s, p99 11.5 s, 955 tok/s). The 4-min sweeps' cheapest point, 32 in flight at $0.197-0.200, carries
a retraction tail over 30 min (p99 37 s), so 28 is the operating point. Chat sessions with 30-60 s of think time cost
~$0.45 per 1M output tokens (~70 sessions per GPU at 30 s, ≥ 118 at 60 s), and there the device prefix cache alone
(`final-mem-c1-c2a`) does as well as HiCache with a 12 GB host pool (section 5).

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
in sessions than in in-flight requests. Host offload of idle histories (HiCache) was the candidate lever for that;
section 5 shows a 12 GB host pool is too small for it to pay at 30-60 s of think time.

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
Fidelity and leg integrity pass. Why small: the default config is already near the HBM bound at decode sizes, so
the tuned tiles take only 2-8% off the kernel. fused_moe is ~33% of GPU time at 12 in flight (the C2-A profile;
the registration's basis assumed ~21%), so E2E moves by about a third of that. Records: `compute/runs/`, vault
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

### C3 Scheduling flags across the cliff — both kept on the base

Nested sweeps base, lpm, chunk 2048, chunk 2048, lpm, base (`compute/sweep_nested.sh`) at 8-20 in flight. The
control reproduced the 10-04 base curve (C12 6.29 s / 569 tok/s), with control drift ≤ 0.9%. Gains are
control/candidate for E2E p90 and candidate/control for output tok/s, geometric mean of the two pairs
(`runs/c3-nested/`):

| in flight | lpm: p90 gain, tok/s gain, hit | chunk 2048: p90 gain, tok/s gain, hit | base hit |
|---|---|---|---|
| 8 | 1.000, 0.999, 0.74 | 1.043, 1.037, 0.79 | 0.74 |
| 12 | 1.001, 0.995, 0.70 | 1.042, 1.026, 0.72 | 0.70 |
| 16 | **1.067**, 1.031, 0.25 | 1.027, 1.045, 0.24 | 0.18 |
| 20 | 1.004 (pairs mixed), **1.073**, 0.10 | 1.011, 1.021, 0.00 | 0.00 |

- **lpm (C3a).** It acts only where turns queue. On the base that is past the cliff: at 16 in flight it admits
  queued turns whose history is still cached first, raising the hit rate from 0.18 to 0.25 and cutting p90 6.3%.
  At 20 the hit rate rises from 0.00 to 0.10 and output tok/s by 7%. Below the cliff it changes nothing.
  The prediction (-5% .. 0% at 16) undershot.
- **Chunk 2048 (C3b).** Every point gains 1-4%. Below the cliff the hit rate rises too (0.74 -> 0.79 at 8 in
  flight), so the gain is mostly less re-prefill, not smoother decode. A plausible mechanism is finer
  sliding-window eviction at chunk boundaries; it is not verified. The prediction (-1% .. +3% p90) was wrong in
  sign. PC measured 2 fewer burst sessions with chunk 2048 on mem-final (27 vs 29).
- **The final stack.** Neither flag joins the final on the base result. Both run overnight against final-hc at
  24 and 32 in flight (vault `g4poc-c3a-hc`, `g4poc-c3b-hc`). A combination joins only if the confirming
  A-B-B-A, the KL check and PC3's HiCache load-back exactness (12/12) all hold.

Vault: `g4poc-c3a`, `g4poc-c3b` (kept on the base, predictions falsified on magnitude).

### C3 flags and C4 on the final stack (overnight, against final-hc)

These are nested sweeps at 24 and 32 in flight, in a 28G scope (`compute/c4_run.sh`; `runs/c4-nested/`). Each
candidate first passed a smoke run. Control drift was ≤ 0.4%.

| flag on final-hc | p90 gain at 24 / 32 | output tok/s gain at 24 / 32 | prefix hit at 24 / 32 | kept |
|---|---|---|---|---|
| C4 `--triton-attention-num-kv-splits 16` | 0.999 / 0.993 | 1.000 / 1.002 | = | no |
| C3a `--schedule-policy lpm` | 1.001 / **1.076** | 1.002 / 1.010 | 0.75 / 0.73 (0.74 / 0.72) | yes |
| C3b `--chunked-prefill-size 2048` | **1.034** / 1.014 | 1.029 / 1.021 | 0.79 / 0.77 (0.74 / 0.72) | yes |

- **C4 (decode KV-split cap).** The backend's split heuristic wants more than 8 splits with the served kv-head
  count. But decode attention speeds up only at small batches (`compute/decode_split_bench.py`: -16% at batch 12,
  0% at 20, -4.4% at 32), so at the final's operating points it is a null. It is retired, as predicted.
- **lpm.** Behind HiCache the device pool still holds ~25 histories, so at 32 in flight some turns wait for
  admission and lpm orders them by cached prefix: -7% p90 at 32, nothing at 24.
- **Chunk 2048.** As on the base: a higher hit rate and +2-3% throughput.

**Combined, `final-hc-cp2048-lpm`.** Confirming A-B-B-A against final-hc (`runs/c4-nested/c4-confirm-*`):

| in flight | p90 (s) | output tok/s | hit |
|---|---|---|---|
| 24 | 7.80 -> 7.53 (gain 1.036) | +2.7% | 0.74 -> 0.79 |
| 32 | 9.45 -> 8.57 (gain 1.103) | +3.0%, 981 | 0.72 -> 0.78 |

That is $0.198 vs $0.204 per 1M output at $0.70, with 1.4 s of p90 headroom under the 10 s SLO. Long role-play
KL check against final-hc: batched KL 0.029 / p99 0.34 vs A/A 0.022 / 0.28, a pass. Promotion into the final also
needs PC3's multi-turn HiCache load-back exactness (12/12) on `final-hc-cp2048-lpm-smallpool`, because chunk size
changes HiCache's SWA admission and chunked prompt nodes, the code path of the race fixed earlier. Vault:
`g4poc-c4` (retired), `g4poc-c3a-hc`, `g4poc-c3b-hc`.

## 3. The study's final stack on bs2

### The HiCache step: final-hc

`final-hc` (gate ref in `hicache/refs.json`; SGLang a0491db764 on `jumanzii/g4poc-final-hicache`; 28G scope)
combines four parts:
- PC's mem-final: the stack1 flags (RoPE tables to 16K, max running 64, mem 0.955, swa ratio 0.268), the L8 FP8
  vocab table, decode graphs to 48 and expandable segments;
- PB's C1 fused_moe config;
- PB's C2-A extend tiles;
- HiCache: a 12 GB pinned host pool (write-through, kernel io, page_first) with PC3's two fixes (SWA admission
  pin, write-through fence).

The coordinator confirmed it after PC3's multi-turn load-back exactness (12/12). With two scheduling flags on
top it became the final config (below).

**In-flight sweep, bs2** (`runs/sweep-final-hc-20261005-194341-build-server-2-8b113a`; 0 failed requests at
every point):

| in flight | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| E2E p90 (s) | 3.59 | 4.54 | 5.44 | 6.25 | 7.27 | 7.78 | 8.83 | 9.50 | 11.41 |
| output tok/s | 324 | 523 | 647 | 754 | 816 | 920 | 927 | 952 | 909 |
| prefix hit | 0.77 | 0.76 | 0.73 | 0.73 | 0.71 | 0.74 | 0.71 | 0.72 | 0.70 |

The host tier holds the histories the device pool cannot, so the hit rate stays at ~0.72 through 40 in flight.
On the base it fell to 0.22 at 16 and 0 from 20. The bs3 replicate (PC2) agrees: C24 7.98 s / 905 tok/s, C32
9.55 s / 942.

**Cost table, bs2** (`compute/cost_table.py`; cheapest point = most output goodput meeting the SLO; prices per
GPU-hour are illustrative):

| stack | E2E p90 SLO | max in flight | cheapest point | out / total tok/s | p90 | hit | $/1M output (0.40 / 0.70 / 1.00 / 1.50) | $/1M total |
|---|---|---|---|---|---|---|---|---|
| base | 6 s | 8 | 8 | 468 / 15,626 | 5.12 s | 0.74 | 0.237 / 0.415 / 0.593 / 0.889 | 0.0071 / 0.0124 / 0.0178 / 0.0267 |
| base | 10 s | 12 | 12 | 568 / 18,005 | 6.31 s | 0.70 | 0.196 / 0.343 / 0.489 / 0.734 | 0.0062 / 0.0108 / 0.0154 / 0.0231 |
| base | 15 s | 20 | 12 | 568 / 18,005 | 6.31 s | 0.70 | 0.196 / 0.343 / 0.489 / 0.734 | 0.0062 / 0.0108 / 0.0154 / 0.0231 |
| final-hc | 6 s | 12 | 12 | 647 / 20,504 | 5.44 s | 0.72 | 0.172 / 0.301 / 0.430 / 0.644 | 0.0054 / 0.0095 / 0.0135 / 0.0203 |
| final-hc | 10 s | 32 | 32 | 952 / 30,313 | 9.50 s | 0.72 | 0.117 / 0.204 / 0.292 / 0.438 | 0.0037 / 0.0064 / 0.0092 / 0.0137 |
| final-hc | 15 s | 40 | 32 | 952 / 30,313 | 9.50 s | 0.72 | 0.117 / 0.204 / 0.292 / 0.438 | 0.0037 / 0.0064 / 0.0092 / 0.0137 |
| **final (cp2048-lpm), rep 1** | 10 s | 36 | 32 | 971 / 30,708 | 8.62 s | 0.78 | 0.114 / **0.200** / 0.286 / 0.429 | 0.0036 / 0.0063 / 0.0090 / 0.0136 |
| **final (cp2048-lpm), rep 2** | 10 s | 40 | 32 | 988 / 30,653 | 8.41 s | 0.79 | 0.113 / **0.197** / 0.281 / 0.422 | 0.0036 / 0.0063 / 0.0091 / 0.0136 |
| final, rep 1 / rep 2 | 15 s | 40 / 40 | 32 / 32 | as at 10 s | | | | |
| **final (cp2048-lpm), bs3** | 6 s | 12 | 12 | 647 / 20,524 | 5.47 s | 0.77 | 0.172 / **0.300** / 0.429 / 0.644 | 0.0054 / 0.0095 / 0.0135 / 0.0203 |

- **final-hc.** At a 10 s p90 SLO it serves 32 requests in flight at 952 output tok/s: $0.204 per 1M output
  tokens at $0.70/GPU-hour (-40.5% vs base; bs3 -41%). At 6 s it is -27% ($0.301 vs $0.415).
- **The final config's 6 s point** (PC2, bs3, `sweep-final-hc-cp2048-lpm-20261006-064035-build-server-3-f18b9e`;
  0 failed, 0 retracted): 12 in flight at p90 5.47 s (p99 6.04 s), 647 output tok/s: **$0.300 per 1M output at
  $0.70, -28% vs the base's $0.415** (8 in flight: p90 4.56 s, 517 tok/s). 16 in flight misses 6 s (p90 6.17 s).

**Memory, deployable rule** (`compute/mem_check.py`, 100 ms samples over the whole sweep). Peak and plateau are
31,514 MiB against bs2's limit of 31,599 MiB (torch capacity 32,111 MiB - 512). The 85 MiB margin is enough, so
mem 0.955 is deployable on bs2 as on bs3. bs2 has no GPU co-tenant and no periodic GPU job.

**Quality** (`gate quality`, greedy, against the base anchor run on bs2 the same evening):

| | GSM8K (full split, 1,319) | tool JSON |
|---|---|---|
| base | 96.13% | 40/40 |
| final-hc | 96.74% | 40/40 |
| paired delta | +0.61 pt, CI95 [-0.07, +1.28], McNemar p 0.12 | 0 |

The pair passes. PC2's bs3 anchor also passed (96.36 vs 96.21). Language adherence of role-play replies is
still under investigation (PC).

### The final config: final-hc-cp2048-lpm

**`final-hc-cp2048-lpm`** (final-hc + `--chunked-prefill-size 2048 --schedule-policy
lpm`; promoted 10-05 ~23:55 after the confirming A-B-B-A, the KL check and HiCache load-back exactness 12/12).
It was replicated twice on bs2 (`runs/sweep-final-hc-cp2048-lpm-20261006-012036-*` and `runs/sweep-final-hc-cp2048-lpm-20261006-015225-build-server-2-813fa1`),
0 failed requests:

| in flight | 16 | 24 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|
| E2E p90 (s), rep 1 / rep 2 | 6.17 / 6.16 | 7.53 / 7.55 | 8.51 / 8.62 | 8.62 / 8.41 | 9.03 / 8.89 | 10.52 / 9.94 |
| output tok/s, rep 1 / rep 2 | 766 / 766 | 942 / 941 | 959 / 953 | 971 / 988 | 956 / 965 | 940 / 949 |
| prefix hit | 0.78 | 0.79 | 0.77 | 0.78 | 0.78 | 0.77 |

- **Headline (bs2).** At a 10 s p90 SLO, one RTX 5090 serves up to 36 requests in flight (40 is on the edge).
  The cheapest point is 32 in flight at 971-988 output tok/s: **$0.197-0.200 per 1M output tokens at
  $0.70/GPU-hour, -42% against the base's $0.343**. The confirming run gave 981 tok/s at p90 8.57 s.
- **Memory.** Peak and plateau are 31,266 MiB, inside bs2's 31,599 MiB rule (`runs/morning-final-hc-cp2048-lpm/`).
  That is 250 MiB below final-hc: chunk 2048 halves the prefill transient.
- **Quality** (paired against the base anchor). GSM8K 96.36% vs 96.13% (+0.23 pt, CI95 [-0.33, +0.78], McNemar
  p 0.58), tool JSON 40/40: pass.
- **Operating point: 30-min soak at 28 in flight** (PC2, bs3, `sweep-final-hc-cp2048-lpm-20261006-052216-build-server-3-d12892`):
  9,656 requests, 0 failed, p50 / p90 / p99 = 5.54 / 8.59 / 11.13 s, 952 tok/s (**$0.204/1M at $0.70, -41% vs
  base**), hit 0.768, 67 retracted (0.69%), GPU memory flat at 31,250 MiB. The same-host bs2 point at 28 (4-min
  window, `runs/sweep-final-hc-cp2048-lpm-20261006-054426-build-server-2-fd138e`): p90 8.57 s, p99 11.47 s,
  955 tok/s, hit 0.773, 11 retracted, 0 failed.
- **30-min soak at 32 in flight** (`runs/sweep-final-hc-cp2048-lpm-20261006-022842-*`): 9,569 requests, 0 failed.
  p50 / p90 / p99 = 5.73 / 9.00 / 37.1 s, 935 tok/s ($0.208/1M at $0.70), hit 0.77. 115 requests (1.2%) were
  retracted, and they make the p99 tail; the 4-min sweep windows do not show it. 32 maximizes p90-bounded goodput
  but carries the tail; at 28 the final retracts 0.69% and p99 stays at 11.1 s (above).
  The tail is the final's flags under overload. final-hc's 30-min soaks on bs3 (PC2) retracted 0.3% at C32 (p99
  11.0 s, 899 tok/s) and 0.2% at C28 (p99 9.9 s, 919 tok/s), against the final's 1.2% and p99 37.1 s at C32: chunk
  2048 + lpm retract more at the edge, another reason the operating point is 28.

**Where the saving comes from, bs2** (10 s p90 SLO, cheapest point; every row measured on bs2):

| stack | adds | capacity | out tok/s | $/1M output @ $0.70 | step |
|---|---|---|---|---|---|
| base (PA's r03) | | 12 in flight | 568 | 0.343 | |
| mem-final (PC) | memory levers: RoPE to 16K, max running 64, mem 0.955, swa ratio 0.268, FP8 vocab table, decode graphs to 48 | 20 | 679 | 0.286 | -17% |
| final-mem-c1-c2a | C1 MoE config + C2-A extend tiles | 24 | 838 | 0.232 | -19% |
| final-hc | HiCache 12 GB host pool, both fixes | 32 | 952 | 0.204 | -12% |
| **final-hc-cp2048-lpm (final)** | C3a lpm + C3b chunk 2048 | 36 (p90 8.9-9.0 s); cheapest at 32 | **971-988** | **0.197-0.200** | -3% |

Base to final: **-42% per 1M output tokens** at the sweeps' cheapest point (32 in flight), **-41% at the
30-min operating point** (28 in flight, $0.204). mem-final's bs2 sweep is `runs/sweep-mem-final-20261006-001140-build-server-2-c8a62b`. bs3 agrees within ~1% at
every shared point.

**What HiCache adds, bs2.** The control is the same stack without HiCache (`final-mem-c1-c2a`,
`runs/sweep-final-mem-c1-c2a-20261005-203524-build-server-2-332ae0`; its memory peak is 31,484 MiB, also inside
the rule):

| in flight | without HiCache: p90, tok/s, hit | final-hc: p90, tok/s, hit | output tok/s |
|---|---|---|---|
| 16 | 6.25 s, 750, 0.70 | 6.25 s, 754, 0.73 | +0.5% |
| 20 | 7.55 s, 791, 0.65 | 7.27 s, 816, 0.71 | +3% |
| 24 | 8.52 s, 838, 0.62 | 7.78 s, 920, 0.74 | +10% |
| 28 | 13.76 s, 610, 0.11 | 8.83 s, 927, 0.71 | +52% |
| 32 | 14.92 s, 607, 0.00 | 9.50 s, 952, 0.72 | +57% |

Without HiCache the device pool (~160K full-layer tokens, ~27 histories) is the cliff at 28 in flight. The
12 GB host pool moves it past 40. At the 10 s SLO, capacity goes from 24 in flight (838 tok/s, $0.232/1M at
$0.70) to 32 (952, $0.204).

### PC4's bug-4 fix (SWA prefill-window margin): an opt-in lever, not promoted on the final (M128)

**What it fixes.** Gemma-4's chat template renders a past assistant turn without the generation prompt's tail, so
the next turn's prompt matches a few tokens short of the inserted one. With no margin, the sliding window behind
that match point is incomplete and SGLang refuses the whole prefix (the ~0.66 think-mode hit ceiling, section 5).
`SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN=128` keeps 128 SWA tokens below the window at tree inserts and holds window +
margin through decode for a branch-inserted prompt (PC4; tree cbf56143b5 = a0491db764 + three commits on
`jumanzii/g4poc-swa-margin`; default off). Ref `final-hc-cp2048-lpm-m128`. Prediction: `compute/PREREG.md` M128,
vault `g4poc-m128f`.

**(a) A-B-B-A at 28 in flight, bs2** (`runs/m128/abba-c28.json`, control drift 0.55%):

| | final | final + margin 128 | gain (geomean of 2 pairs) |
|---|---|---|---|
| E2E p90 (s) | 8.58 / 8.53 | 8.36 / 8.41 | 1.020 (pairs 1.026, 1.015) |
| E2E p99 (s) | 11.47 / 12.34 | 13.18 / 12.42 | 0.930 (+7.5%; pairs +15%, +0.7%) |
| output tok/s | 955 / 953 | 971 / 963 | +1.3% |
| prefix hit | 0.773 / 0.768 | 0.798 / 0.799 | |
| retracted per 240 s window | 11 / 14 (~1.0%) | 25 / 34 (~2.3%) | 2.4x |
| failed | 0 / 0 | 0 / 0 | |

The gain is a third of PC4's on final-hc (+7.1% tok/s, hit 0.71 -> 0.80): chunk 2048 already took most of the hit
headroom (0.77). Retractions rise 2.4x and set the p99; PC4 attributes them to the extra SWA tokens the margin holds
through decode (the pool runs out mid-decode more often).

**(b) Quality** (paired against the base anchor, bs2): GSM8K (full 1,319) 96.44% vs 96.13% (+0.30 pt, CI95
[-0.23, +0.84], McNemar p 0.39), tool JSON 40/40: pass.

**(c) 30-min soak at 28 in flight** (pass, fixed before the run: 0 failed, <= 1.39% retracted, p99 <= 14.47 s, i.e.
twice the final's soak retraction rate and 1.3x its p99):

| 30 min at 28 in flight | requests | failed | p90 | p99 | out tok/s | hit | retracted |
|---|---|---|---|---|---|---|---|
| final (bs3, PC2) | 9,656 | 0 | 8.59 s | 11.13 s | 952 | 0.768 | 67 (0.69%) |
| final + margin (bs3, PC2) | 9,694 | 0 | 8.47 s | 12.43 s | 958 | 0.797 | **259 (2.67%)** |
| final + margin (bs2) | 9,801 | 0 | 8.37 s | 12.59 s | 968 | 0.797 | **251 (2.56%)** |

GPU memory on bs2 stayed flat at 31,262 MiB, inside the rule (`runs/m128/soak-mem.json`). Both hosts retract ~3.8x
the final's rate, past the 1.39% limit, for +0.6-1.7% output tok/s and -1.4..-2.6% p90. **Not promoted: the final
stays `final-hc-cp2048-lpm`.** The margin's retraction cost interacts with chunk 2048 + lpm, which already retract
more under load than final-hc (above); on final-hc alone PC4 measured +7.1% tok/s and -7.3% p90 at 28 in flight with
retractions 2-3 -> 7-8 per 240 s. It ships as an opt-in lever (env, default off): worth it on final-hc, and wherever
the second-turn prefix miss matters more than the retraction tail (think-time traffic: PC4, hit +0.044 at 30 s x 48
sessions). Vault: `g4poc-m128f` (parked), PC4's `g4poc-sw4`.

## 4. Prefill/decode split model (`gate pd-measure`, `gate pd-model`)

Rates are measured on one GPU. **Prefill:** uncached prompts of 5,120 and 10,240 tokens at 1-8 in flight.
**Decode:** from the server's full-batch decode steps; see the harness fix in section 6. The model takes the
colocated sweep's point at the SLO for the workload (prompt, hit rate, output):

| stack | prefill tok/s (5,120) | decode: batch sustained, tok/s, p90 TPOT | SLO point | P:D GPUs | disagg / colocated out tok/s per GPU | $/1M out @0.70, disagg vs colocated |
|---|---|---|---|---|---|---|
| base | 22,585 | 17, 1,070, 16.4 ms | C12 (10 s) | 0.44 | 745 / 568 (1.31x) | 0.261 vs 0.343 |
| final-hc | 30,700 | 24, 1,484, 16.7 ms | C32 (10 s) | 0.42 | 1,048 / 952 (1.10x) | 0.186 vs 0.204 |
| final-hc | | | C12 (6 s) | 0.41 | 1,054 / 647 (1.63x) | 0.184 vs 0.301 |

- **KV transfer.** Per request this is all full-layer KV plus the last 1,024 sliding tokens, ~160 MB. On a
  10 Gb/s link that is 130 ms and 7.7 requests/s; on 100 Gb/s, 13 ms and 77/s. It is not charged GPU time.
- **Where the split pays.** With the final stack it pays most under a tight SLO (6 s: 1.63x), where colocated
  decode steps wait behind prefill chunks. At 10 s, colocated batching already recovers most of it (1.10x).
- **Assumptions.** The model assumes the prefill side keeps the cross-turn prefix cache, i.e. sticky routing
  to a prefill node with HiCache. It is a model, not a disaggregated deployment.

## 5. Fleet model: sessions with think time (`compute/fleet_model.py`)

**The customer message.** The in-flight sweeps have no think time. Real chat sessions idle between turns, and an
idle session's history still has to live somewhere or be recomputed. The zero-think headline ($0.197-0.200 per
1M output tokens at $0.70) is therefore a floor.

**At 30 s of mean think time a GPU serves ~70 live sessions under a 10 s p90 SLO (measured: p90 8.97 s at 67, 10.41 s
at 73), at ~$0.45 per 1M output tokens: 2.3x the floor.** At 60 s it serves ≥ 118 at ~$0.45. Session traffic is
recompute-bound there: ~2.3-2.5 turns/s per GPU at p90 ~9-10 s. A 12 GB host pool does not help at these think times. How much of the gap more host RAM
would close is a model result (below).

**Which config for which traffic:**

| traffic | config | measured at a 10 s p90 SLO |
|---|---|---|
| requests always in flight (no think time) | `final-hc-cp2048-lpm` (section 3) | capacity 36 in flight; cheapest sweep point 32 ($0.197-0.200/1M at $0.70); operating point 28 (30-min soak: p90 8.59 s, p99 11.1 s, $0.204/1M) |
| chat with ≥ 30 s mean think, ~12 GB host RAM per GPU | `final-mem-c1-c2a`: device prefix cache only, default chunking | T30 ~70 sessions ($0.44-0.45/1M); T60 ≥ 118 ($0.446/1M) |
| chat at 30 s think, ≥ 48 GB host RAM per GPU | `final-hc` with a larger host pool | **model only** (retention model below): ~107 sessions/GPU at 48 GB, $0.30/1M |

Think-time cost at every price (measured poisson points at a 10 s p90 SLO; output tok/s per GPU):

| config, mean think | sessions/GPU | out tok/s | $/1M output (0.40 / 0.70 / 1.00 / 1.50 per GPU-hour) |
|---|---|---|---|
| device-only, 30 s | 67 | 430 | 0.258 / 0.452 / 0.646 / 0.969 |
| device-only, 60 s | 118 | 436 | 0.255 / 0.446 / 0.637 / 0.956 |
| final-hc, 30 s | 67 | 430 | 0.258 / 0.452 / 0.646 / 0.969 |
| final-hc-cp2048-lpm, 30 s | 65 | 394 | 0.282 / 0.494 / 0.705 / 1.058 |

At these think times the HiCache configs are no better than device-only. final-hc ties device-only at T30 and runs
slightly worse at T60 (p90 10.43 vs 9.37 s at 119 live). Chunk 2048 + lpm is worse again at T30 (below). The device
pool alone is the simplest and costs the least.

**Measured with poisson session arrivals** (loads `pthink30`/`pthink60`: independent sessions, no bursts, queue max
≤ 4, at most 1 failed request per point; PC2 traced its failures to a client keep-alive race, section 6; the reference for think-time capacity). PC2 ran bs3; the final's sweep ran on bs2
(`runs/sweep-final-hc-cp2048-lpm-20261006-030116-build-server-2-654b74`). Rows with the same `C` replay the same
arrival plan on both hosts:

| config | mean think | C | live sessions | turns/s | E2E p90 | prefix hit |
|---|---|---|---|---|---|---|
| final-hc (bs3) | 30 s | 48 | 41 | 1.54 | 5.15 s | 0.46 |
| final-hc (bs3) | 30 s | 64 / 80 | 63 / 64 | 2.28 / 2.37 | 8.05 / 9.02 s | 0.12 / 0.11 |
| final-hc (bs3) | 30 s | 72 / 76 | 67 / 73 | 2.45 / 2.51 | 9.03 / 10.73 s | 0.09 / 0.07 |
| **final-hc-cp2048-lpm** (bs2) | 30 s | 48 | 42 | 1.54 | 5.59 s | 0.24 |
| **final-hc-cp2048-lpm** (bs2) | 30 s | 64 / 80 | 65 / 66 | 2.28 / 2.37 | **9.31** / 10.46 s | 0.04 / 0.04 |
| final-hc-cp2048-lpm (bs2) | 30 s | 96 | 114 | 2.98 | 21.5 s (overloaded) | 0.01 |
| final-hc-cp2048-lpm (bs3) | 30 s | 64 | 64 | 2.26 | 9.40 s | 0.03 |
| final-mem-c1-c2a (no HiCache, bs3) | 30 s | 48 | 42 | 1.54 | 5.93 s | 0.05 |
| final-mem-c1-c2a (bs3) | 30 s | 64 / 80 | 64 / 64 | 2.28 / 2.37 | 8.19 / 8.99 s | 0.005 / 0.003 |
| final-mem-c1-c2a (bs3) | 30 s | 72 / 76 | **67** / 73 | 2.45 / 2.52 | **8.97** / 10.41 s | 0.003 / 0.004 |
| final-mem-c1-c2a (bs3) | 60 s | 96 | 83 | 1.70 | 6.47 s | 0.002 |
| final-mem-c1-c2a (bs3) | 60 s | 120 / 144 | 113 / **118** | 2.30 / 2.41 | 8.65 / **9.37 s** | 0.002 |
| final-hc (bs3) | 60 s | 96 | 82 | 1.68 | 6.70 s | 0.04 |
| final-hc (bs3) | 60 s | 120 / 144 | 113 / 119 | 2.31 / 2.43 | 9.73 / 10.43 s | 0.01 / 0.01 |

**C sets the arrival rate, but the plan's random draw sets the offered load.** `gate sweep` seeds each C's plan
separately (`sweep-<C>`). Replaying a plan's turn schedule with a constant E2E (`compute/plan_offer.py`) reproduces
the measured points: pthink30 C80 offers only ~66 live sessions (2.37 turns/s), a second sample of C64's load, not
an edge. C96 offers ~99 (3.25 turns/s) and C112 ~113 (3.98), both far past the ~2.4 turns/s a GPU sustains. The
plans predicted the edges in between, C72 (~70-72 live, 2.45 turns/s) and C76 (~73-75 live, 2.50); PC2 measured 67
and 73 live at 2.45 and 2.51-2.52 turns/s. For pthink60 the edge plan is C144 (~120 live, 2.42).

**The final is behind final-hc under think time (HS2, confirmed).** Same host and plan (bs3, PC2, C64): the final
runs p90 9.40 s at hit 0.034, final-hc 8.05 s at 0.116 (+17%). The final's bs2 point on that plan (9.31 s, 0.038)
matches its bs3 point, so the bs2 rows above compare configs, not hosts. Two gaps:

- **At 48 sessions: half the hit rate** (0.24 vs 0.46, p90 5.59 vs 5.15 s). Turn rates (1.538 vs 1.544/s) and device
  evictions (3.47M vs 3.51M tokens) match, so this loss is in host-tier hits. Read through the retention model below,
  the final writes ~0.34 GB of host pool per turn against ~0.22 for final-hc (retention ~23 s vs ~35 s at 12 GB).
  Proposed mechanism: chunk 2048 splits a ~4.3K-token uncached prefill into 3 chunks. Every chunk boundary leaves a
  sliding-window node of up to ~1K tokens (~0.1 GB), and write_through copies it to the SWA host pool. Fewer hits
  mean more uncached tokens and more chunks. In flight (no think time) the same boundaries raise the *device* hit
  (0.74 -> 0.79, C3), so the flag's sign flips with think time.
- **At the 10 s edge: +14-16% E2E p90** (9.31 / 10.46 s against final-hc's 8.05 / 9.02 s at C64 / C80). The
  stack without HiCache, at near zero hit there, runs 8.19 / 8.99 s. The host is not the cause: per-step decode
  rates from the two server logs match at every batch size from 4 to 16 (bs2 within +1% of bs3, e.g. 77.1 vs
  76.3 steps/s at batch 8). The prefill work differs. In C64's timed window the final ran **1.79x the prefill
  passes** (3,227 vs 1,805; 1,815 vs 2,996 new tokens per pass) and 8% more uncached tokens (5.86M vs 5.41M, hit
  0.04 vs 0.12). Near saturation that extra prefill time goes straight into queueing. On the base at 20 in flight
  and zero hit, chunk 2048 was not slower (C3). Why the extra passes cost more under think time (11-14 in flight) is
  not verified.

lpm acts only on a waiting queue, which averaged 0.03 requests at 48 sessions, so chunk 2048 carries both gaps. HS2
(`compute/PREREG.md`, vault `g4poc-hs2`, retired): its prediction (+3 .. +20% p90 at 48 sessions) held at +8.7%
across hosts and +16.6% on one host at 64. **So the recommendation splits:** chunk 2048 + lpm for in-flight traffic
(-3% $/1M at 32 in flight), default chunking for chat with think time.

**2,200 sessions at a 10 s p90 SLO** (zero think: the final; with think time: device-only `final-mem-c1-c2a`,
measured; T30 interpolated between the 67- and 73-session points):

| mean think | sessions per GPU | GPUs for 2,200 | $/1M output @ $0.70 |
|---|---|---|---|
| 0 (2,200 requests in flight) | 36 (C36 meets 10 s; cheapest C32; soak operating point C28) | 62 at C36 (69 at C32, 79 at C28) | 0.197-0.200 |
| 30 s | ~70 (67 meets 10 s, 73 does not; model 73) | ~32 (33 at 67; model 31) | 0.44-0.45 |
| 60 s | ≥ 118 (model 127) | ≤ 19 (model 18) | 0.446 |

`runs/fleet/fleet-v4-poisson.json` holds every poisson point above. The model (the retention model at 12 GB, derated
for poisson arrivals) checks against the measured edges of final-hc, the config it describes: at T30 final-hc crosses 10 s at ~70.5
sessions (interpolated between 67 and 73) and the model says 73, +3.5%; at T60 it crosses at ~115 (between 113 and
119) and the model says 127, +10%. Device-only crosses at ~71 at T30 and had not crossed at 118 at T60.

**Measured, first pass** (PC2, bs3, load `think30`; 0 failures). These ran **before the slots window-cut fix**
(PC2's dff92efc2b), which made every point pessimistic: each slot fired an uncached turn 0 in the window's last
think period. Each slot is a live session with lognormal think time (mean
30 s), and a closed population: a new session starts when one ends, and its turn 0 has no think time, so the
mean think per turn is 30 x (1 - 1/5.15) = 24 s.

| config | sessions | in flight | out tok/s | p90 (s) | prefix hit |
|---|---|---|---|---|---|
| final-hc | 48 | 6.9 | 334 | 7.07 | 0.286 |
| final-hc | 72 | 14.5 | 434 | 11.41 | 0.076 |
| final-hc | 96 | 29.1 | 536 | 16.55 | 0.013 |
| final-hc | 120 | 48.4 | 532 | 24.3 | 0.002 |
| final-mem-c1-c2a (no HiCache) | 48 | 7.7 | 326 | 7.50 | 0.028 |
| final-mem-c1-c2a | 72 | 14.3 | 435 | 11.12 | 0.004 |
| final-mem-c1-c2a | 96 | 27.2 | 552 | 15.07 | 0.002 |

On these pre-fix points a GPU holds 48 sessions at a 10 s p90 SLO (64 interpolated), with or without HiCache: a
lower bound. After the fix, 64 sessions meet it at p90 7.8 s (HS1', below). The poisson think-time runs above give
the corrected capacity.

**Why HiCache barely helps at 30 s think: storage.** PC4's code read (SGLang a0491db764): under write_through the
host pool is an inclusive mirror of the device (host eviction only removes nodes already evicted from the
device). So one GPU holds the larger of what the device and the host pool hold, not their sum.

PC4 measured (final-hc, think30 x 48, bs3) that an idle session costs **~0.23 GB of host pool**: a third
full-layer KV, two thirds sliding-window KV (1.4-1.8K window tokens at 102 KB per token). The window is that
large because windows are node-granular, the dead reply leaf is kept, and chunk boundaries leave windows. So:

| | sessions held |
|---|---|
| device alone (160K tokens, ~6.5K per stored session) | ~25 |
| 12 GB host pool, perfect packing | ~52 |
| 12 GB host pool, with ended-session garbage | ~44 |

At 48 sessions both host pools run at 0.1-0.5% free. The turns split as 22% first turns (nothing to hit), 27%
hits, 28% misses because the sliding window was gone (full prefix still on host), and 23% misses because the
full KV was gone. Misses are all-or-nothing.

From 72 sessions both configs are near zero hit, and every turn re-prefills ~5.8K tokens. Past the storage bound
a HiCache server is just recompute-bound: ~3 turns/s and ~17K uncached prefill tok/s at saturation (closed loop;
~2.4 turns/s with poisson arrivals). That sets the 10 s capacity at ~64 sessions at 30 s think whatever the cache
does.

**Drop-idle, measured on bs2** (`final-mem-c1-c2a` with the radix cache off, max running 24; every turn
re-prefills its whole history; `runs/fleet/`):

| in flight | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|
| E2E p90 (s) | 4.38 | 6.00 | 7.67 | 9.55 | 11.00 |
| turns/s | 1.53 | 2.11 | 2.60 | 2.91 | 3.03 |

At a 10 s SLO the drop-idle GPU runs 16 in flight at 2.9 turns/s. By Little's law that is ~86 sessions at 24 s
of think per turn, an overestimate: independent session arrivals queue more than a closed in-flight loop. PC2's
poisson runs sustain ~2.35 turns/s at p90 ~9 s (0.81 of the closed loop), i.e. ~70 sessions.

**The lever: host RAM per GPU.** A host pool keeps an idle session's cache for a roughly fixed retention time,
not a fixed number of sessions (PC4's SW1). Retention is the host bytes over the aggregate write rate; the
sliding-window share binds first, at ~0.6 of the full-layer retention. At 12 GB with 30 s think, retention is
~25 s at 48 sessions and ~12 s at 72. A returning turn hits if its idle gap (think + E2E) beats the retention:

| idle gap (48 sessions) | < 20 s | 20-30 s | 30-40 s | > 40 s |
|---|---|---|---|---|
| hit | 70-80% | 57% | 16% | 0 |

**Retention model** (`fleet_model.retention_capacity`):
- **Hit rate:** h_max x P(think + E2E < retention), with think drawn from the session file's lognormal (median
  15 s, clipped 2-120 s, then scaled).
- **Retention:** host GB x turn interval / (sessions x w). The per-turn host write w = 0.28 GB is calibrated on
  SW1's 48-session point.
- **GPU turn rate at the SLO:** interpolated by hit rate between final-hc's cached point and the no-cache point,
  derated by 0.81 for poisson session arrivals. That is the ratio of PC2's poisson turn rate at p90 ~9 s (2.35/s)
  to the closed-loop no-cache sweep's (2.91/s at p90 9.55 s).

It reproduces 0.28 at 48 sessions (measured 0.286) and 0.10 at 72 (0.076). On HS1'' (below) it predicts the
6 GB arm (0.11 vs 0.10) and is conservative on the 12 GB arm (0.44 vs 0.56).

| host pool per GPU | 30 s think: sessions/GPU, hit, GPUs for 2,200, $/1M out @0.70 | 60 s think: sessions/GPU, hit, GPUs, $/1M |
|---|---|---|
| 12 GB (as tested) | 73, 0.08, 31, 0.442 | 127, 0.01, 18, 0.462 |
| 24 GB | 87, 0.33, 26, 0.371 | 136, 0.12, 17, 0.431 |
| 48 GB | 107, 0.58, 21, 0.301 | 163, 0.37, 14, 0.360 |
| 96 GB | 121, 0.69, 19, 0.266 | 198, 0.59, 12, 0.296 |

- **At 12 GB HiCache adds almost nothing at 30 s think.** The GPU then runs close to drop-idle, the same as
  without HiCache, which is what was measured.
- **Host RAM buys capacity gradually.** Even 96 GB stays short of the ~187-session compute bound, because long
  thinkers outlive the retention.
- **Upper bound.** The earlier hard cap (0.23 GB of host pool per stored session, ~43 GB for 187 sessions at
  T30) is the bound with no think-time spread.
- **Calibration caveats.** w is calibrated on bs3 final-hc. With chunk 2048 (the final) the per-turn write reads as
  ~0.34 GB instead of ~0.22 (HS2 above), so the table does not apply to the final config.
- **The hit ceiling with think time is ~0.66, not ~0.8** (PC4, 10-06). In think mode every session's second turn
  misses its whole prefix: the cached match ends 1,020 contiguous sliding-window tokens into the window, short of
  the 1,023 SGLang needs (`free_out_of_window_slots` at prefill plus the template's 4-token cut). This holds with
  or without HiCache and is part of every measured think-time hit rate. The model's h_max (0.72, from the
  zero-think sweep) therefore overstates the think-mode ceiling, and the hit rates in the table above are a little
  optimistic. PC4's margin switch (`SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN=128`, opt-in, section 3) fixes it: +0.044 hit at
  30 s x 48 sessions on final-hc; not promoted on the final because of its retraction cost.

**HS1'' (bs2, registered before the run).** final-hc at 30 s think x 36 sessions (slots with the window-cut fix),
6 GB vs 12 GB host pool. 36 sessions lies between the two storage bounds: 6 GB adds nothing over the device
(~25 sessions) and 12 GB holds ~50.

| host pool | hit | E2E p90 | out tok/s |
|---|---|---|---|
| 12 GB | 0.56 | 4.07 s | 234 |
| 6 GB | 0.10 | 5.00 s (+23%) | 231 |

The prediction held (vault `g4poc-hs1b`). Throughput barely moves at this load (3-4 in flight): losing the cache
costs latency and prefill work, not goodput, until the load nears the SLO edge.

**HS1'** (the same arms at 64 sessions, registered before PC4's mirror finding). It predicted 12 GB holds,
assuming ~77 histories (device plus host). It is **falsified**: both arms thrash (hit 0.11 vs 0.01, p90 7.77 vs
8.52 s, 409 vs 406 tok/s; vault `g4poc-hs1`). The retention model predicts 0.14 for the 12 GB arm.

With PC2's window-cut fix, **64 sessions at 30 s think meet the 10 s SLO** on final-hc (p90 7.8 s, 10 in flight).
The 48-64 sessions read from the earlier slots points was a lower bound.

- **No config lever.** The host pool's full/SWA split is near balance, and write_back is not safe (SWA tombstoning
  drops windows without a backup).
- **Code levers (open).** ~1/3 of the SWA host writes are dead: decode-output leaves the template never reuses,
  and chunk-boundary windows. Every miss's re-prefill rewrites ~2K SWA tokens, which feeds the thrash.
  Exclusive tiering for hybrid sliding-window models would add ~+50% distinct capacity.
- **Scope.** Larger pools need a memory scope above the 28G host-safety cap, the user's decision. A customer
  server with 64-128 GB of host RAM per GPU would sit on the right side of it.

## 6. Harness fixes found on the way (2026-10-05/06)

The first two broke the gate's first use on this SGLang commit (91132098df) and were fixed before any gated
number. The others were found later; none changes a reported in-flight number.

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
- **P/D decode rates lost their prefix cache.** `gate pd-measure` timed its decode points by wall clock over a
  second pass that assumed the first pass's prefixes stayed cached. On Gemma-4's hybrid sliding-window pool they
  did not (hit ~0 from batch 16), so every point's rate included re-prefill and `pd-model` refused to run. The
  rate now comes from the server's per-step decode log, at the largest running count the pool sustained for 50
  steps. On the base, batch 8 runs 626 tok/s (p90 TPOT 13.1 ms), and the pool holds 17 requests of 5,120
  tokens: 1,070 tok/s at 16.4 ms. `compute/pd_steps_from_log.py` re-derived the base run from its log.
- **HiCache starts can fail transiently.** SGLang's HiCache host-memory check fails 2 of 7 final-hc starts on
  bs3 in a 28G scope ("Not enough host memory available"), on transient charges in the scope. `Server.__enter__`
  retries exactly that failure, up to 3 tries, and keeps each failed log (`server.log.start-tryN`).
- **Think-time slots started a fresh session after a window cut** (PC2's fix, dff92efc2b, applied here before
  HS1''). With think time, a session whose next turn fell after the window handed its slot to a new session at
  once, so every slot fired an uncached turn 0 in the window's last think period. Every slots think-time point
  measured before the fix is pessimistic. Zero-think in-flight loads never take that path.
- **A failed request's cause was not kept.** Run summaries counted failures only. Each record now carries
  `error_type` (the exception class, `abort` or `abandoned`), and sweep and smoke summaries keep the first 20
  failed records with their error text and `finish_reason`.

- **Failed think-time requests were a client keep-alive race** (PC2's fix, ffa983fd37, picked into this branch and
  deployed on bs2 10-06 ~04:45; deployment note in `memory/REPORT.md`, 8cd3b3e339). SGLang closes an idle
  keep-alive connection after 5 s. With think time, a pooled connection idled past that and aiohttp reused the
  closed socket: `ServerDisconnectedError` 0.5 ms after the send. The client's keep-alive is now 2 s. PC2 traced
  its overnight think-time failures to this. The final's single failure at C96 (bs2) ran before failed records
  kept their error, so its cause is not recorded. A client that sends turns after more than 5 s idle needs the
  same setting.
