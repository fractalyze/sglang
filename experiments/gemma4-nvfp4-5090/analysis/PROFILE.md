# gemma4nv W2: profile and SOL byte model

Model `nvidia/Gemma-4-26B-A4B-NVFP4` (HF snapshot `a19cfe00be84`) on 1x RTX 5090 (SM120,
32 GB). SGLang tree `a9871012a` (this branch's base). Host build-server-2.

> **Status: measured on bs2 (2026-10-02 16:57-17:09 KST, driver 595.91.07, unlocked clocks,
> host load1 < 1.5 during every timed leg).** §3 has the measured component tables, §4 the
> knob screen and §5 what the measurements change. §2 is the analytic byte model the tables are built on. Single unpaired runs are
> labelled "screen, unpaired": nothing here is a gated gain. bs2 numbers are bs2 numbers;
> compare them to bs3 only by delta.

## 1. Serving configuration and what runs per op

Baseline flags are W1's pinned `base` ref (`gate/refs.json`): `--moe-runner-backend
flashinfer_cutlass --cuda-graph-max-bs-decode 32` (+ `--decode-log-interval 1`). My measured
job uses exactly these flags. The two bs2 bring-up launches in `evidence/` also passed
`--context-length 8192 --max-running-requests 32`.

| Setting | Resolved value | Source |
|---|---|---|
| Quantization | `modelopt_fp4`. **Only the routed experts are NVFP4.** Attention projections, dense MLP, router and `lm_head` are BF16 (`ignore` list in `hf_quant_config`). | checkpoint |
| MoE runner | `auto` resolves to `flashinfer_trtllm` on SM120 and **crashes at load**: `AttributeError: 'FusedMoE' object has no attribute 'g1_scale_c'` (`results/logs/default.log`, 14:11). `flashinfer_cutlass` is required. | `modelopt_quant.py:2972-2978` |
| Other NVFP4 MoE runners | `marlin` and `flashinfer_cutedsl` reject `gelu` (Gemma's GeGLU). `humming` uses erf-GELU, not tanh, which is a numerics mismatch. | `moe_runner/marlin.py:142`, `flashinfer_cutedsl.py:230` |
| Attention | `triton` for prefill and decode. Gemma4 accepts only `trtllm_mha`/`triton` on CUDA, and `trtllm_mha` is the SM100 default. | `model_overrides/gemma4.py:28`, `model_hook.py:590` |
| **KV cache dtype** | `auto` resolves to **FP8 e4m3 for all 30 layers**, the 5 full-attention layers included, because the checkpoint declares `kv_cache_quant_algo: FP8`. Log: `Full KV ... dtype: torch.float8_e4m3fn`, `SWA KV ... float8_e4m3fn`. | `kv_cache_dtype.py:39-46` |
| KV pools | Hybrid SWA pool. Full pool 36,594 tokens (0.34 GB); SWA pool 29,275 tokens (2.8 GB). **Full layers store K and V as two copies even with `k_eq_v`.** K gets k_norm + RoPE, V gets v_norm and no RoPE (`gemma4_causal.py:461-507`), and both are written. So a "one copy because K=V" bound undercounts full-layer KV bytes by 2x (about 45 MB per step at B=8, FP8). | log, `gemma4_causal.py` |
| CUDA graph | Decode: full graph, `bs=[1,2,4,8,12,16,24,32,40,48]`. **Prefill graph disabled**, because Gemma4ForConditionalGeneration is a multimodal arch. | log `cuda_graph_config` |
| Chunked prefill | 4096 (32 GB-GPU default). A B=8x1024 prefill is 2 chunks. `max_prefill_tokens=16384`. | log |
| Sampling | Greedy `torch.argmax` over fp32 logits after an in-place softcap(30). | `logits_processor.py:916`, `sampler.py:179` |

### Kernel order of one decoder layer (decode)

From `gemma4_causal.py:654-767`. Every layer has both a dense MLP and a routed MoE.

| # | Op | Implementation | Launches |
|---|---|---|---|
| 1 | input RMSNorm | `rmsnorm` | 1 |
| 2 | qkv_proj (BF16) | cuBLAS/nvjet GEMM. On full layers the V shard is a copy of K (`:1271`), so it repeats K's math. | 1 |
| 3 | q/k/v norm | `gemma_qkv_rmsnorm` (fused Triton) | 1 |
| 4 | RoPE | `rotary_emb`. RoPE+KV-write fusion is hard-disabled (`can_fuse=False`, `:495`). | 1 |
| 5 | KV write | `set_kv_buffer` (FP8 quantize + store) | 1+ |
| 6 | attention | Triton decode: stage1 (split-KV) and stage2 (reduce) | 2 |
| 7 | o_proj (BF16) | GEMM | 1 |
| 8 | post-attn norm, then pre-FF norm | `rmsnorm` + `fused_add_rmsnorm`: two launches over the same row | 2 |
| 9 | dense MLP | gate_up GEMM, `gelu_tanh_and_mul`, down GEMM | 3 |
| 10 | router | `Gemma4RMSNorm`, then BF16 GEMM N=128 | 2 |
| 11 | pre-FF norm 2 | `rmsnorm` over the **same input** as the router norm | 1 |
| 12 | top-k | `gemma4_fused_routing` (`per_expert_scale` folded in) | 1 |
| 13 | routed experts | flashinfer CUTLASS fused MoE (W4A4: static-scale FP4 activation quant, then grouped GEMMs, GeGLU, finalize) | ~4-6 |
| 14 | norms + residual + layer scalar | `gemma_dual_rmsnorm_residual_scalar` (one kernel) | 1 |

That is about 22-25 launches per layer, or about 700 per decode step including the head.
Visible glue waste: the duplicate norm (rows 10 and 11), the two-launch norm in row 8, unfused
RoPE and KV write, and the dense MLP serialized before the MoE.

## 2. Byte / FLOP model and SOL per component (analytic; §3 re-bases it on measurements)

`scripts/sol_model.py`. SOL = max(FLOPs/peak, bytes/BW) per component (SOL-ExecBench
SOLAR style). The table uses `sol_fraction` = SOL / achieved, never a SOL "score".
Placeholder ceilings: BW 1792 GB/s, BF16 209.5 TFLOP/s, FP4 838 TFLOP/s. They get replaced
by `scripts/microbench.py` measurements.
Distinct experts per layer use the uniform-routing expectation 128·(1−(1−8/128)^B) = 8 / 51.6 / 111.8 at
B = 1 / 8 / 32. Real routing is skewed, so the measured value (`scripts/experts.py`, diverse
prompts) should be lower. KV is FP8 (1 B/elem), context about 1088 tokens.

Expert bytes: 5.947 M params per expert (gate/up 2816x1408 + down 704x2816) at 0.5625 B/param
(FP4 + one FP8 scale per 16) is 3.345 MB per expert. Yukon measured 3.35 MB.

### Decode, B=8 (the W8 shape)

| Component | Bytes/step | SOL µs | Share of SOL | Achieved µs | sol_fraction | Gap µs |
|---|---:|---:|---:|---:|---:|---:|
| routed experts (NVFP4, ~52 distinct/layer) | 5.18 GB | 2893 | 47.1% | unmeasured | — | — |
| lm_head (tied, BF16, 262144x2816) | 1.50 GB | 838 | 13.7% | unmeasured | — | — |
| qkv_proj (BF16) | 1.45 GB | 808 | 13.2% | unmeasured | — | — |
| dense MLP (BF16) | 1.08 GB | 601 | 9.8% | unmeasured | — | — |
| attention, sliding (FP8 KV, window 1024) | 0.84 GB | 470 | 7.7% | unmeasured | — | — |
| o_proj (BF16) | 0.81 GB | 453 | 7.4% | unmeasured | — | — |
| attention, full (FP8 KV) | 0.09 GB | 50 | 0.8% | unmeasured | — | — |
| router GEMM (N=128) | 0.02 GB | 13 | 0.2% | unmeasured | — | — |
| norms / RoPE / glue | 0.02 GB | 12 | 0.2% | unmeasured | — | — |
| **total** | **11.0 GB** | **6137** | | unmeasured | | |

### Decode, B=1 and B=32

| Component | B=1 bytes | B=1 SOL µs | B=32 bytes | B=32 SOL µs |
|---|---:|---:|---:|---:|
| routed experts | 0.80 GB | 448 (14%) | 11.2 GB | 6266 (56%) |
| lm_head | 1.48 GB | 826 (26%) | 1.58 GB | 880 |
| qkv_proj + o_proj | 2.25 GB | 1256 (39%) | 2.29 GB | 1275 |
| dense MLP | 1.07 GB | 598 (19%) | 1.09 GB | 610 |
| attention (sliding + full) | 0.12 GB | 65 | 3.73 GB | 2082 |
| **total** | **5.75 GB** | **3206** | **20.0 GB** | **11177** |

What the byte model already says:
- **At B=1, the BF16 non-expert weights are 86% of the bytes.** These are attention projections
  2.25 GB, lm_head 1.48 GB and dense MLP 1.07 GB. The NVFP4 experts are only 14%. Decode at
  B=1 is a BF16-GEMV problem, and the biggest analytic levers are FP8 for those projections
  and for lm_head. Yukon §4 saw the same thing: once experts are 4-bit, BF16 attention and
  lm_head become the dominant byte pool.
- **At B=8, the routed experts are 47% of bytes** (Yukon measured 53% of decode time on MLX).
  The next pools are the BF16 projections (2.26 GB) and lm_head (1.50 GB). Sliding-window KV
  is 0.84 GB even at FP8, and it would be 1.68 GB at BF16. The FP8 KV default is already
  worth about 470 µs of SOL per step.
- **At B=32, experts and sliding KV dominate** (56% and 17%).
- Full-attention KV is small: 0.09 GB at B=8 FP8, 0.18 GB at BF16. Keeping those 5 layers at
  BF16 for fidelity would cost about +50 µs of SOL per step (about 0.8%).

### Prefill, B=8 x 1024 (8192 tokens, 2 chunks of 4096)

Compute-bound. Total SOL is 183 ms at the placeholder BF16 peak. In FLOP terms qkv_proj is
31%, dense MLP 23%, o_proj 18%, routed experts 15% (at the FP4 peak) and attention 7%.
Activation glue is about 22 GB of unfused norm/residual traffic, or 12 ms of memory SOL.
Prefill weighs 0.25 in the composite, and good GEMMs already run within 1.5-2x of SOL at
this shape (crawler feed #1), so prefill headroom is mostly the chunk count, the norm glue,
and the BF16-vs-FP8 GEMM rate.


## 3. Measured component tables (bs2, torch profiler, decode CUDA graph on)

Method: `scripts/drive.py` streams B diverse 1024-token prompts, the profiler captures 7 decode
steps, `scripts/classify_trace.py` buckets every kernel into a component
(`results/components_b{1,8}.json`), and `scripts/sol_table.py` joins those buckets with the
§2 byte model (`results/sol_table_b{1,8}.json`). Ceilings are measured on this card by
`scripts/microbench.py` (`results/microbench_bs2.json`): read BW 1621 GB/s (copy 1527), and
the SOL BW used below is the best streaming read seen, lm_head at M=8, **1674 GB/s**. Dense
BF16 235 TFLOP/s, FP8 669 TFLOP/s. A graph-replayed empty launch costs 0.81 µs.

`sol_fraction` = SOL µs / achieved µs. `share` = achieved / traced step. `rank` =
share x (1 - sol_fraction), the ordering rule from crawler feed #2. Distinct experts per
layer are measured with `--enable-return-routed-experts` (`results/experts_b*.json`):
**8.0 / 35.4 / 71.0 at B = 1 / 8 / 32**, against 8 / 51.6 / 111.8 for uniform routing. Real
routing touches about 31% fewer experts than the §2 placeholder at B=8, so the §2 expert row
(5.18 GB) re-bases to 3.55 GB.

### B=8 decode: timed 9.30 ms/step, SOL 5.60 ms, **e2e sol_fraction 0.60**

| Component (sorted by rank) | Launches | SOL µs | Achieved µs | Share | sol_fraction | Gap µs | Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| routed experts (NVFP4 grouped GEMM + glue) | 210 | 2123 | 3065 | 33.3% | 0.69 | 943 | 0.103 |
| norms, RoPE, KV write | 392 | 13 | 840 | 9.1% | 0.02 | 522 | 0.090 |
| step setup + idle gaps | 48 | 0 | 547 | 6.0% | 0.00 | 508 | 0.060 |
| dense MLP (BF16 GEMMs + GeGLU) | 90 | 643 | 1029 | 11.2% | 0.63 | 386 | 0.042 |
| o_proj (BF16) | 30 | 485 | 832 | 9.0% | 0.58 | 347 | 0.038 |
| attention, sliding (FP8 KV) | 50 | 503 | 697 | 7.6% | 0.72 | 194 | 0.021 |
| router GEMM + top-k | 90 | 14 | 187 | 2.0% | 0.07 | 114 | 0.019 |
| qkv_proj (BF16) | 30 | 865 | 975 | 10.6% | 0.89 | 111 | 0.012 |
| attention, full (FP8 KV) | 10 | 54 | 120 | 1.3% | 0.45 | 66 | 0.007 |
| lm_head (BF16) + sampling | 1 | 897 | 901 | 9.8% | 1.00 | 3 | 0.000 |
| **step (traced)** | **~950** | **5596** | **9194** | | **0.61** | **3598** | |

The routed-experts row splits into grouped GEMMs 2537 µs (60 launches) and MoE glue 528 µs
(150 launches: input expand + FP4 quantize, activation, finalize/un-permute, stride setup).
The glue row splits into norms 483 µs (211 launches) and RoPE + FP8 KV write 358 µs (181
launches; the FP8 quantize is a separate ATen `ConvertToFloat8E4M3` elementwise kernel).

### B=1 decode: timed 5.99 ms/step, SOL 3.43 ms, **e2e sol_fraction 0.57**

| Component (sorted by rank) | Launches | SOL µs | Achieved µs | Share | sol_fraction | Gap µs | Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| routed experts | 210 | 480 | 1280 | 19.7% | 0.37 | 800 | 0.123 |
| norms, RoPE, KV write | 392 | 2 | 632 | 9.7% | 0.00 | 313 | 0.097 |
| step setup + idle gaps | 48 | 0 | 533 | 8.2% | 0.00 | 494 | 0.082 |
| attention, sliding | 50 | 63 | 342 | 5.3% | 0.18 | 279 | 0.043 |
| dense MLP | 90 | 640 | 912 | 14.0% | 0.70 | 272 | 0.042 |
| router GEMM + top-k | 60 | 13 | 164 | 2.5% | 0.08 | 115 | 0.023 |
| qkv_proj | 30 | 862 | 1006 | 15.5% | 0.86 | 144 | 0.022 |
| o_proj | 30 | 483 | 584 | 9.0% | 0.83 | 101 | 0.016 |
| attention, full | 10 | 7 | 106 | 1.6% | 0.06 | 98 | 0.015 |
| lm_head + sampling | 1 | 884 | 941 | 14.5% | 0.94 | 57 | 0.009 |
| **step (traced)** | | **3433** | **6499** | | **0.53** | | |

The B=1 traced step (6.50 ms) is longer than the timed one (5.99 ms) because profiler
instrumentation inflates the many short kernels; shares are taken from the trace, and the
timed number is the one quoted.

### Kernel identity of the BF16 GEMMs

Every BF16 GEMM in the decode graph (qkv_proj, o_proj, dense gate_up/down, router, lm_head)
runs `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x{1,2}_tn_align8`: cuBLAS's
**SM80 WMMA fallback** with 16x16 tiles, not an SM120 kernel. It is fine where N is huge
(lm_head 1.00, qkv 0.89), but **o_proj drops from 0.83 at B=1 to 0.58 at B=8** (584 -> 832 µs
for the same 0.81 GB of weights) and the dense MLP from 0.70 to 0.63. o_proj is K=4096
(8192 on the 5 full layers) -> N=2816, the shape with the fewest output tiles per weight
byte. Why its time grows 42% from M=1 to M=8 at constant bytes is not profiled yet (no ncu);
the candidate explanations are tile occupancy and the 16-row WMMA issue cost.

## 4. Knob screen (screen, unpaired: single launch, 3 reps each, nothing here is a gain)

`results/screen_summaries.txt`. One server launch per config with the `base` flags plus the
knob; B=8 x 1024 x 128 and B=1 x 1024 x 128, radix cache flushed per rep. The base repeat
sizes launch-to-launch drift: B=8 decode 9.318 vs 9.288 ms (0.3%), B=1 5.988 vs 5.994.

| Config | Flags | B=8 prefill s | Δ | B=8 decode ms | Δ | B=1 decode ms | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| base | — | 0.2858 | | 9.318 | | 5.988 | |
| base_repeat | — | 0.2867 | +0.3% | 9.288 | −0.3% | 5.994 | +0.1% |
| chunk8k | `--chunked-prefill-size 8192 --mem-fraction-static 0.718` | 0.2780 | −2.8% | 8.909 | **−4.2%** | 5.998 | +0.1% |
| chunk16k | `--chunked-prefill-size 16384 --max-prefill-tokens 16384 --mem-fraction-static 0.718` | 0.2786 | −2.6% | 8.896 | −4.4% | 6.008 | +0.3% |
| splits16 | `--triton-attention-num-kv-splits 16` | 0.2855 | −0.1% | 9.201 | −1.1% | 5.869 | **−2.0%** |
| splits4 | `--triton-attention-num-kv-splits 4` | 0.2873 | +0.5% | 9.403 | +1.0% | 6.275 | +4.8% |
| cg_bs8 | `--cuda-graph-max-bs-decode 8` | 0.2866 | +0.3% | 9.236 | −0.8% | 5.990 | 0.0% |
| contdec4 | `--num-continuous-decode-steps 4` | 0.2862 | +0.1% | 9.305 | −0.1% | 5.990 | 0.0% |
| nooverlap | `--disable-overlap-schedule` | 0.2911 | +1.9% | 9.747 | +4.7% | 6.350 | +6.0% |
| nocg | `--disable-cuda-graph` | 0.3161 | +10.6% | 14.592 | +56.6% | 13.521 | +126% |
| kv_bf16 | `--kv-cache-dtype bf16` | 0.2949 | +3.2% | 9.647 | +3.7% | 5.920 | −1.2% |

Open questions the screen raises (resolved only by the gate, `trials/`):
- **chunk8k moves B=8 decode by −4.2%** though chunk size should not touch a decode step.
  Two confounds: the chunk runs also passed `--mem-fraction-static 0.718` (the first launch
  without it failed), which shrinks the KV pool, and they are the only configs with
  `cache_hit_tok=0` at B=8 (base saw 11-12). A one-chunk prefill also admits all 8 requests
  in one wave, so the first decode steps run at full B=8 instead of a ragged tail; whether
  the measured "decode ms/step" absorbs that is a workload-shape effect, not a kernel one.
  B=1 (one chunk either way) shows nothing, consistent with a wave artefact. T1 tests it.
- **splits16 helps B=1 (−2.0%) more than B=8 (−1.1%)**, matching the B=1 sliding-attention
  sol_fraction of 0.18: at one sequence the split count is the only parallelism.
- **kv_bf16 is +3.7% at B=8 and −1.2% at B=1**: FP8 KV saves bytes when attention bytes
  matter (B=8) and costs the separate FP8 quantize kernels when they do not (B=1).
- **Launch/CPU share:** no CUDA graph costs +57% / +126%; overlap scheduling is worth 4.7% /
  6.0%. The idle-gap row (0.5 ms) is what remains with both on.

## 5. What the measurements change

1. **Decode runs at 0.60 of SOL at B=8 and 0.57 at B=1.** Total recoverable is about 3.6 ms
   at B=8 and 3.1 ms (traced) at B=1, so no single lever beyond the experts is worth more
   than ~10% of the step.
2. **Routed experts stay first, but smaller than §2 predicted:** 33% of the B=8 step (not
   47% of bytes), because routing is skewed (35 distinct, not 52) and the grouped GEMM already
   runs at 0.69. Their gap is 0.94 ms, of which 0.53 ms is MoE glue launches. At B=1 the
   experts run at 0.37 (8 experts, 1 token each: padding-dominated tiles).
3. **Glue is second and almost all latency:** 392 launches for 13 µs of bytes
   (sol_fraction 0.02). Vault C14 holds: these intermediates sit in L2, so fusion saves
   launches, not DRAM. Removing half the launches is worth ~0.4 ms (~4.5%).
4. **Idle gaps (0.5 ms, 5-8%)** are the launch/graph floor that CUDA graph and overlap do not
   remove; the screen shows the knobs (cg_bs8, contdec4) do not move it.
5. **The SM80 WMMA fallback is a B=8-specific loss on o_proj and dense MLP** (0.73 ms gap
   combined). qkv_proj and lm_head are already at 0.89-1.00, so H2 (FP8 weights) is the only
   way to go below their byte floor, while a better small-M BF16 kernel is a pure-speed,
   no-fidelity-risk lever on o_proj/dense MLP.
6. **lm_head is at SOL (1.00).** Only fewer bytes (H3, FP8 or exact screening) helps it.
