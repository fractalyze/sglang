# Where the NVFP4 MoE time goes at MTP verify widths (W12, bs2)

**Question.** W10's B=1 profile of `base4-spec-fp8head` (k=5) puts 2.28 ms of a ~10.3 ms round in the routed MoE. Where does it go at M = (1 + k) * B, and what is the most promising MoE trial?

**Status:** analysis plus one preregistered trial (T-MOE1). No kernel work without coordinator approval.

## Sources

- **Trace:** W10's torch-profiler trace `runs/w10-prof-fp8head/trace_b1/*DECODE*` (12 verify rounds, B=1, k=5, M=6), re-read per kernel inside `step[VERIFY bs=1]`.
- **Routing:** W8's recorded routing (`runs/w8-experts-104023/routing_{hidden,timing}.npz`, `[tokens, 30 layers, top-8]`), windowed into verify groups of 6 consecutive tokens. For B=8, eight streams' windows are unioned at the same step.
- **Tactics:** FlashInfer's SM120 grouped-GEMM candidate list (`cutlass_heuristic.cpp`, `get_candidate_configs_sm120`) in the bs2 venv.
- **Model:** 128 experts, top-8, hidden 2816, `moe_intermediate_size` 704. Per expert, fc1 (gate+up) is 1408 x 2816 and fc2 is 2816 x 704, in E2M1 with a UE4M3 scale per 16 values.

## 1. Kernel breakdown, B=1 verify (median of 12 rounds, 30 MoE layers)

| kernel | ms per round | per layer |
|---|---:|---:|
| CUTLASS grouped GEMM, fc1 (`128x32x128` CTA tile, 6 stages) | 1.13 | 37.8 µs |
| CUTLASS grouped GEMM, fc2 (`128x32x256` CTA tile, 3 stages) | 0.65 | 21.6 µs |
| `finalizeMoeRoutingKernel` (weighted un-permute + reduce) | 0.18 | 5.9 µs |
| `expandInputRowsKernel` (permute + FP4-quantize activations) | 0.10 | 3.2 µs |
| `doActivationKernel` (GELU-gate between fc1 and fc2) | 0.08 | 2.6 µs |
| `computeStridesTmaWarpSpecializedKernel` | 0.06 | 2.1 µs |
| `fusedBuildExpertMapsSortFirstTokenKernel` | 0.06 | 2.0 µs |
| `_gemma4_routing_kernel` (router top-8 + softmax) | 0.05 | 1.7 µs |
| **total** | **2.33** | **77.6 µs** |

**The grouped GEMMs are 77% of the MoE (1.80 ms).** The other 23% is six small launches per layer (0.53 ms).

## 2. How many experts a verify touches

| verify | (token, expert) pairs | distinct experts per layer, mean (p10-p90) | rows per active expert |
|---|---:|---|---:|
| plain decode, B=1 | 8 | 8.0 | 1 |
| **B=1, k=5 (M=6)** | 48 | **23.3** (17-30), hidden set; 23.9 on the timing corpus | **2.0** |
| B=8, k=5 (M=48) | 384 | 83.6 (73-94), hidden set; 66.9 (55-80) on the timing corpus | 4.6-5.7 |

**Consecutive tokens of one stream share few experts.** Six verify tokens touch 3x the experts of one token, not 6x, and B=8 touches 52-65% of all 128 experts per layer. So the bytes the MoE must stream grow almost linearly with the verify width.

## 3. Bytes vs time (B=1, M=6)

Per active expert: fc1 1.98 MB + 0.25 MB of scales, fc2 0.99 MB + 0.12 MB of scales.

Weight bytes per layer at 23.5 experts, against the measured-bandwidth SOL (1.65 TB/s, `gate peaks`):

| GEMM | MB per layer | SOL µs | measured µs | SOL fraction |
|---|---:|---:|---:|---:|
| fc1 | 52.4 | 31.8 | 37.8 | 0.84 |
| fc2 | 26.2 | 15.9 | 21.6 | 0.74 |
| both, 30 layers | 2,357 | 1.43 ms | 1.80 ms | 0.79 |

**Where the 0.37 ms gap comes from: tile padding, not empty runs.**

- **Every SM120 FP4 grouped-GEMM tactic has a CTA M of 128 or 256.** The list is 128x{32,64,128}x{64,128}B, 128x128x256B, 256x128x{64,128}B and 128x256x64B.
- **The autotuner picked the smallest, 128x32, and not a swapped one.** FlashInfer's candidate generator (`moe_gemm_template_dispatch.h`, about lines 646-674) copies every TMA warp-specialized config with `swap_ab = true`, and adds FINALIZE-fused copies when `supports_finalize_fusion`. The SM120 launcher has a `SwapAB` instantiation.
- **The trace shows neither in use:** an un-swapped `Shape<128,32,128>` grouped GEMM, and a separate `finalizeMoeRoutingKernel`. Unverified: whether the SM120 FP4 swap-AB / FINALIZE tactics are compiled into the JIT module, and whether the autotuner timed them at M=6 or skipped them (`calcMaxWorkspaceSize` drops invalid tiles silently).
- **So each active expert's 2 rows are padded to 128.** Of the MMA work, 98.4% multiplies zero rows. The padded fc1 is 1034 tiles x 128 x 32 x 2816 x 2 = 23.9 GFLOP per layer.
  - At the 5090's dense FP4 rate that costs about 14 µs if FP32 accumulation runs at full rate, or about 28 µs if GeForce halves it.
  - Either way it is of the same order as the 32 µs weight stream. The warp-specialized pipeline overlaps the two only partly, because each CTA tile runs only 22 K-iterations before its epilogue.
- **RUNSKIP-style empty runs (Yukon §5) do not apply here.** The grouped GEMM gets one problem per expert, and only active experts carry M > 0. The waste is the padding inside each expert's single tile, Yukon §5's "tiles are mostly padding", not runs over empty experts.
- **The glue (0.53 ms) is launch- and latency-bound.** Each of the six kernels moves under 100 KB per layer at M=6.

**At B=8 (M=48)** an expert has about 5 rows, so it is still 96% padding. The ~75 active experts stream about 250 MB per layer, so bytes dominate more there.

## 4. Candidate trials, ranked

| # | trial | mechanism | B=1 saving, ms per round | cost / risk |
|---|---|---|---:|---|
| **1** | **T-MOE1: swap-AB NVFP4 grouped GEMM (plus FINALIZE fusion if valid) at verify widths.** Weights take the MMA M dimension, and each expert's few tokens take the N dimension. It runs through FlashInfer's own swap-AB tactic if SM120 FP4 compiles one, else through a new kernel. | removes the padded MMA work, so the GEMMs approach the byte SOL; FINALIZE also drops `finalizeMoeRoutingKernel` (0.18 ms) | 0.20 … 0.45 (GEMMs 1.80 → 1.45-1.60, finalize up to -0.18) | **tactic selection** (no kernel work) if FlashInfer's SM120 swap-AB tactic exists and wins at step 0; otherwise **kernel work** (CUTLASS SM120 block-scaled, or Triton `tl.dot_scaled`). Numerics stay in the same tier: the same FP4 weights and activations, but the K-sum order changes. |
| 2 | Fuse the activation into fc1's epilogue | cuts one glue launch and one activation round-trip per layer | 0.05 … 0.10 | kernel work: the SM120 TMA-WS launcher accepts only NONE or FINALIZE, so a gated-activation epilogue would be new. |
| 3 | `computeStrides` + `buildExpertMaps` merged, or folded into routing | 2 launches → 1 | 0.05 … 0.08 | small, but it touches FlashInfer's runner |

**Most promising: T-MOE1 (rank 1).**
- It attacks the 79%-of-SOL GEMMs directly.
- It also helps B=8, where the padding is still 96%.
- Step 0 may turn it into a tactic-selection change with no kernel work, because FlashInfer already generates swap-AB and FINALIZE variants of every SM120 config.

## 5. T-MOE1 preregistration (frozen; do not implement before coordinator approval)

- **Change:** at verify widths (B=1 up to k=7, and B=8), run fc1 and fc2 as a swap-AB NVFP4 grouped GEMM, with fc2's FINALIZE fusion where it is valid. The weights are the MMA's M operand, and each expert's tokens are N. Larger batches keep the current tactic.
  - First choice: FlashInfer's own SM120 swap-AB / FINALIZE tactics, pinned through the MoE runner's tactic selection for these token counts. That is config or dispatch work behind a new default-off `SGLANG_OPT_*` switch.
  - Only if those tactics do not exist or lose does it become kernel work, which needs separate approval.
- **Step 0** (a microbench, before any gate or kernel):
  1. List every tactic FlashInfer's SM120 FP4 MoE runner can build, timing each at 6 and 48 tokens with the routing of §2. That covers swap-AB, FINALIZE and the tile list.
  2. Time the current tactic at 23-24 active experts with 1, 2, 4, 8 and 128 rows each.
  - **Falsified before any work** if the current tactic is within 5% from 1 to 8 rows (padding is free) and no listed tactic beats it by 5%.
- **Control:** the bs2 reference at the time of the gate (`base4-spec-fp8head`, or T-SPEC6b's candidate if kept).

| metric (bs2, gate) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-2 … -4.5%** |
| W8 composite (guard) | 0 … +3% |
| W32 tok/s (guard) | 0 … +2% (rows per expert exceed 8 at B=32, so most W32 steps keep CUTLASS) |
| Fidelity | approx tier; decode KL within +0.005 of the control |

**Falsified if:**
- step 0 shows the grouped GEMM within 5% from 1 to 8 rows per expert;
- the gate's W1 gain is below 1.015;
- either guard regresses beyond its bar.

### Gaps

> [!gap] Whether the 5090 runs block-scaled FP4 MMA with FP32 accumulation at the full or half GeForce rate. Step 0 measures the padding's cost directly instead of inferring it.

> [!gap] The B=8 split between fc1 and fc2 is extrapolated from the routing union, not profiled. No B=8 verify trace exists yet.
