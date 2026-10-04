# T3d (gemma4nv-b2-t3d): the FP8 o_proj on the Triton GEMM above 32 rows, in M tiers

**Status: retired (W13, 2026-10-03).** Gate `T3d-20261003-165907-build-server-2-521351` vs base4-spec-fp8head-fp8lmhead:

| metric | gain | 95% CI |
|---|---:|---|
| W8 composite (deciding) | 1.0046 | 1.0004-1.0087 |
| W1 TPOT | 1.100 (2.732 → 2.484 ms) | |
| W32 | 1.023 | 0.948-1.104 |

- **Fidelity passed** and integrity held.
- **The W8 composite is below the 1% bar,** and the prediction (+2.5 … +6%) is falsified.
- **The W1 and W32 moves are acceptance on divergent greedy text, not speed.** B=1 per-round kernel time is unchanged: verify 6.406 vs 6.411 ms. See `REPORT-bs2-w13.md`.

Below is the frozen registration.

## Change

- **Code:** commit `350d053100`. There is no new switch: the change sits under the existing `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM`, the same way T-SPEC6b's M blocks did. The control and the candidate differ only by commit.
- **Before.** base4's FP8 E4M3 weight-only o_proj (T3b) ran the small-M Triton kernel up to M=32. Above that it upcast the whole weight to bf16 and called cuBLAS.
  - W12 measured that route at 2.1-2.5x cuBLAS BF16 on the qkv shapes.
  - It is the path of every B=8 verify (M=48 at k=5), every W32 verify (M=192), and every prefill.
- **After.** The two o_proj shapes carry M tiers. Each tier caps BLOCK_M, so a larger M runs one program per M block and re-reads the E4M3 weight from L2.

  | M | tile (BLOCK_M cap, BLOCK_N, BLOCK_K, stages, warps) |
  |---|---|
  | ≤ 32 | unchanged (next_pow2(M), 32, 256, 4, 4) |
  | ≤ 64 | (16, 32, 128, 4, 4) |
  | ≤ 2048 | (32, 64, 128, 3, 4) |
  | ≤ 8192 | (64, 128, 64, 3, 8) |
  | > 8192 | the upcast route, unmeasured and not reached: chunked prefill is at most 8192 |

- **One E4M3 copy stays,** so memory does not change. Dequant-once (a second, BF16 copy, +0.41 GB) would win only above M ≈ 4096, by 5-7%, so it was not taken.
- **Numerics.** The tier is unchanged: an exact bf16 upcast per tile and fp32 accumulation. The per-channel scale is now applied in the fp32 epilogue rather than after a bf16 rounding of the unscaled output, and the K-sum order differs from cuBLAS's.
- **Tests:** `test/registered/gemm/test_triton_small_m_bf16_gemm.py`, 5 cases, all passing on the bs2 5090, and `test_gemma4_fp8_lm_head.py` (6) still passes.
  - The FP8 GEMM stays within one bf16 rounding of the dequantized fp32 reference at every tier edge (M = 32, 33, 48, 64, 65, 192, 2048, 2049, 8192).
  - The route takes M=48 on the Triton kernel and M=8193 on the upcast.

## Evidence: microbench (`trials/t3/oproj_large_m_bench.py` → `t3/results/w13-oproj-large.json`)

Method as T3: CUDA graph of 64 calls, best of 7, weights rotated over 256 MB, on the bs2 5090. Times in µs.

| shape | M | cuBLAS BF16 (dequant-once bar) | upcast + cuBLAS (control) | Triton FP8 tier (candidate) |
|---|---:|---:|---:|---:|
| o_proj sliding 2816 x 4096 | 48 | 18.2 | 41.7 | 12.8 |
| | 192 | 26.7 | 57.4 | 29.5 |
| | 1024 | 131.3 | 170.1 | 131.3 |
| | 8192 | 852.6 | 1015.6 | 925.4 |
| o_proj full 2816 x 8192 | 48 | 32.8 | 79.5 | 24.9 |
| | 192 | 51.8 | 114.1 | 59.8 |
| | 1024 | 241.7 | 313.5 | 266.5 |
| | 8192 | 1763.1 | 1927.6 | 1880.1 |

- The candidate beats the control at every measured M from 48 to 8192.
- `torch._scaled_mm` W8A8 (activation quantization, a different numerics tier) was timed for information only. It loses below M=384 and above M=2048.

## Prediction (frozen)

Each forward runs 25 sliding and 5 full o_proj calls. The saving per target forward is:

| forward | saving |
|---|---|
| B=8 verify (M=48) | 0.99 ms of a ~16 ms round |
| W32 verify (M=192) | 0.97 ms |
| prefill at M=1024 | 1.2 ms |
| prefill at M=8192 | 2.4 ms |

- **B=1 is unaffected:** its verify is M=6, under the unchanged tier. Decode at B < 6 is unaffected too.
- **Acceptance at B=1 cannot move,** for the same reason. Acceptance at B=8 can move only through the scale-epilogue and K-order rounding.

| metric (bs2, gate ratio of sums vs base4-spec-fp8head-fp8lmhead) | predicted |
|---|---|
| **W8 composite (deciding)** | **+2.5 … +6%** (decode +3 … +6.5%, prefill +1 … +5%) |
| W1 TPOT (guard) | -0.3 … +0.3% (gain 0.997-1.003) |
| W32 tok/s (guard, checked against its bar by hand; the default rule guards only W1) | +0.5 … +2.5% |
| Fidelity | pass; decode and forced KL within the control's A/A spread (approx tier, no precision change) |
| GPU memory / KV pool | unchanged |

- **Decision rule:** `--decide-on w8_composite`, with W1 as the guard.
- **Expected verdict: kept.**

**Falsified if:**
- the W8 composite does not clear its bar, or is below 1.015;
- W1 or W32 regresses past its bar;
- fidelity fails;
- the server fails to load.
