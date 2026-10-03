# T-SPEC6a (gemma4nv-b2-tspec6a): the small-M BF16 GEMM on qkv_proj and the target lm_head at verify widths

**Status: registered, not gated (W12, 2026-10-03).** The frozen prediction is below the gate's 1% W1 bar, so a gate run could not keep it; the coordinator agreed to register it and not gate it, as T3c was retired for the same reason.

Registered 2026-10-03 by W12 (bs2) after the microbench and before any code. No code exists for it.

## Question

W10's B=1 profile of `base4-spec-fp8head` (k=5) puts 1.96 ms of a ~10.3 ms round on cuBLAS's SM80 WMMA fallback: the target's qkv_proj (25 sliding, 5 full) and its lm_head, all at M = 1 + k = 6. T3c moved the same layers to T3's single-pass Triton BF16 kernel at plain decode M and was retired at -0.92% W1. Are verify widths (M = 6..48) a different shape class for that kernel?

## Evidence (W12 microbench, `trials/spec/verify_gemm_bench.py`)

Results: `trials/spec/results/w12-verify-gemm.json` (bs2 RTX 5090, CUDA graph of 64 calls, best of 7 replays, weights rotated over >= 256 MB; T3c's repeats put the noise at 0.25%). Best BF16 Triton tile per cell against cuBLAS:

| shape (N x K) | M=6 | M=12 | M=24 | M=48 |
|---|---:|---:|---:|---:|
| qkv sliding (8192 x 2816), cuBLAS µs | 31.2 | 31.2 | 31.2 | 31.9 |
| Triton BF16 | 29.5 (-5.6%) | 29.6 (-5.2%) | 30.2 (-3.2%) | 31.9 (+0.1%) |
| qkv full (10240 x 2816), cuBLAS µs | 37.7 | 37.7 | 37.6 | 39.1 |
| Triton BF16 | 36.1 (-4.2%) | 36.3 (-3.8%) | 36.8 (-2.0%) | 38.3 (-2.1%) |
| lm_head (262144 x 2816), cuBLAS µs | 880.6 | 884.1 | 909.0 | 925.2 |
| Triton BF16 | 876.9 (-0.4%) | 880.1 (-0.5%) | 883.0 (-2.9%) | 900.3 (-2.7%) |

- cuBLAS already streams the 1.48 GB lm_head at about 93% of DRAM bandwidth at M=6, so a BF16 kernel has nothing to win there.
- The qkv gain is the same 4-6% T3c saw at M=1-16, and it fades by M=36-48 (the B=8 verify).

## Prediction (frozen)

Per B=1 round at k=5 (M=6): 25 x 1.7 + 5 x 1.6 + 3.7 = 54 µs of ~10.3 ms.

| metric (bs2, gate ratio of sums vs base4-spec-fp8head) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-0.3 … -0.8%** |
| W8 composite | 0 … +0.3% (gains vanish at M=48) |
| W32 tok/s | 0% (M=192 keeps cuBLAS) |
| Fidelity | reorder tier (fp32 accumulation in K order; bf16 rounding of the same sum) |

**Expected verdict: retired without a gate**, because the deciding interval lies entirely below the 1% W1 bar. **Falsified if** a later measurement of this exact change shows a W1 TPOT gain of 1% or more.

The FP8 weight-only variant of the same kernel is a different trial: it halves the weight bytes (qkv 15.5 / 19.2 µs, lm_head 440 µs at M=6) and changes the target numerics. That is T-SPEC6b (qkv_proj) and T-SPEC6c (target lm_head).
