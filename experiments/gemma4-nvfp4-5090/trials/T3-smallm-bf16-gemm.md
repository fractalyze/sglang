# T3 (gemma4nv-w3-t3): replace the SM80 WMMA fallback for small-M BF16 o_proj and dense MLP

**Status: preregistered, not implemented. Waiting for the coordinator's go before any edit
under `python/sglang` or kernels.** Top code-level lever of the measured re-rank
(`analysis/HYPOTHESES.md` R3), chosen over R1/R2 for the first code trial because it carries
no precision change, a small blast radius, and a cheap config-only probe comes first.

## Evidence (bs2 profile, `analysis/PROFILE.md` §3)

- Every BF16 GEMM in the decode graph runs cuBLAS's
  `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x{1,2}_tn_align8`, an SM80 WMMA
  kernel, on SM120. `--bf16-gemm-backend` offers nothing else here: `auto` picks `cutedsl`
  only on SM10x, and `gemv` raises outside SM90 (`layers/quantization/unquant.py:214-226`).
- o_proj: sol_fraction 0.83 at B=1 but **0.58 at B=8** (584 → 832 µs, same 0.81 GB of
  weights). Dense MLP gate_up/down: 0.70 → 0.63 (912 → 1029 µs). qkv_proj (0.89) and lm_head
  (1.00) are fine on the same kernel family, so the loss is shape-specific (N=2816 output,
  few tiles).
- Combined B=8 gap of o_proj + dense MLP: **733 µs of a 9.30 ms step**.

## Plan (after go)

0. **Config-only probe first (microbench, no server):** time `F.linear` at M ∈ {1, 8, 16, 32}
   for the o_proj (2816x4096, 2816x8192) and dense (2x2816→ gate_up, down) shapes under
   `TORCH_BLAS_PREFER_CUBLASLT=1` / `torch.backends.cuda.preferred_blas_library("cublaslt")`
   and with cuBLASLt heuristics enumerated. If cuBLASLt already picks an SM120 kernel at
   ≥ 0.8 sol_fraction, T3 becomes a config-only env trial (no code).
1. Otherwise: an SM120 small-M (M ≤ 32) BF16 split-K GEMV/GEMM for these shapes as a JIT
   kernel (`add-jit-kernel` skill), dispatched from `UnquantizedLinearMethod.apply` behind a
   `--bf16-gemm-backend` choice, with unit tests against `F.linear`.

## Prediction (frozen at registration)

| metric (bs2, gate ratio of sums) | predicted |
|---|---|
| W8 decode | −2.5 … −5% (recover 30-60% of the 733 µs gap, minus a clock-bin give-back, vault C2) |
| W8 prefill | 0 ± 0.5% (prefill M ≥ 4096 keeps cuBLAS) |
| **W8 composite gain** | **1.02 … 1.04** |
| W1 TPOT | −0.5 … −2% (B=1 gap is 373 µs) |
| Fidelity | reorder tier (split-K changes the fp32 accumulation order); within the calibrated launch-to-launch KL |

Falsified if the step-0 microbench finds no kernel faster than the WMMA fallback by ≥ 20% at
M=8 on the o_proj shape, or if the gate's W8 composite gain is below 1.01.
