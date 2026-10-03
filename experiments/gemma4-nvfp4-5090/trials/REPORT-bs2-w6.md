# W6 report: T3, small-M BF16 GEMM off cuBLAS's SM80 WMMA fallback (gemma4nv, build-server-2)

Branch `jumanzii/gemma4nv-analysis`; host build-server-2. Every timing is a gate verdict
(paired ABBA, ratio of sums) against bs2 controls; the microbench numbers are kernel-level
and decide only the kernel choice.

**Verdict: kept.** Gate run `T3-20261003-092422-build-server-2-fa9e0f` promotes (integrity
ok, fidelity pass). The W8 composite gain is **1.0445**.

## 1. Step 0: microbench (`t3/gemm_microbench.py`, `t3/results/mb1.json`, `t3/results/mb2-fp8.json`)

Method:
- Each candidate is timed inside a CUDA graph of 64 calls, best of 7 replays.
- Weights rotate over at least 256 MB of copies, so each call reads its weight from DRAM, as the 30-layer model does.
- SOL = (weight + activation bytes) / 1.65 TB/s.
- At M=1024 the GEMM is compute-bound, so its sol_fraction (about 0.17) is not a meaningful bound.

Candidates and their status on SM120:
- cuBLAS default.
- `torch` with the cuBLASLt preference.
- FlashInfer `mm_bf16` with the `cublaslt` (heuristic search), `cudnn` and `tinygemm` backends. Its `cutlass`, `tgv` and `cute-dsl` backends refuse SM120.
- A Triton GEMM sweep: BLOCK_N {16, 32, 64} x BLOCK_K {128, 256} x split-K {1, 2, 4} x stages {3, 4}.
- sgl-kernel has no SM120 BF16 dense GEMM.

Every row below is from `mb1.json`. Times are in µs; sol_fraction is in parentheses.

| shape (N x K) | M | SOL | cuBLAS | cuBLASLt heur. | cuDNN | FI tinygemm | best Triton (sweep) | **T3 kernel** | vs cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| o_proj sliding (2816x4096) | 1 | 14.0 | 17.1 (0.82) | 17.1 | 17.5 | 18.7 | 15.4 | **15.4** (0.91) | -10% |
| | 8 | 14.0 | 22.4 (0.63) | 22.5 | 22.5 | 18.2 | 15.5 | **15.5** (0.91) | **-31%** |
| | 16 | 14.1 | 22.5 (0.63) | 22.8 | 22.8 | 22.6 | 15.7 | **15.8** (0.89) | -30% |
| | 32 | 14.2 | 18.1 (0.79) | 18.2 | 18.3 | 33.9 | 16.3 | **16.3** (0.87) | -10% |
| | 1024 | 22.6 | 131.1 | 131.2 | 131.2 | 699.2 | - | cuBLAS kept | - |
| o_proj full (2816x8192) | 1 | 28.0 | 28.8 (0.97) | 28.8 | 28.8 | 35.3 | 29.5 | **29.6** (0.94) | +3% |
| | 8 | 28.1 | 43.7 (0.64) | 43.7 | 43.9 | 35.0 | 29.7 | **29.8** (0.94) | **-32%** |
| | 16 | 28.2 | 43.9 (0.64) | 44.1 | 44.1 | 42.4 | 29.9 | **30.0** (0.94) | -32% |
| | 32 | 28.4 | 32.6 (0.87) | 32.8 | 32.8 | 62.7 | 30.5 | **30.5** (0.93) | -6% |
| | 1024 | 41.6 | 239.5 | 239.8 | 239.7 | 1323.2 | - | cuBLAS kept | - |
| dense gate_up (4224x2816) | 1 | 14.4 | 17.9 (0.81) | 17.9 | 17.9 | 17.0 | 15.8 | **16.1** (0.90) | -10% |
| | 8 | 14.5 | 18.7 (0.77) | 19.1 | 19.1 | 16.8 | 16.1 | **16.3** (0.89) | -13% |
| | 16 | 14.6 | 18.9 (0.77) | 19.2 | 19.3 | 22.4 | 16.4 | **16.6** (0.88) | -12% |
| | 32 | 14.7 | 18.7 (0.78) | 19.3 | 19.3 | 35.5 | 17.1 | **17.5** (0.84) | -6% |
| | 1024 | 23.2 | 132.4 | 133.0 | 133.0 | 869.0 | - | cuBLAS kept | - |
| dense down (2816x2112) | 1 | 7.2 | 10.3 (0.70) | 10.3 | 10.4 | 14.0 | 9.4 | **9.4** (0.77) | -9% |
| | 8 | 7.3 | 13.2 (0.55) | 13.3 | 13.3 | 13.1 | 9.5 | **9.5** (0.77) | **-28%** |
| | 16 | 7.3 | 13.2 (0.55) | 13.5 | 13.5 | 17.8 | 9.5 | **9.5** (0.77) | -28% |
| | 32 | 7.4 | 12.9 (0.57) | 13.3 | 13.3 | 25.8 | 10.2 | **10.2** (0.73) | -21% |
| | 1024 | 13.3 | 77.2 | 77.6 | 77.6 | 590.1 | - | cuBLAS kept (cuBLASLt 68.8) | - |
| qkv sliding (8192x2816), reference | 8 | 28.1 | 31.0 (0.90) | 31.2 | 31.3 | 31.8 | 30.1 | not routed | - |

What the table shows:
- **Config-only fails.** cuBLASLt's heuristic search and cuDNN land on the same kernel speed as cuBLAS at every small M. Step 0's "config-only" exit does not apply.
- **The falsifier is cleared.** At M=8 on o_proj the T3 kernel is 31% faster than the fallback; the preregistered bar was 20%.
- **The winner per shape is the single-pass Triton GEMM** (split-K 1) for M ≤ 32.
  - Split-K bought at most 2% (gate_up), and it needs a workspace and a second launch, so it was dropped.
  - The one regression is o_proj full at M=1 (+3%, 0.8 µs x 5 layers). It is kept for simplicity and covered by the W1 guard below.
- **Prefill stays on cuBLAS.**
- **FP8 W8A8 reference** (`torch._scaled_mm`, per-tensor scales; `mb2-fp8.json`; changes numerics, so not part of T3): o_proj sliding 12.3 µs, o_proj full 20.0 µs and down 9.7 µs at M=8. That is a further 20-33% on o_proj and nothing on down. It is a candidate for a separate T3b with its own prediction. Not started.

## 2. Implementation (commit `1fd77e64b0`, on base `a9871012` with nothing else under `python/`)

- `python/sglang/kernels/ops/gemm/triton_small_m_bf16_gemm.py`: the kernel and a 4-shape (N, K) allowlist. Each shape has its tuned (BLOCK_N, BLOCK_K, stages). `MAX_M` = 32.
- `UnquantizedLinearMethod`:
  - `__init__` reads the switch once.
  - `apply` routes to the kernel when the switch is on and the call is eligible: bf16 and contiguous, no bias, not compiling, batch-invariant mode off, M ≤ 32, and an allowlisted shape.
  - Every other call keeps `F.linear`.
  - `apply_into` and `apply_with_addend` are unchanged; Gemma-4 does not call them for these layers.
- Env kill switch `SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM = EnvBool(False)` in `environ.py`, next to the BF16 split-K switch. It is default off.
- Tests: `test/registered/gemm/test_triton_small_m_bf16_gemm.py`. They pass on the bs2 5090 (`/home/jooman/gemma4nv/w6/ut2/run.log`).
  - **Numerics:** every element of both the Triton kernel and cuBLAS lies within half a bf16 ulp of the fp32 reference, plus 2·K·2⁻²⁴·Σ|x||w| of fp32 summation-order error. The stated contract is therefore "within bf16 rounding of the same fp32 dot product". It is not bitwise.
    - At M=8 and 16 the microbench outputs were bitwise equal to cuBLAS.
    - At M=1 and 32, where cuBLAS uses another kernel, they differ by up to one bf16 step.
    - A first draft asserted "at most 1 ulp of the output" and failed near cancelling outputs, so the bound is now the summation-order one.
  - **Switch:** off means zero kernel calls. On means one call for the M=8 o_proj shape and none for M=1024 or a non-allowlisted shape.
- `model_runner.py` and the other frozen files are untouched.

## 3. Gate (`T3-20261003-092422-build-server-2-fa9e0f`, 4 ABBA pairs)

Setup:
- Control: `w32b-mem076`, which is base2 (base + `--mem-fraction-static 0.76`, W5's pinned ref).
- Candidate: `smallm-gemm`, which is base2 on commit `1fd77e64b0` plus the switch.
- `gate/refs.json` merges `jumanzii/gemma4nv-gate` at `7a45e7f4c6`, so W5's refs and integrity fixes are in.
- Deployed to bs2 `src-gate` (`DEPLOYED.txt`).
- The candidate prebuild `prebuild-smallm-gemm-20261003-092326-build-server-2-36e495` ran no compilers.

| metric | control | candidate | gain | per-pair σ | bar | predicted (frozen) |
|---|---:|---:|---:|---:|---:|---|
| **W8 composite** | | | **1.0445** | 0.09% | 1% | 1.02 … 1.04 |
| W8 decode | | | 1.0603 | 0.06% | 1% | −2.5 … −5% time |
| W8 prefill | | | 0.9984 | 0.23% | 1% | 0 ± 0.5% |
| W1 TPOT (guard) | 5.993 ms | 5.874 ms | 1.0202 (−1.98%) | 0.04% | 1% | −0.5 … −2% |
| W32 tok/s (guard, ungated) | 1541 | 1568 | 1.0174 | 0.33% | - | - |

Checks:
- **Fidelity:** pass.
  - Free-running hidden-set KL mean 0.0130 / p99 0.318 vs the control's 0.0155 / 0.390.
  - The teacher-forced pass is identical to the control, because prefill batches never take the new path.
- **Integrity:** ok. Timed-output agreement is 0.55, and there are no undeclared argument diffs.
- **Prediction:** the composite came in 0.45 points above the frozen interval, and W1 sits at its edge.
  - The microbench sum predicts the size. At M=8 it saves about 425 µs per W8 decode step: o_proj 242 µs, down 111, gate_up 72. That is 4.6% of a 9.3 ms step, against 6.0% measured.
  - The preregistration assumed a recovery of only 30-60% of the 733 µs profile gap.
- No separate base2 A/A on bs2: the gate only needs `reference/noise.json` (base A/A, σ 0.02%), and this run's own per-pair σ is under 0.1% on every gated metric. Every bar sits at the 1% floor.

**Host-safety record.** Peaks per phase:

| phase | min MemAvailable | peak tree RSS | peak load | compilers |
|---|---:|---:|---:|---:|
| microbench (Triton sweep, ptxas only) | 53.8 GB | 1.5 GB | 0.4 | 2 (0.24 GB) |
| gate weight load | 50.3 GB | 7.9 GB | 1.5 | 0 |
| gate autotune | 49.6 GB | 6.2 GB | 1.4 | 0 |
| gate serving / timed | 49.3 GB | 6.6 GB | 1.4 | 0 |

- Every engine process ran under host.lock in the 24G scope, with `MAX_JOBS=2` for my own runs.
- Weight load is the RSS peak: 7.9 GB here, and 11.5 GB in W3's earlier prebuild.
- The FlashInfer JIT cache held, so nothing compiled.
- These are in this report rather than `BASELINE.md`, because the bs3 workers own that file.

## 4. Vault (shared checkout, not pushed; only my paths committed)

Commits:
- `f061b10`: raw import of `gemma4nv-bs2` (snapshot `20261003T0033Z-norev`). It holds the T3 run, the T3 prebuild and the T2 re-evaluation reports already on bs2.
- `38843e2`: verdict **kept** and the new claim `c-gemma4nv-sm120-small-m-bf16-gemm-wmma-fallback` (mechanism, measured-direct, supports).
- `64ec4f7` and `5ea9f0d`: the trial page prose.

Notes:
- bs2 still has no `meta/ledgers.yaml` entry, so the measurements are `--manual-measurement` rows citing `report.json`.
- No `result_stack` was written. The vault requires a stack that this trial generates, and a "base2 + T3" stack page is left to whoever pins the next base.

## 5. Next

1. **Promote T3 into the next base** (base3 = base2 + `SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM=1` on `1fd77e64b0` or later). Later trials then stack on it.
2. **T3b:** FP8 W8A8 for o_proj, which would save a further ~20-33% of the o_proj GEMM. It changes numerics, so it needs its own prediction and the full fidelity gate. Not started.
3. **Extend the allowlist** to other narrow-N small-M BF16 linears, after measuring them:
   - the router (N=128);
   - qkv_proj, which gained 3% at M ≤ 16 in the microbench.
