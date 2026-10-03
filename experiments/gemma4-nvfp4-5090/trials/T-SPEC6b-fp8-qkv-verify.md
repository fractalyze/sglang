# T-SPEC6b (gemma4nv-b2-tspec6b): the target's qkv_proj as FP8 weight-only at verify widths

Registered 2026-10-03 by W12 (bs2) after the microbench and the unit tests, before any screen or gate run of the change. The coordinator approved it as the first of two FP8 trials. T-SPEC6c (the target lm_head) waits for this verdict.

## Change

- **Code:** commits `b9449472fd` and `c9b595d5a3`, behind the existing switch `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM`, which base4 already turns on for o_proj. So this is a commit change, not a new switch.
- **Allowlist.** Both Gemma-4 qkv_proj shapes, sliding (8192 x 2816) and full (10240 x 2816), join T3b's FP8 E4M3 weight-only allowlist with tile (64, 128, 4). At load they are quantized per output channel with T3b's quantizer.
- **Route.** M up to 256 runs the Triton kernel with BLOCK_M capped at 32. M above that 32-row cap runs one program per M block, and those later weight reads come from L2. M above 256 (prefill) keeps T3b's bf16-upcast cuBLAS route.
- **Why the cap.** Above a tile's max_m, the old route costs 2.1-2.5x cuBLAS BF16 (127 vs 59 µs for qkv sliding at M=192). An MTP verify at B=32 runs M=192.
- **o_proj is unchanged.** Its max_m stays 32, so its own B=8 verify (M=48) still takes the upcast route. That is a separate finding for a later trial.
- **Python diff against the control** (`3d1732c505`): `triton_small_m_bf16_gemm.py` (+20 lines) and `unquant.py` (+2/-2) only.
- **Tests:** `test/registered/gemm/test_triton_small_m_bf16_gemm.py`, 5 cases, all passing on the bs2 5090 in the pinned tree.
  - FP8 results stay within one bf16 rounding of the dequantized reference at M ∈ {1, 6, 8, 17, 32, 48, 200, 256}. 48 and 200 cross the 32-row M blocks.
  - Each shape routes to the kernel at its max_m, and to the upcast at max_m + 1.

## Evidence (microbench, bs2 RTX 5090)

Method as T3c: CUDA graph of 64 calls, best of 7, weights rotated over >= 256 MB, noise 0.25%. Results: `trials/spec/results/w12-verify-gemm.json` and `w12-verify-gemm-large.json`.

| M (verify at k=5) | qkv sliding, cuBLAS BF16 → FP8 | qkv full, cuBLAS BF16 → FP8 |
|---|---|---|
| 6 (B=1) | 31.2 → 15.7 µs | 37.7 → 19.6 µs |
| 48 (B=8) | 31.8 → 21.4 µs | 39.0 → 23.2 µs |
| 192 (B=32) | 59.4 → 54.0 µs | 62.1 → 66.3 µs |

## Control and candidate

- **Control:** `base4-spec-fp8head`, the bs2 reference (T-SPEC3 + T-SPEC4).
- **Candidate:** `base4-spec-fp8head-qkvfp8`, which is the control's flags on `c9b595d5a3`. `weight_layout_change` is true.
- **Gate:** the W12 gate (`fa10617a79` or later). It flushes the radix cache before the forced pass and times fixed W8 prompts.
- **Decision:** `--decide-on w1_tpot_gain`. The W8 composite and W32 are guards.
- **Quality** (coordinator rule): full GSM8K plus tool-JSON, paired against the control.
  - GSM8K's paired 95% CI lower bound must stay above -1.0 pt.
  - **Tool-JSON must not drop.** Attention-projection requantization is the crawl's tool-calling failure case, so a tool-JSON drop parks the trial even if timing wins.
- **Memory:** the KV pool (full / sliding tokens) of both arms is reported.

## Prediction (frozen)

**Mechanism.** Per B=1 round at k=5, qkv saves 25 x 15.5 + 5 x 18.1 = 478 µs of about 10.3 ms, or 4.6%. T-SPEC4 realized about 80% of its microbench saving in the gate.

**Prefill.** It stays on the upcast route, now for qkv too. That costs roughly 60 µs per qkv call at a 1024-token chunk, about 1.8 ms per chunk.

| metric (bs2, gate ratio of sums vs base4-spec-fp8head) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-2.5 … -5.5%** |
| W8 decode gain | +1 … +3% (339 µs of a ~16 ms B=8 round) |
| W8 prefill gain | 0.96 … 1.00 (upcast on qkv) |
| W8 composite (guard) | -0.5 … +2.5% |
| W32 tok/s (guard) | -2 … +1% (qkv is roughly a tie at M=192; prefill pays the upcast) |
| GPU memory | -721 MB of weights. KV pool grows by about 7.5k full / 6k sliding tokens. |
| Fidelity | lossy tier. Decode KL mean rises by +0.003 … +0.015 over the control's (T3b's o_proj FP8 added 0.005), and the gate passes. |
| Full GSM8K, paired | delta within [-1.0, +0.5] pt, and the CI lower bound is above -1.0 pt |
| Tool-JSON (40) | 40 / 40 |

**Expected verdict: kept.**

**Falsified if:**
- the W1 TPOT gain is below 1.02;
- the W8 composite or W32 regresses beyond its bar;
- fidelity fails;
- the GSM8K CI lower bound is below -1.0 pt;
- tool-JSON drops below 40 / 40.
