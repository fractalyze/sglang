# T3b (gemma4nv-b2-t3b): FP8 E4M3 weight-only o_proj on the small-M Triton GEMM

Registered 2026-10-03 by W7 (bs2) before any gate run. The code is behind a new switch,
`SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM` (default off), and changes numerics.

## Control and candidate

The control is fixed by T3c's verdict, which is decided before this trial's gate runs:
- if T3c is kept, the control is gate ref `smallm-gemm-t3c`;
- otherwise it is `smallm-gemm` (base2 + T3).

The candidate is the control's flags plus the new switch, on the T3b commit.

## Change

When the switch is on, `UnquantizedLinearMethod.process_weights_after_loading` stores the two
o_proj shapes (2816x4096, 2816x8192) as E4M3 with a per-output-channel absmax scale.
- **Decode (M ≤ 32)** runs the T3 Triton kernel, which upcasts each weight tile to bf16 and applies the scale in
  the fp32 epilogue.
- **Prefill** runs cuBLAS on the exact bf16 upcast and then scales its output. The scale factors out of
  the K sum, so both paths compute the same dequantized product up to bf16 rounding.
- o_proj weights shrink from 0.81 to 0.41 GB. `--mem-fraction-static` stays fixed, so the KV pool gains
  about 0.4 GB. That affects only W32 retraction headroom.

## Evidence (W7 microbench `t3/results/w7-t3b.json`, same method as T3)

| shape | M | T3 BF16 (control) µs | FP8 weight-only µs | W8A8 `_scaled_mm` rowwise / per-tensor / sgl `fp8_scaled_mm` µs |
|---|---:|---:|---:|---|
| o_proj sliding | 1 | 14.7 | 8.9 (-39%) | 35.9 / 21.4 / 49.9 |
| | 8 | 14.8 | 9.1 (-38%) | 38.2 / 24.2 / 50.0 |
| | 32 | 15.2 | 12.4 (-18%) | 40.6 / 25.0 / 50.0 |
| o_proj full | 1 | 29.0 | 15.7 (-46%) | 59.3 / 28.2 / 96.0 |
| | 8 | 29.0 | 15.8 (-45%) | 59.5 / 32.3 / 95.8 |
| | 32 | 29.3 | 24.0 (-18%) | 60.3 / 33.0 / 96.0 |

- **W8A8 loses at decode.** Dynamic activation quantization costs more than the halved weight reads save. Vault
  trial qi21ed-ed10 saw the same on the 5090 at ~50 tokens.
- **Weight-only is the candidate.** Its output error vs fp32 on Gaussian weights is rel-L2 0.027, against 0.038 for W8A8.

## Prediction (frozen)

The microbench deltas are multiplied by 25 sliding and 5 full o_proj per step:
- W8 (M=8): 209 µs per step;
- W1 (M=1): 211 µs;
- W32: 97 µs.

The prefill upcast reads 1 B and writes 2 B per weight element per prefill forward, about 0.8 ms.

| metric (bs2, gate ratio of sums) | predicted |
|---|---|
| **W8 composite (deciding)** | **+1.0 … +2.5%** (decode about +2.4%, prefill 0 … -1%) |
| W1 TPOT (guard) | -2.5 … -4.5% |
| W32 tok/s (guard) | 0 … +3% |
| Fidelity | approx tier. Teacher-forced and free-running KL rise above the control's but stay inside the gate thresholds (forced KL mean ≤ 0.089, decode KL mean ≤ 0.050); low-medium confidence |
| `gate quality` | GSM8K and tool-JSON each within 1 point of the control's baseline on bs2 |

Decision rule: default `--decide-on w8_composite`, with W1 as the guard. A fidelity fail rejects the trial, and so does a quality fail. Both are run because
the crawler prior saw NVFP4/FP8 attention requantization break tool calling.

**Falsified** if any of these holds:
- the W8 composite gain is below 1.01;
- fidelity fails;
- either quality task drops more than 1 point.
