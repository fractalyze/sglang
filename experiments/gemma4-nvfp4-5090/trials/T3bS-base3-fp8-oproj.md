# T3bS (gemma4nv-b3-t3b): FP8 E4M3 weight-only o_proj stacked on base3, build-server-3

**Status: kept and adopted into base4 (W9, 2026-10-03).**
- **Gate** `gemma4nv-b3-t3b-20261003-103822-build-server-3-012d22`: W8 composite 1.0142,
  W1 TPOT -4.03%, W32 0.9997, decode KL 0.0253 (p99 0.45). The decode KL check exercises the
  M <= 32 kernel.
- **Full-GSM8K quality:** paired delta -0.23 pt, 95% CI [-0.92, +0.47]; tool-JSON 100 -> 100.

See `REPORT-bs3-w9.md`. Below is the frozen registration.

Registered before any run (W9, 2026-10-03). Variant of `gemma4nv-b2-t3b` (bs2, W7, kept against
`smallm-gemm` = base2 + T3; see `trials/T3b-fp8-weight-oproj.md` on `jumanzii/gemma4nv-analysis`).

## Control and candidate

- **Control:** gate ref `base3` (base2 + split-KV 16 + Triton small-M BF16 GEMM, commit 1fd77e64b0).
- **Candidate:** gate ref `base3-t3b` = base3's flags and env plus
  `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM=1`, on commit 36aa977541. That commit is
  1fd77e64b0 plus cherry-picks of 826d472504, bbbf5e4d46 and ed0aefcd40, so its `python/` tree is
  byte-identical to the reviewed T3b commit ed0aefcd40. Branch `jumanzii/gemma4nv-b3-t3b` on
  `fractalyze`. The ref declares `weight_layout_change` (o_proj is stored as E4M3).

## Change

Both o_proj shapes (2816x4096 sliding, 2816x8192 full) are stored as E4M3 with a per-output-channel
absmax scale. Decode (M <= 32) runs the T3 Triton kernel with an in-tile bf16 upcast and the scale in
the fp32 epilogue; prefill runs cuBLAS on the exact bf16 upcast and scales the output.

## Prediction (frozen)

Basis: the bs2 gate `T3b-20261003-101044-build-server-2-df2cee` measured W8 composite 1.0136 (decode
1.0244, prefill 0.9819), W1 TPOT -3.92%, W32 1.0007 against base2 + T3. base3 differs from that
control only by split-KV 16, which changes decode attention and no GEMM, so the o_proj saving per
step (about 209 us at M=8, 211 us at M=1) carries over. base3's W8 decode step (8.76 ms) is close
to bs2's control, so the relative gain should match within the host change.

| metric (bs3, gate ratio of sums) | predicted |
|---|---|
| **W8 composite (deciding)** | **+1.4%, interval [+0.8, +2.2]%** (decode +2.0 ... +2.8%, prefill -1 ... -2.5%) |
| W1 TPOT (guard) | -3.9%, interval [-2.8, -4.8]% |
| W32 tok/s (guard) | 0 ... +2% |
| Decode-path KL vs the base reference | mean 0.015 ... 0.030 (base3 measured 0.0112 in T3S; bs2 T3b added about +0.012), under the 0.050 limit; p99 under 0.95 |
| Teacher-forced KL | rises a little above base3's 0.0324 (prefill o_proj now reads the dequantized weight), under the 0.089 limit |

The bar for the deciding metric is 1.01, so the low end of the interval can fail; the bs2 margin
was 0.36 points.

**Falsified if** the W8 composite gain is below 1.01, W1 or W32 regresses past its bar, or fidelity fails.

## Quality rule (frozen before any quality run)

Each arm runs `gate quality` with the full GSM8K test split (1,319 items) and the 40-item
tool-call JSON set, greedy, on bs3:
1. `base3` (A),
2. `base3-t3b`,
3. `base3` again (A'), an A/A that measures how many items flip from batch-composition noise alone.

The GSM8K delta is paired (same items in both arms). Its 95% interval is the Agresti-Min adjusted
Wald interval for a paired difference, from `gate quality-compare`; the exact McNemar p-value is
reported beside it. The decision compares `base3-t3b` against run 1 (A).

- **Adopt T3b into base4** only if the gate verdict is kept, the GSM8K delta's 95% CI lower bound is
  >= -1.0 pt, and tool-JSON accuracy does not drop (delta >= 0).
- **Quality prediction:** GSM8K delta -0.3 pt, between -1.0 and +0.5; tool-JSON 100 -> 100.
- **If the A/A (A vs A') fails the same rule,** the rule cannot separate a real drop from greedy
  batching noise at this n. Then I do not adopt or reject on quality; I report it and ask the
  coordinator.
