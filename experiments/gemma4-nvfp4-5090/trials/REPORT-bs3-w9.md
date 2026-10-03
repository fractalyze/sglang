# W9 report: T3b on base3 with a full-GSM8K quality check, base4 pinned, glue fusion preregistered (build-server-3)

**Outcome: done.**
- T3b (FP8 E4M3 weight-only o_proj) was kept on base3 and adopted into base4, which is pinned
  with a 6-pair A/A.
- The glue-fusion trial T4 is preregistered and not implemented; it waits for the coordinator's go.
- Full numbers are in `BASELINE.md` (base4 section).

## 1. T3b brought onto base3

- **Code:** branch `jumanzii/gemma4nv-b3-t3b` (fractalyze), commit `36aa977541`.
  - It is base3's `1fd77e64b0` plus cherry-picks of `826d472504`, `bbbf5e4d46` and `ed0aefcd40`.
  - Its `python/` tree hash equals that of the reviewed `ed0aefcd40`, so no new SGLang code runs.
- **Gate ref:** `base3-t3b` (base3 + `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM=1`, declares
  `weight_layout_change`).
- **Deploys:** `/data/jooman/gemma4nv/src-w9/gemma4-nvfp4-5090` on bs3.
  - `9e7e8ae8c0` stamped the T3b gate and the quality runs.
  - `b4e832f1a` stamped the base4 A/A.
  - 56 unit tests pass on bs3.
- **Preregistered before any run:**
  - `trials/T3bS-base3-fp8-oproj.md` (`9e7e8ae8c0`).
  - Vault `gemma4nv-b3-t3b`, frozen at `b055a99`, variant of `gemma4nv-b2-t3b`.
  - Deciding metric W8 composite; W1 and W32 guard.
  - The quality rule was frozen in the same file.

## 2. Gate: T3b vs base3 (bs3)

`gemma4nv-b3-t3b-20261003-103822-build-server-3-012d22`, 6 ABBA pairs, `--decide-on w8_composite`.

| metric | gain | pair range | predicted | |
|---|---:|---|---|---|
| **W8 composite (deciding)** | **1.0142** | 1.0069 - 1.0179 | [+0.8, +2.2]% | held |
| W8 decode | 1.0262 | 1.016 - 1.031 | +2.0 ... +2.8% | held |
| W8 prefill | 0.9791 | 0.977 - 0.980 | -1 ... -2.5% | held |
| W1 TPOT (guard) | 1.0420 (5.827 -> 5.592 ms) | 1.0416 - 1.0427 | [-2.8, -4.8]% | held |
| W32 (guard) | 0.9997 | 0.996 - 1.002 | 0 ... +2% | held (flat) |

**Verdict: promote** (integrity OK, fidelity pass).
- One of the six pairs fell below the 1% bar on the W8 composite.
- W8 decode's per-pair sigma was 0.53%, against 0.03% in the base3 A/A.
- The verdict is on the ratio of sums, as for every earlier trial.
- Timed-output agreement was 0.10. It is reported only, because the ref changes numerics.

### Decode-path KL (the check that sees decode-only kernels)

| vs the `base` reference | base3 (control) | base3 + T3b | limit |
|---|---:|---:|---:|
| decode KL mean | 0.0112 | **0.0253** | 0.050 |
| decode KL p99 | 0.235 | **0.451** | 0.95 |
| teacher-forced KL mean | 0.0324 | 0.0340 | 0.089 |
| teacher-forced min / mean top-1 | 0.917 / 0.969 | 0.917 / 0.966 | 0.90 |

- The decode check runs the 22 hidden prompts in one batch (M = 22 <= 32), so it goes through
  the FP8 small-M kernel. Its KL doubled and stays at half the limit, inside the predicted
  0.015-0.030.
- **T3S check:** the decode KL did run for T3S (`gemma4nv-b3-t3s-...-fe1492`): candidate base3
  0.0112 / 0.235 against control base2 + split-KV 16 0.0147 / 0.396, a pass. No rerun was needed.

## 3. Quality: full GSM8K (1,319) + tool-call JSON (40)

- **Harness:** `gate quality --gsm8k-n all` now saves per-item correctness.
- **Comparison:** `gate quality-compare` pairs two runs item by item. It reports the delta with
  an Agresti-Min adjusted-Wald 95% CI for a paired difference, and the exact McNemar p.
- **Why paired:** an unpaired CI at p = 0.965 and n = 1,319 is about ±1.4 pt, so it could never
  clear a -1 pt floor.
- **A/A:** base3 ran twice to measure batching noise.

| pair | GSM8K | delta | 95% CI | lost / gained | McNemar p | tool-JSON |
|---|---|---:|---|---|---:|---|
| base3 A -> base3 A' (A/A) | 96.51 -> 96.59% | +0.08 pt | [-0.55, +0.71] | 8 / 9 | 1.00 | 100 -> 100 |
| **base3 A -> base3 + T3b** | **96.51 -> 96.29%** | **-0.23 pt** | **[-0.92, +0.47]** | 12 / 9 | 0.66 | **100 -> 100** |
| base3 A' -> base3 + T3b | 96.59 -> 96.29% | -0.30 pt | [-0.95, +0.34] | 11 / 7 | 0.48 | 100 -> 100 |

**Rule (frozen):** adopt if the CI lower bound is >= -1.0 pt and tool-JSON does not drop.
- -0.92 >= -1.0 and 100 -> 100, so **T3b is adopted into base4**. The prediction was -0.3 pt in
  [-1.0, +0.5]; it held.
- The A/A passes the same rule, so the rule can tell a real drop from noise at this n.
- Greedy batching alone flips 17 of 1,319 items (1.3%). The n = 200, 1-point check W7 used
  sits inside that noise.
- **Thin margin:** the lower bound is 0.08 pt from the floor. A further FP8 change stacked on
  base4 needs its own full-set run.

Runs: `quality-full-base3-A-...-272124`, `quality-full-base3-A2-...-ad25c8`,
`quality-full-base3-t3b-...-b032cc`. Each pairwise result is
`quality_compare-vs-<control run>.json` in the candidate's run dir.

## 4. base4 pinned

- **Ref:** `base4` = `base3-t3b`.
- **A/A:** `AA-base4-20261003-110525-build-server-3-ae60a0`, 6 pairs, verdict no promotion as an
  A/A must, agreement 1.0 in every pair.
- **Noise:** the A/A was set as `reference/noise.json`. base3's file is kept as `noise.base3.json`.
  - W8 prefill sigma is 0.41%, so its bar is now 1.23%. Every other bar stays at 1%.
- **Vault:** stack `stack-36aa97754-gemma4nv-base4` (parent base3) is T3b's result stack.

| | base4 | base3 | base |
|---|---|---|---|
| W8 decode step | **8.54 ms** (sol_fraction 0.65) | 8.76 ms (0.63) | 9.29 ms (0.60) |
| W1 TPOT | **5.598 ms** (0.57) | 5.833 ms (0.55) | 6.075 ms (0.52) |
| W32 | **1554 tok/s** | 1551 | 1021 |
| W8 prefill (sum of 8 TTFTs) | 1.785 s | 1.744 s | 1.740 s |

These are separate A/As on one host. sol_fraction is against the unchanged SOL tables, which
still count o_proj as BF16.

## 5. Next trial preregistered: T4 glue fusion (`trials/T4-base4-glue-fusion.md`, vault `gemma4nv-b3-t4`)

**Re-derived from the bs2 B=8 decode trace** (PROFILE.md's trace, classify_trace rules, 7 steps):
the 392 glue launches are 13 per layer.

| group | launches/layer | traced us/layer |
|---|---:|---:|
| A. `_gemma_qkv_rmsnorm_kernel` + `sglang::fused_rope_kernel` + 4 ATen elementwise (FP8 KV quantize in `MHATokenToKVPool.set_kv_buffer`) + `store_kvcache_kernel` | 7 | 13.4 |
| B. `RMSNorm` (post-attn) + `FusedAddRMSNorm` (pre-FF) | 2 | 5.1 |
| C. two `RMSNorm`s over `moe_input` (router, pre-FF 2) | 2 | 4.8 |
| D. `input_layernorm` after `_gemma_dual_rmsnorm_residual_kernel` | 1 | 2.3 |

**Change:** behind one switch, four fusions.
- **A** becomes one Triton kernel: q/k/v norm, RoPE, fp32 scale, E4M3 store at `out_cache_loc`.
- **B** and **C** each become one two-output norm kernel.
- **D** goes into the dual-norm kernel's epilogue.
- **270 of 392 launches are removed (69%).**

**Prediction** (control base4, bs3, 6 pairs):

| metric | predicted |
|---|---|
| **W8 composite (deciding)** | **+2.8%, interval [+1.5, +4.0]%** |
| W1 TPOT (guard) | -4.0%, interval [-2.5, -5.5]% |
| W32 | +1 ... +2.5% |
| decode KL | within 0.005 of base4's |

**Risks:**
- The tree hard-disables its own RoPE + KV-write fusion for an accuracy regression, so group A
  needs a byte-level KV-cache unit test first.
- The attribution of the 4 elementwise kernels needs one stack-traced profile to confirm.

## Host safety

- Every engine process ran under the protocol through the gate's `host_lock` + capped scope +
  watchdog: 1 prebuild, 12 + 12 gate legs, and 3 quality servers.
- **Peaks:**
  - weight load: 11.2 GB tree RSS (min MemAvailable 51.0 GB);
  - autotune/capture: 6.5 GB;
  - serving: 7.0 GB (min MemAvailable 49.3 GB);
  - peak load1: 5.4, at the start of the T3b gate; otherwise <= 1.7;
  - swap: stayed at 0.13 GB;
  - the new FP8 Triton kernel compiled in-process with no nvcc.
- No watchdog trips.

## Vault (`$WORLD_MODEL_PATH`, my paths only, not pushed; `git status` clean before each step)

- `b055a99`: T3bS prediction frozen.
- `7de6939`: claim `c-gemma4nv-greedy-gsm8k-batch-noise-floor` (method-rule).
- `4f50c11`, `0220dd6`, `a88c03f`: T3b recorded kept, with prose and result stack base4.
- `ac91bfa`: raw imports of `gemma4nv-bs3`, two snapshots.
- `cd6370fee`: stack base4.
- `af3b36b`: ledger ref map `base3-t3b`.
- `386636e`: T4 stub, prediction frozen (`gemma4nv-b3-t4`, `no-evidence`: no glue-fusion prior
  on a launch-bound decode stack).
- **Open lint:** `c-gemma4nv-sm120-fp8-weight-only-beats-w8a8-at-decode`'s evidence does not
  yet cite `gemma4nv-b3-t3b`, which supports it on base3. A claim's evidence list is written
  only by the claim tooling, so I left it for wm-maintain.

## Harness changes (branch `jumanzii/gemma4nv-gate`)

- `9e7e8ae8c0`: `--gsm8k-n`, per-item correctness, `quality-compare` (paired CI, McNemar,
  adoption rule). The n = 200 baseline verdict now refuses mismatched sample sizes. Adds 6
  tests, the `base3-t3b` ref and the T3bS prediction.
- `b4e832f1a`: `base4` ref. `quality-compare` output is now keyed by control run; the first
  version overwrote one file when two comparisons shared a candidate.

## Left

- T4 implementation, on the coordinator's go. The go question was asked as orchestration
  message `msg_31838e110ae5`; there was no reply after 20 min. Implementing T4 edits `python/sglang`,
  so it needs a new dispatch.
- `quality_baseline.json` is still base's n = 200 run. Paired full-set comparisons replace it
  for numerics trials.
- The SOL tables still model o_proj as BF16. A re-based byte model would lower base4's floor by
  about 0.4 GB per step.
