# W13 report: o_proj FP8 above 32 rows (T3d) and MoE tactics at verify widths (T-MOE1 step 0), build-server-2

- **Branch:** `jumanzii/gemma4nv-analysis`, pushed to `fractalyze`.
- **Host:** build-server-2.
- **Control for both:** the bs2 reference `base4-spec-fp8head-fp8lmhead` (T-SPEC6c).
- **Timings** are gate verdicts (paired ABBA, ratio of sums, 4 pairs), unless labelled a screen (unpaired, one server lifetime) or a microbench.

| trial | deciding metric | predicted (frozen) | measured | verdict |
|---|---|---|---|---|
| T3d: FP8 o_proj on the Triton GEMM above M=32, in M tiers | W8 composite | +2.5 … +6% | **+0.46%** (CI +0.04 … +0.87%); W1 TPOT -9.1% and W32 +2.3%, both from acceptance on divergent text | **retired** |
| T-MOE1: swap-AB / FINALIZE MoE tactics at verify widths | W1 TPOT | -2 … -4.5% | step 0 hit its own falsifier: padding is free, and the autotuner already picks swap-AB | **retired at step 0** (coordinator) |

- **The bs2 reference is unchanged:** `base4-spec-fp8head-fp8lmhead`.
- **The branch's python and test trees again equal `dd5361a7bc`.**
  - T3d's code (`350d053100`) was reverted after the verdict.
  - Its microbench, registration and probes stay in the tree.

## 1. T3d: o_proj FP8 weight-only above M=32

### Microbench

Script: `trials/t3/oproj_large_m_bench.py`. Results: `trials/t3/results/w13-oproj-large.json`. Method: CUDA graph of 64 calls, best of 7, weights rotated over 256 MB. Times in µs.

| shape | M | cuBLAS BF16 (dequant-once) | upcast + cuBLAS (control) | Triton FP8, capped BLOCK_M (best tier) | W8A8 `_scaled_mm` (other tier, informational) |
|---|---:|---:|---:|---:|---:|
| o_proj sliding | 48 | 18.2 | 41.7 | **12.8** | 41.5 |
| | 192 | 26.7 | 57.4 | **29.5** | 44.9 |
| | 1024 | 131.3 | 170.1 | **131.3** | 139.4 |
| | 8192 | 852.6 | 1015.6 | **925.4** | 1184.9 |
| o_proj full | 48 | 32.8 | 79.5 | **24.9** | 61.4 |
| | 192 | 51.8 | 114.1 | **59.8** | 70.4 |
| | 1024 | 241.7 | 313.5 | **266.5** | 274.8 |
| | 8192 | 1763.1 | 1927.6 | **1880.1** | 2421.5 |

**The Triton kernel beats the upcast route at every M from 48 to 8192.** Its tiers were picked from this microbench as the best worst case over both shapes:

| M | BLOCK_M cap | tile (BLOCK_N, BLOCK_K, stages, warps) |
|---|---:|---|
| ≤ 64 | 16 | (32, 128, 4, 4) |
| ≤ 2048 | 32 | (64, 128, 3, 4) |
| ≤ 8192 | 64 | (128, 64, 3, 8) |

**Dequant-once would keep a second, BF16 copy (+0.41 GB).** It wins only above M ≈ 4096, by 5-7%, so it was not taken.

**The FP8 lm_head needs no large-M path.** Above 48 rows it already uses the BF16 table.

### Code (`350d053100`, now reverted)

- **The tiers.** The o_proj entries in `_FP8_WEIGHT_TUNED_SHAPES` became M tiers, and `_TileConfig` gained `block_m_cap` and `num_warps`.
- **No new switch:** the change ran under the existing `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM`.
- **Tests:** `test_triton_small_m_bf16_gemm.py`, 5 cases, all passing on the bs2 5090, and `test_gemma4_fp8_lm_head.py` (6) still passes.
  - At every tier edge (M = 33 … 8192) the output stays within one bf16 rounding of the dequantized fp32 reference.
  - M=48 routes to Triton and M=8193 to the upcast.

### Gate `T3d-20261003-165907-build-server-2-521351`

| metric | control | candidate | gain | 95% CI |
|---|---:|---:|---:|---|
| **W8 composite (deciding)** | | | **1.0046** (decode 1.0024, prefill 1.0110) | 1.0004-1.0087 |
| W1 TPOT | 2.732 ms | 2.484 ms | 1.100 | 1.093-1.108 |
| W32 | 1701 tok/s | 1740 tok/s | 1.023 | 0.948-1.104 |
| fidelity | | | pass | |
| integrity | | | ok, all 8 legs | |

### Why W1 moved 9% when B=1 kernels did not change

The B=1 verify is M=6, under the unchanged tier, so W1 was predicted at ±0.3%. The screens and profiles below are unpaired, and their files are under `trials/t3/results/w13-probes/`.

| check | control | candidate |
|---|---:|---:|
| B=1 verify kernels per round (profile, 12 rounds) | 6.406 ms | 6.411 ms |
| B=1 draft-loop kernels per round | 2.839 ms | 2.843 ms |
| W1-prompt τ (gate-shape) | 3.562 | **3.916** |
| B=8 timing-corpus τ (spec) | 3.21 | **2.99** |
| B=32 timing-corpus τ (spec) | 2.837 | **3.282** |
| hidden-set τ, all 22 prompts (spec) | 3.464 | 3.638 |
| hidden-set τ per category | | -31% (long_docs_16k) … +245% (long_json) |
| W1 greedy outputs identical between arms (gate) | | **0 / 24**; control vs control across pairs 24 / 24 |

**The change sends every timed prompt down different greedy text.**
- It changes target numerics in every prefill (M up to 4096) and every wide verify.
- The drafter attends over the target's KV (`num_kv_shared_layers: 4`), and the text diverges within the 256 generated tokens.

**Acceptance on that text swings ±10% or more, in either direction per workload.**
- The W1 -9.1% is that swing, since the B=1 GPU time per round did not change.
- At B=8 the corpus τ fell 6.7%, about as much as the ~1 ms per round o_proj saving (≈6% of a 16 ms round). That is consistent with W8 decode netting +0.24%.

**The gate's paired CI cannot see this.** Each arm reproduces its own outputs exactly across pairs, so the CI measures only server-lifetime noise.

**New claim:** `c-gemma4nv-gate-spec-timing-tracks-divergent-text-acceptance`.

**Consequence for the gate.** This affects every numerics-changing trial under MTP. The gate's W1 and W8 TPOT on fixed prompts carries an acceptance lottery of about ±10%, which is larger than most kernel trials' predicted gains.
- Earlier verdicts are not re-judged.
- T-SPEC6c's W1 gain points the same way as its profiled kernel saving, at a lower τ, so it does not depend on the lottery.

**Proposed fix (not implemented; it changes the gate).** For numerics-changing trials under speculation, decide on per-round time × τ measured separately:
- **Per-round time:** from profiles or `decode_steps_in_window`.
- **τ:** on a large prompt set, or with the drafter run teacher-forced on the control's text.
- **Alternative:** time on the control's outputs in both arms (forced continuation), so the text is fixed.

## 2. T-MOE1 step 0: FlashInfer's MoE tactics at verify widths

- **Script:** `analysis/scripts/moe_tactic_bench.py`. **Results:** `analysis/results/w13-moe-tactics.json`.
- **The full table and the tactic-index map** are in `analysis/MOE-AT-VERIFY.md` §6.
- **Method:** every FlashInfer 0.6.18 SM120 NVFP4 tactic forced through `profile_ids`, with W8's recorded routing in 6-token verify windows. Times are for one whole MoE layer.

| case | tuned (as served) | best forced pair | tuned with FINALIZE |
|---|---:|---:|---:|
| B=1 verify (M=6) | 78.1 µs | 79.6 µs (swap-AB 128x32x64B, both GEMMs) | 76.2 µs (-2.5%) |
| B=8 verify (M=48) | 165.6 µs | 165.7 µs | 160.4 µs (-3.1%) |

| rows per expert (24 active experts, tuned) | 1 | 2 | 4 | 8 | 128 |
|---|---:|---:|---:|---:|---:|
| µs per layer | 75.4 | 75.8 | 75.7 | 76.4 | 124.9 |

- **Padding is free.** 1 to 8 rows per expert move the cost by 1.3%.
- **The autotuner already runs swap-AB 128x32 tiles.** GEMM1 tactics 10-19 are the swap-AB copies of tiles 0-9. With the same GEMM2 fixed, unswapped 128x32 costs 155-160 µs per layer against 89 µs swapped.
- **W12's §3 inference was wrong.** It read "unswapped" from the trace's CTA shape name, which is the same for both.
- **The preregistered step-0 falsifier holds:** within 5% from 1 to 8 rows, and no tactic 5% better.
- **FINALIZE fusion** (existing `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1`) is the only config-only gain.
  - It is worth about -0.6% W1 TPOT and +0.75% W8, both below the 1% bar.
  - Its epilogue reduces the top-8 by bf16 atomic adds (`sm90_visitor_scatter.hpp`), so the sum is nondeterministic.
  - The coordinator chose not to gate it.
- **New claims:** `c-gemma4nv-moe-verify-tactic-already-swap-ab` and `c-gemma4nv-moe-fused-finalize-below-bar`.

### Costed proposal: the MoE glue, not the GEMM (MOE-AT-VERIFY §7; nothing started)

**B=1 MoE: 2.33 ms per round.**

| part | time | assessment |
|---|---|---|
| grouped GEMMs | 1.80 ms | at the tactic optimum, 79-84% of byte SOL; beating it needs a new grouped GEMM, 3-5 days, low confidence |
| glue | 0.53 ms | six launches per layer: router, expert maps, strides, expand + FP4 quantize, activation, finalize |

**The cheaper target is the glue:** a persistent prep kernel that folds strides, expert maps and expand into routing, plus the activation fused into fc1's epilogue.
- **Saving:** 0.2-0.3 ms per round, about -2 … -3% W1 TPOT.
- **Cost:** about a week of kernel work.
- **Needs approval.**
- **Under §1's finding,** its gate would need the per-round-time decision rule, since it changes the reduction order.

## 3. Vault (shared checkout; only my paths committed; not pushed)

| commit | what |
|---|---|
| `3ef2abc` | T3d prediction (frozen before the gate) |
| `45c7066` | T-MOE1, stubbed from its W12 preregistration (`2dc43b7228`) |
| `ebd55bb`, `d9d4309`, `3174b38` | claims: MoE tactic already swap-AB; FINALIZE below the bar; gate timing tracks divergent-text acceptance |
| `ad2f640`, `60c6dc6` | T-MOE1 retired, plus its prose |
| `a13910d`, `c118505` | T3d prose, then T3d retired |

- **Measurements are `source: manual`.** There is still no gemma4nv-bs2 ledger root.
- **Lint leaves "stale" warnings** on the new claims against base5/base6, as in W10 and W12.
- **Lint also leaves a "fidelity unstated" warning on T3d.**

## 4. Host safety

Per-phase peaks are in `BASELINE.md` § "W13 on bs2".
- **Every engine ran alone** under `host.lock`, in the 24G no-swap scope with the watchdog.
- **No trips.** Serving RSS was 19.5 GB. No nvcc or cicc ran.
- **Two launch mistakes of mine.** I wrapped `gate.hostwatch` in an outer `flock` on host.lock and then on gpu.lock, and it deadlocked twice. hostwatch takes both locks itself. Nothing ran during the deadlock, and it is fixed in the W13 queue scripts.
- **The first T3d gate attempt timed out at the quiescence rule.** A co-tenant's CPU-bound `cargo-zisk-dev` held a 614 MB CUDA context on an idle GPU. The queue retried, and the gate ran once the process exited.

## 5. Next (for the coordinator)

1. **Fix the gate before more numerics-changing kernel trials under MTP (§1).**
   - The fixed-prompt TPOT carries a ±10% acceptance lottery that the paired CI does not show.
   - T3d's microbench saving is about 1 ms per B=8 round, roughly 6% of the round, but no per-round B=8 measurement exists yet. Re-gating it under a per-round-time rule would test that. The code is at `350d053100`.
2. **MoE: the glue is the target (§2), not tactics.** That is about a week of kernel work and needs approval.
