# T4 (gemma4nv-b3-t4): fuse the per-layer decode glue (q/k/v norm + RoPE + FP8 KV write, norm pairs)

Registered before any code exists (W9, 2026-10-03). Control `base4` (base3 + T3b, pinned by W9). **Not implemented; waiting for the coordinator's go.**

## Why this, now

The measured ranking (PROFILE.md section 3 on `jumanzii/gemma4nv-analysis`, bs2 B=8 decode trace,
rank = share x (1 - sol_fraction)) puts "norms, RoPE, KV write" second: 392 launches per step,
840 us traced, 13 us of DRAM bytes, sol_fraction 0.02. The first row (routed experts) is
FlashInfer's CUTLASS MoE and a larger project. Since that profile, T2/T3/T3b changed attention
splits and the o_proj/dense GEMMs only; none of them touched the glue kernels, so its launch count
is unchanged on base4.

## What runs today (per decoder layer, decode; re-read from the B=8 trace, 7 steps, 30 layers)

| group | kernels (launches per layer) | B=8 traced us per layer |
|---|---|---:|
| A. q/k/v norm, RoPE, KV write | `_gemma_qkv_rmsnorm_kernel` (1), `sglang::fused_rope_kernel` (1), 4 x ATen `elementwise_kernel`, attributed to the FP8 conversion in `MHATokenToKVPool.set_kv_buffer` (`cache_k.div_(k_scale)`, `cache_v.div_(v_scale)`, `.to(float8_e4m3fn)` for K and V) (4), `sglang::store_kvcache_kernel` (1) | 13.4 |
| B. post-attention norm pair | flashinfer `RMSNormKernel` (`post_attention_layernorm`) then `FusedAddRMSNormKernel` (`pre_feedforward_layernorm` + residual add) on the same row (2) | 5.1 |
| C. MoE-input norm pair | `RMSNormKernel` for the router's `Gemma4RMSNorm` and `RMSNormKernel` for `pre_feedforward_layernorm_2`, both over the same `moe_input` (2) | 4.8 |
| D. next layer's input norm | `RMSNormKernel` (`input_layernorm`) right after `_gemma_dual_rmsnorm_residual_kernel` (1) | 2.3 |
| (kept) | `_gemma_dual_rmsnorm_residual_kernel` (1) | 2.2 |

That is 13 glue launches per layer (390 per step plus the final norm), matching PROFILE's 392.
Triton decode passes `layer.k_scale` / `layer.v_scale` to `set_kv_buffer`, so each `div_` is a
real scale and each cast is the `ConvertToFloat8E4M3` kernel PROFILE names. The trace shows the four
ATen kernels sitting between RoPE and the store; it does not name which source line launched each one.
The first implementation step is to confirm that attribution with `TORCH_SHOW_CPP_STACKTRACES` or
record_function ranges.

## Change (one trial, one switch)

New switch `SGLANG_OPT_GEMMA4_FUSED_DECODE_GLUE` (default off), in `gemma4_causal.py` and
`gemma4_fused_ops.py`:

1. **Group A, 7 -> 1 launch.** Extend `_gemma_qkv_rmsnorm_kernel` into one Triton kernel that,
   per token row: RMS-normalizes q and k (learned weights) and v (no weight); applies
   neox-style RoPE to q and k from the existing cos/sin cache at `positions`; writes q back
   in place; divides k and v by their scales in fp32, rounds to E4M3, and stores them into the
   layer's K/V buffers at `out_cache_loc` (the SWA-translated loc on sliding layers). Attention
   is then called with `save_kv_cache=False`. Full layers (`k_eq_v`) take v from the raw k
   shard before k's norm and RoPE, as today.
2. **Group B, 2 -> 1.** One kernel: `h = rmsnorm(attn_out, w_post)`, `residual += h`,
   `out = rmsnorm(residual, w_pre)`.
3. **Group C, 2 -> 1.** One kernel reads `moe_input` once and writes both normalized outputs
   (router weights with the folded scale, and `pre_feedforward_layernorm_2`).
4. **Group D, 1 -> 0.** `_gemma_dual_rmsnorm_residual_kernel` also writes the next layer's
   `input_layernorm(out)` as a second output; the last layer writes the final model norm.

Launches removed per step: A 6 x 30 = 180, B 30, C 30, D 30 = **270 of 392 (69%)**.

## Prediction (frozen when registered)

Basis: measured mechanism. At B=8 these launches average 2.0-2.3 us traced each and move almost
no DRAM bytes (vault C14: the intermediates sit in L2), so the saving is launch and tail
latency. Per layer, the fused kernels are estimated at 4 us for A (against 13.4) and about
2.8 us each for B and C (against 5.1 and 4.8). D costs about 0.5 us of epilogue (against 2.3).
That saves about 15.5 us per layer, or 465 us per traced step. The profiler inflates short
kernels; the traced B=1 step was 8.5% longer than the timed one. So I take 60-75% of the traced
saving as the timed saving: 280-350 us of base4's W8 decode step (about 8.54 ms: base3's 8.76 ms
less T3b's 2.6%), or 3.3-4.1%. At B=1 the same 55% of the traced glue time (632 us) is 350 us
traced, 210-260 us timed, of base4's 5.59 ms W1 TPOT.

| metric (bs3, gate ratio of sums) | predicted |
|---|---|
| **W8 composite (deciding)** | **+2.8%, interval [+1.5, +4.0]%** (decode -3.2 ... -4.0%; prefill 0 ... +1%, since the prefill path also drops the four elementwise KV-quantize passes over 8192 rows) |
| W1 TPOT (guard) | -4.0%, interval [-2.5, -5.5]% (same 270 launches at B=1) |
| W32 tok/s (guard) | +1 ... +2.5% |
| Decode KL vs the base reference | within 0.005 of the control's. The fp32 divide-then-round can differ from today's bf16 `div_` then cast by one E4M3 step on rare elements. |
| Teacher-forced KL | within 0.002 of the control's (prefill uses the same fused path) |

**Falsified if** the W8 composite gain is below 1.01, W1 or W32 regresses past its bar, or fidelity
fails. Group A also needs a byte-exact unit test: the KV-cache bytes must match the unfused path,
or differ only where the fp32 and bf16 divide round differently, and every such element must
be counted. The tree hard-disables the existing RoPE + KV-write fusion
(`can_fuse = False`, "accuracy regression in launch_server path"), so this risk is known.

**Decision rule:** `gate run --control base4 --candidate base4-glue --pairs 6` on bs3,
deciding on the W8 composite with W1 as the guard; fidelity must pass. A `gate quality` run on
the full GSM8K set is needed only if the unit test shows non-exact KV bytes.
