# gemma4nv W2: ranked hypotheses

Target metric: the W8 composite from W1's gate, `prefill_gain^0.25 x decode_gain^0.75` at
B=8 x 1024 prompt x 128 decode with diverse prompts. B=1 decode is tracked separately.
Baseline: checkpoint defaults plus `--moe-runner-backend flashinfer_cutlass`. That baseline
has triton attention, FP8 KV on all layers, decode CUDA graph, and chunked prefill 4096
(see `PROFILE.md` §1).

> **Status: the predictions are analytic, and the screen is unmeasured.** bs2 lost its NVIDIA
> module after a reboot, and bs3 is offline (see `REPORT.md`). Each prediction comes from the
> byte model in `PROFILE.md` §2: decode Δ ≈ −(SOL removed / total SOL), times the fraction of
> the step that is memory-bound. The real lever sizes depend on each component's measured
> `sol_fraction`, which the staged profile produces. Re-rank once it lands. The
> "Screen" column is filled from `scripts/run_all.sh`. Single runs there are labelled
> "screen, unpaired" and never quoted as gains.

World-model consult: one call, `consult_raw.txt` (14 changes). It is a **COLD START for this
model**: no LLM trial is in the vault. Every row is from diffusion models (Qwen-Image 2.1,
Wan 2.2, FLUX.2) on the same RTX 5090 SM120. The rows below are priors on the GPU, not
evidence about this model. Those that bear on these candidates:
- **T12 `qi21ed-ed10`:** W8A8 per-token FP8 on text-encoder linears at about 50 tokens was
  **+19% slower**. Activation quant plus rowwise `_scaled_mm` cost more than the halved weight
  reads saved. Weight-only e4m3 (T13) won −34%. This argues that decode FP8 should be
  weight-only, not W8A8.
- **T37 `qi21ed-ed04`:** `torch.compile` with FP8 tensors fails on sm_120. Inductor lowers
  `torch.empty(float8_e4m3fn)` to a Triton NaN fill. The baseline KV is FP8.
- **C14 `c-rtx5090-fusion-needs-l2-spill`:** fusion removes DRAM traffic only when the
  intermediate spills the 96 MB L2. Decode intermediates are B x 2816 (tiny), so norm/glue
  fusions save launch and latency only.
- **C2 `c-…-fusion-gains-lose-a-clock-bin-to-the-power-cap`:** on the 575 W 5090, removing
  memory-bound time raises power, and GEMMs drop about one SM clock bin. Expect realized gains
  below the byte model.
- **C10 `c-20260930-flux2-fp8-error-sits-in-double-stream-blocks`:** FP8 error concentrates in
  specific blocks. This matches the Yukon prior that precision sensitivity falls about 15x
  with depth. Mixed-precision rollouts should go layer by layer behind a KL gate.
- **T26 `qi21mk-h16`:** cuBLASLt nvjet block-scaled FP8 (982 TF) beat CUTLASS FP8 (494 TF) on
  sm_120. Use it for any prefill FP8 GEMM.

## Ranked list

Rank = expected W8 composite gain x confidence. This is the analytic stand-in until the
profile lands. Then the coordinator's rule applies (crawler feed #2): order by
time_share x (1 - sol_fraction), then confidence, and close every kernel claim end to end.
SOL-ExecBench saw an 84%-faster attention kernel give only 3% TTFT inside SGLang. Δ is the predicted change in step time:
negative is faster. "Composite" is the predicted W8 composite gain.

| # | Hypothesis | Mechanism / evidence | Pred. B=8 decode Δ | Pred. prefill Δ | Pred. composite | Conf. | Effort | Screen |
|---|---|---|---|---|---|---|---|---|
| H1 | **W4A16 decode MoE path.** (a) Cheap: add tanh-GeGLU to the in-tree `marlin` NVFP4 MoE runner, which today rejects gelu (`moe_runner/marlin.py:142`), and use it for decode M≤16. (b) Full: an expert-major gather-GEMV that reads each active expert once, skips empty tiles (RUNSKIP), and pairs same-expert tokens. | Experts are 47% of B=8 decode bytes (5.2 GB, SOL 2.9 ms). B=8 x top-8 gives 64 assignments over about 52 experts, about 1.2 tokens/expert, so 16-row grouped tiles are mostly padding (Yukon §5). Crawler prior: decode-shaped NVFP4 projections run at median sol_frac 0.25. | −15% … −30% (if measured expert sol_frac ≤ 0.4) | 0 (decode-only dispatch) | **+12 … +25%** | med-low | (a) S, (b) L | moe_marlin: expected launch reject (gelu) |
| H2 | **FP8 weight-only (W8A16) for the BF16 projections:** qkv_proj, o_proj, dense MLP. Excluded layers get an FP8 copy plus a per-channel scale. Decode uses a dequant-in-GEMV kernel; prefill uses an FP8 GEMM (cuBLASLt block-scaled, T26). | They are 3.3 GB of BF16 per decode step (30% at B=8, 58% at B=1). Halving saves about 930 µs SOL at B=8. Vault T12: W8A8 *lost* at about 50 tokens, so weight-only. Fidelity risk: NVIDIA deliberately left these BF16, and the FP8 error concentrates in specific layers (C10). Roll out last layer first. | −10% … −15% (B=1: −20% … −29%) | −15% … −25% (FP8 GEMM rate) | **+9 … +14%** | med-low | M | — (code) |
| H3 | **Cheaper lm_head.** Use an FP8 weight-only copy of the tied 262144x2816 embedding for logits. Alternative: a certified coarse-screen exact-greedy head (Yukon §4, greedy only). | 1.50 GB per step, 14% of B=8 and 26% of B=1 decode bytes. FP8 saves about 420 µs SOL. Costs +0.74 GB of HBM for the FP8 copy. | −5% … −7% (B=1 −10% … −13%) | ~0 | **+4 … +5%** | medium | S-M | — (code) |
| H4 | **Glue fusion at decode.** One shared norm for the router and `pre_feedforward_layernorm_2` (same input). Merge `post_attention_layernorm` with `fused_add_rmsnorm`. Fuse RoPE with the FP8 KV write (`can_fuse=False` today). Fuse the router GEMM (N=128) with `gemma4_fused_routing`. | About 5 launches per layer x 30, roughly 150 of about 700 launches per step. Crawler: standalone norm at decode runs 6-55x above SOL, and the skinny router GEMM about 20x (≈10 µs on a 0.5 µs bound). Vault C14: no DRAM saving (L2-resident), so this is latency only. C2: expect a clock-bin give-back. | −3% … −6% | −1% … −2% | **+2.5 … +5%** | medium | M | — (code) |
| H5 | **Dense MLP and routed MoE on two streams** inside the decode graph. They are independent given `moe_input`. | Both are small-M and leave SMs idle at B=8. The dense MLP SOL is about 600 µs. Overlap hides part of the shorter one. Vault: no direct row (T29-T31 are graph/launch bounds, all ≤ 0.5% on diffusion). | −2% … −6% | ~0 | **+1.5 … +4.5%** | low-med | M | — (code) |
| H6 | **Decode attention kernel.** Pass the window length into the split-KV count for SWA layers. Today `window_num_kv_splits` is computed but `forward_decode` uses the full-length `num_kv_splits` (`triton_backend.py:2279`). Tune `--triton-attention-num-kv-splits`. Port FlashInfer decode for hd256 SWA (Gemma4 asserts it out, `model_hook.py:590`). | Attention is about 520 µs of SOL at B=8 FP8. A Triton grouped kernel with `BLOCK_DMODEL=512` on full layers is unverified for SM120 smem. The window-split fix matters for long contexts, and is about 0 at 1024. | −1% … −5% | −0 … −3% | **+1 … +4%** | low-med | S (splits) / L (FlashInfer) | splits4 / splits16: pending |
| H7 | **One-chunk prefill.** `--chunked-prefill-size 8192` or `16384`, so the 8x1024 prefill is 1 chunk instead of 2. | Removes a second pass over all weights and the per-chunk scheduler step. Memory headroom: about 8.8 GB free after the KV pool. | 0 | −2% … −6% | **+0.5 … +1.5%** | med-high | knob | chunk8k / chunk16k: pending |
| H8 | **FP4 (nvfp4) KV for sliding layers only**, keeping full layers at FP8 or BF16. | SWA KV is 0.84 GB per step at FP8, and FP4 saves about 235 µs SOL. Yukon: a 4-bit g64 mirror was accepted (+2%), but the NF8 and q4 K+V mirrors failed fidelity. Needs per-pool dtype (one dtype per `SWAKVPool` today) and Triton FP4 KV support. | −3% … −4% | ~0 | **+2 … +3%** | low | L | — (code) |
| H9 | **BF16 KV on the 5 full-attention layers** (fidelity, not speed). The checkpoint default puts FP8 on all layers. | Yukon: full-attention KV quant failed fidelity 3 times. Cost is about +50 µs SOL (+0.8%). Required only if W1's long-context fidelity gate fails on the baseline. | +0.5% … +1% (slower) | ~0 | −0.5 … −0.8% | high (cost) / unknown (need) | M (per-pool dtype) | kv_bf16 (all layers): pending |
| H10 | **MTP speculative decoding** (`FROZEN_KV_MTP`, `google/gemma-4-26B-A4B-it-assistant` BF16 draft): **off at B≥8, test at B=1.** | B=8: Yukon never won. An 8x(1+k) verify roughly doubles the experts touched (52→112). B=1: 86% of the step is BF16 dense bytes, so a k=3 verify grows expert bytes only 0.8→2.9 GB (+37% step) for about 2-3 tokens per round. Crawler: MTP gain shrinks after NVFP4 (1.70x bf16 → 1.23x NVFP4). | B=8: +10 … +40% (slower) | — | W8: **≤ 0** (keep off). B=1: +30 … +80% tok/s | B=1 low-med | S (flags + draft download) | — |
| H11 | **MoE GeGLU + finalize fused into the CUTLASS grouped path** (only if H1 fails). | Removes the separate activation/finalize kernels per layer. Yukon 64df0c99: fused gate/up/GeGLU + compact down grid gave +0.3%; 5bb15364 fused down + weighted reduce gave +3.5%. | −1% … −3% | −1% | **+1 … +2.5%** | low-med | L | — |
| H12 | **`--enable-torch-compile` on decode** (bs ≤ 8). | It could fuse elementwise glue. Vault T37: inductor breaks on sm_120 FP8 `empty`, and the KV is FP8. | 0 … −3% (or fails) | 0 | **0 … +2%** | low | knob | tcompile: pending |
| H13 | **Launch and CUDA-graph knobs.** `--cuda-graph-max-bs 8`, `--num-continuous-decode-steps 4`, overlap scheduler on/off. Controls: no CUDA graph. | Decode is already fully graphed at bs=8. Prefill graphs are disabled for this multimodal arch, which doesn't matter at 4096-token chunks. These screens size the CPU/launch share of decode. | 0 … −2% | 0 | **0 … +1.5%** | low | knob | cg_bs8 / contdec4 / nooverlap / nocg: pending |

Not candidates (checked):
- **MoE runner choice:** `auto` resolves to `flashinfer_trtllm` and crashes on SM120 (`'FusedMoE' object has no attribute 'g1_scale_c'`). `marlin` and `flashinfer_cutedsl` reject gelu. `humming` uses erf-GELU, a numerics mismatch with Gemma's tanh GeGLU. **`flashinfer_cutlass` is the only working NVFP4 MoE runner today**, which is why H1(a) is a code change and not a knob.
- **Attention backend knob:** Gemma4 accepts only `triton`/`trtllm_mha` on CUDA, and `trtllm_mha` is SM100-only. Screen `attn_trtllm` is expected to fail at launch. FlashInfer needs code (H6).
- **FP8 KV vs BF16 KV as a speed lever:** the baseline is already FP8 everywhere. `kv_bf16` sizes what FP8 buys, predicted at about +7-8% B=8 decode time for BF16.
- **Skipping the fp32 logits copy and softcap for greedy:** 8x262144x4 B is about 25 MB per step, about 14 µs. Negligible.

## Knob screen (screen, unpaired: nothing here is a gain until it passes W1's gate)

Configs are staged in `scripts/run_all.sh` and run through `scripts/job.sh` (W1's hostwatch
protocol). Each is one server launch with W1's baseline flags plus the knob, then B=8 x3 and
B=1 x3 reps over diverse prompts. The radix cache is flushed per rep, and the prefix-cache hit
tokens are recorded per stream (they must be 0).

| Config | Flags (on top of baseline) | B=8 prefill s | B=8 decode ms/step | B=1 decode ms/step | Note |
|---|---|---|---|---|---|
| base | — | unmeasured | unmeasured | unmeasured | |
| kv_bf16 | `--kv-cache-dtype bf16` | unmeasured | unmeasured | unmeasured | sizes the FP8-KV value |
| chunk8k | `--chunked-prefill-size 8192` | unmeasured | unmeasured | unmeasured | H7 |
| chunk16k | `--chunked-prefill-size 16384 --max-prefill-tokens 16384` | unmeasured | unmeasured | unmeasured | H7 |
| splits4 / splits16 | `--triton-attention-num-kv-splits 4/16` | unmeasured | unmeasured | unmeasured | H6 |
| cg_bs8 | `--cuda-graph-max-bs 8` | unmeasured | unmeasured | unmeasured | H13 |
| nocg | `--disable-cuda-graph` | unmeasured | unmeasured | unmeasured | control: launch share |
| nooverlap | `--disable-overlap-schedule` | unmeasured | unmeasured | unmeasured | control |
| contdec4 | `--num-continuous-decode-steps 4` | unmeasured | unmeasured | unmeasured | H13 |
| tcompile | `--enable-torch-compile` | skipped | | | H12. Dropped from the screen: vault T37 (inductor FP8 breaks on sm_120), and inductor compile fan-out is a host-RAM risk under the host-safety protocol. Run only as a deliberate trial. |
| attn_trtllm / moe_cutedsl / moe_marlin | | skipped | | | Settled from code (SM100-only kernels; gelu rejected). Launch attempts would add JIT risk for no information. |

Hand-off: each hypothesis that the gate takes goes through wm-preregister → implement → W1
gate → wm-record. That gives prediction vs result per row, using the "Pred." columns above as
the registered prediction once they are re-based on the measured profile.
