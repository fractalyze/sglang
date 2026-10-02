# gemma4nv W2: profile and SOL byte model

Model `nvidia/Gemma-4-26B-A4B-NVFP4` (HF snapshot `a19cfe00be84`) on 1x RTX 5090 (SM120,
32 GB). SGLang tree `a9871012a` (this branch's base). Host build-server-2.

> **Status: analytic only. The measured columns are UNMEASURED.** bs2 went down mid-bring-up.
> It flapped on the network, rebooted at ~15:36 into kernel 7.0.0-34, and that kernel has no
> NVIDIA module, so `nvidia-smi` fails. bs3 was offline as well. Every number below comes
> from `config.json` / `hf_quant_config.json`, the code, and the server log of the one launch
> that reached KV allocation. The scripts in `scripts/` fill in the measured columns. See
> "How to finish" at the end.

## 1. Serving configuration and what runs per op

Baseline flags (same as W1): defaults plus `--moe-runner-backend flashinfer_cutlass`. My
runs also pass `--context-length 8192 --max-running-requests 32`.

| Setting | Resolved value | Source |
|---|---|---|
| Quantization | `modelopt_fp4`. **Only the routed experts are NVFP4.** Attention projections, dense MLP, router and `lm_head` are BF16 (`ignore` list in `hf_quant_config`). | checkpoint |
| MoE runner | `auto` resolves to `flashinfer_trtllm` on SM120 and **crashes at load**: `AttributeError: 'FusedMoE' object has no attribute 'g1_scale_c'` (`results/logs/default.log`, 14:11). `flashinfer_cutlass` is required. | `modelopt_quant.py:2972-2978` |
| Other NVFP4 MoE runners | `marlin` and `flashinfer_cutedsl` reject `gelu` (Gemma's GeGLU). `humming` uses erf-GELU, not tanh, which is a numerics mismatch. | `moe_runner/marlin.py:142`, `flashinfer_cutedsl.py:230` |
| Attention | `triton` for prefill and decode. Gemma4 accepts only `trtllm_mha`/`triton` on CUDA, and `trtllm_mha` is the SM100 default. | `model_overrides/gemma4.py:28`, `model_hook.py:590` |
| **KV cache dtype** | `auto` resolves to **FP8 e4m3 for all 30 layers**, the 5 full-attention layers included, because the checkpoint declares `kv_cache_quant_algo: FP8`. Log: `Full KV ... dtype: torch.float8_e4m3fn`, `SWA KV ... float8_e4m3fn`. | `kv_cache_dtype.py:39-46` |
| KV pools | Hybrid SWA pool. Full pool 36,594 tokens (0.34 GB); SWA pool 29,275 tokens (2.8 GB). Full layers store K and V separately even with `k_eq_v`, because V = v_norm(k_raw) without RoPE. | log, `swa_memory_pool.py` |
| CUDA graph | Decode: full graph, `bs=[1,2,4,8,12,16,24,32,40,48]`. **Prefill graph disabled**, because Gemma4ForConditionalGeneration is a multimodal arch. | log `cuda_graph_config` |
| Chunked prefill | 4096 (32 GB-GPU default). A B=8x1024 prefill is 2 chunks. `max_prefill_tokens=16384`. | log |
| Sampling | Greedy `torch.argmax` over fp32 logits after an in-place softcap(30). | `logits_processor.py:916`, `sampler.py:179` |

### Kernel order of one decoder layer (decode)

From `gemma4_causal.py:654-767`. Every layer has both a dense MLP and a routed MoE.

| # | Op | Implementation | Launches |
|---|---|---|---|
| 1 | input RMSNorm | `rmsnorm` | 1 |
| 2 | qkv_proj (BF16) | cuBLAS/nvjet GEMM. On full layers the V shard is a copy of K (`:1271`), so it repeats K's math. | 1 |
| 3 | q/k/v norm | `gemma_qkv_rmsnorm` (fused Triton) | 1 |
| 4 | RoPE | `rotary_emb`. RoPE+KV-write fusion is hard-disabled (`can_fuse=False`, `:495`). | 1 |
| 5 | KV write | `set_kv_buffer` (FP8 quantize + store) | 1+ |
| 6 | attention | Triton decode: stage1 (split-KV) and stage2 (reduce) | 2 |
| 7 | o_proj (BF16) | GEMM | 1 |
| 8 | post-attn norm, then pre-FF norm | `rmsnorm` + `fused_add_rmsnorm`: two launches over the same row | 2 |
| 9 | dense MLP | gate_up GEMM, `gelu_tanh_and_mul`, down GEMM | 3 |
| 10 | router | `Gemma4RMSNorm`, then BF16 GEMM N=128 | 2 |
| 11 | pre-FF norm 2 | `rmsnorm` over the **same input** as the router norm | 1 |
| 12 | top-k | `gemma4_fused_routing` (`per_expert_scale` folded in) | 1 |
| 13 | routed experts | flashinfer CUTLASS fused MoE (W4A4: static-scale FP4 activation quant, then grouped GEMMs, GeGLU, finalize) | ~4-6 |
| 14 | norms + residual + layer scalar | `gemma_dual_rmsnorm_residual_scalar` (one kernel) | 1 |

That is about 22-25 launches per layer, or about 700 per decode step including the head.
Visible glue waste: the duplicate norm (rows 10 and 11), the two-launch norm in row 8, unfused
RoPE and KV write, and the dense MLP serialized before the MoE.

## 2. Byte / FLOP model and SOL per component

`scripts/sol_model.py`. SOL = max(FLOPs/peak, bytes/BW) per component (SOL-ExecBench
SOLAR style). The table uses `sol_fraction` = SOL / achieved, never a SOL "score".
Placeholder ceilings: BW 1792 GB/s, BF16 209.5 TFLOP/s, FP4 838 TFLOP/s. They get replaced
by `scripts/microbench.py` measurements.
Distinct experts per layer use the uniform-routing expectation 128·(1−(1−8/128)^B) = 8 / 51.6 / 111.8 at
B = 1 / 8 / 32. Real routing is skewed, so the measured value (`scripts/experts.py`, diverse
prompts) should be lower. KV is FP8 (1 B/elem), context about 1088 tokens.

Expert bytes: 5.947 M params per expert (gate/up 2816x1408 + down 704x2816) at 0.5625 B/param
(FP4 + one FP8 scale per 16) is 3.345 MB per expert. Yukon measured 3.35 MB.

### Decode, B=8 (the W8 shape)

| Component | Bytes/step | SOL µs | Share of SOL | Achieved µs | sol_fraction | Gap µs |
|---|---:|---:|---:|---:|---:|---:|
| routed experts (NVFP4, ~52 distinct/layer) | 5.18 GB | 2893 | 47.1% | unmeasured | — | — |
| lm_head (tied, BF16, 262144x2816) | 1.50 GB | 838 | 13.7% | unmeasured | — | — |
| qkv_proj (BF16) | 1.45 GB | 808 | 13.2% | unmeasured | — | — |
| dense MLP (BF16) | 1.08 GB | 601 | 9.8% | unmeasured | — | — |
| attention, sliding (FP8 KV, window 1024) | 0.84 GB | 470 | 7.7% | unmeasured | — | — |
| o_proj (BF16) | 0.81 GB | 453 | 7.4% | unmeasured | — | — |
| attention, full (FP8 KV) | 0.09 GB | 50 | 0.8% | unmeasured | — | — |
| router GEMM (N=128) | 0.02 GB | 13 | 0.2% | unmeasured | — | — |
| norms / RoPE / glue | 0.02 GB | 12 | 0.2% | unmeasured | — | — |
| **total** | **11.0 GB** | **6137** | | unmeasured | | |

### Decode, B=1 and B=32

| Component | B=1 bytes | B=1 SOL µs | B=32 bytes | B=32 SOL µs |
|---|---:|---:|---:|---:|
| routed experts | 0.80 GB | 448 (14%) | 11.2 GB | 6266 (56%) |
| lm_head | 1.48 GB | 826 (26%) | 1.58 GB | 880 |
| qkv_proj + o_proj | 2.25 GB | 1256 (39%) | 2.29 GB | 1275 |
| dense MLP | 1.07 GB | 598 (19%) | 1.09 GB | 610 |
| attention (sliding + full) | 0.12 GB | 65 | 3.73 GB | 2082 |
| **total** | **5.75 GB** | **3206** | **20.0 GB** | **11177** |

What the byte model already says:
- **At B=1, the BF16 non-expert weights are 86% of the bytes.** These are attention projections
  2.25 GB, lm_head 1.48 GB and dense MLP 1.07 GB. The NVFP4 experts are only 14%. Decode at
  B=1 is a BF16-GEMV problem, and the biggest analytic levers are FP8 for those projections
  and for lm_head. Yukon §4 saw the same thing: once experts are 4-bit, BF16 attention and
  lm_head become the dominant byte pool.
- **At B=8, the routed experts are 47% of bytes** (Yukon measured 53% of decode time on MLX).
  The next pools are the BF16 projections (2.26 GB) and lm_head (1.50 GB). Sliding-window KV
  is 0.84 GB even at FP8, and it would be 1.68 GB at BF16. The FP8 KV default is already
  worth about 470 µs of SOL per step.
- **At B=32, experts and sliding KV dominate** (56% and 17%).
- Full-attention KV is small: 0.09 GB at B=8 FP8, 0.18 GB at BF16. Keeping those 5 layers at
  BF16 for fidelity would cost about +50 µs of SOL per step (about 0.8%).

### Prefill, B=8 x 1024 (8192 tokens, 2 chunks of 4096)

Compute-bound. Total SOL is 183 ms at the placeholder BF16 peak. In FLOP terms qkv_proj is
31%, dense MLP 23%, o_proj 18%, routed experts 15% (at the FP4 peak) and attention 7%.
Activation glue is about 22 GB of unfused norm/residual traffic, or 12 ms of memory SOL.
Prefill weighs 0.25 in the composite, and good GEMMs already run within 1.5-2x of SOL at
this shape (crawler feed #1), so prefill headroom is mostly the chunk count, the norm glue,
and the BF16-vs-FP8 GEMM rate.

## 3. Measured profile (UNMEASURED: to fill)

The plan is staged in `scripts/run_config.sh` / `scripts/run_all.sh`:
- `base profile`: B=8 and B=1 timing, 3 reps each, diverse prompts. Then a torch-profiler
  trace with `profile_by_stage`, 8 steps, at B=8 and at B=1.
  `scripts/classify_trace.py` gives per-kernel time, and per-layer position labels give
  per-component time. The skill script `llm-torch-profiler-analysis` gives the
  kernel/overlap/fuse tables.
- `experts`: `--expert-distribution-recorder-mode per_token`, B=1/8/32 x 64 decode steps over
  64 diverse prompts (wikitext-103, SGLang Python source, Markdown; `scripts/make_prompts.py`).
  Output is distinct experts per layer and the run-length histogram.
- `microbench.py`: copy and read GB/s, BF16 and FP8 TFLOP/s at 8192³, and lm_head GEMV at M=1
  and M=8. These are the measured ceilings for the SOL column.

## How to finish (when a GPU is back)

1. Restore the NVIDIA module on bs2 (admin), then take `host.lock` per the coordinator's
   safe-launch protocol.
2. `bash scripts/run_all.sh` on bs2, in order: microbench, base profile, experts, knob screen.
3. Fill the "Achieved / sol_fraction / Gap" columns from `results/base/` and update
   `HYPOTHESES.md` predictions.
