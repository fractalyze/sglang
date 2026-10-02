# gemma4nv pinned baseline

**Status: configuration pinned, numbers NOT yet measured.** Both study hosts went offline on
2026-10-02 right after a server launch (see "Host incident" below), so the A/A run, noise, SOL
fractions and GSM8K are still to be produced. Every table cell marked `pending` is filled only
from a gate run json under `/data/jooman/gemma4nv/runs/`; nothing here is quoted from a single
unpaired run.

## Pinned configuration (`gate/refs.json`, ref `base`)

| item | value |
|---|---|
| SGLang commit | `a9871012acb768dc94a43a6542cc32626c7b7b0b` (study branch point; `python/` unmodified) |
| Model | `nvidia/Gemma-4-26B-A4B-NVFP4` @ `a19cfe00be84568a6867111c9a68c9c44fdcffe6` (ModelOpt 0.43.0rc2, `nvfp4_experts_only`) |
| Server flags | `--moe-runner-backend flashinfer_cutlass` (+ gate-fixed `--decode-log-interval 1`, port) |
| Quantization | `modelopt_fp4`, auto-detected from `hf_quant_config.json` |
| KV cache | FP8 E4M3 (`kv_cache_dtype=auto` adopts the checkpoint's `kv_cache_quant_algo: FP8`) |
| Env (bs3) | torch 2.13.0+cu130, sgl_kernel 0.4.7, flashinfer 0.6.18, Python 3.12.3, driver 610.43.02, CUDA 13.3 UMD |
| Env (bs2) | W2's venv, same pins from the same commit's `pyproject.toml` (re-recorded per run in `meta.json`) |

### Which kernels serve what (from the server log of the bs3 bring-up)

| component | backend | evidence |
|---|---|---|
| Routed experts (NVFP4 W, NVFP4 A) | FlashInfer CUTLASS FP4 MoE (`flashinfer_cutlass`), set explicitly | the tree's `auto` picks `flashinfer_trtllm` |
| Attention (sliding + full) | Triton (`attention_backend=triton`) | gemma4 override: `trtllm_mha` only on SM100, else triton |
| Dense GEMMs (attention projections, dense MLP, router, lm_head) | BF16, `bf16_gemm_backend=auto` (cuBLAS through torch) | these modules are excluded from quantization |
| Sampling | FlashInfer | server_args |
| CUDA graphs | decode only, bs 1-48 (`[1,2,4,8,12,16,24,32,40,48]`); prefill graph disabled ("Breakable CUDA graph is incompatible with multimodal model") | server log |
| Chunked prefill | 4096 tokens | server_args default |

The table is re-derived on every gate leg (`leg.json` -> `backends`) and must be re-checked
against the first bs2/bs3 baseline log. The `flashinfer_cutlass` row is pending confirmation:
the bs3 bring-up that used it was the launch after which bs3 went offline.

### Why `--moe-runner-backend flashinfer_cutlass` (smallest working config, no code change)

With no MoE flag, `ModelOptNvFp4FusedMoEMethod.create_moe_runner`
(`python/sglang/srt/layers/quantization/modelopt_quant.py`) resolves `auto` to Marlin only for
SM80-SM90, and to `flashinfer_trtllm` for everything else, SM120 included. The trtllm-gen FP4 MoE
kernels are SM100-only and the SM120 weight path never builds `g1_scale_c`, so the first
CUDA-graph warm-up dies with
`AttributeError: 'FusedMoE' object has no attribute 'g1_scale_c'` (bs3 log
`/data/jooman/gemma4nv/scratch/smoke1.log`). The Gemma-4 model override
(`arg_groups/model_overrides/gemma4.py`) only sets an MoE backend on SM100. FlashInfer's CUTLASS
FP4 MoE supports SM120, so a server flag suffices and `python/sglang` stays untouched. Marlin
(W4A16) is the fallback if CUTLASS fails. The upstream fix (teach `auto` about SM120) is out of
this task's scope.

## Gate protocol (summary; code in `gate/`)

- Workloads: W8 = 8 concurrent streams x 1024-token prompt x 128 greedy tokens, 4 reps per leg;
  W1 = 1 stream x 1024 x 256, 3 reps; W32 = 32 streams x 1024 x 128, 1 rep (ungated).
- Score: W8 composite = prefill_gain^0.25 x decode_gain^0.75, gains = control/candidate ratio of
  sums over all pairs (prefill = per-stream TTFT summed; decode = per-stream e2e - TTFT summed).
- Legs: one server lifetime per leg, ABBA order, >= 4 pairs, under `flock gpu.lock` for the whole
  sequence; warm-up of every timed shape before the window; per-pair skew reported.
- Bar: max(3 x per-pair sigma of ln(gain) from an A/A run, 1%) per metric (`reference/noise.json`).
- Refusals: decode step outside a CUDA graph in the window; foreign GPU process in the window;
  thermal/HW clock throttling; weights checksum changed during the leg; undeclared server-arg
  difference between arms; timed-window outputs of the two arms disagree on > 10% of tokens.
- Fidelity: host-only hidden set (22 prompts, sha256 `424d7d5f...`): 17 short (code, math, chat,
  Korean x2, Japanese, Spanish, tool JSON x3) + 5 long (8.0k, 10.0k, 12.5k, 14.1k, 15.5k tokens).
  Per-prompt token match >= 90%; top-20 KL mean and p99 <= 3x the baseline-vs-baseline values at a
  different batch composition (serial vs all-at-once), with floors 1e-3 / 1e-2.
- Quality: GSM8K first 200 test questions + 40 tool-call JSON requests (English and Korean),
  greedy, tolerance -1.0 pt vs baseline (`gate quality`).
- Host safety: server in a `systemd-run --user --scope` with MemoryMax=28G and no swap; refuse
  to launch below 30 GB MemAvailable.

## Baseline numbers

| metric | value | source run |
|---|---|---|
| W8 prefill (sum of per-stream TTFT, per rep) | pending | |
| W8 decode (sum of per-stream e2e - TTFT, per rep) | pending | |
| W8 decode step | pending | |
| W1 TPOT | pending | |
| W32 throughput (tok/s) | pending | |
| A/A W8 composite per-pair sigma / bar | pending | |
| A/A W1 TPOT per-pair sigma / bar | pending | |
| Fidelity A/A (control vs reference) | pending | |
| KL calibration (serial vs batched): mean / p99 | pending | |
| GSM8K (200) / tool JSON (40) | pending | |

## SOL (speed of light)

Decode bound = compulsory bytes per step / bandwidth (1792 GB/s datasheet, and the `gate peaks`
measured read bandwidth). Bytes come from the safetensors headers: BF16 attention weights, BF16
dense MLP, router, norms, the BF16 tied lm_head (1.48 GB, read every step), the routed experts
counted as **distinct experts the router actually picked per layer** (measured with SGLang's
expert distribution recorder over the hidden prompts at B=8) x per-expert NVFP4 bytes, and the
KV read (sliding layers capped at the 1024 window; full layers one copy because K=V). Prefill
bound = FLOPs / tensor peak (two views: precision as served, and all GEMMs at NVFP4 with
attention at FP8), floored by one weight read. Reported as sol_fraction = t_sol / t_achieved; a
time below its bound triggers an audit.

| workload | component table | sol_fraction decode | sol_fraction prefill |
|---|---|---|---|
| W8 | pending (`reference/sol/sol.json`) | pending | pending |
| W1 | pending | pending | pending |
| W32 | pending | pending | pending |

## vLLM reference (ungated)

vLLM 0.20.0 (the card's version) is installed on bs2 at `/home/jooman/gemma4nv/vllm-venv`
(torch 2.11.0+cu130). Not run yet: both hosts went down before it could be launched.

## Host incident (2026-10-02)

- 13:51 bs3: first launch (default flags) crashed in CUDA-graph warm-up (g1_scale_c); host fine.
- ~13:53 bs3: second launch with `flashinfer_cutlass`; bs3 stopped answering ssh and ping
  (tailscale) from both the laptop and bs2 within ~1 min and has not come back.
- ~14:55 bs2: first launch with `flashinfer_cutlass` after a co-tenant job freed the GPU (host had
  swap 7/7 GB used, ~23 GB available); ssh closed by the remote within ~1-2 min, host unreachable
  since.
- Cause unknown (host OOM during weight load vs GPU/driver fault, driver 610.43.02 on both). The
  harness now caps server memory and refuses to launch on low host RAM (commit 2778e4245).
