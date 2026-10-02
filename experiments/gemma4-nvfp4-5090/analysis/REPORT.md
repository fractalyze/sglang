# W2 report: profile, SOL byte model, ranked hypotheses (gemma4nv)

Branch `jumanzii/gemma4nv-analysis`, base `a9871012a`. Host build-server-2 (bs2).

## Done

- **bs2 bring-up.** venv at `/home/jooman/gemma4nv/venv` (torch 2.13.0+cu130, flashinfer
  0.6.18, sglang editable from `/data/jooman/gemma4nv/src-analysis`, built with
  `SGLANG_BUILD_RUST_EXTS=none` because the Rust manifest isn't synced). Model at
  `/data/jooman/gemma4nv/hf` (18.8 GB on /data, which had 32 GB free). 64 diverse 1024-token
  prompts in `/data/jooman/gemma4nv/results/prompts_1024.json`.
- **Serving facts from the launches** (`evidence/`):
  - `--moe-runner-backend auto` resolves to `flashinfer_trtllm` on SM120 and crashes at load
    (`'FusedMoE' object has no attribute 'g1_scale_c'`). `flashinfer_cutlass` is required, and
    it is the only in-tree NVFP4 MoE runner that accepts Gemma's GeGLU.
  - `--kv-cache-dtype auto` resolves to **FP8 e4m3 on all 30 layers** (checkpoint
    `kv_cache_quant_algo: FP8`), the full-attention layers included.
  - Attention is `triton`. The prefill CUDA graph is disabled (multimodal arch). Chunked
    prefill is 4096, and decode graph bs runs up to 48.
- **Code survey** of the per-layer kernel order, glue waste, the SWA split-KV issue, and the
  MTP path: `PROFILE.md` §1.
- **Analytic byte/FLOP/SOL model** (`scripts/sol_model.py`) at B=1/8/32 decode and B=8x1024
  prefill: `PROFILE.md` §2.
- **Ranked hypotheses** (13 candidates plus the checked non-candidates) with predicted deltas,
  and **one `wm consult`** (COLD START for this model; GPU-level priors folded in):
  `HYPOTHESES.md`, raw output in `consult_raw.txt`.
- **Measurement tooling**, staged on bs2 and bs3: `scripts/{make_prompts,drive,experts,classify_trace,microbench}.py`
  plus `scripts/{run_config,job,run_all}.sh`. The order is microbench → JIT prebuild → base
  profile → expert counts → 10-config knob screen. Every step runs under W1's `gate.hostwatch`
  (host.lock, 24G scope, watchdog) with W1's exact baseline flags.
- **Bound correction for W1:** full-attention layers store K and V separately (K is roped, V is
  not), so the KV bound needs two copies, not one (`PROFILE.md` §1).

## Not done (blocked)

- **No measured numbers.** bs2 flapped (ssh timeouts, load avg 124-180, kernel OOM kills
  15:04-15:05 per the coordinator), then rebooted at ~15:36 into kernel `7.0.0-34-generic`.
  NVIDIA modules exist only for 7.0.0-27/-28 (`linux-modules-nvidia-595-open-7.0.0-28-generic`),
  so `nvidia-smi` fails. bs3 was offline too. Fixing this needs a sudo user: install
  `linux-modules-nvidia-595-open-7.0.0-34-generic`, or boot into 7.0.0-28.
- Missing as a result: the measured component table (achieved µs, sol_fraction, gap), the
  distinct-experts measurement, and the knob-screen results. Every "unmeasured" cell in
  `PROFILE.md` / `HYPOTHESES.md` comes from `bash scripts/run_all.sh` on bs2 once the GPU is
  back, or on a bs3 slot, once the coordinator gives the go.

## Top of the list (analytic)

1. H1: W4A16 decode MoE (enable tanh-GeGLU in the marlin NVFP4 MoE runner, or a gather-GEMV).
   Experts are 47% of B=8 decode bytes.
2. H2: FP8 weight-only for the BF16 qkv/o/dense-MLP projections. They are 30% of B=8 and 58% of
   B=1 decode bytes. Weight-only, because vault T12 shows W8A8 losing at small M on this GPU.
3. H3: FP8 or exact-greedy lm_head (14% of B=8 and 26% of B=1 decode bytes).
4. H4: decode glue fusion (about 150 launches per step), latency-only per vault C14.
