# gemma4nv pinned baseline (build-server-3)

Every number below comes from a gate run json under `/data/jooman/gemma4nv/runs/` on
build-server-3. Nothing is quoted from a single unpaired run. Compare hosts by delta only.

- **Decision run:** A/A `AA-20261003-020310-build-server-3-cb509e`
  - 6 clean ABBA pairs, harness 8f5e3a589.
  - Re-evaluated at c16a12cf5 after `startup_time` was marked volatile; the first report is kept as `report.v1.json`.
- **Verdict:** integrity OK, fidelity pass, no promotion (as an A/A must).

## Pinned configuration (`gate/refs.json`, ref `base`)

| item | value |
|---|---|
| SGLang commit | `a9871012acb768dc94a43a6542cc32626c7b7b0b` (study branch point; `python/sglang` unmodified) |
| Model | `nvidia/Gemma-4-26B-A4B-NVFP4` @ `a19cfe00be84568a6867111c9a68c9c44fdcffe6` (ModelOpt 0.43.0rc2, `nvfp4_experts_only`); safetensors sha256 match the revision's LFS hashes (`reference/checkpoint_sha256.json`) |
| Server flags | `--moe-runner-backend flashinfer_cutlass --cuda-graph-max-bs-decode 32`, plus the gate-fixed `--decode-log-interval 1` and port |
| KV cache | FP8 E4M3 for all layers (`kv_cache_dtype=auto` adopts the checkpoint's `kv_cache_quant_algo: FP8`; this is what NVIDIA ships and is kept as the baseline) |
| Env | torch 2.13.0+cu130, sgl_kernel 0.4.7, flashinfer 0.6.18, transformers 5.12.1, triton 3.7.1, Python 3.12.3 |
| GPU | RTX 5090, driver 610.43.02, VBIOS 98.02.2E.40.7D, power limit 575 W |
| Host | build-server-3: 32 cores, 60 GB RAM shared with other users, kernel 7.0.0-34 |
| Harness protocol | host.lock; `systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0`; JIT `MAX_JOBS=2` (see the incident section); watchdog every 2 s |

### Which kernels serve what (server log of every leg; re-derived per leg in `leg.json`)

| component | backend |
|---|---|
| Routed experts (NVFP4 weights and activations, static input scales) | FlashInfer CUTLASS fused MoE, JIT module `fused_moe_120` (sm_120f), autotuned at launch |
| Attention, sliding and full layers | Triton (the Gemma4 override picks `trtllm_mha` only on SM100) |
| Dense GEMMs (q/k/v/o, dense GeGLU MLP, router, tied lm_head) | BF16 via torch/cuBLAS (`bf16_gemm_backend=auto`); these modules are excluded from quantization |
| KV cache | FP8 E4M3; sliding pool 29,664 tokens, full pool 37,081 tokens (`mem_fraction_static` 0.718) |
| CUDA graphs | decode only, bs {1,2,4,8,12,16,24,32}; prefill graph disabled ("Breakable CUDA graph is incompatible with multimodal model") |
| Chunked prefill | 4096 tokens |

### Smallest working config on SM120

The `--moe-runner-backend flashinfer_cutlass` flag is required, and it needs no code change.

- With no MoE flag, `ModelOptNvFp4FusedMoEMethod.create_moe_runner` resolves `auto` to `flashinfer_trtllm` on every device outside SM80 to SM90, SM120 included.
- The trtllm-gen FP4 MoE kernels are SM100-only, so the first CUDA-graph warm-up fails with `AttributeError: 'FusedMoE' object has no attribute 'g1_scale_c'`.
- With the flag, the bs3 baseline served every gate workload. Every decode step in every timed window ran inside a CUDA graph.

## Host OOM incident (2026-10-02) and root cause

On 2026-10-02:

- **bs3:** went down at about 13:53, after an uncapped server launch.
- **bs2:** went down at about 14:55 after my launch, and W2's launch at 15:18 died during FlashInfer autotune. The bs2 journal shows kernel OOM kills at 15:04 to 15:05, and load 124 to 180 before its reboot at 15:36.
- **bs3 afterwards:** its FlashInfer JIT cache held zero compiled `.so` files.

**Root cause, measured under the protocol.** At launch, FlashInfer's autotune pass JIT-compiles the 97-unit SM120 CUTLASS fused-MoE module, and FlashInfer runs ninja without `MAX_JOBS`. That means nproc + 2 = 34 parallel nvcc jobs on a 60 GB host shared with other users.

Peaks per phase, taken from the 2 s watchdog CSVs (`runs/prebuild-base-*`):

| run | phase | duration | min MemAvailable | peak tree RSS | peak load | compilers (procs) | their RSS | largest single compiler |
|---|---|---|---|---|---|---|---|---|
| prebuild 1: server, cold cache, MAX_JOBS=4, 24G scope | weight load | 18 s | 42.9 GB | 12.9 GB | 8.9 | 0 | - | - |
| | **autotune (JIT)** | 423 s, then stopped | **28.3 GB** | **24.9 GB** | 24 | 8 | **19.5 GB** | **9.6 GB** |
| prebuild 2: standalone JIT module, MAX_JOBS=2 | build | 685 s | 31.6 GB | 16.8 GB | 19 | 4 | 16.0 GB | 9.1 GB |
| prebuild 2: server, warm cache | weight load | 14 s | 51.6 GB | 12.4 GB | 2.4 | 0 | - | - |
| | autotune | 22 s | 50.3 GB | 6.3 GB | 2.1 | 4 | 0.3 GB | 0.3 GB |
| | serving and first requests | 20 s | 49.8 GB | 6.9 GB | 1.9 | 0 to 2 | 0.2 GB | 0.2 GB |

Prebuild 1 hit the 24 GB scope limit with the server resident, and systemd stopped the scope. The host itself never dropped below 28 GB available.

- **Measurement:** with 4 jobs, the compilers together held 19.5 GB, about 4.9 GB per job. Single cicc processes reached 9.6 GB.
- **Uncapped estimate:** 34 parallel jobs need about 34 x 4.9 = **166 GB**, and 34 x 9.6 = 326 GB at the per-job peak. That is far beyond a 60 GB host.
- **Conclusion:** this confirms the JIT-parallelism hypothesis. Weight load is not the cause: it peaks at 12.9 GB RSS from mmap plus ModelOpt post-processing.
- **Autotune, once the module is cached:** benign (22 s, 0.3 GB).
- **Remedy now in the harness:**
  - `gate prebuild` builds `fused_moe_120` alone at `MAX_JOBS=2` in the capped scope. Worst case is about 19 GB.
  - Servers inherit `MAX_JOBS=2`.
  - The JIT cache persists under `/data/jooman/gemma4nv/cache/flashinfer`.
- **Watchdog record across every run since then:** 36 CSVs, minimum MemAvailable **28.3 GB**, swap never used. The watchdog killed one SOL recording launch because other tenants' Rust builds pushed load to 55 (our tree held 7 GB). Engines now start only below load 24, and a leg whose window saw load above 24 is rerun.

## Baseline numbers (A/A, 6 pairs, control legs)

| metric | value | definition |
|---|---|---|
| **W8 prefill** | **1.740 s** per rep (sum of 8 per-stream TTFTs); batch prefill (max TTFT) 291 ms | 8 streams, 1024-token prompt each |
| **W8 decode** | **9.442 s** per rep (sum of 8 per-stream e2e - TTFT); 9.29 ms per stream-token | 127 decode steps after the first token |
| **W1 TPOT** | **6.075 ms** | summed (e2e - TTFT) / summed (tokens - 1), 1 x 1024 x 256 |
| W1 prefill (TTFT) | 48.4 ms | |
| **W32 throughput** | **1020.6 tok/s** (ungated) | 32 x 1024 x 128; the 29.7k-token sliding KV pool forces SGLang to retract and recompute requests at this load (32 x 1152 tokens) |

A/A gains (control/candidate, ratio of sums over 6 pairs; ideal is 1.0):

| metric | A/A gain | per-pair sigma of ln(gain) | promotion bar max(3 sigma, 1%) |
|---|---|---|---|
| W8 composite (prefill^0.25 x decode^0.75) | 0.99978 | 0.056% | **1.0%** |
| W8 prefill | 0.99925 | 0.121% | 1.0% |
| W8 decode | 0.99995 | 0.049% | 1.0% |
| W1 TPOT | 1.00018 | 0.033% | **1.0%** |
| W32 throughput | 0.99973 | 0.069% | 1.0% (ungated) |

The largest per-pair W8 composite deviation is 0.10% (pair 2). The bars sit at the 1% floor because 3 sigma is about 0.17%. These bars are in `reference/noise.json`.

**Integrity in every leg:**

- No eager decode step in the window.
- No foreign GPU process.
- No thermal or hardware throttling.
- Host load stayed quiet.
- Prefix-cache hit rate at most 0.06% (BOS only).
- Timed outputs of the two arms agreed 100%.
- No undeclared server-arg difference.

## Fidelity (hidden set, 22 prompts, sha256 `424d7d5f...`)

**Calibration** (`runs/calibrate-*-86420a`, baseline vs baseline on one server):

- **Reproducibility:**
  - Repeating the same 22-prompt batch flips greedy near-ties on 4 prompts, with KL mean 6e-4.
  - Serial vs batched flips 17 prompts (free-running match 0.37 on average).
  - The baseline is therefore not run-to-run deterministic, so the token check is teacher-forced: the reference tokens are fed back as input and each position's top-1 is compared.
- **Teacher-forced serial vs batched:**
  - top-1 agreement min 0.92, mean 0.97.
  - KL mean 0.030, p99 0.53.
- **Prefill vs decode:** the reference's own teacher-forced top-1 matches its decode-path greedy tokens 97% of the time (min 0.94).

Thresholds (3x calibration, with floors):

| check | threshold |
|---|---|
| teacher-forced top-1 agreement per prompt | >= 0.90 |
| forced KL mean / p99 | <= 0.0895 / 1.594 |
| decode-path KL (up to first divergence) mean / p99 | <= 0.0502 / 0.950 |

**A/A result:** both arms vs the reference: forced top-1 min 0.9375, forced KL mean 0.031, decode KL mean 0.018. Pass.

**Finding:** a fresh server at the same batch composition sits at the serial-vs-batched level against the reference, not at the same-server repeat level. Outputs depend on the launch. This is consistent with FlashInfer autotune choosing timing-dependent tactics on each launch. Trials that change kernels should expect KL of this size from the launch alone.

## Quality (`gate quality --ref base --set-baseline`, `reference/quality_baseline.json`)

| task | accuracy |
|---|---|
| GSM8K, first 200 test questions, greedy, chat template, thinking off | **96.0%** |
| Tool-call JSON, 40 requests (English and Korean) | **100.0%** |

A candidate fails if either drops more than 1.0 pt.

## Measured peaks (`gate peaks`, `reference/peaks.json`)

| quantity | measured | datasheet |
|---|---|---|
| DRAM read BW (2 GB bf16 sum) | 1685 GB/s | 1792 GB/s |
| DRAM copy BW (read + write) | 1529 GB/s | |
| BF16 GEMM 8192^3 | 232.8 TFLOP/s | |
| FP8 `_scaled_mm` 8192^3 | 719.1 TFLOP/s | |
| NVFP4 GEMM | not measured (`sgl_kernel` has no `cutlass_scaled_fp4_mm` in 0.4.7); SOL uses 838 TFLOP/s, unverified | |
| Launch floor | eager 2.34 us per kernel; 0.81 us per CUDA-graph node | |

## SOL (speed of light) and sol_fraction

**Decode bound.** Compulsory bytes per step at 1792 GB/s. The bytes come from the safetensors headers:

- Routed experts: the measured distinct experts per layer x 3.345 MB per expert (NVFP4 plus FP8 block scales).
- KV read: sliding layers capped at the 1024 window; full layers counted as one copy because K = V.

**Router measurement.** `--enable-return-routed-experts` at 128 decode steps, taking the per-step union over the batch.

| source | mean distinct experts per layer |
|---|---|
| W8, hidden prompts (used for the SOL) | **42.7** |
| W8, corpus prompts | 34.1 |
| W1 | 8.0 |
| W32, corpus prompts | 62.4 |

Mean bytes per decode step (GB):

| component | W8 | W1 | W32 |
|---|---|---|---|
| attention weights (BF16) | 2.220 | 2.220 | 2.220 |
| dense GeGLU MLP (BF16) | 1.071 | 1.071 | 1.071 |
| lm_head = tied embedding (BF16) | 1.476 | 1.476 | 1.476 |
| router + norms | 0.023 | 0.023 | 0.023 |
| routed experts (NVFP4, distinct only) | 4.283 | 0.803 | 6.260 |
| KV read (FP8) | 0.883 | 0.111 | 3.534 |
| **total** | **9.958** | **5.704** | **14.587** |

On this checkpoint, BF16 attention, dense MLP and lm_head make up 4.77 GB of W8's 9.96 GB step (48%), more than the routed experts (43%).

| workload | achieved decode step | SOL step | **decode sol_fraction** | implied BW | achieved batch prefill | prefill SOL (NVFP4/FP8 view / as served) | **prefill sol_fraction** (NVFP4/FP8 / as served) |
|---|---|---|---|---|---|---|---|
| W8 | 9.29 ms | 5.56 ms | **0.60** | 1072 GB/s | 291 ms | 63.7 / 154.9 ms | **0.22 / 0.53** |
| W1 | 6.07 ms | 3.18 ms | **0.52** | 939 GB/s | 48.4 ms | 9.8 / 19.4 ms | **0.20 / 0.40** |
| W32 | 17.05 ms | 8.14 ms | **0.48** | 856 GB/s | 2615 ms | 254.6 / 619.6 ms | **0.10 / 0.24** |

How to read the prefill columns:

- **NVFP4/FP8 view:** every GEMM at the NVFP4 peak and attention at the FP8 peak.
- **As served:** BF16 non-expert GEMMs at the measured BF16 peak.
- Both are floored by one 17.6 GB weight read.

Caveats:

- No achieved time is below its bound, so no audit was triggered.
- W8 and W32 achieved decode steps include decode steps that overlapped other streams' prefill chunks.
- The W32 time also includes KV-pool retractions.
- The per-step union assumes the batch decodes in lockstep. Long hidden prompts can be one prefill chunk out of step.

## vLLM reference (ungated, `runs/vllm-ref-20261003-042242-build-server-3-e964bc`)

**Setup:**

- vLLM 0.20.0 (the card's version) in `/data/jooman/gemma4nv/vllm-venv`.
- transformers 5.12.1: vLLM's resolver picked 4.57.6, which rejects model type `gemma4`.
- The card's flags, plus `--max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 4096 --limit-mm-per-prompt {image:0,video:0} --gpu-memory-utilization 0.85`.
- Same protocol (scope, watchdog), same prompt generator and workloads, one leg after SGLang's A/A on the same box.
- vLLM chose the `VLLM_CUTLASS` NvFp4 MoE backend.
- Peak tree RSS was 10.8 GB at startup and no JIT compilers ran.

| metric | SGLang baseline | vLLM 0.20 reference |
|---|---|---|
| W8 prefill (sum of per-stream TTFT per rep) | 1.740 s | 1.701 s |
| W8 decode (sum of per-stream e2e - TTFT per rep) | 9.442 s | 9.518 s |
| W1 TPOT | 6.075 ms | 6.199 ms |
| W32 throughput | 1020.6 tok/s | **1530.2 tok/s** |

**Reading:** this is a single unpaired vLLM leg, so it is indicative only and not gated.

- W8 and W1 are within about 2% of each other: the SGLang baseline is not weak on the decision workloads.
- At W32, vLLM is about 50% ahead. That is consistent with SGLang's sliding-window KV pool forcing retractions at 32 x 1152 tokens. That pool size is a candidate trial.

## Known limitations

- **Load-time weight hash.**
  - SGLang's `/weights_checker` has no comparable form for the NVFP4 fused-MoE method, and Gemma4 has no `get_weights_by_name`. An in-memory hash after load is therefore unavailable without model-code changes.
  - The gate instead pins the on-disk bytes (sha256 vs the HF LFS hashes) and refuses undeclared server-arg changes.
  - A post-load transform inside a candidate's code would only be caught by fidelity.
- **Routing measurement.**
  - SGLang's expert-distribution recorder needs a model hook that Gemma4 lacks, so routing comes from `--enable-return-routed-experts`.
  - The capturer reads `num_experts_per_tok`, which Gemma4 calls `top_k_experts`. The SOL recording server, which is never timed, passes the alias through `--json-model-override-args`.
- **No NVFP4 peak measurement:** the prefill NVFP4/FP8 view uses an unverified 838 TFLOP/s.
- **Earlier A/A attempt:** `AA-20261002-164516` (5 pairs, then died on the `/flush_cache` 400 race, since fixed) has no report and is not used.
