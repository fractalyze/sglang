# gemma4nv pinned baseline (build-server-3)

Current pinned base: **base4** (section below). The first base is `base`.

Every number below comes from a gate run json under `/data/jooman/gemma4nv/runs/` on
build-server-3. Nothing is quoted from a single unpaired run. Compare hosts by delta only.

- **Decision run:** A/A `AA-20261003-020310-build-server-3-cb509e`
  - 6 clean ABBA pairs, harness 8f5e3a589.
  - Re-evaluated at c16a12cf5 after `startup_time` was marked volatile; the first report is kept as `report.v1.json`.
- **Verdict:** integrity OK, fidelity pass, no promotion (as an A/A must).

## base4 (current pinned base, 2026-10-03)

`base4` = `base3` plus T3b, kept on bs3 (W9):

- **FP8 E4M3 weight-only o_proj:** SGLang commit `36aa977541` with
  `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM=1`, from T3bS (`gemma4nv-b3-t3b`).
  - The commit is base3's `1fd77e64b0` plus cherry-picks of the reviewed T3b commits; its
    `python/` tree is identical to `ed0aefcd40`. It is on branch `jumanzii/gemma4nv-b3-t3b`.
  - Decided on the W8 composite: 1.0142 against base3, with W1 TPOT -4.03%.
  - Adopted on the full GSM8K test split: the paired delta is -0.23 pt, 95% CI [-0.92, +0.47].
    Tool-JSON stayed at 100%.

In the vault it is `stack-36aa97754-gemma4nv-base4`, with parent `stack-1fd77e64b-gemma4nv-base3`.

- **Decision run:** A/A `AA-base4-20261003-110525-build-server-3-ae60a0`.
  - 6 clean ABBA pairs.
  - Harness b4e832f1a (deploy stamp).
- **Verdict:** integrity OK, fidelity pass, no promotion.
- **Noise:** `reference/noise.json` now comes from this A/A. base3's file is kept as
  `reference/noise.base3.json`.
  - W8 prefill's sigma rose to 0.41%, so its bar is now 1.23% (3 sigma). Every other bar
    stays at 1%.
- **Timed-output agreement:** 1.0 in every pair, so the threshold stays at 0.98.

| metric | base4 | base3 | base | definition |
|---|---|---|---|---|
| **W8 prefill** | **1.785 s** per rep; batch 297 ms | 1.744 s; 291 ms | 1.740 s | sum of 8 TTFTs |
| **W8 decode** | **8.678 s** per rep; 8.54 ms per stream-token | 8.900 s; 8.76 ms | 9.442 s; 9.29 ms | sum of 8 (e2e - TTFT) |
| **W1 TPOT** | **5.598 ms** | 5.833 ms | 6.075 ms | 1 x 1024 x 256 |
| W1 prefill (TTFT) | 49.5 ms | 48.0 ms | 48.4 ms | |
| **W32 throughput** | **1554.4 tok/s** (ungated) | 1551.2 | 1020.6 | 32 x 1024 x 128 |

These columns are separate A/As on one host; the gated delta is the paired T3bS run.
T3b's prefill cost (bf16 upcast before cuBLAS) shows in W8 prefill and W1 TTFT.

A/A gains and noise (6 pairs):

| metric | A/A gain | per-pair sigma of ln(gain) | bar |
|---|---|---|---|
| W8 composite | 1.00021 | 0.080% | 1.0% |
| W8 prefill | 1.00133 | 0.411% | 1.23% |
| W8 decode | 0.99983 | 0.041% | 1.0% |
| W1 TPOT | 1.00006 | 0.110% | 1.0% |
| W32 throughput | 1.00019 | 0.044% | 1.0% |

The per-pair W8 composite ranges from 0.9994 to 1.0016.

sol_fraction (`gate sol-report` against the unchanged SOL tables, which still count o_proj as
BF16; base4's real byte floor is about 0.4 GB lower per step, so these fractions understate how
close the served kernels are to their own floor):

| workload | achieved decode step | **decode sol_fraction** (base3 / base) | implied BW | prefill sol_fraction (NVFP4/FP8 / as served) |
|---|---|---|---|---|
| W8 | 8.54 ms | **0.65** (0.63 / 0.60) | 1166 GB/s | 0.21 / 0.52 |
| W1 | 5.60 ms | **0.57** (0.55 / 0.52) | 1019 GB/s | 0.20 / 0.39 |
| W32 | 15.55 ms | **0.52** (0.52 / 0.48) | 938 GB/s | 0.22 / 0.54 |

**Fidelity** (pair 0 control vs the `base` reference):

- Teacher-forced: min top-1 0.917, mean 0.966, KL mean 0.034, p99 0.72 (base3: 0.969, 0.032,
  0.59). Prefill o_proj now reads the dequantized FP8 weight.
- Decode-path KL: mean 0.025, p99 0.45, against base3's 0.011 and 0.23. The limits are 0.050
  and 0.95, so this passes. The check runs the 22 hidden prompts in one batch (M = 22), so it
  covers the small-M FP8 kernel.

**Quality** (full GSM8K test split, n = 1,319, plus tool-JSON, n = 40; greedy; `gate quality
--gsm8k-n all`, paired by `gate quality-compare`):

| run | GSM8K | tool-JSON |
|---|---|---|
| base3 (A) `quality-full-base3-A-20261003-105333-build-server-3-272124` | 96.51% | 100% |
| base3 (A') `quality-full-base3-A2-20261003-110057-build-server-3-ad25c8` | 96.59% | 100% |
| base3 + T3b `quality-full-base3-t3b-20261003-105714-build-server-3-b032cc` | 96.29% | 100% |

| pair | GSM8K delta | 95% CI (Agresti-Min paired) | lost / gained | McNemar p |
|---|---|---|---|---|
| A -> A' (A/A) | +0.08 pt | [-0.55, +0.71] | 8 / 9 | 1.00 |
| **A -> T3b (deciding)** | **-0.23 pt** | **[-0.92, +0.47]** | 12 / 9 | 0.66 |
| A' -> T3b | -0.30 pt | [-0.95, +0.34] | 11 / 7 | 0.48 |

**Host peaks per phase** (max over the 12 A/A legs; the T3bS gate, the 3 quality runs and the
prebuild stayed within these, apart from a 5.4 peak load1 at the start of the T3bS gate):

| phase | min MemAvailable | peak tree RSS | peak load1 | compilers |
|---|---|---|---|---|
| weight load | 51.0 GB | 11.2 GB | 1.7 | 0 |
| autotune / graph capture | 49.8 GB | 6.5 GB | 1.6 | 2 (Triton) |
| serving / timed | 49.3 GB | 7.0 GB | 1.5 | 2 / 0 |

Swap stayed at its 0.13 GB starting value. `gate prebuild --ref base3-t3b` peaked at 7.0 GB
tree RSS and load 0.8; the FP8 Triton kernel compiled in-process, with no nvcc.

## base3 (pinned 2026-10-03, superseded by base4)

`base3` = `base2` plus two trials, both kept on bs3:

- **Split-KV 16:** `--triton-attention-num-kv-splits 16`, from T2S (`gemma4nv-b3-t2s`). Decided
  on W1 TPOT: -1.94% against base2.
- **Triton small-M BF16 GEMM:** SGLang commit `1fd77e64b0` with
  `SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM=1`, from T3S (`gemma4nv-b3-t3s`). Decided on the W8
  composite: 1.0457 against base2 + split-KV 16.

In the vault it is `stack-1fd77e64b-gemma4nv-base3`, with parent
`stack-a9871012a-gemma4nv-base2`.

- **Decision run:** A/A `AA-base3-20261003-101154-build-server-3-5644da`.
  - 6 clean ABBA pairs.
  - Harness de9b44278 (deploy stamp).
- **Verdict:** integrity OK, fidelity pass, no promotion.
- **Noise:** `reference/noise.json` now comes from this A/A. The base2 noise file is kept as
  `reference/noise.base2.json`.
- **Timed-output agreement:** 1.0 in every pair. The 0.98 hard threshold set from the base2
  A/A held.

| metric | base3 | base2 | base | definition |
|---|---|---|---|---|
| **W8 prefill** | **1.744 s** per rep; batch 291 ms | 1.742 s | 1.740 s | sum of 8 TTFTs |
| **W8 decode** | **8.900 s** per rep; 8.76 ms per stream-token | 9.427 s; 9.28 ms | 9.442 s; 9.29 ms | sum of 8 (e2e - TTFT) |
| **W1 TPOT** | **5.833 ms** | 6.066 ms | 6.075 ms | 1 x 1024 x 256 |
| W1 prefill (TTFT) | 48.0 ms | 48.0 ms | 48.4 ms | |
| **W32 throughput** | **1551.2 tok/s** (ungated) | 1527.4 | 1020.6 | 32 x 1024 x 128 |

These columns are absolute control-leg numbers from three separate A/A runs on one host. The
gated deltas are the paired trial runs (T-W32b, T2S, T3S), not differences between these
columns.

A/A gains and noise (6 pairs):

| metric | A/A gain | per-pair sigma of ln(gain) | bar |
|---|---|---|---|
| W8 composite | 0.99970 | 0.039% | 1.0% |
| W8 prefill | 0.99928 | 0.154% | 1.0% |
| W8 decode | 0.99984 | 0.032% | 1.0% |
| W1 TPOT | 0.99961 | 0.044% | 1.0% |
| W32 throughput | 1.00004 | 0.025% | 1.0% |

The largest per-pair W8 composite deviation is 0.07%.

sol_fraction (`gate sol-report` against the unchanged SOL tables):

| workload | achieved decode step | **decode sol_fraction** (base2 / base) | implied BW | prefill sol_fraction (NVFP4/FP8 / as served) |
|---|---|---|---|---|
| W8 | 8.76 ms | **0.63** (0.60 / 0.60) | 1137 GB/s | 0.22 / 0.53 |
| W1 | 5.83 ms | **0.55** (0.52 / 0.52) | 978 GB/s | 0.21 / 0.40 |
| W32 | 15.68 ms | **0.52** (0.51 / 0.48) | 930 GB/s | 0.23 / 0.55 |

**Fidelity** (pair 0 control vs the `base` reference):

- Teacher-forced: min top-1 0.917, mean 0.969, KL mean 0.032, p99 0.59. The forced pass is a
  prefill (M > 32), which neither base3 change touches, so it equals base2's to four digits.
- Decode-path KL: mean 0.011, p99 0.23, against base2's 0.015 and 0.39. Pass.

**Host peaks per phase** (max over the 12 legs):

| phase | min MemAvailable | peak tree RSS | peak load1 | compilers |
|---|---|---|---|---|
| weight load | 51.0 GB | 10.8 GB | 3.0 | 0 |
| autotune / graph capture | 49.5 GB | 6.5 GB | 2.8 | 2 (Triton) |
| serving / timed | 49.2 GB | 7.0 GB | 2.4 | 2 / 0 |

`gate prebuild --ref base2-splits16-smallm` stayed under 7.0 GB tree RSS at load 1.0. Its new
Triton GEMM compiled in-process, with no nvcc.

## base2 (pinned 2026-10-03, superseded by base3)

`base2` = `base` plus `--mem-fraction-static 0.76`, promoted from T-W32b (`gemma4nv-b3-w32b`,
kept on W32 x1.49 with W8 and W1 neutral and fidelity passing). In the vault it is
`stack-a9871012a-gemma4nv-base2` (parent `stack-a9871012a-gemma4nv`). The `base` sections below
remain the record of the first base.

- **Decision run:** A/A `AA-base2-20261003-092146-build-server-3-dce578`.
  - 6 clean ABBA pairs.
  - Harness a7ece6d73 (the deploy stamp names it; see `bin/deploy`).
- **Verdict:** integrity OK, fidelity pass, no promotion (as an A/A must).
- **Noise and bars:** `reference/noise.json` now comes from this A/A. The `base` noise file is
  kept as `reference/noise.base.json`.
- **Timed-output agreement:** 1.0 in every pair, so the hard threshold for numerics-unchanged
  refs is calibrated at 0.98 (lowest pair minus max(3 sigma, 0.02)).
- **KV pools:** FP8. Full attention holds 51,893 tokens and sliding window holds 41,514 tokens.
  Under `base` they held 37,081 and 29,664.

| metric | base2 (control legs) | base | definition |
|---|---|---|---|
| **W8 prefill** | **1.742 s** per rep; batch prefill 291 ms | 1.740 s; 291 ms | sum of 8 TTFTs |
| **W8 decode** | **9.427 s** per rep; 9.28 ms per stream-token | 9.442 s; 9.29 ms | sum of 8 (e2e - TTFT) |
| **W1 TPOT** | **6.066 ms** | 6.075 ms | 1 x 1024 x 256 |
| W1 prefill (TTFT) | 48.0 ms | 48.4 ms | |
| **W32 throughput** | **1527.4 tok/s** (ungated) | 1020.6 tok/s | 32 x 1024 x 128; no retraction now |

A/A gains and noise (6 pairs):

| metric | A/A gain | per-pair sigma of ln(gain) | bar |
|---|---|---|---|
| W8 composite | 1.00000 | 0.041% | 1.0% |
| W8 prefill | 1.00000 | 0.078% | 1.0% |
| W8 decode | 1.00001 | 0.037% | 1.0% |
| W1 TPOT | 1.00005 | 0.041% | 1.0% |
| W32 throughput | 0.99972 | 0.047% | 1.0% |

The largest per-pair W8 composite deviation is 0.05%.

sol_fraction, from `gate sol-report` on this A/A against the unchanged SOL tables in `reference/sol/sol.json`:

| workload | achieved decode step | decode sol_fraction | achieved batch prefill | prefill sol_fraction (NVFP4/FP8 / as served) |
|---|---|---|---|---|
| W8 | 9.28 ms | **0.60** | 291 ms | 0.22 / 0.53 |
| W1 | 6.07 ms | **0.52** | 48.0 ms | 0.21 / 0.40 |
| W32 | 16.01 ms | **0.51** (base 0.48) | 1124 ms (base 2615) | 0.23 / 0.55 (base 0.10 / 0.24) |

W32 changed and W8 and W1 did not. Retraction under `base` recomputed prefill and stretched
decode steps, and the larger pool removes it.

**Fidelity** (pair 0 control vs the `base` reference; the reference is not re-pinned, so stacked
trials are still judged against the original model behaviour):

- Teacher-forced: min top-1 agreement 0.917, mean 0.969, KL mean 0.032, KL p99 0.59. Pass.
- Free-running decode path: KL mean 0.015, p99 0.39. Pass.
- Free-running token match is 0.32 mean. That is reported only, since greedy near-ties flip
  across launches.

**Host peaks per phase** (max over the 12 A/A legs, from the 2 s watchdog; 24G scope, no swap):

| phase | min MemAvailable | peak tree RSS | peak load1 | compilers (peak count / RSS) |
|---|---|---|---|---|
| start | 51.7 GB | 4.5 GB | 5.9 | 0 |
| weight load | 51.0 GB | **10.9 GB** | 5.6 | 0 |
| autotune / graph capture | 49.8 GB | 6.4 GB | 5.2 | 2 / 0.35 GB (Triton) |
| serving (warm-up) | 49.3 GB | 7.1 GB | 4.3 | 2 / 0.32 GB |
| timed | 49.8 GB | 6.7 GB | 3.2 | 0 |

The weight-load phase is the RSS peak, at 10.9 GB (mmap plus modelopt post-processing), and it
falls back to about 6.5 GB once loading ends. With a warm JIT cache, no nvcc runs at launch.

The `gate prebuild --ref base2-splits16` step (warm FlashInfer cache, new Triton split-16
kernels) peaked as follows:

- Tree RSS 10.5 GB.
- Load1 0.66.
- Two Triton compilers at 0.28 GB.

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

### W10 on bs2 (speculative stack on base4, 2026-10-03)

Peak tree RSS per phase, from the 2 s watchdog CSVs. Every engine ran alone under `host.lock` in the 24G no-swap scope. No watchdog tripped, swap peaked at 0.88 GB, and no compiler other than Triton's ptxas ran.

| run | min MemAvailable | weight load | autotune | graph capture | serving | peak load |
|---|---:|---:|---:|---:|---:|---:|
| prebuild base4 | 44.4 GB | 8.4 GB | 6.1 GB | - | 6.6 GB | 2.0 |
| prebuild base4-spec | 41.3 GB | 12.5 GB | 6.3 GB | 7.1 GB | 19.5 GB | 2.0 |
| gate T-SPEC3 (8 legs) | 48.0 GB | 14.7 GB | 6.0 GB | 7.4 GB | 19.5 GB | 1.4 |
| quality, full GSM8K, base4 / base4-spec | 48.3 GB | 13.7 / 7.9 GB | 6.0 GB | - / 6.3 GB | 6.6 / 19.5 GB | 1.3 |
| screens and profiles, base4-spec and base4-spec-fp8head | 48.8 GB | 6.9-13.4 GB | 5.9-6.1 GB | 6.3-19.2 GB | 19.5-19.6 GB | 3.5 |
| FP8-head unit tests and microbench (3 runs) | 52.7 GB | - | - | - | 1.7-2.0 GB | 3.2 |

- **The 19.5 GB serving RSS appears whenever the MTP draft loop is graphed,** as W8 found. It held at 19.6 GB with the FP8 head. It sits 4.4 GB under the scope cap.
- **Weight-load spikes vary between 6.9 and 14.7 GB from launch to launch** for the same ref. That spread is mmap page-cache attribution, not a change in the model.
- **Two FP8-head launches failed at load** with a Python error, not a memory event (`runs/contaminated-w10-*-loadfail`, peak 11.4 GB).

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
- KV read: sliding layers capped at the 1024 window; full layers count two copies (K and V). `attention_k_eq_v` shares only the projection weight: the cached K is k_norm + RoPE and the cached V is v_norm of the same projection, so both are stored and read. (The first version of this table counted one copy; recomputed from the same router data in `reference/sol/sol.v2.json`.)

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
| KV read (FP8) | 0.928 | 0.117 | 3.712 |
| **total** | **10.003** | **5.710** | **14.766** |

On this checkpoint, BF16 attention, dense MLP and lm_head make up 4.77 GB of W8's 10.00 GB step (48%), more than the routed experts (43%).

| workload | achieved decode step | SOL step | **decode sol_fraction** | implied BW | achieved batch prefill | prefill SOL (NVFP4/FP8 view / as served) | **prefill sol_fraction** (NVFP4/FP8 / as served) |
|---|---|---|---|---|---|---|---|
| W8 | 9.29 ms | 5.58 ms | **0.60** | 1077 GB/s | 291 ms | 63.7 / 154.9 ms | **0.22 / 0.53** |
| W1 | 6.07 ms | 3.19 ms | **0.52** | 940 GB/s | 48.4 ms | 9.8 / 19.4 ms | **0.20 / 0.40** |
| W32 | 17.05 ms | 8.24 ms | **0.48** | 866 GB/s | 2615 ms | 254.6 / 619.6 ms | **0.10 / 0.24** |

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
