# gemma4nv wrap-up: Gemma-4-26B-A4B (NVFP4) on one RTX 5090, 2026-10-02 to 10-03

This branch (`jumanzii/gemma4nv-wrapup`) collects the code, the gate harness and every trial
record of the gemma4nv study, so the next study can start from it. The study was stopped on
2026-10-03; this file is the entry point. Details live in the files it links.

## Result

Reference host build-server-3 (bs3), every step decided by the paired gate below, quality
checked on the full GSM8K test split (1,319 q) and a 40-item tool-call JSON set.

| metric | `base` (stock SGLang `a9871012a`) | `base6` (pinned, provisional) | speedup |
|---|---|---|---|
| W1 TPOT (1 stream, 1024 in / 256 out) | 6.075 ms | 2.626 ms | 2.31x |
| W8 decode per stream-token (8 streams, 1024 in / 128 out) | 9.29 ms | 4.61 ms | 2.02x |
| W32 output throughput (32 concurrent) | 1020.6 tok/s | 1748 tok/s | 1.71x |
| GSM8K / tool-JSON | baseline | within tolerance at every step | held |

vLLM 0.20 on the same box (one unpaired leg, reference only): W8/W1 within ~2% of `base`,
W32 1530 tok/s.

`base6` is pinned on an A/A with one co-tenant telemetry sample (see BASELINE.md, base6
section); a clean confirming A/A was pending when bs3's GPU went into "GPU requires reset".

## What base6 is (each step behind a default-off switch or a server flag)

| step | change | switch / flag | gate result | record |
|---|---|---|---|---|
| base2 | larger KV pool: sliding-window pool was retracting requests at W32 | `--mem-fraction-static 0.76` (0.78 with spec) | W32 x1.49 | trials/W32-DIAG.md |
| base3 | Triton decode attention split-KV count 16 (tree default 8) | `--triton-attention-num-kv-splits 16` | W1 -1.9% | trials/T2S-base2-splits16.md |
| base3 | Triton small-M BF16 GEMM for o_proj + dense MLP (cuBLAS/cuDNN/FlashInfer all fall to an SM80 WMMA kernel on SM120) | `SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM=1` | W8 x1.046 | trials/T3S-*.md, T3-*.md |
| base4 | FP8 E4M3 weight-only o_proj on the same kernel | `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM=1` | W8 x1.014, W1 -4.0% | trials/T3bS-base3-fp8-oproj.md |
| base5 | fused decode glue: qkv RMSNorm + RoPE + FP8 KV quantize + KV store, norm pairs, router + norm (270 of 392 launches/step removed; KV bytes bit-exact) | `SGLANG_OPT_GEMMA4_FUSED_GLUE=2` | W8 x1.063, W1 -7.6% | trials/T4-base4-glue-fusion.md |
| base6 | MTP speculative decoding with `google/gemma-4-26B-A4B-it-assistant`, k=5 + split-KV Triton TARGET_VERIFY attention | `SGLANG_OPT_USE_TRITON_SPLITKV_VERIFY_CUDA=1` + spec flags in refs `base5-spec` | W8 x1.349, W1 -44.9% | trials/T-SPEC5-base5-spec.md |
| base6 | drafter's tied LM head as FP8 per row | `SGLANG_OPT_MTP_FP8_LM_HEAD=1` | W1 -7.7% | trials/T-SPEC4b-*.md |

Exact flags and pinned commits per ref: `gate/refs.json` (`base` ... `base6` and every trial
ref). base6 as measured is commit `701947e266`; this branch is base6 plus the build-server-2
work below, all default off.

## Retired or open (do not reuse without reading the record)

| trial | what | outcome | why |
|---|---|---|---|
| T1 | chunked prefill 8192 | retired | screen gain was a prefill-wave timing artefact |
| T3c | Triton GEMM for qkv/router/lm_head at decode M | retired | W1 -0.92% < 1% bar |
| T-SPEC6a/6b | BF16 / FP8 qkv at verify widths | retired | 6b: verify -0.41 ms but MTP acceptance fell 4.4% |
| T-SPEC7 | MTP k=6 | retired | W1 +0.3% with the FP8 heads |
| T3d | FP8 o_proj M tiers (large-M path) | retired | W8 +0.46%; W1 -9.1% was the confound below |
| T-MOE1 | MoE tactic / padding at verify widths | retired at step 0 | padding free; autotuner already picks swap-AB 128x32 |
| **T-SPEC6c** | FP8 copy of the target's tied LM head (`SGLANG_OPT_GEMMA4_FP8_LM_HEAD`) | kept on bs2, **suspect** | target numerics change under MTP: see the confound |

## Lessons that transfer (the vault holds them as claims)

1. **Host OOM from JIT.** FlashInfer's launch-time autotune JIT-compiles the 97-unit SM120
   CUTLASS MoE module with `nproc+2` ninja jobs (one cicc 9.6 GB, ~166 GB uncapped); it took
   two 60 GB hosts down. Prebuild with `MAX_JOBS<=2`, run every engine in
   `systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0`, one engine per host
   (`gate/hostwatch.py`, `gate/jit_prebuild.py`; BASELINE.md "Host OOM incident").
2. **SM120 + SGLang defaults:** `--moe-runner-backend auto` picks SM100-only
   `flashinfer_trtllm` and crashes; use `flashinfer_cutlass`. The checkpoint sets FP8 KV for all
   layers, including the 5 full-attention layers.
3. **KV capacity is throughput.** The W32 gap to vLLM was the sliding-window pool retracting
   requests, not a kernel.
4. **Kernel wins shrink end to end.** T6b: -50% microbench, -0.45% e2e. Always gate e2e.
5. **Speculative timing confound.** Under MTP, any change to target numerics changes the
   greedy text, the drafter's acceptance moves on the new text, and timing moves with it while
   per-round kernel time does not (T3d: W1 -9.1% at identical round time). The paired CI cannot
   see it. Decompose speedup = round-time ratio x acceptance ratio; W14 started that gate rule
   (`gate/specrule.py`, `gate/spec_replay.patch`, `trials/spec/`), unfinished.
6. **Teacher-forced fidelity is a prefill pass.** Decode-only kernels need the decode-path KL
   check; flush the radix cache before the forced pass (`runner.fidelity_passes`).
7. Shared hosts: co-tenant GPU jobs and CI CPU load contaminate legs; the gate refuses or
   discards those pairs. Speculative W1 is sensitive to host CPU load (+3-10% at load 4-14).

## How to run

- Environment: `env/setup_env.sh`, `env/env.sh` (all caches on `/data`; bs3's root disk is full).
- Deploy the experiments tree to a host: `bin/deploy <host> <dir>` (stamps the tree hash the
  gate records).
- Gate: `python -m gate run --control <ref> --candidate <ref> [--decide-on <metric>]`, plus
  `calibrate`, `set-noise`, `quality [--gsm8k-n all]`, `quality-compare`, `sol`, `spec-run`
  (see `gate/__main__.py`). Workloads, bars and the hidden fidelity set: BASELINE.md.
- Tests: `python tests/test_gate.py` (79, harness); GPU tests under `test/registered/`
  (`gemm/test_triton_small_m_bf16_gemm.py`, `gemm/test_*fp8*head.py`,
  `kernels/ops/layernorm/test_gemma4_fused_*.py`, `attention/test_verify_splitkv.py`,
  `spec/test_spec_replay.py`): 126 pass on an RTX 5090 at this branch's head.

## Map

| path | content |
|---|---|
| BASELINE.md | every pinned base, A/A noise, fidelity, quality, SOL, vLLM reference, host incident |
| analysis/PROFILE.md, HYPOTHESES.md | measured component profile and sol_fraction; ranked hypotheses |
| analysis/MOE-AT-VERIFY.md | MoE at verify widths; remaining MoE glue 0.53 ms per round (costed, not started) |
| trials/*.md | one page per trial: frozen prediction, gate run ids, verdict |
| trials/REPORT-*.md | each worker's report, in order |
| gate/ | the harness |

The world-model vault (`fractalyze/optimization-world-model`) has the study page, stacks
`stack-*-gemma4nv-base2..base6`, every trial with its frozen prediction, and the claims; its
local commits were not pushed when the study stopped.

## Applying this to a different workload

The harness, the host-safety protocol and the trial discipline carry over unchanged. The
optimisation results are specific to this setup: NVFP4 experts, batch <= 32, 1K-token prompts,
cache-flushed single-turn requests. For FP8 official checkpoints, long prompts (5-10K),
multi-turn chat with high concurrency and a cost-per-token objective, expect the large levers to
be different and unmeasured here: KV capacity and GPU layout (TP/DP), prefix-cache reuse across
turns, and prefill efficiency. Speculative decoding should be re-judged at that concurrency, with
the confound in lesson 5 handled first.
