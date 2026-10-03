# W7 report: T3 follow-ups, T3c (qkv/router/lm_head) and T3b (FP8 o_proj) (gemma4nv, build-server-2)

- Branch: `jumanzii/gemma4nv-analysis`. Host: build-server-2.
- Control for both trials: base2 + T3 (gate ref `smallm-gemm`, commit `1fd77e64b0`, switch on).
- Every timing below is a gate verdict: paired ABBA, ratio of sums, 4 pairs.
- The microbench numbers are kernel-level and decide only what gets built.

| trial | deciding metric | predicted (frozen) | measured | verdict |
|---|---|---|---|---|
| T3c qkv_proj + router + lm_head on the small-M kernel | W1 TPOT | -1.5 … -3.5% | **-0.92%** (bar 1%) | **not promoted** (falsified) |
| T3b FP8 E4M3 weight-only o_proj | W8 composite | +1.0 … +2.5% | **+1.36%** (bar 1%) | **kept** (fidelity and quality pass) |

## 1. Microbench per layer (`t3/gemm_microbench_w7.py`)

Method:
- Same harness as T3: a CUDA graph of 64 calls, best of 7 replays.
- Weights rotated over at least 256 MB, so every call streams from DRAM.
- SOL is computed at 1.65 TB/s.
- Three repeat runs agree within 0.25% on every cell, so a delta above about 0.5% is real at kernel level.

Files: `t3/results/w7-t3c.json` (full sweep), `w7-t3c-narrow.json` (router split-K sweep), `w7-t3c-rep{1,2,3}.json` and `w7-t3b.json`.

### T3c shapes: µs, with the change vs cuBLAS

| layer (N x K) | M | SOL | cuBLAS | cuBLASLt | FI tinygemm | Triton single-pass (BN32 BK128 s4) | Triton split-K 8 (BN16 BK128) |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv sliding (8192x2816) | 1 | 28.0 | 32.2 (0.87) | 32.2 | 31.4 | **29.4 (-8.7%)** | 31.5 |
| | 8 | 28.1 | 31.4 (0.90) | 31.4 | 32.0 | **29.6 (-5.7%)** | 32.3 |
| | 16 | 28.2 | 31.3 | 31.4 | 43.9 | **29.7 (-5.3%)** | 32.8 |
| | 32 | 28.4 | 31.5 | 38.8 | 70.9 | **30.4 (-3.6%)** | 35.2 |
| qkv full (10240x2816) | 1 | 35.0 | 39.4 (0.89) | 39.4 | 37.8 | **35.9 (-8.9%)** | 38.3 |
| | 8 | 35.1 | 37.9 (0.93) | 37.8 | 37.8 | **36.2 (-4.3%)** | 39.3 |
| | 16 | 35.2 | 37.8 | 37.8 | 53.5 | **36.4 (-3.8%)** | 40.2 |
| | 32 | 35.5 | 37.8 | 44.9 | 87.7 | **37.0 (-2.1%)** | 44.2 |
| router (128x2816) | 1 | 0.4 | 2.45 (0.18) | 2.4 | 3.6 | 5.15 (+110%) | **2.26 (-7.8%)** |
| | 8 | 0.5 | 3.57 (0.13) | 3.6 | 3.6 | 4.96 (+39%) | **2.57 (-28.2%)** |
| | 16 | 0.5 | 3.73 | 3.7 | 3.6 | 5.23 | **2.66 (-28.8%)** |
| | 32 | 0.6 | 3.72 | 4.0 | 3.7 | 7.19 | **2.87 (-22.9%)** |
| lm_head (262144x2816) | 1 | 895 | 933.2 (0.96) | 931.6 | 881.6 | **872.6 (-6.5%)** (BK256) | - |
| | 8 | 897 | 881.4 (1.02) | 881.6 | 885.0 | 877.8 (-0.4%) | - |
| | 16 | 900 | 884.9 | 885.7 | 1767.8 | 881.3 (-0.4%) | - |
| | 32 | 905 | 912.9 | 890.9 | 3532.5 | **886.8 (-2.9%)** | - |

All three layers beat the current path by more than noise at some decode M, so all three went into T3c:
- **qkv_proj**: -4…-9% at M ≤ 16.
- **router**, with split-K only: a single pass cannot fill 170 SMs from 4-8 N tiles.
- **lm_head**, at M=1 and M=32. cuBLAS is already at SOL at M=8 and M=16.
- The router sweep tried split-K {2,4,8,11,16,22}, BLOCK_N {16,32,64} and BLOCK_K {64,128}. Split 8 with BN16 BK128 was the best on average over M.

### T3b shape (o_proj): µs, with the change vs the T3 BF16 kernel

| layer | M | T3 BF16 | **FP8 weight-only Triton** (BN32 BK256 s4) | W8A8 `_scaled_mm` rowwise | W8A8 `_scaled_mm` per-tensor | sgl-kernel `fp8_scaled_mm` |
|---|---:|---:|---:|---:|---:|---:|
| o_proj sliding (2816x4096) | 1 | 14.7 | **8.9 (-39%)** | 35.9 | 21.4 | 49.9 |
| | 8 | 14.8 | **9.1 (-38%)** | 38.2 | 24.2 | 50.0 |
| | 16 | 14.8 | **9.0 (-39%)** | 40.5 | 26.4 | 50.1 |
| | 32 | 15.2 | **12.4 (-18%)** | 40.6 | 25.0 | 50.0 |
| | 1024 | 131.1 (cuBLAS) | - | 139.2 | 108.9 | 99.3 |
| o_proj full (2816x8192) | 1 | 29.0 | **15.7 (-46%)** | 59.3 | 28.2 | 96.0 |
| | 8 | 29.0 | **15.8 (-45%)** | 59.5 | 32.3 | 95.8 |
| | 16 | 29.1 | **15.9 (-46%)** | 59.8 | 32.4 | 96.0 |
| | 32 | 29.3 | **24.0 (-18%)** | 60.3 | 33.0 | 96.0 |
| | 1024 | 239.6 (cuBLAS) | - | 273.8 | 207.9 | 197.8 |

- **Error.** Output error vs fp32 on Gaussian weights is rel-L2 0.027 for weight-only and 0.037 for W8A8.
- **Decode.** W8A8 is slower than the BF16 kernel, because dynamic activation quantization costs 2-4 launches per call. The vault saw the same on this GPU (qi21ed-ed10). So T3b is weight-only.
- **Prefill.** T3b keeps cuBLAS on the bf16 upcast. W8A8 would be 17-25% faster at M=1024, but its numerics differ from the decode path's.

## 2. T3c: verdict not promoted

Change:
- Commit `826d472504`. Same switch `SGLANG_OPT_USE_TRITON_SMALL_M_BF16_GEMM`, with three more allowlist shapes.
- A split-K path (fp32 partials and a reduce launch) for the router.
- The lm_head route in `LogitsProcessor._compute_lm_head`.
- Prediction: `trials/T3c-smallm-qkv-router-lmhead.md`, vault `gemma4nv-b2-t3c`, frozen in vault `415af77`.

Gate `T3c-20261003-095735-build-server-2-fb82c2`:
- control `smallm-gemm`, candidate `smallm-gemm-t3c`, `--decide-on w1_tpot_gain`;
- prebuild `prebuild-smallm-gemm-t3c-20261003-095652-build-server-2-e57c3f`.

| metric | control | candidate | gain | per-pair σ | bar | predicted |
|---|---:|---:|---:|---:|---:|---|
| **W1 TPOT (deciding)** | 5.873 ms | 5.819 ms | **1.0092 (-0.92%)** | 0.04% | 1% | -1.5 … -3.5% |
| W8 composite (guard) | | | 1.0083 | 0.56% | 1% | +0.3 … +1.2% |
| W8 decode | | | 1.0102 | 0.64% | 1% | |
| W8 prefill | | | 1.0025 | 0.53% | 1% | |
| W32 tok/s (guard) | 1572.5 | 1569.8 | 0.9983 | 0.59% | 1% | 0 … +2% |

Checks:
- **Fidelity: pass.** Free-running KL mean is 0.0164 against the control's 0.0130, and p99 is 0.353 against 0.318. Teacher-forced KL is identical to the control, because prefill batches stay on cuBLAS.
- **Integrity: ok.** Timed-output agreement is 0.16. The gate only gates it where numerics are unchanged, and split-K reorders the router sum.
- **Timing: not promoted.** W1 clears the no-regression check but not the 1% bar. Everything else is neutral.

Measured vs predicted: W1 saved 54 µs per step against the microbench sum of 154 µs (35%). T3 had recovered 130% of its microbench sum.

> [!gap] Where the missing 100 µs went is unmeasured. Candidates:
> - lm_head may not take the route at decode. The route check needs a contiguous 2D hidden state.
> - Kernel-level gains may not survive in the graph, when the previous op leaves part of the weight's L2 sets warm for cuBLAS.
>
> A W1 kernel trace of the candidate would settle it. It was not run, because bs2's GPU went to T3b.

**Code after the verdict.** Commit `ed0aefcd40` removes T3c's three allowlist entries, the lm_head route and the split-K path. The branch keeps only kept or pending code. T3c's code stays reachable at `826d472504`, and gate ref `smallm-gemm-t3c` pins it.

## 3. T3b: FP8 E4M3 weight-only o_proj, kept

### Change

- The final code is commit `ed0aefcd40`. The original commit `bbbf5e4d46` plus the T3c removal leaves T3b alone against base2 + T3.
- New switch `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM = EnvBool(False)` in `environ.py`.
- When it is on, `UnquantizedLinearMethod.process_weights_after_loading` stores the two o_proj shapes as E4M3 with a per-output-channel absmax scale (`weight_scale`).
- `apply` handles the two batch sizes differently:
  - **M ≤ 32:** the T3 Triton kernel, with a scale epilogue. It upcasts each weight tile to bf16 (`w.to(x.dtype)`), so the BF16 path is unchanged.
  - **Prefill:** `F.linear(x, w8.to(bf16)) * scale`. The upcast is exact, and the scale factors out of the K sum, so both paths compute the same dequantized product up to bf16 rounding.
- Tests in `test/registered/gemm/test_triton_small_m_bf16_gemm.py` pass on the bs2 5090 (`/home/jooman/gemma4nv/w7/ut2/run.log`, 4/4):
  - the kernel and the prefill path stay within 1 and 3 bf16 half-ulps of the fp32 dequantized product, plus summation-order error;
  - the scale maps each row's absmax to 448;
  - the switch converts only allowlisted shapes and routes decode to the kernel and prefill away from it.
- Prediction: `trials/T3b-fp8-weight-oproj.md`, vault `gemma4nv-b2-t3b`, frozen in vault `59027c9`.

### Gate

Gate `T3b-20261003-101044-build-server-2-df2cee`:
- control `smallm-gemm` (T3c was not kept); candidate `smallm-gemm-t3b`, which declares `weight_layout_change`;
- default decision on the W8 composite;
- prebuild `prebuild-smallm-gemm-t3b-20261003-100754-build-server-2-be0f2e`.

| metric | control | candidate | gain | per-pair σ | bar | predicted |
|---|---:|---:|---:|---:|---:|---|
| **W8 composite (deciding)** | | | **1.0136** | 0.54% | 1% | +1.0 … +2.5% |
| W8 decode | | | 1.0244 | 0.80% | 1% | about +2.4% |
| W8 prefill | | | 0.9819 | 0.38% | 1% | 0 … -1% (missed: -1.8%) |
| W1 TPOT (guard) | 5.873 ms | 5.643 ms | 1.0408 (-3.92%) | 0.05% | 1% | -2.5 … -4.5% |
| W32 tok/s (guard) | 1566.7 | 1567.8 | 1.0007 | 0.63% | 1% | 0 … +3% |

Checks:
- **Fidelity: pass.** It is measured against the base reference.

  | KL vs the base reference | control | candidate | limit |
  |---|---:|---:|---:|
  | free-running mean | 0.0130 | **0.0251** | 0.050 |
  | free-running p99 | 0.318 | 0.451 | 0.950 |
  | teacher-forced mean | 0.0324 | 0.0342 | 0.089 |
  | teacher-forced p99 | 0.590 | 0.716 | 1.594 |

  The lowest per-prompt teacher-forced top-1 agreement is 0.917, against a minimum of 0.90. The same prompt (h12) scores 0.917 on the control too.
- **Integrity: ok.**
- **Quality** (`gate quality`, greedy, tolerance 1 point):

  | | control (set as the bs2 baseline) | candidate |
  |---|---:|---:|
  | run | `quality-smallm-gemm-20261003-100836-build-server-2-1705e3` | `quality-smallm-gemm-t3b-20261003-100942-build-server-2-abf6a3` |
  | GSM8K (n=200) | 97.0% | 96.0% |
  | tool-JSON (n=40) | 100% | 100% |

  It passes. **The GSM8K drop sits exactly at the tolerance edge (2 of 200 items).** One gate run cannot separate that from greedy-decode noise. Any further FP8 linear stacked on T3b needs its own quality run with a larger GSM8K sample.

Measured vs predicted:
- W8 composite, W1 and W32 land inside their frozen intervals.
- Prefill cost 1.8%, not 0-1%. The `w8.to(bf16)` upcast before cuBLAS writes a 23-46 MB bf16 copy per layer per prefill forward, and W8 prefill is short enough that this shows.
- Decode gained 2.44% against the microbench estimate of about 2.4%.

## 4. Host-safety record

Host peaks per phase, over all hostmem CSVs of each run:

| run | min MemAvailable | peak tree RSS (phase) | peak load | compilers |
|---|---:|---:|---:|---:|
| microbench t3c sweep | 53.6 GB | 1.6 GB | 1.1 | 2 (ptxas, 0.25 GB) |
| microbench mb2 (router sweep, reps, t3b) | 53.9 GB | 1.6 GB | 0.9 | 0 |
| unit tests (ut1, ut2) | 53.6 GB | 1.7 GB | 0.3 | 0 |
| prebuild T3c | 49.2 GB | 11.3 GB (weight load) | 0.5 | 0 |
| gate T3c | 49.0 GB | **14.8 GB (weight load)** | 1.3 | 0 |
| prebuild T3b | 49.1 GB | 12.1 GB (weight load) | 0.6 | 0 |
| quality control / candidate | 49.2 GB | 7.2 / 6.6 GB | 1.0 | 0 |
| gate T3b | 49.0 GB | 6.9 GB | 3.5 | 0 |

- Every engine process ran under `host.lock`, through the gate's own `systemd-run` 24G scope and watchdog. The microbenches and unit tests used `python -m gate.hostwatch`, with `MAX_JOBS` set.
- No watchdog tripped and swap held at 0.7 GB. The FlashInfer JIT cache held, so nothing compiled.
- Weight load remains the RSS peak. It reached 14.8 GB in one T3c leg; W6 saw 7.9-11.5 GB.
- bs2 had no foreign GPU processes during the runs.

## 5. Vault (shared checkout; not pushed; only my paths committed)

- `415af77`, `59027c9`: T3c and T3b predictions, frozen before their gate runs.
- `e07c3af`: raw import of `gemma4nv-bs2`, snapshot `20261003T0121Z-norev`. It holds both gate runs, their prebuilds and the ledger.
  - The quality runs are not in the import list of `meta/raw-imports.yaml`. Their numbers are on the T3b page's `accuracy_note` and in this report.
- `1b0e27e`: T3c **retired** (outcome both-wrong).
- `eb4955d`: claim `c-gemma4nv-sm120-fp8-weight-only-beats-w8a8-at-decode` (mechanism, measured-direct).
- `fcaccda`: T3b **kept**, which supports that claim.
- `8f143b7`: result prose.
- One lint warning is left: T3c is "retired on an implementation of unstated fidelity", because the page has no `action.implementation.fidelity`.

## 6. Next

1. **Promote T3b into the next base.** On bs3, base3 = base2 + T3 (+ T2) + `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM=1`, on `ed0aefcd40` or later. The coordinator should re-run `gate quality` there with a larger GSM8K sample, because of the edge result above.
2. **Recover T3b's prefill cost (1.8%).** Either upcast once into a persistent buffer per prefill forward, or use a Triton dequant-GEMM for large M with the same numerics. W8A8 for prefill only is faster still, but it splits the prefill and decode numerics.
3. **Extend FP8 weight-only to other linears, gated on quality each time.**
   - Dense MLP: down is 0.55 sol at M=8 even after T3.
   - qkv_proj: 1.45 GB, the largest BF16 pool after lm_head. FP8 would halve it, where T3c's routing could only shave 4-9%.
4. **Revisit T3c's router split-K only with an in-graph trace.** The kernel is at `826d472504`.
