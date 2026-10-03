# W11 report: speculative decoding on base5 (build-server-3)

**Outcome: done, with one provisional item.**
- **T-SPEC5 kept** (base5 + MTP k=5 + split-KV verify + pool 0.78): W8 composite **1.349**, W1 TPOT **-44.9%** (5.18 -> 2.86 ms), W32 **+3.2%**. Fidelity passes (decode KL 0.016 vs control 0.021), full GSM8K -0.23 pt (CI [-0.86, +0.40]), hidden-set acceptance length **3.29**.
- **T-SPEC4b kept** (W10's FP8 MTP head on top, at the coordinator's request): W1 TPOT **-7.7%**, W8 +5.4%, W32 +5.0%; hidden tau 3.49.
- **base6 pinned, provisional** = base5 + both, from a 6-pair A/A whose integrity is flagged on one leg (one co-tenant telemetry sample). W8 decode 4.61 ms per stream-token (base5 7.97), W1 TPOT **2.626 ms** (base5 5.167, base 6.075), W32 **1748 tok/s** (base5 1617, base 1021). Full GSM8K vs base5 -0.30 pt (CI [-0.95, +0.34]). The confirming A/A died when bs3's GPU went into "GPU requires reset" (escalated; needs root).
- **SOL tables now count o_proj as FP8** (v3). base5 decode sol_fraction W8 0.672 / W1 0.573 / W32 0.539. Under speculation the per-token SOL no longer bounds decode (base6 reads 1.16 at W8).

## 1. Code and harness

- **T-SPEC5 tree:** `jumanzii/gemma4nv-b3-tspec5` @ `a053c1bd0f` (fractalyze) = base5's `1d859709ef` + `git cherry-pick -x c99575c4f52f` (split-KV Triton verify attention, W8). Clean. Under `python/` and `test/` it differs from `1d859709ef` by exactly `36aa977541..c99575c4f52f`: the two diffs are identical.
- **T-SPEC4b tree:** `jumanzii/gemma4nv-b3-tspec4b` @ `701947e266` = `a053c1bd0f` + cherry-picks of W10's `4ebe6175af`, `262f327c28`, `3d1732c505` (FP8 MTP head). Clean. It adds exactly `c99575c4f52f..3d1732c505`; the only diff-of-diffs difference is one `environ.py` hunk offset.
- No other `python/sglang` edit. The speculative-naming skill was read first; the cherry-picks add no identifiers.
- **Harness (gate branch):** merged W10's gate v2 (`ff10d37648`: W1 on 24 fixed prompts with a per-pair 95% CI, speculative refs and probe) and later W10's T-SPEC4 harness (`b1b46de51b`). New refs `base5-spec`, `base5-spec-fp8head`, `base6`. New `trials/spec/verify_width_kv.py`.
- **Drafter on bs3:** `google/gemma-4-26B-A4B-it-assistant` downloaded to `/data/jooman/gemma4nv/models/` (0.83 GB).

## 2. Does T4 compose with the drafter and the verify pass?

T4 runs at every forward mode, so the verify forward runs it at M = (1 + k) * B rows.

- **Drafter.** `Gemma4AssistantForCausalLM` marks every drafter attention layer `is_kv_shared_layer`. T4's fused q/k/v norm + RoPE + KV store skips shared layers, so the drafter never writes the target's frozen KV through T4. The drafter does take T4's level-2 norm pairs (within 1 bf16 ulp). The hidden-set acceptance length is the end-to-end check on that.
- **Verify write location.** `TritonAttnBackend` sets `swa_out_cache_loc` for TARGET_VERIFY in eager mode and in CUDA-graph capture and replay. T4 writes sliding layers there.
- **Unit tests, bs3, tree `a053c1bd0f`:** `test_verify_splitkv.py` + the three gemma4 fused-op test files, **104 passed** (`runs/w11-unit-20261003-130120`, under host.lock + 24G scope + watchdog; peak RSS 2.0 GB).
- **T4's byte-level KV test at the verify widths** (M = 6, 12, 48, 192 for B = 1, 2, 8, 32; both layer shapes; scales None, 1.0, 0.37): **24/24 bit-exact** for q/k/v and every K/V byte.
- **Decode KL through the speculative path:** see §3. It passes, and is lower than the control's.

## 3. T-SPEC5: base5 + speculative stack vs base5

Registered before any gate run (`trials/T-SPEC5-base5-spec.md`, harness `fc3639f966`; vault `gemma4nv-b3-tspec5`, `5b4bd71`). Variant of T-SPEC2b. W1 on W10's gate v2 design (24 fixed prompts), which had landed.

**Gate** `gemma4nv-b3-tspec5-r2-20261003-134245-build-server-3-f33cc2`: 6 pairs, `--decide-on w8_composite`, harness `4f00766728`.

| metric | gain | 95% CI | per-pair sigma | predicted (frozen) | |
|---|---:|---|---:|---|---|
| **W8 composite (deciding)** | **1.349** | 1.329-1.370 | 1.4% | +30 … +48% | inside |
| W8 decode / prefill | 1.625 / 0.772 | | | +55 … +85% / 0.70 … 0.84 | inside |
| W1 TPOT (guard) | 1.816 (5.184 -> 2.855 ms, **-44.9%**) | 1.752-1.884 | 3.5% | -35 … -51% | inside |
| W32 (guard, by hand) | 1.032 (1606 -> 1657 tok/s) | 0.999-1.066 | 3.1% | +1 … +14% | inside, guard passes |

**Verdict: kept.**
- **Fidelity: pass.** Decode KL through the speculative path 0.0162 (p99 0.34) against the control's 0.0207 (0.48). Teacher-forced KL 0.0307 vs 0.0305, min top-1 0.917 vs 0.922.
- **Integrity: ok.** No undeclared server-arg diffs (checked beforehand on two `info` launches). Timed-output agreement 0.12, reported only (reduction order differs).
- **First run** `gemma4nv-b3-tspec5-20261003-130735-build-server-3-c82e63`: the same gains (W8 1.356, W1 1.863, W32 1.047), but pair5-candidate's timed window saw a co-tenant GPU process (`proofman-setup`, 614 MiB). Integrity failed, so it was rerun in full; it is recorded as contaminated.
- **W10's forced-pass note** (the forced pass reuses radix-cached speculative KV): it does not move this trial's forced numbers, which match the control's to 0.0002 KL. Decode KL is the deciding fidelity signal for speculative trials.

**Quality** (full GSM8K 1,319 + tool-JSON 40, base5 arm = W9b's `quality-full-base4-glue2` run):

| | GSM8K | delta | 95% CI (paired) | lost / gained | McNemar p | tool-JSON |
|---|---|---:|---|---|---:|---|
| base5 -> base5-spec | 96.36 -> 96.13% | -0.23 pt | [-0.86, +0.40] | 10 / 7 | 0.63 | 100 -> 100 |

**Acceptance length** (screen `runs/w11-screen-20261003-130252/spec`, greedy, 256 new tokens, tau = tokens per verify round, bonus included):

| set | tau |
|---|---:|
| **hidden (22 prompts)** | **3.29** |
| per hidden category | chat 2.69, code 3.89, korean 2.84, long_code 4.41, long_docs 3.37, long_docs_16k 3.77, long_json 5.22, long_korean 3.20, math 5.02, multilingual 2.72, tool_json 2.48 |
| timing corpus B=1 / B=8 / B=32 | 4.36 / 3.29 / 2.98 |

bs2's T-SPEC1 measured 3.51 on the hidden set with base3; my preregistered band was 3.3 … 3.8, so 3.29 sits at its lower edge.

## 4. T-SPEC4b: the FP8 MTP head on base5-spec (coordinator follow-up)

After T-SPEC5's verdict the coordinator asked for W10's T-SPEC4 on base5 + spec so base6 could carry it. Registered before any run (`trials/T-SPEC4b-base5-spec-fp8head.md`, `4f00766728`; vault `gemma4nv-b3-tspec4b`, `152fb10`).

**Gate** `gemma4nv-b3-tspec4b-20261003-140249-build-server-3-900d19`: 6 pairs, `--decide-on w1_tpot_gain`.

| metric | gain | 95% CI | predicted (frozen) | |
|---|---:|---|---|---|
| **W1 TPOT (deciding)** | **1.0836** (2.863 -> 2.642 ms, -7.7%) | 1.046-1.122 | -4 … -8% | inside |
| W8 composite (guard) | 1.054 | 1.017-1.093 | +1 … +5% | above |
| W32 (guard, by hand) | 1.050 (1685 -> 1769 tok/s) | 1.028-1.072 | 0 … +4% | above |
| hidden tau (screen) | 3.29 -> 3.49 | | -2 … +1% | above (falsifier was a >4% drop) |

**Verdict: kept.** Fidelity pass (decode KL 0.0178 vs 0.0162), integrity ok, timed-output agreement 0.88. Per hidden category tau moved both ways (tool_json 2.48 -> 4.22, long_code 4.41 -> 2.94). One changed draft argmax changes every later round of that prompt, so a 22-prompt mean is not a precise acceptance measure for drafter-only changes.

## 5. base6 pinned

Ref `base6` = `base5-spec-fp8head` (commit `701947e266`, `SGLANG_OPT_USE_TRITON_SPLITKV_VERIFY_CUDA=1`, `SGLANG_OPT_MTP_FP8_LM_HEAD=1`, MTP k=5, `--mem-fraction-static 0.78`).

- **A/A** `AA-base6-20261003-142757-build-server-3-fd0d70`: 6 pairs, verdict no promotion, timed-output agreement 1.0, fidelity pass (decode KL 0.0178, p99 0.34).
- **Integrity flagged on one leg:** pair5-control's window saw a co-tenant `/usr/local/bin/python` (526 MiB) in 1 of 106 telemetry samples. The ledger keeps the run `contaminated`.
- **Pin is provisional** (coordinator decision A): confirm with a clean A/A after the bs3 GPU reset. The rerun `AA-base6-r2-20261003-145007` finished 2 pairs (gains within 0.1%) before the fault.

| metric | base6 | base5 | base |
|---|---|---|---|
| W8 prefill (sum of 8 TTFTs) | 2.246 s | 1.733 s | 1.740 s |
| **W8 decode** | 4.680 s; **4.61 ms** per stream-token | 8.100 s; 7.97 ms | 9.442 s; 9.29 ms |
| **W1 TPOT** | **2.626 ms** | 5.167 ms | 6.075 ms |
| **W32** | **1747.6 tok/s** | 1616.6 | 1020.6 |

- **Per-pair sigma** (A/A, quiet host at load1 <= 1.9): W8 composite 0.064%, W8 prefill 0.29%, W8 decode 0.057%, W1 0.031%, W32 0.069%. Every bar is the 1% floor. `noise.json` is from this A/A; base5's is kept as `noise.base5.json`.
- **Composed against base5** (product of the two gated ratios; separate A/As agree): W8 composite 1.349 x 1.054 = 1.42, W1 1.816 x 1.084 = 1.97x (A/As 5.167 -> 2.626 ms = 1.97x), W32 1.032 x 1.050 = 1.08 (A/As 1617 -> 1748 = 1.08).
- **Quality** base5 -> base6: GSM8K 96.36 -> 96.06% (-0.30 pt, CI [-0.95, +0.34], 11 lost / 7 gained, McNemar p 0.48), tool-JSON 100 -> 100. Pass.
- **BASELINE.md** has the base6 section, with the caveat written out.

**bs3 GPU fault.** Between 14:54:56 (pair1-candidate of the rerun shut down normally) and 14:56:42, the RTX 5090 went into "GPU requires reset". `nvidia-smi -q` shows fan, power, remapped rows and repair status as "GPU requires reset", and compute-apps lists `[N/A]` rows. No engine of ours was running at that point. The kernel log needs root, so I cannot see an Xid. The gate then crashed parsing the `[N/A]` row; `7bb6eafba5` makes it refuse cleanly instead (`GpuUnhealthy`, an unresolvable pid counts as foreign; 2 tests, checked live against the faulted GPU). Escalated as `msg_bcea3839ce8e`.

## 6. Findings worth carrying

- **Speculative B=1 decode is host-CPU bound on a shared host.** In T-SPEC5's rerun the candidate's per-prompt W1 TPOT was identical (sum 66.6-66.7 ms over the 24 prompts) in the three legs whose timed window peaked at load1 <= 2. It rose uniformly over all 24 prompts by +4%, +3% and +10% in legs at load1 4.4, 5.9 and 13.7. The gate only reruns a leg above load1 24. W1 under speculation needs a much lower host-load limit, or the per-pair CI (3.5% here vs 0.03% in the quiet A/A) absorbs it. The base6 A/A ran at load1 <= 1.9 and shows W1 per-pair sigma 0.03%.
- **Co-tenants touch the GPU briefly.** Two of my four gate runs had a foreign GPU process (526-614 MiB) in one leg's window, each seen in a single telemetry sample. The gate's integrity rule caught both and I reran both in full.
- **Spec memory.** The candidate's KV pool is 50.0k tokens at 0.78 with the drafter (base5 55.5k at 0.76); 5.58 GB GPU memory free after graph capture vs base5's 5.23 GB. No OOM in any pass, including the logprob pass.

## 7. SOL tables: o_proj as FP8 (W9b left it BF16)

- `gate sol` and the new `gate sol-retable` take the served format from the ref: `SGLANG_OPT_USE_TRITON_SMALL_M_FP8_WEIGHT_GEMM=1` counts every o_proj as FP8 E4M3 (1 byte per weight) plus an fp32 scale per output row. Prefill FLOPs keep o_proj in the bf16 bucket (cuBLAS on the upcast). Two new unit tests.
- `reference/sol/sol.v3.json` = v2's router data, served formats of base4 on. Attention weights per decode step drop 2.220 -> 1.817 GB; SOL decode step W8 5.582 -> 5.357 ms, W1 3.186 -> 2.961, W32 8.240 -> 8.015. `sol-retable --ref base3` reproduces v2 exactly.
- So base4/base5 sol_fractions were overstated, not understated as W9b wrote: base5 decode sol_fraction W8 **0.672** (0.697 on v2), W1 **0.573** (0.616), W32 **0.539** (0.547); base4 W8 0.627 (0.651). BASELINE.md has the v3 table under base5 and the bytes in the SOL section.
- `sol-report --sol <file>` now writes `sol_fractions.<stem>.json` for non-default tables. My first re-report replaced base5's A/A `sol_fractions.json`; I regenerated it from the default tables (identical to W9b's 0.697 / 0.616 / 0.547).

## Host safety

Every engine process ran through the gate's `host_lock` (host.lock + gpu.lock) + `systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0` + the 2 s watchdog: 1 prebuild, 5 probe launches (2 info, 3 spec), 3 trial gate runs (40 leg attempts incl. 4 host-load retries), 2 A/As (17 leg attempts), 2 full-GSM8K quality servers, and the unit tests. No nvcc ran (FlashInfer cache warm; Triton JIT only). No watchdog trip.

| phase | peak tree RSS | min MemAvailable | peak load1 (host) |
|---|---:|---:|---:|
| JIT prebuild + warm-up (base5-spec) | 8.0 GB | 50.0 GB | 0.8 |
| weight load | 9.8-14.4 GB | 44.3 GB | 18.2 (co-tenant) |
| autotune | 6.8 GB | 34.4 GB (co-tenant build) | 20.5 |
| CUDA-graph capture | 7.4 GB | 36.3 GB | 28.9 (co-tenant) |
| serving (draft loop graphed) | **20.0 GB** | 40.1 GB | 30.9 (co-tenant) |
| unit tests | 2.0 GB | 54.4 GB | 0.7 |

- Serving RSS under speculation is 20.0 GB in every run, 4 GB under the scope cap (W10 saw 19.5 GB on bs2).
- Weight-load RSS varies 9.8-14.4 GB between launches of the same ref (page-cache attribution, as W10 found).
- Host swap peaked at 0.5 GB, already held before my runs started.
- Load peaks above 24 came from root-owned `rustc`/`cargo` co-tenant builds. The gate retried 4 legs for them and kept the earlier attempts.

## Vault (shared checkout; only my paths; not pushed)

| commit | what |
|---|---|
| `0881272`, `3ac1aed` | ledger map `base5-spec` -> `gemma4nv-b3-tspec5`, `base5-spec-fp8head` -> `gemma4nv-b3-tspec4b` |
| `5b4bd71` | T-SPEC5 prediction frozen |
| `152fb10` | T-SPEC4b prediction frozen |
| `23fffa6`, `7ad560b`, `59e4008` | T-SPEC5 recorded kept (deciding = rerun; run 1 contaminated), raw import, prose |
| `420f5ac`, `385a867`, `c0d714e` | T-SPEC4b recorded kept with raw import, prose, result stack |
| `95a3fd3` | raw import: base6 A/As, base6 quality, noise, SOL v2/v3 tables and v3 fractions (import list gains `sol.v*.json`, `sol_fractions.*.json`) |
| `1be61eb` | stack `stack-701947e26-gemma4nv-base6` (parent base5), provisional gap noted |

Lint warnings left: `c-gemma4nv-mtp-verify-shares-experts-wins-b1-b8` is marked stale (established on base3) and "unlinked" for T-SPEC2b/3/5, whose evidence does not cite it. I did not supersede it: its mechanism held on base5.

## Left

- W10's forced-pass radix-cache note is not fixed in the gate. Flushing before the forced pass would change the forced numbers against a reference calibrated without a flush, so it needs a recalibration. The decode-KL check is unaffected.
- The gate's load-24 rerun limit is too loose for speculative W1 (§6). Suggest a separate, lower limit for the timed window.
- **Confirm base6 with a clean 6-pair A/A once bs3's GPU is reset** (then `gate set-noise` from it and update the vault stack's gap).
- A speculative SOL model (bytes per verify round / tokens per round) so base6's decode gets a sol_fraction again.
- The upstream `can_fuse = False` path stays disabled and untouched.
