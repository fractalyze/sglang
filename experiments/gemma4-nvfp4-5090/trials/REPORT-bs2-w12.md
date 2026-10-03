# W12 report: the next B=1 costs under speculation (gemma4nv, build-server-2)

- **Branch:** `jumanzii/gemma4nv-analysis`, pushed to `fractalyze`.
- **Host:** build-server-2.
- **Timings:** gate verdicts (paired ABBA, ratio of sums, 4 pairs) unless labelled a screen. A screen is unpaired, one server lifetime.

| trial | control | deciding metric | predicted (frozen) | measured | verdict |
|---|---|---|---|---|---|
| T-SPEC6a: T3's BF16 small-M GEMM on qkv_proj + target lm_head at verify widths | base4-spec-fp8head | W1 TPOT | -0.3 … -0.8% | not gated (microbench: ~54 µs of a 10.3 ms round) | **retired, not gated** (coordinator) |
| T-SPEC6b: target qkv_proj as FP8 weight-only up to M=256 | base4-spec-fp8head | W1 TPOT | -2.5 … -5.5% | **-0.45%**; W8 composite **-3.9%**, W32 **-5.3%** | **retired** |
| **T-SPEC6c: FP8 copy of the target's tied LM head for ≤ 48-row logits batches** | base4-spec-fp8head | W1 TPOT | -2 … -4.5% | **-2.85%** (2.800 → 2.720 ms); W8 composite **+3.0%**; W32 **+0.1%**; GSM8K **+0.08 pt** [-0.44, +0.59]; tool-JSON 40/40 | **kept: new bs2 reference** |
| T-SPEC7: MTP k=6 (+ FP8 head max_m 64) | T-SPEC6c's candidate | W1 TPOT | -1 … -4% | **+0.29%**; W8 composite **-2.1%**, W32 **-5.0%** | **retired** |

- **New bs2 reference:** `base4-spec-fp8head-fp8lmhead`, which is commit `dd5361a7bc` with `SGLANG_OPT_GEMMA4_FP8_LM_HEAD=1` on top of base4-spec-fp8head.
- **Against base4:** W1 TPOT 5.525 → 2.720 ms (gain about 2.03). This is composed from the T-SPEC3, T-SPEC4 and T-SPEC6c gates.
- **The branch's python tree at `17ccadd47c` equals `dd5361a7bc`'s.**

## 0. Gate integrity (commit `fa10617a79`)

**1. Forced-pass flush.**
- `runner.fidelity_passes` flushes the radix cache between the free-running and the teacher-forced fidelity passes. The forced pass no longer reuses KV written by verify rounds (W10's `c-gemma4nv-gate-forced-pass-reuses-spec-kv`).
- In the two W12 gates whose target path is unchanged below the head (T-SPEC6c, T-SPEC7), the control's and the candidate's forced KL match exactly: 0.0329 / 0.0329. W10's T-SPEC4 gate had 452 differing positions.

**2. Fixed W8 prompts.**
- W8 now times a fixed set (seed `W8-fixed-v3`, 4 reps of 8 distinct windows) in every pair, as W1 has since v2.
- W8 composite's per-pair 95% CI half-width fell from ±2.7% in T-SPEC3 to ±0.1-0.8% in the three W12 gates.

**Tests:** `tests/test_gate.py`, 60 cases, all passing on bs2.
- Fixed seeds hold across pairs for W1 and W8.
- W32 still varies per pair.
- The fidelity passes run in the order run → flush → forced.

**Older verdicts are not re-judged.**

## 1. T-SPEC6: the verify-width GEMMs

### Microbench

Script: `trials/spec/verify_gemm_bench.py`. Results: `trials/spec/results/w12-verify-gemm{,-large}.json`. Method: CUDA graph, best of 7, weights rotated over ≥ 256 MB.

| shape | M=6 cuBLAS BF16 | BF16 Triton | FP8 weight-only Triton |
|---|---:|---:|---:|
| qkv sliding 8192 x 2816 | 31.2 µs | 29.5 (-5.6%) | 15.5 (-50%) |
| qkv full 10240 x 2816 | 37.7 | 36.1 (-4.2%) | 19.2 (-49%) |
| target lm_head 262144 x 2816 | 880.6 | 876.9 (-0.4%; cuBLAS at ~93% of DRAM BW) | 440.5 (-50%) |

- **BF16 is T3c's shape class again.** About 54 µs of a 10.3 ms round, so T-SPEC6a was registered and not gated.
- **Large-M finding.** Above a tile's `max_m`, the existing FP8 weight-only route (bf16 upcast, then cuBLAS) costs 2.1-2.5x cuBLAS BF16. At M=192 that is 127 vs 59 µs.
  - base4's FP8 o_proj therefore pays the upcast at every B=8 verify (M=48 > 32) and at W32.
  - Nobody has measured that cost. It is a candidate trial: o_proj on 32-row M blocks, the mechanism T-SPEC6b built and then removed with it.

### T-SPEC6b: FP8 qkv_proj, retired

**Gate:** `T-SPEC6b-20261003-142707-build-server-2-d70081`.

| metric | gain | 95% CI |
|---|---:|---|
| W1 TPOT (deciding) | 1.0045 | 1.0039-1.0051 |
| W8 composite | 0.961 (decode 0.965, prefill 0.948) | 0.953-0.968 |
| W32 | 0.947 | 0.920-0.975 |

**Why it missed by 5x:**
- **The kernels delivered.** In the B=1 profile (`runs/w12-prof-qkvfp8` vs W10's `w10-prof-fp8head`), the verify span fell 7.36 → 6.95 ms. The WMMA fallback went 1.96 → 0.98 ms and the Triton GEMMs rose 1.08 → 1.57 ms.
- **Acceptance fell.** On the gate's 24 W1 prompts (`gate-shape` probe), τ dropped 3.623 → 3.463 (-4.4%). The FP8 attention projections perturb the hidden states the MTP drafter reads, so the drafter agrees less often.
- **Prefill paid too.** qkv prefill on the upcast route cost W8 prefill 5%.
- **New claim:** `c-gemma4nv-target-precision-cut-costs-mtp-acceptance`. A target-side precision cut is priced by its acceptance loss, not by its kernel saving.
- **Clean-up.** The qkv entries were removed in `dd5361a7bc`.
- **Quality was never paired,** because timing had already failed. The control's run that started was killed and marked `ABORTED.txt`.

### T-SPEC6c: FP8 copy of the target's tied head, kept

**Code** (`cb7cf71a39`, `aa2c5be1ef`, `dd5361a7bc`):
- **Switch:** `SGLANG_OPT_GEMMA4_FP8_LM_HEAD`, default off.
- **Where the hook lives:** `gemma4_mm.py`. The checkpoint serves `Gemma4ForConditionalGeneration`, not `Gemma4ForCausalLM`, so the coordinator's `gemma4_causal.py` would never run.
- **Routing:**
  - Batches of up to 48 rows use the FP8 copy on the T-SPEC4 kernel.
  - Wider batches use the BF16 table through LogitsProcessor's own matmul, bit-identical to the control.
  - The hook goes through the existing lm_head `quant_method` hook, so LogitsProcessor is unchanged.
- **Tests:** `test/registered/gemm/test_gemma4_fp8_lm_head.py`, 6 cases. These, T3's and T-SPEC4's test files all pass on the bs2 5090.

**Gate:** `T-SPEC6c-20261003-150155-build-server-2-0cb85a`.

| metric | control | candidate | gain | 95% CI |
|---|---:|---:|---:|---|
| **W1 TPOT (deciding)** | 2.800 ms | 2.720 ms | **1.0293** | 1.0289-1.0297 |
| W8 composite | | | 1.0298 (decode 1.038, prefill 1.005) | 1.026-1.034 |
| W32 | 1757.8 | 1759.4 | 1.0009 | 0.986-1.016 |

**Head fidelity** (the coordinator's rule):
- **The gate's fidelity passes barely use this head.** They run at concurrency 64, where verify batches exceed 48 rows. Its forced pass is one wide prefill, and its forced KL was identical.
- **So I added `spec_probe --mode forced-c1`.** It runs teacher-forced top-20 on the hidden set, one logits row per forward, so every row goes through the decode and verify head.

| forced-c1, 22 hidden prompts, 3,550 positions | control | candidate |
|---|---:|---:|
| top-1 agreement with the reference tokens | 96.75% | 96.68% |
| KL vs the BF16 reference rows | 0.0314 | 0.0317 |
| argmax agreement, candidate vs control | | **99.49%** (3,532 / 3,550) |
| KL(control ‖ candidate), mean / p99 | | **0.00043 / 0.0078** |

**Acceptance.** W1-prompt τ went 3.623 → 3.562, -1.7%. That is just outside the frozen -1.5 … 0%, which was not a falsifier.

**Quality** (paired; `runs/quality-all-base4-spec-fp8head-20261003-151844-…` vs `…-fp8lmhead-20261003-152048-…`):
- GSM8K: 96.66% → 96.74%, +0.08 pt, CI [-0.44, +0.59]. 6 / 5 answers discordant, McNemar p = 1.0.
- Tool-JSON: 40/40 → 40/40.

**Memory.** GPU +740 MB. The KV pool went 52,469 → 45,092 tokens, and W32 still fits.

**New claim:** `c-gemma4nv-target-head-fp8-copy-verify`.

### Memory deltas (coordinator ask)

| trial | weights | KV pool |
|---|---|---|
| T-SPEC6b | -721 MB (qkv BF16 → FP8) | not read from the logs; its gate passed integrity |
| T-SPEC6c | +740 MB (FP8 copy beside the BF16 table) | 52.5k → 45.1k tokens |

## 2. k re-sweep and T-SPEC7

### Hidden-set acceptance

`spec_probe --mode spec`, `runs/w12-k/k*`, on base4-spec-fp8head. 22 prompts, 256 tokens.

| k | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|
| τ | 3.049 | 3.354 | 3.655 | 3.767 | 3.837 |
| correct-drafts histogram (0 … k) | 327/277/213/1030 | 354/250/205/174/696 | 314/240/228/142/100/517 | 308/314/186/127/93/70/397 | 321/286/186/168/93/89/57/268 |

The spec mode's B=1 TPOT uses only 3 prompts, and their spread (2.2-4.0 ms) swamps k. So I added `spec_probe --mode gate-shape`, which times the gate's own workloads once on its fixed prompts.

### Gate-shape screens on base4-spec-fp8head

| k | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|
| W1 TPOT ms | 2.822 | 2.798 | **2.723** | 3.112 |
| W1-prompt τ | 3.479 | 3.623 | 3.977 | 3.730 |

### T-SPEC7: k=6 on the T-SPEC6c reference, retired

- **The enabler.** It raised the FP8 head's `max_m` to 64, so a B=8 verify of 56 rows stays on the FP8 head. It was reverted in `17ccadd47c`.
- **Gate** `T-SPEC7-20261003-153035-build-server-2-ca1bf6`:

  | metric | value |
  |---|---|
  | W1 TPOT (deciding) | 2.7205 → 2.7284 ms (gain 0.9971) |
  | W8 composite | 0.979 |
  | W32 | 0.950 |
  | fidelity | pass |

- **Why.** With the FP8 target head, k=6's W1-prompt τ is 3.814, against 3.977 with the BF16 head.
  - The deepest drafts are the ones most exposed to near-tie flips of the target argmax.
  - k=6 and T-SPEC6c each buy about 2.8% at B=1, and they do not compose.
  - The gate-shape probe on the new reference agrees with the gate (2.727 vs 2.721 ms).
- **New claim:** `c-gemma4nv-k6-and-fp8-head-do-not-compose`.
- **k=5 stays.**

## 3. MoE at verify widths (note + preregistration, no kernel work)

The full note is `analysis/MOE-AT-VERIFY.md` (`2dc43b7228`).

**Where the time goes at B=1, k=5: 2.33 ms per round.**

| part | time | detail |
|---|---|---|
| CUTLASS grouped GEMMs | 1.80 ms | 79% of the weight-byte SOL: fc1 37.8 µs vs 31.8 SOL, fc2 21.6 vs 15.9 |
| six glue launches per layer | 0.53 ms | finalize 0.18, expand 0.10, activation 0.08, strides 0.06, sort 0.06, router 0.05 |

**Why the GEMMs fall short.**
- A 6-token verify touches 23.5 experts per layer at 2 rows each (W8's routing). B=8 touches 67-84.
- Every SM120 FP4 tactic has CTA M ≥ 128, so each tile is 98% padding.
- This is padding inside each expert's tile, not Yukon §5's RUNSKIP empty runs.

**FlashInfer may already have the fix.**
- Its candidate generator duplicates every SM120 TMA-WS config with `swap_ab` and FINALIZE variants.
- The trace shows the autotuner chose the un-swapped `128x32` tile, with a separate finalize kernel.

**T-MOE1 (preregistered in the note, frozen, not implemented).**
- **Change:** a swap-AB NVFP4 grouped GEMM, plus the FINALIZE fusion where valid, at verify widths. W1 TPOT is predicted at -2 … -4.5%.
- **Step 0:** list and time FlashInfer's SM120 tactics at 6 and 48 tokens, and time the current tactic at 1-128 rows per expert. That may make it a tactic-selection change with no kernel work.
- **Kernel work needs approval.** Also, after T-SPEC6b's lesson, it should be checked for acceptance: the numerics tier is unchanged, but the K order changes.

## 4. Vault (shared checkout; only my paths committed; not pushed)

| commit | what |
|---|---|
| `98a6e11`, `78b2216`, `94e624a`, `502f509` | T-SPEC6a, 6b, 6c and 7 predictions |
| `21d659f`, `940a437`, `64a85cb` | claims: target precision cut vs acceptance; FP8 target head; k6 vs FP8 head |
| `f901b36`, `08a194f`, `d6006bb`, `d6f6ecd` | verdicts: 6b retired, 6c kept, 6a retired (not gated), 7 retired |

- **Measurements are `source: manual`.** `meta/ledgers.yaml` has no gemma4nv-bs2 ledger root, so `wm record --import` cannot import bs2 runs. This matches W7-W10.
- **Lint leaves "stale" warnings** on the new claims against base5, as in W10.
- **T-MOE1's prediction is frozen in `analysis/MOE-AT-VERIFY.md`** and not stubbed in the vault, because no implementation is approved.

## 5. Host safety

The per-phase peaks are in `BASELINE.md` § "W12 on bs2".
- **Every engine ran alone** under `host.lock`, in the 24G no-swap scope with the watchdog, and no watchdog tripped.
- **Serving RSS stayed at 19.5 GB.**
- **One probe launch died of CUDA OOM.** A co-tenant `zisk-worker` (29.6 GB of GPU memory) started at the same moment. It was rerun.
- **The W12 queues now wait for a quiet GPU and at least 32 GB of host memory before each launch,** because the co-tenant pushed MemAvailable down to 15-22 GB at times.

## 6. Next (for the coordinator)

1. **Promote T-SPEC6c on bs2** (`base4-spec-fp8head-fp8lmhead`).
   - For bs3's base6 line: apply `git diff 3d1732c505 dd5361a7bc -- python test`, then set `SGLANG_OPT_GEMMA4_FP8_LM_HEAD=1`.
   - That diff is the FP8 head plus the per-tile `max_m` plumbing it uses. It touches `gemma4_mm.py`, `environ.py`, `triton_small_m_bf16_gemm.py`, `unquant.py` (no behaviour change) and two test files.
2. **o_proj FP8 at M > 32** (found in §1).
   - Every B=8 verify and every W32 step takes the 2-2.5x upcast route on 30 o_proj calls.
   - A 32-row M-block route recovers it, with no numerics change relative to base4's o_proj FP8. It is acceptance-neutral and worth a trial.
3. **T-MOE1 step 0.** It is a microbench only and needs no approval.
4. **Remaining B=1 costs on the new reference** (round about 9.9 ms):

   | part | cost |
   |---|---|
   | MoE | 2.3 ms |
   | qkv WMMA | ~0.98 ms (precision cuts there cost acceptance) |
   | drafter layer GEMVs | 0.91 ms |
   | head | 0.44 ms |
   | full-vocab softmax / max in the draft loop | 0.32 ms (an argmax would do for topk=1) |
