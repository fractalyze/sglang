# W10 report: the speculative stack on base4, quality at scale, the draft loop (gemma4nv, build-server-2)

- **Branch:** `jumanzii/gemma4nv-analysis`, pushed to `fractalyze`. It merges `jumanzii/gemma4nv-gate` at `39c744d467`, which carries base4, `gate quality --gsm8k-n all` and the T4 refs.
- **Host:** build-server-2.
- **Timings** in §2 and §4 are gate verdicts: paired ABBA, ratio of sums, 4 pairs.
- **Screens** are labelled; they are unpaired, one server lifetime each.

| trial | control | deciding metric | predicted (frozen) | measured | verdict |
|---|---|---|---|---|---|
| **T-SPEC3**: T-SPEC2b's stack on base4 (`base4-spec`) | base4 | W8 composite | +34 … +50% | **+36.6%** (1.366); W1 TPOT **-46.4%**; W32 **+6.9%**; full GSM8K **+0.23 pt**, CI [-0.33, +0.78] | **kept: promote candidate** |
| **T-SPEC4**: MTP assistant head in FP8 (`base4-spec-fp8head`) | base4-spec | W1 TPOT | -3 … -8% | **-5.5%** (2.964 → 2.800 ms, gain 1.0584); W8 composite **+3.4%**; W32 **+2.1%**; hidden τ 3.648 → 3.655 | **kept** |

## 1. Building `base4-spec` (step 1)

- **Tree.** The branch's `python/` was already base4 plus the split-KV verify switch: `git diff 36aa977541 c99575c4f52f -- python` touches only `verify_splitkv.py`, `environ.py` and `triton_backend.py`. So the ref pins **`c99575c4f52f`** with no cherry-pick.
- **Ref `base4-spec`.** It is `base4` with:
  - the MTP flags of `base3-mtp5` (assistant drafter, k=5, topk 1, `--max-running-requests 48`, decode graphs up to 32);
  - `SGLANG_OPT_USE_TRITON_SPLITKV_VERIFY_CUDA=1`;
  - `--mem-fraction-static 0.78`.
- **Deploy stamp:** `b6cebfedf2` (tree `daef5c2881ba…`) in `src-w10` for the T-SPEC3 gate. Quality ran from the same directory under stamp `87e5f2b21c`, whose gate files are identical.
- **Prebuilds:**
  - `prebuild-base4-20261003-122632-build-server-2-38260a`;
  - `prebuild-base4-spec-20261003-122710-build-server-2-0efafb`.
  - Neither compiled anything: the JIT cache already held both builds.
- **KV pools** (full / sliding tokens):

  | ref | pool |
  |---|---|
  | base4 | 55.3k / 44.2k |
  | base4-spec | 49.8k / 39.8k |
  | base4-spec-fp8head | 52.5k / 42.0k |

  W32 needs 36.9k sliding tokens, so all three hold it.

## 2. T-SPEC3 gate and the W1 design change (step 2)

### W1 design v2 (gate commit `f0ed9bd64f`)

**Diagnosis.** I re-read T-SPEC2b's 12 W1 streams one by one (`T-SPEC2b-20261003-115404-build-server-2-6c0718`):
- The control's TPOT was 5.750-5.759 ms on every prompt.
- The candidate's per-prompt gain ranged from 1.19 to 2.74, a log σ of 0.28.
- So the 23.7% per-pair spread was prompt sampling. With 3 fresh prompts per pair, the pairs drew different acceptance.

**Change.** W1 now times **the same 24 prompts** (seed `W1-fixed-v2`) in every leg of every pair, so the per-pair spread is timing noise again. The summary gains a per-pair 95% Student-t interval for each metric, and the verdict reports `ci95_narrower_than_bar` for each.

**Older verdicts are unchanged.**
- The CI field is reported, not checked, so `gate reevaluate` of an old run gives the old verdict.
- Old runs keep their old W1 numbers.
- The estimand is now the W1 gain on that fixed set. Prompt-to-prompt variation is still real: T-SPEC2b's per-prompt σ implies about ±11% on a fresh 24-prompt draw.

**Tests.** `tests/test_gate.py` gains 3 cases (59/59 pass, also on bs2):
- W1 seeds are identical across pairs, and W8 seeds differ;
- the t-interval math;
- the CI is reported but never a check.

### Verdict

**Registration:** `trials/T-SPEC3-base4-spec.md` (`ff10d37648`), vault `gemma4nv-b2-tspec3`, frozen in `52023a8`.

**Gate** `T-SPEC3-20261003-123016-build-server-2-057827`, default rule. The rule decides on W8 and guards W1; W32 is guarded by hand, as preregistered.

| metric | control (base4) | candidate (base4-spec) | gain | per-pair σ | 95% CI | bar | predicted |
|---|---:|---:|---:|---:|---|---:|---|
| **W8 composite (deciding)** | | | **1.366** | 1.65% | 1.331-1.402 | 1% | +34 … +50% |
| W8 decode / prefill | | | 1.648 / 0.777 | 2.2% / 0.5% | | | +62 … +90% / 0.72 … 0.84 |
| **W1 TPOT (guard)** | 5.525 ms | 2.960 ms | **1.866 (-46.4%)** | **0.029%** | **1.8654-1.8671** | 1% | -36 … -52% |
| W32 tok/s (guard) | 1548.3 | 1654.6 | 1.069 | 3.5% | 1.012-1.129 | 1% | +2 … +14% |

- **Fidelity: pass.** Decode KL mean is 0.0204 against the control's 0.0253, and p99 is 0.283 against 0.451. The forced checks all pass.
- **Integrity: ok.** There are no undeclared arg diffs.
- **Timed-output agreement is 0.12,** reported only, because speculation changes the numerics.
- **The W1 CI half-width is 0.05%, under the 1% bar.** It was 24% per pair in T-SPEC2b. The design change was sufficient on its own: W1 needed no extra pairs.
- **Every metric landed inside its frozen interval.**
- **W8's own CI (±2.7%) is wider than its 1% bar.** W8 still draws fresh prompts per pair, and under speculation its decode gain follows acceptance too. That does not matter at +37%, but a small W8 effect under speculation would need the same fix.

## 3. Quality at scale (step 3)

Full GSM8K (1,319 questions) and tool-JSON (40), paired, `gate quality --gsm8k-n all` then `quality-compare`.
- Control run: `quality-all-base4-20261003-125528-build-server-2-2d81c9`.
- Candidate run: `quality-all-base4-spec-20261003-125850-build-server-2-21d3af`.
- The comparison is saved in the candidate run's directory.

| | base4 | base4-spec | delta | paired 95% CI | discordant (candidate-only / control-only correct) | McNemar p |
|---|---:|---:|---:|---|---|---:|
| GSM8K (n=1319) | 96.29% | 96.51% | **+0.23 pt** | **[-0.33, +0.78]** | 8 / 5 | 0.58 |
| tool-JSON (n=40) | 100% | 100% | 0 | | 0 / 0 | 1.0 |

- **Verdict: pass.** The CI lower bound is -0.33 pt against the -1.0 pt rule, and tool-JSON did not drop.
- **Only 13 of 1,319 answers change correctness,** which is what greedy verify should give: output preserved up to numerics.
- **W8's edge case is resolved.** W8's GSM8K 96 vs 97 on n=200 was sampling. Nothing needed investigating in the drafter KV or the verify numerics.

## 4. The draft loop (step 4)

### Profile

`trials/spec/draft_loop_profile.py` reads the B=1, k=5 torch-profiler traces. Each column is the median of 12 rounds.
- `base4-spec` trace: `runs/w10-prof-spec`.
- `base4-spec-fp8head` trace: `runs/w10-prof-fp8head`.

| part, ms per round | base4-spec | base4-spec-fp8head |
|---|---:|---:|
| **draft loop, span** | **3.65** | **2.92** |
| - lm_head (one full-vocab GEMV per draft step) | **1.67** | **0.82** |
| - drafter layer GEMVs | 0.91 | 0.91 |
| - softmax and max over 262,144 logits | 0.32 | 0.32 |
| - cuBLAS WMMA down_proj + split-K | 0.21 | 0.30 |
| - norms, RoPE, glue | 0.28 | 0.29 |
| - attention | 0.19 | 0.20 |
| **verify, span** | **7.36** | **7.36** |
| - NVFP4 MoE (CUTLASS) | 2.28 | 2.27 |
| - cuBLAS WMMA fallback (qkv, lm_head at M=6) | 1.96 | 1.96 |
| - small-M Triton GEMMs | 1.08 | 1.08 |
| - norms, RoPE, glue | 1.03 | 1.03 |
| - split-KV verify attention | 0.47 | 0.47 |

- **The head was 47% of the draft loop.** The 26B assistant ties its head to its own 262144 × 1024 BF16 `embed_tokens` (512 MB). Every draft step read all of it, at about 87% of DRAM bandwidth (335 µs).
- **The checkpoint has no centroid head** (`use_ordered_embeddings: false`, no `centroids` tensor). The tree's sparse-centroid path serves only the E2B/E4B assistants.
- **Choice of trial (coordinator GO):**
  - FP8 weight-only head on T3b's reviewed small-M kernel.
  - Rejected: a smaller k, because W8's τ still grows to k=7 at B=1 and the verify is now cheap.
  - Rejected: an NVFP4 head, which needs a new M=1 FP4 kernel.
  - Rejected: vocabulary pruning, a risk for multilingual and tool output.

### T-SPEC4: change

**Code.**
- Commits `4ebe6175af`, `262f327c28` and `3d1732c505`. The switch is `SGLANG_OPT_MTP_FP8_LM_HEAD`, default off.
- At load, the head is quantized to E4M3 with per-row absmax scales, using T3b's quantizer in 16k-row chunks. The BF16 tensor is released, which frees 256 MB, about +2.7k / +2.1k KV tokens.
- Draft logits go through `LogitsProcessor`'s quant hook into `triton_small_m_fp8_vocab_head`. The tile is (128, 128, 3), batches above 32 rows are split, and the tile table is separate from the linear allowlist.

**Bugs the screens caught.**
- My first tile, (64, 256, 4), overflowed shared memory at 48 rows.
- The model loader calls `process_weights_after_loading` on every `quant_method`, which my method lacked.
- Both are fixed, and each has a test.

**Tests:** `test/registered/gemm/test_mtp_fp8_vocab_head.py`, 5 cases, all passing on the bs2 5090.
- FP8 logits stay within E4M3's 2^-4 relative bound of the BF16 head at M ∈ {1, 8, 32, 48, 100}.
- A separated top-3 is preserved exactly, and so is top-1.
- Chunked quantization equals one-shot quantization.
- The loader's post-load pass leaves the head intact.
- `LogitsProcessor` takes the hook.

The T3 small-M test file still passes.

**Microbench** (`trials/spec/fp8_head_bench.py`), head GEMM time:

| M | BF16 cuBLAS | FP8 |
|---:|---:|---:|
| 1 | 355 µs | 186 µs |
| 8 | 353 µs | 192 µs |
| 32 | 364 µs | 198 µs |

**Registration:** `trials/T-SPEC4-mtp-fp8-head.md` (`7c177ddf24`), vault `gemma4nv-b2-tspec4`, frozen in `d45c365` before any screen of the change.

### Acceptance on the hidden set (22 prompts, 256 new tokens, greedy; contents never read)

| | base4-spec | base4-spec-fp8head |
|---|---:|---:|
| **τ, all prompts** | **3.648** | **3.655** (+0.2%) |
| chat / Korean / multilingual | 2.75 / 2.75 / 2.72 | 2.81 / 2.93 / 2.71 |
| code / long code | 3.97 / 4.66 | 4.06 / 4.66 |
| math / tool-JSON / long JSON | 4.96 / 4.52 / 4.57 | 4.92 / 4.52 / 4.57 |
| long docs / long docs 16k / long Korean | 5.02 / 3.66 / 3.37 | 5.02 / 3.24 / 3.01 |

- **Per prompt** (`hidden_per_prompt`, by index):
  - 13 of 22 prompts have identical τ.
  - 4 rose: #3 3.08→3.33, #7 2.81→3.01, #10 2.91→3.37 and #13 3.05→3.16.
  - 5 fell: #6 4.49→4.41, #11 2.61→2.59, #12 2.46→2.37, #19 3.37→3.01 and #20 3.66→3.24.
- **The two long-context falls cancel against the rises.** τ moves only where FP8 rounding flips a near-tied draft argmax.

**Screens** (`w10-scr-spec`, `w10-scr-fp8head`), unpaired, base4-spec → fp8head:

| | base4-spec | fp8head |
|---|---:|---:|
| B=1 TPOT | 4.09 ms | 3.26 ms |
| B=8 TPOT | 5.22 ms | 4.92 ms |
| B=32 throughput | 1677 tok/s | 1702 tok/s |

The B=1 screen moved more than the profile can explain (0.74 of an ~11 ms round). It is 3 single-stream reps on prompts whose τ also moved (3.50 → 3.59). The gate decides.

### T-SPEC4 gate

**Gate** `T-SPEC4-20261003-130844-build-server-2-596d2b`, `--decide-on w1_tpot_gain`, base4-spec against base4-spec-fp8head, harness stamp `1f480cbeb2`:

| metric | control | candidate | gain | per-pair σ | 95% CI | bar | predicted |
|---|---:|---:|---:|---:|---|---:|---|
| **W1 TPOT (deciding)** | 2.964 ms | 2.800 ms | **1.0584 (-5.5%)** | 0.03% | 1.0579-1.0589 | 1% | -3 … -8% |
| W8 composite (guard) | | | 1.034 | 1.2% | 1.016-1.054 | 1% | +1 … +4.5% |
| W8 decode / prefill | | | 1.045 / 1.002 | | | | +2 … +6% / - |
| W32 tok/s (guard) | 1665.9 | 1701.6 | 1.021 | 1.3% | 1.000-1.043 | 1% | 0 … +4% |

- **The per-pair W1 gains were 1.0587, 1.0580, 1.0586 and 1.0584.** That is what the fixed-prompt design is for.
- **Every metric landed inside its frozen interval.** The hidden-set τ (+0.2%) also stayed inside its interval.
- **Fidelity: pass.** Decode KL mean is 0.0177 against the control's 0.0204, within the preregistered ±0.01.
- **Integrity: ok.** Timed-output agreement is 0.86.

**Decode/forced KL: investigated, because the target path is unchanged and should give identical numbers** (coordinator's rule).
- **What the gate showed.** The gate's teacher-forced logprobs matched the control bit for bit on 19 of 22 prompts. They differed at every position of 3 long prompts (h18-h20), which is 452 of 3,550 positions. Base4-spec against itself across two gates matched on all 3,550.
- **Diagnosis** (`trials/spec/forced_probe.py`, `runs/w10-forced-*`). I reran the gate's forced pass on fresh servers, at the gate's concurrency (64) and at 1:

  | comparison | prompts that differ |
  |---|---|
  | base4-spec vs fp8head, concurrency 64 | none |
  | base4-spec with its KV pool set to the fp8head's 52,469 tokens vs fp8head, concurrency 1 and 64 | none |
  | base4-spec at its own pool vs fp8head, concurrency 1 | h20 only, a cache-eviction difference from the 256 MB larger pool |

- **The cause is the gate, not the change.** The gate runs its forced pass right after the free-running fidelity pass and does not flush the radix cache (`runner.py:108-109`). Under speculation, the forced pass therefore reuses KV written by verify rounds, and their shapes follow the drafts.
- **So the target numerics are unchanged.** Vault claim `c-gemma4nv-gate-forced-pass-reuses-spec-kv`.

**Verdict: kept.** The drafter head is now 0.82 of the 2.92 ms draft loop.

## 4b. Vault (shared checkout; only my paths committed; not pushed)

| commit | what |
|---|---|
| `52023a8` | T-SPEC3 prediction |
| `d929747` | raw import: T-SPEC3 gate, base4/base4-spec prebuilds, ledger |
| `9dad67d` | T-SPEC3 verdict **kept**, with the full-GSM8K accuracy note |
| `d45c365` | T-SPEC4 prediction |
| `99a278d` | raw import: T-SPEC4 gate, ledger |
| `9685637` | claim `c-gemma4nv-mtp-draft-head-half-of-draft-loop` |
| `873525b` | claim `c-gemma4nv-gate-forced-pass-reuses-spec-kv` |
| `898bc27` | T-SPEC4 verdict **kept** |
| `d95ce8d` | result prose for both trials |

- The bs2 measurements are `source: manual`, as in W7 and W8.
- The quality JSONs are not in `raw/`, because the import list does not cover `runs/quality-*`. They are cited by path.
- Lint leaves "stale" warnings on the new claims against base5, which bs3 pinned after these runs.

## 5. Host safety

The per-phase peaks are in `BASELINE.md` § "W10 on bs2".
- Every engine ran alone under `host.lock`, in the 24G no-swap scope with the watchdog.
- The lowest MemAvailable was 41.3 GB, during the base4-spec prebuild.
- Swap peaked at 0.88 GB, load at 3.5, and no watchdog tripped.
- The FP8 head does not change the 19.5 GB serving RSS of a graphed draft loop.
- **One co-tenant delay.** A co-tenant `cargo-zisk-dev` process held 620 MiB of GPU memory at 0% utilization for about 13 minutes (12:30-12:43). The gate's quiescence rule held the T-SPEC3 start until it left.
- **No legs were contaminated.**

## 6. Next (for the coordinator)

1. **Promote T-SPEC3 + T-SPEC4 together on bs2.** They are kept against base4 and against base4-spec respectively. The composed stack against base4:

   | metric | change |
   |---|---:|
   | W1 TPOT | 5.525 → 2.800 ms (gain about 1.97) |
   | W8 composite | about 1.41 |
   | W32 | about +9% |

   Full GSM8K holds. The switch to add is `SGLANG_OPT_MTP_FP8_LM_HEAD=1` on `3d1732c505` or later; it does nothing without the Gemma-4 assistant. The bs3 base5-spec line (`gemma4nv-b3-tspec5`) needs three commits cherry-picked to pick it up: `4ebe6175af`, `262f327c28` and `3d1732c505`. They touch `gemma4_mtp.py`, `environ.py` and `triton_small_m_bf16_gemm.py`, plus one test, and nothing on the target path.
2. **Gate change to decide on:** `flush_cache()` between the free-running and the forced fidelity passes. Speculative refs would then measure the target alone. It changes the forced numbers of every future run, so it needs a recalibration, and older verdicts should be re-evaluated only for the record.
3. **W8 under speculation** still draws fresh prompts per pair. Its CI half-width here was 2.7% in T-SPEC3, against a 1% bar. Fixing its prompts, as W1's now are, matters only for small W8 effects under speculation.
4. **Next B=1 costs** (fp8head round of about 10.3 ms):
   - **Verify (7.36 ms):**
     - NVFP4 MoE 2.28 ms;
     - cuBLAS WMMA-fallback qkv and lm_head at M=6, 1.96 ms. T3c's small-M routes for these did not pay at decode M, but at verify M=6 to 48 they are a different shape class. A verify-only route is the next candidate.
   - **Draft loop (2.92 ms):**
     - drafter layer GEMVs 0.91 ms (BF16, small-M Triton or FP8 candidates);
     - full-vocab softmax + max 0.32 ms. For topk=1 greedy drafting, an argmax would do.
5. **k re-sweep.** A cheaper draft step moves the k optimum up. k=6 or 7 may now win at B=1, but B=8 τ saturates near 3.4 (W8 sweep). That is a config-only screen.
