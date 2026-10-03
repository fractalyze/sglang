# T-SPEC5 (gemma4nv-b3-tspec5): T-SPEC2b's speculative stack on base5 (build-server-3)

Registered 2026-10-03 by W11 (bs3) before any gate run. Variant of T-SPEC2b (`gemma4nv-b2-tspec2b`); the bs3 counterpart of W10's T-SPEC3 (base4, bs2), which has no verdict yet.

## Control and candidate

- **Control:** `base5` (base4 + T4 fused decode glue, commit `1d859709ef`, `SGLANG_OPT_GEMMA4_FUSED_GLUE=2`), pinned by W9b on bs3.
- **Candidate:** `base5-spec`: base5 on commit `a053c1bd0f` plus T-SPEC2b's stack:
  - MTP assistant drafter `google/gemma-4-26B-A4B-it-assistant`, k=5, topk 1 (flags as `base3-mtp5`), from `/data/jooman/gemma4nv/models/` on bs3;
  - `SGLANG_OPT_USE_TRITON_SPLITKV_VERIFY_CUDA=1`;
  - `--mem-fraction-static 0.78`.
- `a053c1bd0f` = `1d859709ef` + `git cherry-pick -x c99575c4f52f` (clean). Under `python/` and `test/` it differs from `1d859709ef` by exactly the diff `36aa977541..c99575c4f52f` (checked by diffing the two diffs).
- Harness: gate branch with W10's gate v2 merged (`f0ed9bd64f`, fixed 24-prompt W1 with per-pair 95% CI), deployed to `src-w11` on bs3.

## Composition checks (before registration)

T4 runs at every forward mode, so the verify forward runs it at M = (1 + k) * B rows.

- **Drafter:** `Gemma4AssistantForCausalLM` marks every drafter attention layer `is_kv_shared_layer`, and T4's fused q/k/v + RoPE + KV store is skipped for shared layers. The drafter never writes the target's frozen KV through T4. It does take T4's level-2 norm pairs, which are within 1 bf16 ulp; acceptance length measures any effect.
- **Verify write location:** `TritonAttnBackend` sets `swa_out_cache_loc` for TARGET_VERIFY in the eager path and in CUDA-graph capture and replay, and T4 writes sliding layers there.
- **Unit tests on bs3, tree `a053c1bd0f`:** `test_verify_splitkv.py` + the three gemma4 fused-op test files, 104 passed (`runs/w11-unit-20261003-130120`).
- **T4's byte-level KV test at the verify widths** (`trials/spec/verify_width_kv.py`, M = 6, 12, 48, 192 for B = 1, 2, 8, 32; both layer shapes; scales None/1.0/0.37): 24/24 bit-exact for q/k/v and every K/V byte.
- **Decode KL at the verify widths:** the gate's decode-KL pass (pair 0 of both arms) decodes through the speculative path, so it covers the verify forward end to end.

## Decision rule

- **Deciding metric: W8 composite** (`--decide-on w8_composite`).
- **Guards: W1 TPOT** (gate check) **and W32 tok/s** (by hand from `report.json`: gain must not fall below `1 - bar`, bar 1%).
- **W1 design:** W10's gate v2 (24 fixed prompts per leg, seed `W1-fixed-v2`), which landed before this trial. The bs3 noise file was calibrated on the old W1 design (base5 A/A, 3 prompts per leg); its 1% floor bar is kept and the v2 per-pair CI is reported next to it.
- **Fidelity must pass, decode KL included.**
- **Quality at scale:** full GSM8K (1,319) + tool-JSON, paired `quality-compare` of base5 against base5-spec. Rule: GSM8K paired 95% CI lower bound >= -1.0 pt, tool-JSON no drop. The base5 arm reuses W9b's full-GSM8K run of base5.
- **Acceptance length:** hidden-set tau (22 prompts, per category, contents never printed) from `trials/spec/spec_probe.py --mode spec`.
- 6 pairs, as W9b's T4 gate.

## Prediction (frozen)

Start: T-SPEC2b's gate ratios against base3 on bs2: W8 composite 1.444 (decode 1.777, prefill 0.775), W1 1.845, W32 1.093.

base5 is faster than base3 on bs3 (A/As): W8 decode step 8.76 -> 7.97 ms (x1.099), W1 5.83 -> 5.167 ms (x1.128), W32 1551 -> 1617 tok/s (x1.043). The candidate keeps only part of that:
- T4 removes a fixed ~0.4-0.6 ms of launches per target forward, so the verify forward keeps it; the drafter keeps only the norm pairs.
- FP8 o_proj and the small-M BF16 GEMM serve M <= 32 only: the B=1 verify (M=6) takes them, the B=8 verify (M=48) does not.
- A spec round is ~13.8 ms at B=1 and at B=8 (T-SPEC2b screens: TPOT x tau). Saving ~0.7-0.8 ms of it gives the candidate ~6% back, against the control's 10-13%.

| metric (bs3, gate ratio of sums vs base5) | predicted |
|---|---|
| **W8 composite (deciding)** | **+30 … +48%** (point 1.41: decode 1.777 / 0.94 / 1.099 = 1.72, prefill 0.775) |
| W8 decode gain | +55 … +85% |
| W8 prefill gain | 0.70 … 0.84 |
| W1 TPOT (guard) | -35 … -51% (point 5.167 -> ~2.94 ms, 1.76x; fixed 24-prompt set, ±11% from prompt sampling) |
| W32 tok/s (guard, by hand) | +1 … +14% (point 1.093 x 1.03 / 1.043 = 1.08) |
| W1 per-pair 95% CI half-width | < 1% |
| Fidelity | approx/reorder. Decode KL mean near T-SPEC2b's 0.026 (base5 alone 0.021). Pass. |
| Full GSM8K | paired delta within ±1 pt, CI lower bound >= -1.0 pt. |
| Hidden-set tau (k=5) | 3.3 … 3.8 (T-SPEC1 3.51 on bs2) |

**Expected verdict: kept against base5 -> base6.**

**Screen seen before freezing** (unpaired, one server lifetime, `runs/w11-screen-20261003-130252/spec`; the intervals above were composed before it and are not changed): hidden tau 3.29 (22 prompts; chat 2.69, tool_json 2.48, math 5.02); timing corpus tau 4.36 / 3.29 / 2.98 at B=1/8/32; client TPOT median 2.42 ms (B=1), 5.27 ms (B=8; bs2's T-SPEC2b screen 4.60 ms), B=32 1757 tok/s wall. B=8 is weaker than on bs2, so W8 may land at or below the interval's low end. Resolved args vs base5: no undeclared diff (`runner.server_arg_diff` on the two `info` launches). KV pool 50.0k tokens (base5 55.5k); 5.58 GB free after graph capture (base5 5.23 GB).

**Falsified if** the W8 composite gain is below 1.28, W1 or W32 regresses beyond its bar, fidelity fails, the full-GSM8K CI lower bound is below -1.0 point, or the server OOMs (including the gate's logprob pass at 0.78).
