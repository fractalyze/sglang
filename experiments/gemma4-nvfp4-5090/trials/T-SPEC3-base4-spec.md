# T-SPEC3 (gemma4nv-b2-tspec3): T-SPEC2b's speculative stack on base4

**Status: kept, promote candidate (W10, 2026-10-03).** Gate `T-SPEC3-20261003-123016-build-server-2-057827` vs `base4`:
- W8 composite 1.366;
- W1 TPOT 5.525 → 2.960 ms (-46.4%, CI half-width 0.05% on W1 design v2);
- W32 +6.9%;
- fidelity pass, integrity ok;
- full GSM8K 96.29 → 96.51% (paired CI [-0.33, +0.78]), tool-JSON 100 → 100.

Every metric landed inside its frozen interval. See `REPORT-bs2-w10.md`. Below is the frozen registration.

Registered 2026-10-03 by W10 (bs2) before any gate run. Config only. Variant of T-SPEC2b (`gemma4nv-b2-tspec2b`).

## Control and candidate

- **Control:** `base4` (base3 + T3b FP8 weight-only o_proj, commit `36aa977541`), pinned by W9 on bs3.
- **Candidate:** `base4-spec`: base4 on commit `c99575c4f52f` plus T-SPEC2b's stack:
  - MTP assistant drafter `google/gemma-4-26B-A4B-it-assistant`, k=5, topk 1 (flags as `base3-mtp5`);
  - `SGLANG_OPT_USE_TRITON_SPLITKV_VERIFY_CUDA=1`;
  - `--mem-fraction-static 0.78`.
- `c99575c4f52f` is `36aa977541`'s python tree plus the split-KV verify switch; nothing else under `python/` differs (`git diff --stat 36aa977541 c99575c4f52f -- python`: `verify_splitkv.py`, `environ.py`, `triton_backend.py`).
- Harness: deploy stamp `b6cebfedf2` (tree `daef5c28…`), `src-w10` on bs2.

## Decision rule

- **Deciding metric: W8 composite** (`gate run`'s default rule).
- **Guards: W1 TPOT and W32 tok/s.** The default rule guards W1 only. W32 is guarded by hand from `report.json`: its gain must not fall below `1 - bar` (bar 1%). I am not changing the default rule, because older verdicts re-evaluate through it.
- **Fidelity must pass, decode KL included** (pair 0 of both arms).
- **Quality at scale, separately:** full GSM8K (1,319 questions) and tool-JSON, paired `quality-compare` of base4 against base4-spec. The rule is the gate's: the GSM8K paired 95% CI lower bound must be ≥ -1.0 point, and tool-JSON must not drop.

## W1 design change (gate v2, before this trial)

- **Old design:** W1 drew 3 fresh prompts per pair from the pair seed.
- **Why it failed:** in T-SPEC2b the control's per-prompt TPOT was 5.750-5.759 ms. The candidate's per-prompt gain ranged from 1.19 to 2.74 (log σ 0.28). So the 23.7% per-pair spread measured which prompts were drawn, not timing.
- **New design:** W1 times the same 24 prompts (seed `W1-fixed-v2`) in every leg of every pair. The summary now reports a per-pair 95% t-interval for each metric, along with whether its half-width is under the bar.
- **What the gate measures now:** the per-pair spread is timing noise, and the W1 gain is defined on that fixed 24-prompt set.
- **How well the set represents prompts in general:** this is reported separately, as a bootstrap over the 24 prompts.
- **Older trials keep their verdicts.** They ran on the old design, and the new CI field is informational, not a check.
- Commit `f0ed9bd64f`.

## Prediction (frozen)

The prediction starts from T-SPEC2b's gate ratios against base3. That trial measured W8 composite 1.444, W1 1.845 and W32 1.093.

base4's FP8 o_proj makes the control about 1.4% faster on W8 and 4% faster on W1 (bs3), so the candidate's ratio shrinks slightly. That holds unless the FP8 o_proj also speeds up the verify (M = 6 or 48 rows), which it may not.

| metric (bs2, gate ratio of sums vs base4) | predicted |
|---|---|
| **W8 composite (deciding)** | **+34 … +50%** |
| W8 decode gain | +62 … +90% |
| W8 prefill gain | 0.72 … 0.84 |
| W1 TPOT (guard) | -36 … -52% (fixed 24-prompt set; ±11% from prompt sampling alone) |
| W32 tok/s (guard, by hand) | +2 … +14% |
| W1 per-pair CI half-width | < 1% (the bar), now that prompts are fixed |
| Fidelity | approx. Decode KL mean is near T-SPEC2b's 0.026. Pass. |
| Full GSM8K | paired delta within ±1 pt, CI lower bound ≥ -1.0 pt. Greedy verify preserves the output up to numerics. |

**Expected verdict: kept against base4.**

**Falsified if:**
- the W8 composite gain is below 1.30;
- W1 or W32 regresses beyond its bar;
- fidelity fails;
- the full-GSM8K CI lower bound is below -1.0 point;
- the server OOMs.
