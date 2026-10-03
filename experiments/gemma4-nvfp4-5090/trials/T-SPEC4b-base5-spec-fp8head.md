# T-SPEC4b (gemma4nv-b3-tspec4b): T-SPEC4's FP8 MTP head on base5-spec (build-server-3)

**Status: kept (W11, 2026-10-03).** Gate `gemma4nv-b3-tspec4b-20261003-140249-build-server-3-900d19` (6 pairs) vs `base5-spec`:
- W1 TPOT 2.863 -> 2.642 ms (**-7.7%**, gain 1.0836, CI 1.046-1.122);
- W8 composite **1.054** (above the predicted +1 … +5%); W32 **+5.0%** (above 0 … +4%);
- fidelity pass (decode KL 0.0178 vs control 0.0162), integrity ok;
- hidden tau 3.29 -> 3.49 (screen), outside the predicted -2 … +1% band upward; per category it moved both ways.

See `REPORT-bs3-w11.md`. Below is the frozen registration.

Registered 2026-10-03 by W11 (bs3) before any screen or gate run of the change on base5-spec. Variant of T-SPEC4 (`gemma4nv-b2-tspec4`, kept on bs2 vs base4-spec). The coordinator asked for it so base6 can be base5 + spec + FP8 MTP head if T-SPEC5 and this trial both keep.

## Control and candidate

- **Control:** `base5-spec` (T-SPEC5's candidate, commit `a053c1bd0f`).
- **Candidate:** `base5-spec-fp8head`: commit `701947e266` with `SGLANG_OPT_MTP_FP8_LM_HEAD=1`.
  - `701947e266` = `a053c1bd0f` + `git cherry-pick -x 4ebe6175af 262f327c28 3d1732c505` (clean). Under `python/` and `test/` it adds exactly the diff `c99575c4f52f..3d1732c505`; diffing the two diffs differs only in one `environ.py` hunk offset, where T4's switch sits above.
  - The MTP assistant's tied 262144 x 1024 head is stored as FP8 E4M3 with per-row scales and draft logits run on the Triton small-M FP8 kernel. The target model and its verify forward are unchanged.

## Decision rule

As T-SPEC4:
- **Deciding metric: W1 TPOT** (`--decide-on w1_tpot_gain`), fixed 24-prompt W1 (gate v2).
- **Guards:** W8 composite (gate check) and W32 tok/s (by hand from `report.json`, bar 1%).
- **Fidelity must pass.** The target path is unchanged; the drafter only changes which tokens get proposed.
- **Hidden-set acceptance length** for control and candidate (`spec_probe --mode spec`).
- Quality at scale is not rerun for this trial alone. The target's greedy output is unchanged up to verify-shape numerics, and W10's composed base4 -> base4-spec-fp8head full GSM8K was 96.29 -> 96.51 (CI [-0.33, +0.78]). The base6 pin runs full GSM8K on the composed stack against base5.

## Prediction (frozen)

T-SPEC4 on bs2: W1 1.0584 (-5.5%), W8 composite 1.034, W32 +2.1%, hidden tau +0.2%. The head saves a fixed ~0.7 ms per B=1 round (the drafter head was 0.74 ms of an ~11 ms round). base5-spec's round is a little shorter than base4-spec's (T4 in the verify forward), so the relative gain is the same or slightly larger.

| metric (bs3, gate ratio of sums vs base5-spec) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-4 … -8%** |
| W8 composite (guard) | +1 … +5% |
| W32 tok/s (guard) | 0 … +4% |
| Hidden-set accept length | -2 … +1% relative to the control |
| Fidelity | pass; decode KL within ±0.01 of the control |

**Expected verdict: kept.**

**Falsified if** the W1 TPOT gain is below 1.02, W8 composite or W32 regresses beyond its bar, the hidden-set accept length drops by more than 4%, fidelity fails, or a launch fails at load.
