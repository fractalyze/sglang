# T-SPEC7 (gemma4nv-b2-tspec7): MTP draft depth k=6 on the bs2 reference

**Status: retired (W12, 2026-10-03).** Gate `T-SPEC7-20261003-153035-build-server-2-ca1bf6` vs base4-spec-fp8head-fp8lmhead:
- W1 TPOT 2.7205 → 2.7284 ms (+0.29%, gain 0.9971);
- W8 composite 0.979;
- W32 0.950;
- fidelity pass.

With the FP8 target head, k=6's W1-prompt τ is only 3.814, against 3.977 on the BF16-head ref the screens used. The `max_m` 64 enabler was reverted in `17ccadd47c`. See `REPORT-bs2-w12.md`. Below is the frozen registration.

Registered 2026-10-03 by W12 (bs2), after the k re-sweep screens and before any gate run.

## Change

- **Config:** `--speculative-num-steps 6 --speculative-num-draft-tokens 7`, replacing 5 / 6.
- **Enabler:** commit `43039cd0f2` raises the FP8 target head's `max_m` from 48 to 64.
  - At k=6 a B=8 verify has 56 logits rows, which would otherwise fall back to the BF16 head and silently undo T-SPEC6c at W8.
  - The tile already runs BLOCK_M=64 for M in 33..64, so there is no new shape.
  - For k ≤ 5 at B ≤ 8 it is a no-op.
  - Head tests pass on the bs2 5090 at M ∈ {1, 6, 8, 32, 48, 56, 64}, and M = 65 is bit-identical to BF16.

## Evidence: the k re-sweep (screens, unpaired, one server lifetime each)

### Hidden set

`spec_probe --mode spec`, `runs/w12-k/k*`, on `base4-spec-fp8head`. 22 prompts, 256 tokens, greedy.

| k | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|
| τ, all prompts | 3.049 | 3.354 | 3.655 | 3.767 | 3.837 |
| correct-drafts histogram (0..k) | 327/277/213/1030 | 354/250/205/174/696 | 314/240/228/142/100/517 | 308/314/186/127/93/70/397 | 321/286/186/168/93/89/57/268 |
| B=8 τ (timing corpus) | 2.60 | 2.84 | 3.41 | 3.26 | 3.08 |

The spec mode's own B=1 TPOT uses 3 prompts. Its per-prompt spread (2.2-4.0 ms at k=5) swamps the k effect, so it was not used.

### Gate's own prompts

`spec_probe --mode gate-shape`: the gate's timed workloads once, after its warm-up (`runs/w12-gs-base4-spec-fp8head`, `runs/w12-kgs/k*`), on `base4-spec-fp8head`.

| k | W1 TPOT ms | W1-prompt τ | W8 decode sum s | W8 prefill sum s | W32 tok/s |
|---|---:|---:|---:|---:|---:|
| 4 | 2.822 | 3.479 | 19.93 | 9.07 | 1669 |
| 5 | 2.798 | 3.623 | 19.30 | 9.07 | 1652 |
| **6** | **2.723** | **3.977** | 19.26 | 9.12 | 1719 |
| 7 | 3.112 | 3.730 | 18.13 | 9.10 | 1656 |

- **k=6 is the only depth that beats k=5 on W1 (-2.7%).** Its W1-prompt τ rises 9.8% and the round costs one more draft step (about 0.58 ms).
- **k=7 loses:** its W1-prompt τ is lower than k=6's on these prompts.
- **These screens ran on `base4-spec-fp8head`.** T-SPEC6c's cheaper verify head leaves the draft cost unchanged and trims the fixed round cost, which favours a deeper k slightly.

## Control and candidate

- **Control:** `base4-spec-fp8head-fp8lmhead`, the T-SPEC6c candidate (kept), at k=5.
- **Candidate:** `base4-spec-fp8lmhead-k6`, which is the control on `43039cd0f2` at k=6.
- **Gate:** the W12 gate, `--decide-on w1_tpot_gain`. The W8 composite and W32 are guards.
- **Reported alongside:**
  - W1-prompt τ (`gate-shape`) and hidden-set τ (`spec`) for both arms;
  - the KV pool of both arms;
  - quality: full GSM8K + tool-JSON, paired. Greedy verify preserves outputs up to numerics, so no change is expected.

## Prediction (frozen)

| metric (bs2, gate ratio of sums vs base4-spec-fp8head-fp8lmhead) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-1 … -4%** |
| W8 composite (guard) | -1.5 … +1.5% |
| W32 tok/s (guard) | -3 … +3% |
| W1-prompt τ | +6 … +12% |
| Fidelity | pass, with decode KL within the control's A/A spread |
| Full GSM8K, paired | CI lower bound above -1.0 pt; tool-JSON 40 / 40 |

**Expected verdict: kept.**

**Falsified if:**
- the W1 TPOT gain is below 1.01;
- the W8 composite or W32 regresses beyond its bar;
- fidelity or quality fails;
- the server fails to load or the KV pool cannot hold W32.
