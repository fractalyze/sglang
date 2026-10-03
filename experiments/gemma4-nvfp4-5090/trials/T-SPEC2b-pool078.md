# T-SPEC2b (gemma4nv-b2-tspec2b): the MTP + split-KV-verify stack with its KV pool restored, against base3

Registered 2026-10-03 by W8 (bs2) before any gate run. Config only.

## Why this replaces T-SPEC1b

T-SPEC1b was meant to run speculation only while the running batch is small. T-SPEC2 removed its premise:
- With split-KV verify attention, speculation wins at B=32 too: TPOT is 10.26 ms with speculation, against 15.97 ms for base3's plain decode.
- What remains of the W32 gap is the KV pool. The drafter's weights now count inside `--mem-fraction-static`, so at 0.76 the sliding pool is 32.3k tokens, below W32's 36.9k.
- A batch-size switch would turn off a winning path at B=32, and the pool would stay small.

The coordinator allowed a pool variant for this case. The T-SPEC1b design is mapped but not built: adaptive speculation's per-batch-size slots, enabled for FROZEN_KV_MTP, about 40 LOC and no scheduler change.

## Control and candidate

- **Control:** `base3`.
- **Candidate:** `base3-mtp5-svk-m078`, which is T-SPEC2's `base3-mtp5-svk` plus `--mem-fraction-static 0.78`.
- This gates the whole stack (T-SPEC1 + T-SPEC2 + pool) against the current base.
- Decision: `--decide-on w1_tpot_gain`. The gate guards the W8 composite and W32.

## Evidence

Screens, unpaired, base3-mtp5-svk at each fraction:

| fraction | full / sliding pool (tokens) | free after graph capture | retractions in the screen | B=32 wall tok/s | B=1 / B=8 TPOT |
|---|---:|---:|---:|---:|---:|
| 0.76 | 40.4k / 32.3k | 6.34 GB | 9 | 1470 | 2.65 / 4.60 ms |
| **0.78** | **48.6k / 38.9k** | **5.67 GB** | 1 | **1718** | 2.65 / 4.60 ms |
| 0.79 | 50.3k / 40.2k | - | 1 | 1716 | 2.65 / 4.61 ms |
| base3 (0.76, no drafter) | 52.0k / 41.5k | 5.22 GB | 1 | 1545 | 5.75 / 8.74 ms |

- **0.78 is the smallest probed fraction whose sliding pool holds W32.**
- **It leaves more memory free after graph capture than base3 does** (5.67 against 5.22 GB). Base3's 5.22 GB passes the teacher-forced logprob pass, the pass that OOMed at 0.80 (T-W32a).
- **A co-tenant contaminated the first 0.78/0.79 screens.** It held 6.2 GB of GPU memory. Those runs are kept in `runs/contaminated/` and are not used.

## Prediction (frozen)

Composed from the T-SPEC1 and T-SPEC2 gate ratios, with W32 from the screen:

| metric (bs2, gate ratio of sums vs base3) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-42 … -55%** (T-SPEC1 1.458 x T-SPEC2 1.470 = 2.14x) |
| W8 decode gain | +70 … +100% |
| W8 prefill gain | 0.70 … 0.82 (TTFT rises under speculation) |
| W8 composite (guard) | +38 … +60% |
| W32 tok/s (guard) | **+4 … +16%** |
| Fidelity | approx/reorder. The fraction does not change numerics, so KL stays near T-SPEC2's. No OOM in the logprob pass. |

**Expected verdict: promoted against base3.** It would be the first stack to clear W1, W8 and W32 together with speculation on.

**Falsified** if any of these holds:
- the W1 TPOT gain is below 1.70;
- W8 composite or W32 regresses beyond its bar;
- fidelity fails;
- the server OOMs.
