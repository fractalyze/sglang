# T-SPEC1 (gemma4nv-b2-tspec1): MTP speculative decoding, 5 drafts, at every batch size

Registered 2026-10-03 by W8 (bs2) before any gate run. Config only: no `python/sglang` change.

## Control and candidate

- **Control:** gate ref `base3` (W5's pin: base2 + split-KV 16 + Triton small-M BF16 GEMM, commit `1fd77e64b0`).
  T3b (FP8 o_proj) is not in it, because it is pending a larger quality run.
- **Candidate:** `base3-mtp5`, which is base3 plus the flags below.
  - `--speculative-algorithm NEXTN --speculative-draft-model-path <google/gemma-4-26B-A4B-it-assistant>`: the tree promotes NEXTN to `FROZEN_KV_MTP` for this drafter.
  - `--speculative-num-steps 5 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1`.
  - Three restated flags: `--max-running-requests 48`, `--speculative-draft-model-revision main` and `--cuda-graph-max-bs-decode 32`.
    They are what the speculative hook resolves anyway. They are flags only so that the gate's `server_info` arg diff sees them declared.
- The drafter is BF16 and 0.83 GB on disk: 4 layers, hidden 1024, a 2048-centroid sparse lm_head, and reads of the target's KV (no own KV pool).
- At `--mem-fraction-static 0.76` the pools shrink from 52.0k full / 41.5k sliding tokens to 40.4k / 32.3k.

Decision: `--decide-on w1_tpot_gain`. The gate then guards the W8 composite and W32 throughput, each at its own bar.

## Evidence

Every number below is a screen: unpaired, one server lifetime, and taken through the gate's own capped server path. The tool is `trials/spec/spec_probe.py`, and the runs are on bs2 under `/data/jooman/gemma4nv/runs/w8-*`.

### Feasibility (step 0)

- The tree serves the target with this drafter on SM120.
  - Verify CUDA graphs are captured at every decode batch size up to 32 (6 tokens per request), and the drafter's decode loop is graphed too.
  - Every timed decode step reports `cuda graph: True`.
- No JIT compile ran: the FlashInfer cache held.
- Peak server tree RSS is 19.5-19.8 GB during graph capture and serving, against 6.6-14.8 GB without the drafter. That is under the 24G scope, but the margin is thin.
- DFlash (`z-lab/gemma-4-26B-A4B-it-DFlash`, `--speculative-algorithm DFLASH`, 16 draft tokens) also serves config-only. The tree has had Gemma-4 DFlash support since `5ea0d1d093`. Its peak RSS is 8.2 GB.

### Accept length τ (tokens per verify round, bonus included)

Measured on the 22-prompt hidden fidelity set (contents never read or printed) and on the gate's timing corpus. Both use greedy decoding with 256 new tokens per prompt.

| config | hidden τ | timing corpus τ (B=1) |
|---|---:|---:|
| MTP k=1 | 1.84 | 1.87 |
| MTP k=2 | 2.45 | 2.52 |
| MTP k=3 | 3.03 | 3.05 |
| MTP k=4 | 3.33 | 3.62 |
| **MTP k=5** | **3.51** (3.68 on base3) | **4.43** (4.20 on base3) |
| MTP k=7 | 4.03 | 5.00 |
| DFlash 16 | 3.04 | 4.21 |

- **The hidden set does not collapse.** Yukon's MoE tracks saw public 0.92-1.0 against hidden 0.43-0.63. Here the hidden τ is 80-100% of the timing corpus τ.
- **Per hidden category at k=3:** multilingual 2.36, chat 2.49, Korean 2.80, code 3.15, tool-JSON 3.73 and math 3.75. The long-context prompts (8-16k) score 2.8-3.6.

### Verify-cost curve (step 1)

Method:
- **Decode steps.** The step time at exactly M running rows comes from the target's per-step decode log (`w8-sweep-*`, base2 + T3, CUDA graphs up to 48).
- **Routing.** The raw routing per token comes from `--enable-return-routed-experts` (`w8-experts-*`).
- **Analysis.** `trials/spec/verify_cost.py` counts the distinct experts per layer over 1+k consecutive tokens of B streams, which is what one verify routes. It fits `step = a + b*rows + c*experts` to the sweep: a = 5.86 ms, b = 0.099 ms/row, c = 0.046 ms/expert, max error 7.8%.

| B | k | rows | distinct experts/layer (timing / hidden) | same rows as independent decode | verify cost / 1-step (model) | decode at same rows / 1-step (measured) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 8.0 / 8.0 | 8.0 | 1.00 | 1.00 |
| 1 | 1 | 2 | 13.2 / 12.5 | 14.1 | 1.05 | 1.15 |
| 1 | 3 | 4 | 19.4 / 18.5 | 23.1 | 1.13 | 1.26 |
| 1 | 5 | 6 | 23.8 / 23.4 | 29.8 | 1.19 | 1.34 |
| 1 | 7 | 8 | 27.2 / 26.9 | 35.1 | 1.25 | 1.39 |
| 8 | 0 | 8 | 35.1 / 45.5 | 34.8 | 1.00 | 1.00 |
| 8 | 1 | 16 | 47.2 / 61.6 | 48.7 | 1.16 | 1.18 |
| 8 | 3 | 32 | 59.6 / 77.6 | 62.7 | 1.42 | 1.44 |
| 8 | 5 | 48 | 66.5 / 86.5 | 70.9 | 1.65 | - (40 rows: 1.61) |
| 8 | 7 | 64 | 71.7 / 92.0 | 76.5 | 1.87 | - |

Findings:
- **Consecutive tokens of one stream share experts.** A 6-row verify touches 24 experts per layer, against 30 for 6 independent rows. The expert term is therefore not the barrier at B=1.
- **At B=8 the hidden mix spreads wider** (86 against 66 experts at k=5), because its 8 streams are different tasks.
- **The real verify costs more than this curve.**
  - A B=1 torch-profiler trace of k=3 (`w8-prof-k3-*`, 12 rounds) puts the verify forward at 11.5 ms per round, against 7.4 ms for a 4-row decode step.
  - The gap is attention. TARGET_VERIFY on the Triton backend runs the unified extend kernel `_fwd_kernel` with grid [1, 16, 1]: 16 CTAs on 170 SMs, with no split over the 1.1k-token KV.
  - That costs 4.7 ms per round, about 157 µs per layer.
  - Decode uses split-KV stage 1/2 at about 50 µs total per step.
- **Draft loop:** 2.2 ms per round for 3 steps. 1.55 ms of that is a cuBLAS gemv.
- **The GPU is 93% busy over the round**, so the round is not launch-bound.

### Timing screens (TPOT from the gate's client: (e2e - ttft) / (tokens - 1))

| ref | B=1 TPOT | B=8 TPOT | B=8 TTFT median | B=32 wall tok/s |
|---|---:|---:|---:|---:|
| base2 + T3, no spec | 5.87 ms | 8.73 ms | | |
| + MTP k=1 / 2 / 3 / 4 / 5 / 7 | 7.29 / 5.06 / 4.34 / 4.13 / 4.03 / 3.95 | 8.65 / 7.04 / 6.70 / 6.27 / 6.21 / 6.80 | | |
| + DFlash 16 | 4.41 | 6.17 | | |
| **base3** | **5.75** | **8.74** | 0.201 s | **1545** |
| **base3-mtp5** | **4.02 (-30%)** | **5.82 (-33%)** | 0.295 s (+47%) | **1205 (-22%)** |

- **k=5** is the best B=8 point and within 2% of the best B=1 point (k=7).
- **At B=32 the candidate retracts.** The server logged 11 retractions, against 1 for base3, because the drafter's 0.4 GB weights and graphs shrink the sliding pool below W32's need. That repeats the mechanism of T-W32b.
- **B=8 TTFT rises.** The cause is unprofiled. One candidate: mixed chunked prefill is disabled under speculation, and later admission waves wait behind 18 ms verify rounds instead of 8.7 ms decode steps.

## Prediction (frozen)

| metric (bs2, gate ratio of sums, 4 pairs) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-22 … -34%** (gain 1.28 … 1.52) |
| W8 decode gain | +30 … +50% |
| W8 prefill gain | 0.60 … 0.85 (TTFT rises) |
| W8 composite (guard) | +10 … +30% |
| W32 tok/s (guard) | **-12 … -30%** (KV retraction), so it is beyond its bar |
| Fidelity | approx tier. The verify path computes the same logits with extend-shaped attention, so free-running KL stays near the control's and well inside the thresholds. Greedy outputs differ from the control's only where bf16 rounding flips an argmax. |

**Expected verdict: not promoted.** The W32 guard is predicted to regress beyond its bar, while W1 clears its bar by a wide margin.

That outcome leads to T-SPEC1b: speculation only while the running batch is small. It needs code, because adaptive speculation with per-batch-size tiers is EAGLE-only in this tree and FROZEN_KV_MTP has no batch-size switch. It goes to the coordinator first.

**Falsified** if any of these holds:
- the W1 TPOT gain is below 1.15;
- W32 does not regress beyond its bar;
- the W8 composite falls below 1.0;
- fidelity fails.
