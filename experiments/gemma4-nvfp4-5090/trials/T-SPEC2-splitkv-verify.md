# T-SPEC2 (gemma4nv-b2-tspec2): split-KV Triton verify attention for MTP on CUDA

**Status: kept (W8, 2026-10-03).** Gate `T-SPEC2-20261003-113757-build-server-2-ae2f98` vs `base3-mtp5`:
- W1 TPOT 4.414 → 3.002 ms (-32.0%);
- W8 composite 1.197;
- W32 +23.4%;
- fidelity pass, integrity ok.

Every metric landed inside its frozen interval. See `REPORT-bs2-w8.md`. Below is the frozen registration.

Registered 2026-10-03 by W8 (bs2) before any gate run. The code change was approved by the coordinator; the switch defaults off.

## Control and candidate

- **Control:** `base3-mtp5`. This is T-SPEC1: base3 plus MTP k=5, commit `1fd77e64b0`. It is parked, because W32 fails against base3.
- **Candidate:** `base3-mtp5-svk`, which is the same flags on commit `c99575c4f52f` with `SGLANG_OPT_USE_TRITON_SPLITKV_VERIFY_CUDA=1`.
- Decision: `--decide-on w1_tpot_gain`. The gate guards the W8 composite and W32.

## Change

TARGET_VERIFY on the Triton backend ran `extend_attention_fwd`. That kernel launches one program per (sequence, head) and walks the 1.1k-token prefix KV serially. At B=1 that is 16 programs on 170 SMs, about 157 µs per layer and 4.7 ms of a 14.8 ms k=3 round.

The tree already had a split-KV verify kernel for gfx95 (`verify_splitkv.py`). With the new switch it also runs on CUDA. On CUDA only:
- **Sliding-window layers.** It serves them with `extend_attention_fwd`'s exact window mask on the same KV slice, on both the prefix rows and the draft-draft block. Gemma-4 has 25 of them.
- **Split count.** It picks the count for about two prefix programs per SM, at most 16.
- **Stage 2.** It is tiled over D and Dv in 64-wide chunks with 4 warps, so the head_dim-512 full layers do not spill.

The ROCm path is unchanged: there the new arguments are off, and the tiles default to full width, which reproduces the old arithmetic.

**Tests.** `test/registered/attention/test_verify_splitkv.py` passes 14/14 on the bs2 5090 (`/home/jooman/gemma4nv/w8/ut2/run.log`). The new cases cover:
- parity with `extend_attention_fwd` on Gemma-4 shapes: 16 query heads, with hd 256 / 8 KV heads and hd 512 / 2 KV heads, at bs 1 and 8;
- windows 1024, 256 and 3 (window 3 hides the whole prefix from rows 3+);
- an FP8 KV cache with scales;
- the split policy;
- the sliding-window opt-in gate.

## Evidence

**Kernel microbench** (`trials/spec/verify_attn_bench.py`, bs2 5090, FP8 KV, ctx 1100). Per verify forward (25 sliding + 5 full layers), extend against split-KV:

| bs | rows | extend | split-KV | max abs err |
|---:|---:|---:|---:|---:|
| 1 | 4 | 4.87 ms | 0.58 ms | ≤ 0.016 |
| 1 | 6 | 4.86 ms | 0.66 ms | ≤ 0.014 |
| 8 | 6 | 5.18 ms | 1.83 ms | ≤ 0.016 |
| 32 | 6 | 17.24 ms | 5.73 ms | ≤ 0.016 |

**In-server screen.** Unpaired: `w8-scr-svk-113351` against `w8-scr-base3mtp5-105310`, with base3 alongside.

| | base3 | base3-mtp5 | base3-mtp5-svk |
|---|---:|---:|---:|
| B=1 TPOT | 5.75 ms | 4.02 ms | **2.65 ms** |
| B=8 TPOT | 8.74 ms | 5.82 ms | **4.60 ms** |
| B=8 TTFT median | 0.201 s | 0.295 s | 0.294 s |
| B=32 wall tok/s | 1545 | 1205 | **1470** |
| hidden τ | - | 3.68 | 3.49 |

- No decode step ran outside a CUDA graph.
- **B=1 profile at k=5** (`w8-prof-svk-*`): a round is 11.8 ms, of which the verify is 7.1 ms and the draft loop 3.6 ms.
  - The verify (6 rows) is now cheaper than a 6-row decode step (7.9 ms).
  - The draft loop is the next cost.

## Prediction (frozen)

| metric (bs2, gate ratio of sums vs base3-mtp5) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-28 … -40%** |
| W8 decode gain | +18 … +32% |
| W8 prefill gain | 0.98 … 1.02 |
| W8 composite (guard) | +12 … +24% |
| W32 tok/s (guard) | +12 … +30% |
| Fidelity | reorder tier. The split-KV LSE merge reorders the attention reduction. Free-running and teacher-forced KL stay within the gate thresholds, near T-SPEC1's. |

- **Expected verdict: kept against base3-mtp5.**
- **The stack is still short of promotion against base3.** Its W32 is projected about 5% below base3, because the drafter still shrinks the KV pool. T-SPEC1b addresses that.

**Falsified** if any of these holds:
- the W1 TPOT gain is below 1.20;
- W8 composite or W32 regresses beyond its bar;
- fidelity fails.
