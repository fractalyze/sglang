# W32 gap diagnosis (build-server-3, 2026-10-03)

**Question.** Why does vLLM 0.20 reach 1530 tok/s on W32 (32 x 1024 prompt x 128 decode) while the SGLang baseline reaches 1020.6 tok/s?

**Answer.** The KV pool is too small, and SGLang retracts and recomputes requests.

- With a pool large enough to never retract, SGLang reaches **1519 tok/s**, within 0.7% of vLLM's unpaired 1530.
- So retraction explains essentially the whole 1.5x.
- The cause is the auto-chosen `mem_fraction_static` 0.718, which leaves 6.56 GB of the GPU unused after CUDA-graph capture. No kernel is involved.

All numbers below are unpaired diagnostic probes, not gate numbers. The decision number comes from the gated trial T-W32a.

## Evidence

### 1. Retractions happen, and they set the wall time

- **Baseline A/A legs** (`AA-20261003-020310-...-cb509e`): every server log contains 14 events of the form `KV cache pool is full. Retract requests. #retracted_reqs: 1`. Each log covers W32 warm-up plus one timed rep.
- **Pool demand:** W32 needs 32 x (1024 + 128) = 36,864 tokens in each pool. Supply:
  - Sliding pool: 29,664 tokens, short by 7,200. The prompts are exactly one window long, so out-of-window eviction frees almost nothing.
  - Full pool: 37,081 tokens, just enough.
- **Pool sizing:** the pools hold only 3.18 GB (`SWAKVPool mem usage: 3.18 GB`), with `max_total_num_tokens=37081` and `available_gpu_mem=6.56 GB` left after capture.
- **Per-token cost (FP8):**
  - Sliding: about 100 KB per token (25 layers x 8 heads x 256 x K and V).
  - Full: about 10 KB per token (5 layers x 2 heads x 512 x K and V).
  - At 3.18 GB, no `--swa-full-tokens-ratio` gives both pools 36,864 tokens. Even ratio 1.0 gives both pools only about 30k. The ratio is not the knob; the pool needs more memory.
- **Timeline of one rep** (A/A pair 0 control, versus vLLM's W32 rep):

  | engine | first stream done | wall | max TTFT |
  |---|---|---|---|
  | SGLang | 2.66 s | 3.99 s | 2.56 s |
  | vLLM | 2.59 s | 2.68 s | 1.10 s |

  - SGLang's non-retracted streams finish at vLLM's pace.
  - The retracted streams restart their prefill after others finish, which adds a tail of about 1.3 s.

### 2. Probes: how much of the gap retraction explains

The script is `trials/diag_w32.py`, at commit `a9871012a`, base ref plus one flag.

- Each config ran in its own server lifetime under host.lock, with the 24G scope and the watchdog.
- Each config ran one W32 warm-up, then 3 timed W32 reps.
- Retractions are counted from the log of the timed window only.
- Run directory: `runs/gemma4nv-b3-diag-w32-20261003-084104`.

| config | full / sliding pool (tokens) | GPU free after capture | retracted reqs (3 reps) | wall per rep | tok/s | vs base |
|---|---|---|---|---|---|---|
| base | 37,081 / 29,664 | 6.56 GB | 18 | 3.97 to 3.99 s | 1030 | 1.00 |
| `--max-running-requests 16` | 37,081 / 29,664 | 8.63 GB | 0 | 3.75 to 3.77 s | 1090 | 1.06 |
| `--mem-fraction-static 0.80` | 66,235 / 52,988 | 4.03 GB | 0 | 2.69 to 2.71 s | **1519** | **1.48** |

- **Fewer in flight:** removes retraction but serves the batch in two waves. Most of the loss comes back as queueing, so the gain is only +6%.
- **Larger pool:** removes retraction and keeps all 32 streams in one decode batch. That reaches vLLM's number.
- **Host side, all three:**
  - Min MemAvailable 49.4 GB.
  - Peak tree RSS 12.6 GB, during weight load.
  - Peak load 1.3.
  - No JIT compilation: the cache was warm.

### 3. What vLLM does differently

Read from `runs/vllm-ref-20261003-042242-...-e964bc/vllm.log`.

| item | vLLM 0.20 reference | SGLang baseline |
|---|---|---|
| KV dtype | `fp8_e4m3` | FP8 E4M3 (same) |
| attention | `TRITON_ATTN` | Triton (same) |
| max running | `--max-num-seqs 32` | 2048 (not the limit here) |
| chunked prefill | 4096 (`max_num_batched_tokens`) | 4096 (same) |
| memory for KV | `--gpu-memory-utilization 0.85`: **7.81 GiB KV**, 68,256 tokens in the hybrid allocator | `mem_fraction_static` 0.718 (auto): **3.18 GB KV** |
| CUDA graphs | decode sizes up to 64 | decode up to 32 |

The two engines run the same kernel classes and the same scheduling limits. vLLM simply gives its KV cache 2.5x the memory.

## Choice for T-W32a

The trial adds `--mem-fraction-static 0.80` and keeps the default swa ratio 0.8. It is the smallest change that removes retraction at W32:

- Pools: 66k full and 53k sliding tokens, which is 1.4x the W32 demand.
- GPU free after capture: 4.0 GB.

Expected effects on the other workloads:

- **W8 and W1:** they never fill the pool, so they should not move. They stay as guard metrics.
- **Fidelity:** the long hidden prompts (8k to 15.5k tokens) prefill in 4096-token chunks with the same kernels. Fidelity should sit at the A/A level.

Out of scope here:

- vLLM's `0.85` (about 0.87 in SGLang's accounting) would leave less headroom on a GPU shared with other users. A larger `mem_fraction_static` is a separate variant if W32 ever needs more than 32 x 1152 tokens.
- No code change in the SWA pool or the scheduler is needed for this gap.
