# T-SPEC4 (gemma4nv-b2-tspec4): the MTP assistant's vocab head in FP8

Registered 2026-10-03 by W10 (bs2) before any screen or gate run of the change. Code change. The coordinator approved it, behind a default-off switch.

## What the draft loop costs (profile)

**Source.** W8's B=1, k=5 torch-profiler trace of base3-mtp5-svk (`w8-prof-svk-113351`, 12 rounds), re-read for the draft loop:

| part of `step[DRAFT_LOOP]`, per round (5 draft steps) | ms |
|---|---:|
| **total kernel time** (window 3.65 ms, 98% busy) | **3.58** |
| **lm_head gemv, grid 32768 = 262144 rows / 8: one per step, 335 µs each** | **1.67** |
| full-vocab softmax (`cunn_SoftMaxForward`, 50 µs per step) + max reduce | 0.28 |
| 4 drafter layers: BF16 gemvs (gate_up 4096-grid, qkv, o, down) | 0.91 |
| down_proj on the cuBLAS WMMA fallback with split-K + reduce | 0.25 |
| attention, norms, RoPE, glue | 0.47 |

- **The head dominates.** The 26B assistant ties its head to its own `embed_tokens`: 262144 × 1024 BF16, 512 MB. Every draft step reads all of it, at about 87% of DRAM bandwidth.
- **The checkpoint has no centroid head.** `use_ordered_embeddings` is false, and `model.safetensors` holds no `centroids` or `token_ordering`. The tree's sparse-centroid path (`_centroid_logits_processor`) is for the E2B/E4B assistants and cannot be used here.

## Change

- Commit `262f327c28`: `4ebe6175af` plus a shared-memory fix. The python tree is `c99575c4f52f`'s plus this change only. It sits behind the switch `SGLANG_OPT_MTP_FP8_LM_HEAD`, default off.
- **At load,** the assistant's tied head is quantized to FP8 E4M3 with a per-row absmax scale, using the same quantizer as T3b. It is quantized in 16k-row chunks, and the BF16 tensor is released, which frees 256 MB of GPU memory.
- **Draft logits** go through `LogitsProcessor`'s quant hook into the T3b Triton small-M kernel's FP8 weight path. The tile is (128, 128, 3), and batches above 32 rows are split into 32-row launches.
- **Only draft tokens can change.** The target, its verify and its sampling are untouched. With greedy verify, outputs are preserved up to the numerics already present in base4-spec, and only acceptance can move.
- **Tests:** `test/registered/gemm/test_mtp_fp8_vocab_head.py`.
  - FP8 logits stay within E4M3's 2^-4 relative bound of the BF16 head, at M ∈ {1, 8, 32, 48, 100}.
  - A separated top-3 is preserved exactly.
  - Chunked quantization equals one-shot quantization.
  - `LogitsProcessor` takes the hook.
- **Microbench** (`trials/spec/fp8_head_bench.py`, bs2 5090), head GEMM time:

  | M | BF16 cuBLAS | FP8 (128,128,3) |
  |---:|---:|---:|
  | 1 | 355 µs | 186 µs |
  | 8 | 353 µs | 192 µs |
  | 32 | 364 µs | 198 µs |

## Control and candidate

- **Control:** `base4-spec` (T-SPEC3's candidate).
- **Candidate:** `base4-spec-fp8head`, which is `base4-spec` on `262f327c28` with `SGLANG_OPT_MTP_FP8_LM_HEAD=1`.
- **Decision:** `--decide-on w1_tpot_gain`. The W8 composite and W32 are guards.
- **Fidelity and decode KL run as usual.** The target path is unchanged, so decode KL must match base4-spec's 0.020 within the A/A spread. It cannot be bitwise identical, because different drafts change which positions are verified together. A large move means a bug.
- **Acceptance length on the hidden set** (22 prompts, mean and per prompt) is reported for control and candidate next to the timing.

## Prediction (frozen)

**Mechanism.** Each draft step saves about 165 µs. That is 0.83 ms per k=5 round, against a B=1 round of about 11.5 ms on base4-spec (TPOT 2.96 ms at τ ≈ 3.9). τ is assumed to fall by at most 2% from FP8 rounding of near-tied draft logits.

| metric (bs2, gate ratio of sums vs base4-spec) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-3 … -8%** |
| W8 decode gain | +2 … +6% (0.8 ms of a ~15 ms B=8 round) |
| W8 composite (guard) | +1 … +4.5% |
| W32 tok/s (guard) | 0 … +4% |
| Hidden-set accept length | -2 … 0% relative to the control |
| Fidelity | pass. Decode KL mean is within ±0.01 of the control's. |

**Expected verdict: kept against base4-spec.**

**Falsified if:**
- the W1 TPOT gain is below 1.02;
- W8 composite or W32 regresses beyond its bar;
- the hidden-set accept length drops by more than 4%;
- fidelity fails;
- the server fails to load or OOMs.
