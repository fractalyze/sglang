# T-SPEC6c (gemma4nv-b2-tspec6c): the target's tied LM head as an FP8 copy at verify widths

**Status: kept (W12, 2026-10-03).** Gate `T-SPEC6c-20261003-150155-build-server-2-0cb85a` vs base4-spec-fp8head:
- W1 TPOT 2.800 → 2.720 ms (-2.85%, gain 1.0293, CI 1.0289-1.0297);
- W8 composite 1.030;
- W32 1.001;
- fidelity pass.

Supporting checks:
- forced-c1: argmax agreement with the BF16 head on 99.5% of 3,550 positions, KL 0.0004;
- W1-prompt τ 3.623 → 3.562 (-1.7%, just outside the frozen -1.5 … 0%);
- full GSM8K +0.08 pt, CI [-0.44, +0.59];
- tool-JSON 40/40;
- KV pool 52.5k → 45.1k tokens.

See `REPORT-bs2-w12.md`. Below is the frozen registration.

Registered 2026-10-03 by W12 (bs2), after T-SPEC6b's verdict and the unit tests, and before any screen or gate run of the change. The coordinator approved the model-code hook, behind its own default-off switch.

## Change

- **Code:** commit `dd5361a7bc`, behind `SGLANG_OPT_GEMMA4_FP8_LM_HEAD` (default off).
- **Hook.** After load, `Gemma4ForConditionalGeneration`, the class this checkpoint serves, keeps an FP8 E4M3 copy with per-row scales of its tied 262144 x 2816 `embed_tokens`.
  - The coordinator named `gemma4_causal.py`, but the hook lives in `gemma4_mm.py` because that is the served class.
  - Logits reach the copy through LogitsProcessor's existing lm_head `quant_method` hook. LogitsProcessor is unchanged.
- **Routing:**
  - Logits batches of at most 48 rows (B=1 and B=8 MTP verifies at k=5, plain decode, last-token prefill logits) run T-SPEC4's Triton small-M FP8 vocab-head kernel with tile (128, 128, 3).
  - Wider batches keep the BF16 table through the same matmul LogitsProcessor runs today. That covers prefill input logprobs, the gate's forced pass and the W32 verify (M=192), so their logits are bit-identical to the control's.
- **Memory:** the BF16 embedding stays for input lookups, so the copy adds 740 MB of GPU memory and the KV pool shrinks.
- **Diff against the control's tree** (`3d1732c505`):
  - this hook;
  - the env var;
  - the per-tile `max_m` plumbing, under which o_proj routes exactly as before;
  - the row-chunked quantizer.
  - T-SPEC6b's qkv entries were removed in `dd5361a7bc`.
- **Tests:** `test/registered/gemm/test_gemma4_fp8_lm_head.py`, 6 cases, all passing on the bs2 5090.
  - FP8 logits stay within E4M3's bound of the BF16 head for M ≤ 48.
  - M = 49 and 192 are bit-identical to the BF16 path.
  - Chunked quantization equals one-shot.
  - The copy shares the embedding's storage.
  - An untied head is left alone.
  - The loader's post-load pass and LogitsProcessor's hook both accept the head.
  - T3's and T-SPEC4's test files still pass.

## Evidence

- **Microbench** (`trials/spec/results/w12-verify-gemm.json`), target head:

  | M | 6 | 8 | 24 | 48 |
  |---|---:|---:|---:|---:|
  | cuBLAS BF16, µs | 880.6 | 881.8 | 909.0 | 925.2 |
  | FP8 (128, 128, 3), µs | 443.9 | 444.7 | 453.1 | 524.8 |

- **T-SPEC6b's lesson** (claim `c-gemma4nv-target-precision-cut-costs-mtp-acceptance`): a target-side precision cut is priced by the acceptance it costs.
  - The head differs from qkv: it changes no hidden state the drafter reads, and moves only near-tied target argmaxes. So the acceptance loss should be much smaller.
  - T-SPEC4's FP8 drafter head moved hidden-set τ by +0.2%.

## Control and candidate

- **Control:** `base4-spec-fp8head`, the bs2 reference, since T-SPEC6b was retired.
- **Candidate:** `base4-spec-fp8head-fp8lmhead`, which is the control's flags on `dd5361a7bc` with `SGLANG_OPT_GEMMA4_FP8_LM_HEAD=1`.
- **Gate:** the W12 gate. `--decide-on w1_tpot_gain`; the W8 composite and W32 are guards.
- **The gate's own fidelity cannot see this head**, because its passes run at concurrency 64 (verify batches above 48 rows) and its forced pass is one wide prefill.
  - The head is therefore measured with `spec_probe --mode forced-c1` on both arms. That is teacher-forced top-20 logprobs on the hidden set, one logits row per forward.
  - It reports per-prompt top-1 agreement with the reference tokens and KL(control ‖ candidate), using `fidelity.compare_forced` with the control's forced-c1 rows as the baseline.
- **Quality:** full GSM8K plus tool-JSON, paired against the control, as T-SPEC6b's rule required.
- **Acceptance:** W1-prompt τ from `spec_probe --mode gate-shape` on both arms.
- **Memory:** the KV pool of both arms is reported.

## Prediction (frozen)

**Mechanism.**
- **B=1:** each round saves about 437 µs of a ~10.3 ms round, or 4.2%. τ is assumed to move by -1.5 … 0%.
- **B=8 (M=48):** 400 µs of a ~16 ms round.
- **W32:** the verify is unchanged (BF16 at M=192), and the pool loses about 6k sliding tokens.

| metric (bs2, gate ratio of sums vs base4-spec-fp8head) | predicted |
|---|---|
| **W1 TPOT (deciding)** | **-2 … -4.5%** |
| W8 composite (guard) | +0.5 … +2.5% |
| W32 tok/s (guard) | -1 … +0.5% |
| W1-prompt τ | -1.5 … 0% |
| forced-c1 top-1 agreement with the reference tokens | within 0.5 pt of the control's |
| forced-c1 KL mean, candidate vs control rows | below 0.01 |
| gate fidelity | pass, with decode / forced KL within the A/A spread of the control's (the head is mostly unused there) |
| full GSM8K, paired | delta within [-1.0, +0.5] pt, and the CI lower bound is above -1.0 pt |
| tool-JSON | 40 / 40 |
| GPU memory | +740 MB, with the KV pool smaller by about 7.8k full / 6.1k sliding tokens |

**Expected verdict: kept.**

**Falsified if:**
- the W1 TPOT gain is below 1.02;
- the W8 composite or W32 regresses beyond its bar;
- fidelity fails;
- the GSM8K CI lower bound is below -1.0 pt;
- tool-JSON drops below 40 / 40;
- forced-c1 top-1 agreement falls more than 1 pt below the control's;
- the server fails to load, or the KV pool cannot hold W32.
