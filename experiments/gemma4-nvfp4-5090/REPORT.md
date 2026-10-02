# W1 / W1b report: verify gate + pinned baseline (gemma4nv)

**Outcome: done.** The full numbers are in `BASELINE.md`.

## Pinned baseline

The baseline is pinned and measured on build-server-3 with a 6-pair A/A gate run (`AA-20261003-020310-build-server-3-cb509e`):

- **Verdict:** integrity OK, fidelity pass, no promotion.
- **W8:** prefill 1.740 s and decode 9.442 s per rep (summed per stream). Decode is 9.29 ms per stream-token, sol_fraction 0.60.
- **W1:** TPOT 6.075 ms, sol_fraction 0.52.
- **W32:** 1020.6 tok/s (ungated).
- **Noise:** per-pair sigma 0.056% for the W8 composite, so the promotion bar is the 1% floor.
- **Quality:** GSM8K 96.0%, tool JSON 100%.
- **vLLM 0.20 reference** (ungated): within about 2% on W8 and W1, and 1.50x on W32.

## Root cause of the host crashes

The 2026-10-02 host crashes came from FlashInfer's launch-time autotune JIT-compiling its 97-unit SM120 CUTLASS MoE module at nproc + 2 = 34 parallel nvcc jobs. Measured:

- A single cicc process reaches 9.6 GB.
- 4 jobs hold 19.5 GB.
- 34 jobs would need about 166 GB, against a 60 GB host.

The fix is the harness protocol:

- host.lock, a 24 GB scope without swap, a 2 s watchdog with kill limits, and a start-only-on-a-quiet-host preflight.
- A standalone `gate prebuild` at MAX_JOBS=2.
- Every launch path is covered: gate legs, calibrate, quality, sol, peaks, and the vLLM reference.

Minimum MemAvailable across all 36 runs since then is 28.3 GB, and swap was never used.

## Gate changes found while running it

These are all committed on `jumanzii/gemma4nv-gate`.

- **Teacher-forced fidelity.**
  - The baseline is not run-to-run deterministic: greedy near-ties flip within one server and across launches.
  - So the per-prompt 10% token budget is now teacher-forced top-1 agreement.
  - KL is gated on both the forced path and the free-running decode path. Free-running match is reported only.
- **Weight hash.** SGLang has no load-time weight hash for NVFP4 MoE or Gemma4. The gate instead pins the on-disk safetensors sha256 against the HF revision, and stating that gap is part of the record.
- **Router measurement.** The expert-distribution recorder needs a Gemma4 model hook that doesn't exist. The SOL routing comes from `--enable-return-routed-experts`, with a `num_experts_per_tok` alias override on the untimed recording server only.
- **Harness bug fixes:**
  - `/flush_cache` returned 400 while the scheduler still held a request; flush now waits for an idle scheduler.
  - `startup_time` is now treated as volatile in the server-arg diff.
  - Disturbed legs are retried.
- **vLLM 0.20.0** needs transformers 5.12.1 (its resolver picks 4.57, which rejects gemma4), plus text-only mode at 4096 batched tokens.

## Vault

Committed in `$WORLD_MODEL_PATH` (my paths only, no push):

- Model, three workload and study pages (W1).
- Source root `gemma4nv-bs3` and its snapshot entry in `meta/raw-imports.yaml`.
- The import `raw/gemma4nv-bs3/20261002T2329Z-norev` (49 files).
- Baseline stack `stack-a9871012a-gemma4nv`.
- Study page update.

## Left for later

- **Trials.** W32 shows SGLang's sliding KV pool retracting at 32 x 1152 tokens; vLLM is 1.5x there.
- **Ledger.** Registering a gemma4nv adapter in `meta/ledgers.yaml` waits for the first trial (schema owner).
- **Fidelity noise from autotune.** Launch-dependent outputs, most likely from FlashInfer autotune tactic choice, mean KL of about 0.03 against the reference comes from the launch alone. Pinning autotune tactics would tighten fidelity thresholds.
