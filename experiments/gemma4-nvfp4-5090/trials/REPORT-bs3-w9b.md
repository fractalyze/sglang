# W9b report: T4 fused decode glue implemented, gated and adopted; base5 pinned (build-server-3)

**Outcome: done.** T4 was kept against base4 at W8 composite **1.0626**. Every gated metric came
in above its predicted interval. Fidelity and full-GSM8K quality pass, and base5 is pinned with
a 6-pair A/A. Full numbers are in `BASELINE.md` (base5 section) and `trials/T4-base4-glue-fusion.md`.

## 1. Before any fusion: why the tree disables its RoPE + KV-write fusion, and the byte-level test

**Why it is disabled.**
- `can_fuse = False  # DISABLED: causes accuracy regression in launch_server path` in
  `Gemma4Attention.forward` arrived already disabled in upstream PR #23280 (`2c8357f794`, XPU
  bring-up of Gemma 4). Nothing else is recorded: no discussion, no test, no issue.
- The CUDA helper it would call (`models/utils.py`) supports only a bf16, non-`SWAKVPool` pool
  with no KV scales.
- It writes at raw `out_cache_loc`, which is the wrong slot for sliding layers of a hybrid pool.
  That is the likely, but inferred, regression.
- This checkpoint has FP8 KV, a hybrid pool and scales, so T4 does not reuse that path.

**Byte-level KV test** (`test_gemma4_fused_qkv_rope_kv.py`, 31 tests):
- The fused kernel is checked against the real unfused ops: Triton qkv norm, then JIT
  `rope.cuh`, then `MHATokenToKVPool.set_kv_buffer` on an FP8 pool.
- Coverage:
  - sliding (16/8 heads, hd 256) and full (16/2, hd 512, proportional RoPE, separate K and V
    copies) layers;
  - M from 1 to 300;
  - scales None, 1.0 and 0.37;
  - scattered slots for the SWA ring.
- **Result: bit-exact for q/k/v and every K/V byte.**
- Getting there took three fixes, each found by the test:
  - a weight-offset bug in my first kernel;
  - torch's `bf16.div_(fp32 0-dim scale)` casts the scale to bf16 first;
  - nvcc contracts rope.cuh to `fma(x,cos,-(y*sin))` / `fma(y,cos,x*sin)`. I matched it by
    simulating each candidate form.
- A profile test pins the four ATen elementwise launches per layer to `set_kv_buffer`'s `div_`
  + `.to(fp8)`.
- **Documented difference:** at scale != 1 the unfused store also divides the k/v activations in
  place, and the fused kernel does not. This checkpoint's scales are 1.0.

## 2. Implementation

- **Code:** `jumanzii/gemma4nv-b3-t4` @ `1d859709ef` (fractalyze), on base4's `36aa977541`.
  - Edits: `gemma4_fused_ops.py`, `gemma4_causal.py`, `environ.py`, and 2 tests. No frozen
    file.
  - pre-commit is clean.
- **Switch:** `SGLANG_OPT_GEMMA4_FUSED_GLUE`, an `EnvInt` over the `Gemma4FusedGlue` IntEnum.
  The default is 0 (off).
  - Level 1 is A: q/k/v norm, RoPE and the FP8 KV store in one kernel. It writes where
    `TritonAttnBackend` writes; a sliding layer uses the backend's `swa_out_cache_loc`.
    Any other pool or backend falls back, with a one-time warning.
  - Level 2 adds B, C and D:
    - B: post-attention RMSNorm + FusedAddRMSNorm in one kernel.
    - C: router norm + pre-FF-2 norm, two outputs from one read.
    - D: the next layer's input norm, or the final norm, as a second output of the dual-norm
      epilogue.
- **Tests:** 90 pass on bs3 (new + existing gemma4 fused-op tests).
  - The B/C/D tests bound the difference from FlashInfer's norms at 1 bf16 ulp. For B's
    residual the bound is 1 ulp of the larger addend.
  - The existing dual-norm output is asserted unchanged.
- **Sub-steps:** refs `base4-glue1` (A) and `base4-glue2` (A-D). Level 2 gated cleanly, so A
  alone was not gated separately.

## 3. Gate: T4 (base4-glue2) vs base4

`gemma4nv-b3-t4-20261003-121505-build-server-3-421028`: 6 pairs, `--decide-on w8_composite`,
harness 44ea04d17.

| metric | gain | pair range | predicted (frozen) | |
|---|---:|---|---|---|
| **W8 composite (deciding)** | **1.0626** | 1.0596 - 1.0646 | [+1.5, +4.0]% | above the interval |
| W8 decode | 1.0737 | 1.070 - 1.076 | -3.2 ... -4.0% step | above |
| W8 prefill | 1.0300 | 1.021 - 1.037 | 0 ... +1% | above |
| W1 TPOT (guard) | 1.0818 (5.594 -> 5.171 ms) | 1.081 - 1.083 | [-2.5, -5.5]% | above |
| W32 (guard) | 1.0389 (1554 -> 1614 tok/s) | 1.024 - 1.054 | +1 ... +2.5% | above |

**Verdict: promote.**
- **Fidelity:** decode KL 0.0207 (base4 0.0253), p99 0.48 (limits 0.050 / 0.95). Teacher-forced
  KL 0.031 (base4 0.034). Pass.
- **Timed-output agreement:** 0.12, reported only, because the norm sums are reordered.
- **Why the prediction was low:** each removed launch saved about its whole traced time, 2.2 us
  at B=8 and 1.6 us at B=1. I had discounted traced time by 25-40% for profiler inflation.
  - New vault claim: `c-gemma4nv-fused-glue-launch-saves-its-traced-time`.
  - Prefill gained because the 4 elementwise KV-quantize passes and 2 norm launches per layer
    also leave the 8192-row prefill.

## 4. Quality (full GSM8K 1,319 + tool-JSON 40)

| | GSM8K | delta | 95% CI (paired) | lost / gained | McNemar p | tool-JSON |
|---|---|---:|---|---|---:|---|
| base4 -> base5 | 96.29 -> 96.36% | +0.08 pt | [-0.55, +0.71] | 8 / 9 | 1.00 | 100 -> 100 |

- Passes the rule: CI low >= -1.0 pt, tool-JSON no drop.
- The discordance equals base3's A/A (8 / 9), which is noise level.
- **Control arm:** the base3+T3b run, which is base4's commit, flags and env, so no extra
  control server was launched.

## 5. base5 pinned

- **A/A:** `AA-base5-20261003-123340-build-server-3-e1f9b9`, 6 pairs, verdict no promotion,
  agreement 1.0.
  - W8 decode step **7.97 ms** (sol_fraction **0.70**; base4 0.65, base 0.60).
  - W1 TPOT **5.167 ms** (0.62).
  - W32 **1617 tok/s**.
- **Noise:** set from this A/A. base4's file is kept as `noise.base4.json`.
  - W8 prefill sigma is 0.61%, so its bar is now 1.84%. Every other bar stays at 1%.
- **Against the original base on bs3** (separate A/As, not a gated delta): W8 decode 9.29 ->
  7.97 ms, W1 6.075 -> 5.167 ms, W32 1021 -> 1617 tok/s.

## Host safety

Every engine process ran through the gate's `host_lock` + `systemd-run` scope (MemoryMax 24G) +
watchdog: 1 prebuild, 24 gate legs, 12 A/A legs and 1 quality server. Unit tests ran under
`host.lock` + `gpu.lock` + the same scope.

- Min MemAvailable was 49.3 GB.
- Peak tree RSS was 11.0 GB, at weight load.
- Peak load1 was 6.4.
- Swap stayed at 0.14 GB.
- No watchdog trips, and no nvcc ran.

## Vault (my paths only, not pushed)

| commit | what |
|---|---|
| `eb06f28` | ledger map `base4-glue2` -> `gemma4nv-b3-t4` |
| `b22d993` | T4 recorded kept, with raw import |
| `3408fbe` + `0b2a96b` | cost-rule claim, linked |
| `6d34844` | prose |
| `b369763` | raw import of the base5 A/A |
| `d71428068` | stack `stack-1d859709e-gemma4nv-base5` (parent base4) |
| `4536b59` | result stack |

## Left

- Not gated on its own: A alone (`base4-glue1`). It isn't needed for the verdict, but it would
  split the gain between the bit-exact KV fusion and the norm pairs.
- The SOL tables still model o_proj as BF16, so base5's sol_fraction understates.
- The upstream `can_fuse = False` path stays disabled and untouched.
