# Task W4 report (build-server-3): the W32 sliding-KV retraction gap

**Outcome.** The 1.5x W32 gap to vLLM is a KV-pool sizing problem, not a kernel problem.

- **Kept:** T-W32b, `--mem-fraction-static 0.76`. It lifts W32 throughput by **+49.1%** through the gate, from 1025 to 1528 tok/s, which matches vLLM's 1530.
- **Guards hold:** W8 and W1 stay within 0.2%, and fidelity passes.
- **No SGLang code change is needed.**

## 0. SOL bound fix

`gate/sol.py` counted one KV copy for full-attention layers because of `attention_k_eq_v`. That attribute shares only the projection weight: the cached K is k_norm + RoPE and the cached V is v_norm of the same projection, so both tensors are stored and read (`Gemma4Attention.forward`).

- **Fix:** commit 910bf86 counts two copies, with the test updated.
- **Recomputed tables:** commit b968b358a, from the same router data (`reference/sol/sol.v2.json` on bs3).

| workload | decode SOL step, old -> new | sol_fraction |
|---|---|---|
| W8 | 5.56 -> 5.58 ms | 0.60 (unchanged) |
| W1 | 3.18 -> 3.19 ms | 0.52 (unchanged) |
| W32 | 8.14 -> 8.24 ms | 0.48 (unchanged) |

## 1. Diagnosis (`trials/W32-DIAG.md`)

- **Retractions confirmed:** every A/A leg log has 14 `Retract requests` events. W32 needs 32 x 1152 = 36,864 tokens per pool, and the sliding pool holds 29,664 tokens (3.18 GB total KV). 6.56 GB of the GPU sits unused after capture.
- **How much of the gap it explains:** unpaired probes, `trials/diag_w32.py`.

  | config | result |
  |---|---|
  | base | 1030 tok/s, 6 retractions per rep |
  | `--max-running-requests 16` | 1090 tok/s: no retraction, but two waves |
  | `--mem-fraction-static 0.80` | 1519 tok/s, zero retractions |

  Retraction explains essentially all of the 1.5x.
- **What vLLM does differently:**
  - The same in both engines: FP8 KV, Triton attention, 32 sequences, 4096-token chunks.
  - The difference: `--gpu-memory-utilization 0.85` gives vLLM 7.81 GiB of KV, against SGLang's 3.18 GB.
- **The ratio knob does not help:** sliding tokens cost about 100 KB each versus 10 KB for full tokens, so no `--swa-full-tokens-ratio` fixes the pool at fixed memory.

## 2. Trials

| trial | change | prediction (frozen) | gate | verdict |
|---|---|---|---|---|
| T-W32a `gemma4nv-b3-w32a` | `--mem-fraction-static 0.80` | +48% W32 (vault 54ccf88) | run `gemma4nv-b3-w32a-20261003-084709-build-server-3-853530`: the candidate **OOMed** in the teacher-forced fidelity pass (2 GiB logits alloc, 1.19 GiB free); no verdict | **retired** |
| T-W32b `gemma4nv-b3-w32b` (variant) | `--mem-fraction-static 0.76` | +47% W32 (vault 7ed2b00) | run `gemma4nv-b3-w32b-20261003-085455-build-server-3-bf804a`, details below | **kept** |

**T-W32b gate result:** 4 ABBA pairs, bar 1%.

| metric | role | gain | per-pair range |
|---|---|---|---|
| **W32 throughput** | deciding | **1.4906** (1025.4 -> 1528.5 tok/s) | 1.482 to 1.495 |
| W8 composite | guard | 0.9986 | |
| W1 TPOT | guard | 1.0020 (6.074 -> 6.062 ms) | |

- **Fidelity:** pass on every check.
- **Retractions:** 0 in the candidate's W32 window, 14 in the control's.
- **Timed-output agreement:** 0.91, above the 0.5 floor.
- **Gate promotion:** `timing_promote` is false by design, because the gate promotes on W8 only. W32 is this trial's deciding metric, as preregistered.

**Integrity note.** The first T-W32b report failed integrity on two "undeclared" server-info diffs.

- **The keys:** `launch_command` and `max_req_input_len`.
- **Why they are false positives:** `launch_command` restates the declared args, which are already compared key by key. `max_req_input_len` is `max_total_num_tokens - 6`, and `max_total_num_tokens` is already volatile.
- **Fix:** harness commit 2f10a17 treats both keys as derived, with a test.
- **Re-evaluation:** `gate reevaluate` rewrote the report from the saved legs and appended a new ledger row. The first report is kept as `report.v1.json`.

**Probe sizing the variant** (`runs/gemma4nv-b3-diag-w32-20261003-085116`, with the gate's fidelity passes):

| fraction | retractions per rep | throughput | fidelity passes | GPU free after capture |
|---|---|---|---|---|
| 0.74 | 1 | 1391 tok/s | survived | |
| 0.76 | 0 | 1522 tok/s | survived | 5.24 GB |

## Vault (`~/fractalyze/optimization-world-model`, shared checkout; commits only, never pushed)

- **Adapter:** the `gemma4nv-gate` ledger adapter in `scripts/ingest_ledger.py`, with a test.
  - It is host-agnostic: bs2 needs only its own `meta/ledgers.yaml` entry.
  - It emits `output_throughput_tok_s`, `tpot_ms` and `w8_composite_gain`.
  - Schema additions: the `tok/s` and `ratio` units, the `gemma4nv-gate` method, and `gemma4nv-b3-*` import patterns.
  - Commits: 967973c, ee1965c, d56eb34, 98fe6ae.
- **Trials:**
  - `gemma4nv-b3-w32a`: retired, with the diagnostic probe as a measurement.
  - `gemma4nv-b3-w32b`: kept, with ledger-backed deciding and guard measurements.
- **Claims:**
  - `c-gemma4nv-w32-gap-is-kv-retraction` (mechanism).
  - `c-gemma4nv-kv-pool-logprob-headroom` (gotcha).
- **Raw import:** `raw/gemma4nv-bs3/20261003T0006Z-norev`.
- **Lint warning left open:** `gemma4nv-b3-w32b` tests the headroom claim, but that claim's evidence cites only w32a. Claims are not edited in place, so this warning stays.

## Host safety

- **Locks and limits:** every engine launch ran under host.lock, in the 24G scope, with the 2 s watchdog.
- **Watchdog record:** across the probes, min MemAvailable was 49.4 GB, peak tree RSS 12.6 GB (weight load), and peak load 1.3.
- **Compilation:** no JIT ran; the cache was warm.
- **The only failure** was the GPU-side OOM above. The host was unaffected.

## What's left

- **Adoption:** `w32b-mem076` could become the study's new base, but that is a coordinator decision.
  - BASELINE.md still pins `base` at the auto 0.718.
  - The next trials should decide which stack they compare against.
- **No code trial is preregistered.** The gap closed with config, so the step-3 code trial is not needed.
- **Optional code follow-up:** SGLang's memory planner does not reserve the input-logprob logits peak of one prefill chunk (2 GiB at chunk 4096 with a 262k vocab). That peak is what caps the KV pool here. It would be a `python/sglang` change and needs a coordinator go.
