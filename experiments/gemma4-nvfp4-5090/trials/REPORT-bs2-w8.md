# W8 report: speculative decoding on Gemma-4-26B-A4B NVFP4 (gemma4nv, build-server-2)

- Branch: `jumanzii/gemma4nv-analysis`. Host: build-server-2.
- Control: gate ref `base3` (W5's pin: base2 + split-KV 16 + T3, commit `1fd77e64b0`), which exists on the gate branch and is copied verbatim here.
- Every timing in §4 is a gate verdict (paired ABBA, ratio of sums, 4 pairs). Everything else is a labelled screen: unpaired, one server lifetime, through the gate's capped server path, from `trials/spec/spec_probe.py`.

| trial | deciding metric | predicted (frozen) | measured | verdict |
|---|---|---|---|---|
| T-SPEC1 MTP assistant, k=5, all batch sizes | W1 TPOT | -22 … -34% | **-31.4%** (5.751 → 3.946 ms) | **not promoted**: W32 guard -20.3% (predicted -12 … -30%) → vault **parked** |
| T-SPEC1b speculation only at small batch | - | not registered | - | needs a `python/sglang` change; asked the coordinator (msg_31153bc67ce0), no answer yet |
| DFlash-16 | - | not registered | screened only | serves config-only; at B=1 and B=8 it is no better than MTP k=5 |

## 1. Feasibility (step 0)

**Drafters.** `google/gemma-4-26B-A4B-it-assistant` (0.83 GB) and `z-lab/gemma-4-26B-A4B-it-DFlash` (0.82 GB) are in `/home/jooman/gemma4nv/models/`, on bs2's root disk.

**MTP.** `--speculative-algorithm NEXTN` with the assistant is promoted to `FROZEN_KV_MTP`.
- The drafter runs in BF16: 4 layers, hidden 1024, a 2048-centroid sparse lm_head.
- It reads the target's FP8 KV, so it has no KV pool of its own.
- On SM120, verify CUDA graphs are captured at every decode bs up to 32, with 1+k tokens per request. The draft decode loop is graphed when k ≥ 2.
- Every timed decode step reports `cuda graph: True`, and no JIT ran.

**Cost.** At `--mem-fraction-static 0.76` the KV pools shrink:

| pool | without the drafter | with the drafter |
|---|---:|---:|
| full | 52.0k tokens | 40.4k tokens |
| sliding | 41.5k tokens | 32.3k tokens |

**Arg-diff integrity.** The speculative hook derives `max_running_requests` = 48, the draft revision and a finer decode-graph bs grid. The candidate ref restates these as flags, so `gate run` sees no undeclared diff. I found this beforehand with the probe's `info` mode and the gate's own `server_arg_diff`.

**DFlash.** It serves config-only (`--speculative-algorithm DFLASH --speculative-num-draft-tokens 16`). The tree has had Gemma-4 DFlash support since `5ea0d1d093`. Peak RSS is 8.2 GB.

**Routed-expert capture on Gemma-4** needs the override W2 used: `--json-model-override-args '{"text_config": {"num_experts_per_tok": 8}}'` and `--disable-cuda-graph`. Without it, `RoutedExpertsCapturer` fails at init on `num_experts_per_tok`.

## 2. Accept length: hidden set vs timing corpus

τ is tokens per verify round, bonus included. Greedy decoding, 256 new tokens.

| config | hidden (22 prompts) | timing corpus B=1 | timing corpus B=8 |
|---|---:|---:|---:|
| MTP k=1 | 1.84 | 1.87 | 1.79 |
| MTP k=2 | 2.45 | 2.52 | 2.29 |
| MTP k=3 | 3.03 | 3.05 | 2.78 |
| MTP k=4 | 3.33 | 3.62 | 3.03 |
| MTP k=5 | 3.51 (3.68 on base3) | 4.43 (4.20) | 3.02 (3.08) |
| MTP k=7 | 4.03 | 5.00 | 3.84 |
| DFlash 16 | 3.04 | 4.21 | 2.87 |

- **The hidden set does not collapse.** In Yukon it fell to 0.43-0.63 of public. Here it stays at 80-100% of the timing corpus.
- **Per hidden category at k=3:**

  | category | τ |
  |---|---:|
  | multilingual | 2.36 |
  | chat | 2.49 |
  | Korean | 2.80 |
  | code | 3.15 |
  | tool-JSON | 3.73 |
  | math | 3.75 |
  | long-context (8-16k) | 2.8-3.6 |

  Prompt contents were never read or printed.

## 3. Verify-cost curve (step 1)

Method:
- **Decode step time** at exactly M running rows comes from the per-step decode log (`w8-sweep-*`). Base2 + T3, graphs up to 48. 40 rows got only 20 steps and 48 none, because of KV admission.
- **Raw routing** comes from `w8-experts-*`: 24 timing streams and 22 hidden, 128 tokens each.
- **The fit**, from `trials/spec/verify_cost.py`: step = 5.86 ms + 0.099 ms·rows + 0.046 ms·(distinct experts per layer), max error 7.8%.

| B | k | rows | experts/layer, verify (timing / hidden) | experts/layer, same rows independent | verify / 1-row-step (model) | decode at same rows (measured) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 13.2 / 12.5 | 14.1 | 1.05 | 1.15 |
| 1 | 3 | 4 | 19.4 / 18.5 | 23.1 | 1.13 | 1.26 |
| 1 | 5 | 6 | 23.8 / 23.4 | 29.8 | 1.19 | 1.34 |
| 1 | 7 | 8 | 27.2 / 26.9 | 35.1 | 1.25 | 1.39 |
| 8 | 1 | 16 | 47.2 / 61.6 | 48.7 | 1.16 | 1.18 |
| 8 | 3 | 32 | 59.6 / 77.6 | 62.7 | 1.42 | 1.44 |
| 8 | 5 | 48 | 66.5 / 86.5 | 70.9 | 1.65 | (40 rows: 1.61) |
| 8 | 7 | 64 | 71.7 / 92.0 | 76.5 | 1.87 | - |

**Prediction from the curve.** At B=8 and k=5, with τ about 3.0, speculation pays if the round costs under 3.0 × 8.7 ms. The verify is about 14 ms. The draft adds about 5 ms. So W8 decode was predicted to win, unlike the Yukon MLX prior.

**The real verify is slower than the curve.** A B=1 torch-profiler trace of k=3 (`w8-prof-k3-*`, 12 rounds) splits each 14.8 ms round as follows. The GPU is 93% busy.

| part | ms per round |
|---|---:|
| **verify forward** | **11.5** |
| - `_fwd_kernel` (Triton extend attention, grid [1,16,1]) | **4.7** |
| - cuBLAS WMMA-fallback BF16 GEMMs (qkv and lm_head at M=4) | 2.4 |
| - NVFP4 MoE (CUTLASS) | 1.6 |
| - small-M Triton GEMM | 1.3 |
| draft loop (3 steps; 1.55 ms of it is a cuBLAS gemv) | 2.2 |
| gaps | ~1 |

The TARGET_VERIFY path of the Triton backend runs the unified extend kernel with no KV split. That leaves 16 CTAs on 170 SMs, at about 157 µs per layer. Split-KV decode attention costs about 0.05 ms per step for the same context. This is vault claim `c-sglang-triton-target-verify-attention-unsplit`.

## 4. T-SPEC1: gate verdict

**Choice of k.** Screens on base2 + T3, TPOT in ms:

| | none | k=1 | k=2 | k=3 | k=4 | **k=5** | k=7 | DFlash-16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B=1 | 5.87 | 7.29 | 5.06 | 4.34 | 4.13 | **4.03** | 3.95 | 4.41 |
| B=8 | 8.73 | 8.65 | 7.04 | 6.70 | 6.27 | **6.21** | 6.80 | 6.17 |

- k=1 loses: a round costs about 13.6 ms for 1.87 tokens.
- k=5 is the best B=8 point and within 2% of the best B=1 point.
- DFlash pays the same unsplit verify attention at 16 rows.

**Screen of the final pair** (base3 vs base3-mtp5):

| | base3 | base3-mtp5 | change |
|---|---:|---:|---:|
| B=1 TPOT | 5.75 ms | 4.02 ms | -30% |
| B=8 TPOT | 8.74 ms | 5.82 ms | -33% |
| B=8 TTFT | | | +47% |
| B=32 wall throughput | | | -22% (11 retractions vs 1) |

**Registration.** `trials/T-SPEC1-mtp5.md`, sglang `4db5b6c644`. Vault `gemma4nv-b2-tspec1`, frozen in `51a621e`. Pattern `request-batching`, because the vault has no speculative-decoding pattern. Agent decision `no-evidence`.

**Gate** `T-SPEC1-20261003-105933-build-server-2-51affe`, `--decide-on w1_tpot_gain`, prebuild `prebuild-base3-mtp5-20261003-105856-build-server-2-e4d9c9`:

| metric | control | candidate | gain | per-pair σ | bar | predicted |
|---|---:|---:|---:|---:|---:|---|
| **W1 TPOT (deciding)** | 5.751 ms | 3.946 ms | **1.458 (-31.4%)** | 6.9% | 1% | -22 … -34% |
| W8 composite (guard) | | | **1.259** | 1.3% | 1% | +10 … +30% |
| W8 decode | | | 1.487 | 1.8% | 1% | +30 … +50% |
| W8 prefill | | | 0.765 | 0.7% | 1% | 0.60 … 0.85 |
| **W32 tok/s (guard)** | 1574.3 | 1254.4 | **0.797 (-20.3%)** | 3.4% | 1% | -12 … -30% |

Checks:
- **Fidelity: pass.** Free-running KL mean is 0.0274 against the control's 0.0112, and p99 is 0.415 against 0.235. Both are inside the thresholds. Teacher-forced checks all pass.
- **Integrity: ok.** There are no undeclared arg diffs. Timed-output agreement is 0.12, reported only, because the numerics change.
- **Timing: not promoted.** W32 regresses beyond its bar. Every metric lands inside its frozen interval.
- **W1 per-pair spread is wide** (1.345-1.591), because τ varies with each pair's fresh prompts.

**Vault** (shared checkout; only my paths committed; not pushed):

| commit | what |
|---|---|
| `51a621e` | prediction |
| `1fb8b8d` | raw import of `gemma4nv-bs2`: the gate run, its prebuild and the ledger |
| `16491bd` | claim `c-gemma4nv-mtp-verify-shares-experts-wins-b1-b8` |
| `3e47fa2` | claim `c-sglang-triton-target-verify-attention-unsplit` |
| `5e0d9ee` | verdict **parked** |
| `5a91b77` | result prose |

- The park condition is speculation only at small batch, plus a KV pool that holds W32 with the drafter loaded.
- The bs2 ledger has no adapter in `meta/ledgers.yaml`, so the measurements are `source: manual`, as W7's were.

## 5. What it means

- **On the 5090 speculation wins at B=8 too, against the Yukon prior.**
  - A verify of one stream's consecutive tokens shares most experts, so an 8x6 verify costs 1.65x one step.
  - The hidden τ holds at about 3.5.
  - The W8 composite is +26% on its own. That is the largest W8 move in the study so far.
- **What blocks promotion is memory.** The drafter costs about 1 GB of KV pool, and W32 retracts again. T-W32b's fix was exactly this headroom.
- **T-SPEC1b's gating alone will not fix W32.** The drafter's weights stay loaded, so the pool stays small. It also needs one of two things:
  - the pool restored. 0.80 OOMed the teacher-forced logprob pass on bs3, so that needs a measured fraction between 0.76 and 0.80, or a smaller `--cuda-graph-max-bs-decode`;
  - speculation off at the batch sizes where retraction starts.
- **The biggest further lever is verify attention** (T-SPEC2). Split-KV for TARGET_VERIFY in the Triton backend would cut about 4.5 ms of a 14.8 ms B=1 round. Predicted W1 TPOT is about 2.9 ms against 3.95 ms. It also helps B=8, where the grid is [8,16,1].

## 6. Host-safety record

Peaks over all hostmem CSVs of each run on bs2 (`/data/jooman/gemma4nv/runs/`):

| run | min MemAvailable | peak tree RSS (phase) | peak load | compilers |
|---|---:|---:|---:|---:|
| spec screens, MTP k=2…7 (8 runs) | 48.8 GB | **19.5-19.8 GB (graph capture / serving)** | 1.3 | 0-2 (ptxas) |
| MTP k=1 (no draft graph) | 49.5 GB | 13.9 GB (weight load) | 1.1 | 0 |
| DFlash-16 | 48.2 GB | 8.2 GB (autotune) | 1.3 | 4 (Triton ptxas) |
| no-spec and base3 screens, info | 49.6 GB | 6.6-6.8 GB | 4.6 | 0 |
| sweep / experts | 49.7 GB | 14.1-14.5 GB (weight load) | 1.1 | 0 |
| profile k=3 | 49.0 GB | 19.4 GB (serving) | 6.2 | 0 |
| prebuild base3-mtp5 | 48.3 GB | 19.4 GB (serving) | 0.6 | 0 |
| gate T-SPEC1 | 48.2 GB | 19.5 GB | 6.0 | 0 |

- **One engine per host.** Every engine ran alone under `host.lock` and the gate's 24G `systemd-run` scope with its watchdog. No watchdog tripped, swap stayed at 0.73 GB and the FlashInfer cache held.
- **New finding: the graphed MTP draft loop raises tree RSS to 19.5 GB.** It appears from k ≥ 2, when the draft decode loop is graphed.
  - MemAvailable falls only about 0.7 GB more than for the control, so most of that RSS is shared or mapped pages, not anonymous host memory.
  - It still sits 4.5 GB under the 24G scope, so it is a margin to watch if speculation joins a base.
- **One aborted launch.** The first `experts` run (`w8-experts-103258`) died at server init on `num_experts_per_tok`. That was a config error, not a memory event.

## 7. Next (decisions for the coordinator)

1. **T-SPEC1b**: a FROZEN_KV_MTP batch-size switch behind a new `SGLANG_OPT_` env, plus a W32 memory fix. This is a `python/sglang` change, so it needs a go.
2. **T-SPEC2**: split-KV Triton TARGET_VERIFY attention. This is also a `python/sglang` change. It is the largest per-round cost found.
3. **Explain W8 TTFT +31%.** One profile would separate the two candidates: mixed chunked prefill disabled, or prefill waves waiting behind verify rounds.
4. **DFlash:** no trial. It is not better than MTP k=5 at either batch, and it pays the same verify-attention cost at 16 rows.
