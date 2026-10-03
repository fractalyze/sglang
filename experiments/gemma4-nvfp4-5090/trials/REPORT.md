# W3 report: analysis write-up, bs2 noise bar, first config-only trials (gemma4nv)

Branch `jumanzii/gemma4nv-analysis` (merges the gate head `c0dc2bc33a`), host build-server-2.
Every timing below is a gate verdict (paired ABBA, ratio of sums). bs2 numbers compare to
bs2 controls only.

## 1. Write-up

- `analysis/PROFILE.md` §3-5: measured B=8 and B=1 component tables (achieved µs, SOL µs,
  sol_fraction, rank = share x (1 - sol_fraction)), the knob screen with deltas, and the
  SM80 WMMA finding: every BF16 GEMM runs `cutlass_80_wmma_..._16x16`, and o_proj drops
  from 0.83 to 0.58 sol_fraction going from B=1 to B=8 for the same bytes.
- `analysis/HYPOTHESES.md`: re-ranked on the measured shares (R1 MoE path, R2 glue fusion,
  R3 small-M BF16 GEMM, R4 FP8 projections, R5 FP8 lm_head, ...). It adds the B=1
  speculative-decoding drafters `google/gemma-4-26B-A4B-it-assistant` (`gemma4_mtp.py`,
  FROZEN_KV_MTP) and `z-lab/gemma-4-26B-A4B-it-DFlash` (`dflash.py`), with a verify-cost
  note: a B=8 step is 1.55x a B=1 step, so k=3 MTP needs τ ≳ 1.5 and DFlash-16 τ ≳ 2. It
  stays off at B=8 (Yukon; the crawler shows MTP gain shrinking after NVFP4).

## 2. bs2 gate bring-up and noise bar

- Harness deployed to `/data/jooman/gemma4nv/src-gate` (record in `DEPLOYED.txt`).
- Prebuild `prebuild-base-20261003-083007-build-server-2-64370b`: no compilers ran (the JIT
  cache from bs3 holds). Peak tree RSS 11.5 GB at weight load, min MemAvailable 49.4 GB, peak
  load 0.8.
- Calibration `calibrate-20261003-083126-build-server-2-fc5d05` reproduces bs3: forced KL
  mean 0.0298 (bs3 0.030), min top-1 0.92.
- **A/A `AA-20261003-083242-build-server-2-6228c0`** (6 pairs):

  | metric | A/A gain | per-pair σ |
  |---|---|---|
  | W8 composite | 0.99995 | 0.020% |
  | W1 TPOT | 1.0001 | 0.050% |

  Integrity and fidelity pass, and every bar sits at the 1% floor. W1 TPOT is 6.00 ms on bs2
  (bs3 6.075).

## 3. Trials (each: frozen prediction → gate → vault record)

| trial | change | prediction (frozen) | gate result | verdict |
|---|---|---|---|---|
| T1 `gemma4nv-b2-t1` | `--chunked-prefill-size 8192` (+ pinned mem fraction) | W8 composite 0.95-0.99 (accounting shift) | `T1-20261003-084551-build-server-2-80d58e`: composite **0.9926**; prefill TTFT-sum +16.8%; decode-sum −4.1%; max TTFT −3.5%; W1 +0.16%; fidelity pass | **retired** |
| T2 `gemma4nv-b2-t2` | `--triton-attention-num-kv-splits 16` | W1 TPOT −1.5…−2.5%; W8 1.004-1.012 | `T2-20261003-085810-build-server-2-22b5a6`: **W1 −1.95%** (σ 0.03%); W8 composite 0.9982; fidelity pass | **parked** (gate integrity false positive) |
| T3 `gemma4nv-b2-t3` | replace the SM80 WMMA fallback for small-M BF16 o_proj and dense MLP | W8 composite 1.02-1.04; W1 −0.5…−2% | not run | **preregistered, awaiting go** |

- **T1 answer:** the screen's −4.2% "decode" was a W8 prefill-wave artefact. Base admits
  1/4/3 streams per wave. One chunk moves the early streams' wait out of the decode term and
  into the summed TTFT. Nothing changes at W1.
- **T2:** a real B=1 gain with W8 neutral. The gate's verdict is promote=false only because
  of integrity checks that do not fit this change. It is parked until those checks are fixed;
  `gate reevaluate --run T2-...` then decides it without a rerun.
- Vault (shared checkout, not pushed):
  - meta `c8b5cdd`: W8 metrics and the gemma4nv-bs2 raw root;
  - raw imports `695acb3` and `ef954e6`;
  - stubs `8c513c7`, `879df61` and `0a2b029`;
  - claims `14eee3e` and `7550a8e`;
  - records `45b73c4` and `021705e`, prose `c8d5a95` and `bd64899`.

  bs2 has no `meta/ledgers.yaml` entry, so the measurements are `--manual-measurement`, each
  citing its report.json.

## 4. Findings for other owners

- **Gate integrity false positives** (reported to the coordinator):
  - `launch_command` should be volatile;
  - arg diffs that follow from declared flags are refused (prefill-graph max_bs from the chunk
    size, `max_req_input_len` from the mem fraction);
  - timed-output agreement < 0.5 fails any change to batch composition or reduction order,
    though fidelity bounds that drift;
  - the ledger's `harness_commit` records src-gate's HEAD, not the rsynced experiments commit.
- **Vault hygiene incident** (reported): my bs2 import commit `ef954e6` included 7
  MANIFEST lines from W4b's in-progress bs3 import. `wm record`'s study-wide ingest also
  appended W4b's w32b ledger run to their page (`021705e`). Nothing was invented. W4b needs
  to commit its bs3 snapshot files.

## 5. Next

1. Coordinator go/no-go on T3 (`trials/T3-smallm-bf16-gemm.md`). Step 0 is a config-only
   cuBLASLt microbench, and it decides whether code is needed.
2. Fix the gate integrity scoping, then run `gate reevaluate` on T2. If kept, consider a
   batch-aware split count (B=1 gain without B=8 cost).
3. B=1 speculative decoding (R10) as its own W1-latency trial. It needs a drafter download
   (bs2 /data has ~56 GB free now).
