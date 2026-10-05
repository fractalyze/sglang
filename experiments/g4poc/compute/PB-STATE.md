# PB state (compute levers on bs2) — read this first after a restart

Owner: PB2 (T3 delegated task, replaces PB thread 6d4ea636, dead 2026-10-05 11:55). Coordinator: e7e4878e-4495-462c-8ed6-666a55c0bdae.
Coordinator rule: never call t3_thread_configure while background commands run (it killed PB).
Timebox ends 2026-10-06 14:00 KST. Branch jumanzii/g4poc-b (push to `fractalyze`).
Harness on bs2: /data/jooman/g4poc/harness-pb (gate/deploy.sh). Runs: /home/jooman/g4poc/runs. Logs: /home/jooman/g4poc/logs.
Never launch a server outside the gate / serve.sh (MemoryMax scope + host.lock).

## Done
- Baseline in-flight sweep (base = PA r03), bs2: runs/sweep-base-20261004-152459-build-server-2-c0acf7.
  Capacity at p90 6/10/15 s = C8 / C12 / C20; goodput peaks at C12 (568 out tok/s, 18.0K total).
  Prefix hit rate collapses above C12 (0.70 -> 0.22 at C16 -> 0.002 at C20+). Sent to coordinator 2026-10-05 ~11:31 KST.
- Fidelity prompts copied from gemma4nv (same tokenizer.json sha cc8d3a0c...) to /data/jooman/g4poc/hidden.
- Checkpoint pinned (reference/checkpoint_pin.json), 2026-10-05.
- Gated load = inflight-C12; noise is used only if measured at the same load.

## In flight / waiting (update on every launch)
- Done 10-05: harness fixes (logprob chunk 128, weights_checker body); calibrate; A/A aa-c12 (bar 1%).
  C1 KEPT (C12 +1.2%, C8 +1.0%; vault g4poc-c1 recorded). C2-A KEPT at C12 (+11.5% p90, +10.6% tok/s;
  runs/c2-gate-20261005-154627); numerics: compute/runs/c2/accuracy.json, kl-c2-extend-tiles*.
- bs2 queue (resumable): compute/pb_queue.sh c2-extend-tiles -> logs/pb-queue.log:
  C2-A sweep_abba 8,16 (runs/c2-c8-c16-abba.json, ~17:30) -> C3 sweep_nested base "c3-lpm c3-cp2048" 8,12,16,20
  (runs/c3-nested/, ~19:50) -> pd-measure base (~20:00) -> "=== ... queue done".
- Next (coordinator-approved 10-05 ~16:40): decode_split_bench (C4 microbench, PYTHONPATH tree 1425761173d3) ->
  compute/final_run.sh final-mem-c1-c2a[+kept C3 flag] (~95 min; memory limit on bs2 31,599 MiB: if the peak
  breaks it, rerun at mem 0.95 and report it as bs2's deployable value) -> C4 overnight vs the final stack
  (--triton-attention-num-kv-splits / split tile size; nested sweeps C12+C20 then gate C12; hard stop 06:00).
- Morning 08:00-11:00: replicate the final sweep (+ kept C4 flag; + --max-running-requests 24 if the
  coordinator says so by ~19:00). HiCache never in a final stack. 11:00-13:30 report, cost table, P/D model, wm-record.
- Final trees on bs2: 1425761173 (mem-final 53752c62aa + C2-A, branch jumanzii/g4poc-final-c2), 57e5f273c0 (exact tiles).

## Queue (GPU, in order)
1. gate calibrate --ref base (fidelity reference + thresholds)
2. A/A: gate run --control base --candidate base --pairs 4 (load inflight-C12), then gate set-noise
3. C1 gate: base vs c1-moe-tuned (env SGLANG_MOE_CONFIG_DIR)
4. C2 prefill attention backend, C3 chunked-prefill size — each preregister -> gate -> record

## Vault notes
- bs1 vault ~/fractalyze/optimization-world-model is authoritative; commit, never push. Commit only PB's paths.
- `wm` must run as `uv run --quiet --project ~/fractalyze/optimization-world-model python $(cat ~/.local/state/world-model/wm-path) ...`
  (system python3's old jsonschema breaks validation and wm does not re-exec under uv).
- `--registered` must be a URL: https://github.com/fractalyze/sglang/blob/<commit>/experiments/g4poc/compute/PREREG.md
- Pages: workload wl-g4poc-rp-inflight-c12, metric e2e_p90_s, baseline stack stack-ac6035c07-g4poc-mem-base (PC's; same r03 config).
