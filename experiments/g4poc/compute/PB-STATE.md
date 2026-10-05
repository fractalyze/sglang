# PB state (compute levers on bs2) — read this first after a restart

Owner: T3 thread 6d4ea636-228f-4a20-9be9-d006d6311a21 (PB). Coordinator: e7e4878e-4495-462c-8ed6-666a55c0bdae.
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
- C1 MoE tune (pruned): bs2 /data/jooman/g4poc/moe-tune/c1/run2.sh -> driver2.log, small/ (M 1-32,
  BLOCK_M 16-64) and large/ (M 256-4096, BLOCK_M 64-256), 648 configs each; started 2026-10-05 11:44 KST,
  ~65 min, holds host.lock per part. Done when driver2.log has "tune2 done". Then merge small+large JSON into
  /data/jooman/g4poc/moe-configs/c1/configs/triton_3_7_1/E=128,N=704,device_name=NVIDIA_GeForce_RTX_5090,dtype=fp8_w8a8,per_channel_quant=True.json
  (The first full 1,920-config run was stopped at 11:43: ~14 min per token count, 4 h total.)
  Watcher: session-local background loop on driver2.log (lost on restart: just re-check the file).
- C1 chain: bs2 harness-pb/compute/c1_chain.sh -> /home/jooman/g4poc/logs/pb-c1-chain.log, started 11:46 KST.
  Steps (=== markers): wait tune -> merge -> kernel bench (moe-tune/c1/bench-tuned.log) -> calibrate ->
  A/A aa-c12 + set-noise -> C1 gate (runs/c1-moe-tuned-*). Expected done ~15:00 KST. Ends with "=== ... done".
  DO NOT gate/deploy.sh to harness-pb while the chain runs (bash reads the script as it goes).
- C1 C8 check: waits for the chain's "done", then compute/sweep_abba.sh base c1-moe-tuned 8 ->
  logs/pb-c1-c8.log and runs/c1-c8-abba.json (confirming, ~30 min). Host one-liner: logs/pb-state.txt.
- Coordinator plan (2026-10-05): C1 <= 3 h; C1/C2 gate at C12 (deciding) + C8; C2 survey-first, no kernel
  project; C3 chunked-prefill + --schedule-policy lpm checked at C8/C12/C16/C20 (sweep_abba); stretch HiCache
  feasibility write-up only if 1-3 finish before 10-06 04:00. No new lever after 10-06 08:00; 08-11 final
  stacked sweep C4-C32 + quality anchor on bs2; 11-13:30 COMPUTE.md report, push.
- C1 prediction frozen: vault trial g4poc-c1 (b21c0d8), compute/PREREG.md (477d5ab0bc): E2E p90 -6..-1%.
- C2 survey: Explore subagent reading SGLang attention backends for SM120 + Gemma-4 (session-local).

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
