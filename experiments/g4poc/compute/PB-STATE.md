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
- 2026-10-05 12:34 calibrate OOMed (input logprobs in 2048-row chunks, 3 GiB fp32 over the 262K vocab with 1.7 GB
  free); fixed by config.SERVER_ENV SGLANG_LOGPROB_CHUNK_SIZE=128 (ed56e65). Then weight_checksum crashed on the new
  /weights_checker body (fixed 8782950). Calibrate passed 13:20 (reference/fidelity_thresholds.json).
- C1 chain (resumable, skips finished steps): bs2 harness-pb/compute/c1_chain.sh -> logs/pb-c1-chain.log, started
  13:18. Steps: A/A aa-c12 + set-noise -> mid tune 768/1536 (moe-tune/c1/mid) -> merge small+large+mid -> kernel bench
  (bench-tuned-merged.log) -> C1 gate c1-moe-tuned at C12 -> C8 A-B-B-A (runs/c1-c8-abba.json) -> "=== ... done".
  Expected ~16:15. Old logs pb-c1-chain-run1/run2.log.
- C2-A share profile: compute/profile_extend_share.py --ref base --concurrency 12 --steps 500 (queued on the host lock,
  runs after the A/A) -> logs/pb-profile-c12.log, runs/profile-base-C12-*/extend_share.json. Coordinator rule: send
  the split + prediction (sum share x (1 - 1/s)) before gating; >= 5% -> C2-A before C3 (KL check, gate C8+C12, one
  C16 point); 3-5% -> C3 first; < 3% -> drop C2-A with numbers. C2-A must start by 10-06 02:00.
- C2 microbench done (c2/bench.json): hd512 best mix 2.93x (32,32,64,8,1; maxdiff 0.031), exact 2.68x (16,32,32,4,1;
  maxdiff 0); hd256 best 2.02x (32,32,32,4,1; maxdiff 0.016), exact 1.62x (32,64,64,8,1).
- C3 registered (PREREG.md 543f9beaad; vault g4poc-c3a ecab5ac, g4poc-c3b d85ff6d, workload wl-g4poc-rp-inflight-sweep):
  compute/sweep_nested.sh base "c3-lpm c3-cp2048" 8,12,16,20 <out dir> (6 sweeps, ~2.3 h).
- C2-A tools: compute/kl_check.py --candidate <ref> (8 long role-play prompts; A/A = base batched vs serial).
- C1 prediction frozen: vault trial g4poc-c1 (b21c0d8), compute/PREREG.md (477d5ab0bc): E2E p90 -6..-1%.

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
