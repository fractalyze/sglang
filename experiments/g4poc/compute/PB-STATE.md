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
- FINAL = final-hc (hicache/refs.json; tree a0491db764; 28G scope). Done on bs2 10-05: final_run final-hc (sweep 4-40:
  C32 952 tok/s p90 9.50 -> $0.204/1M @0.70, -40.5% vs base; memory peak 31,514 <= 31,599 OK at mem 0.955; quality
  GSM8K 96.74 vs base 96.13 pass; final-mem-c1-c2a 16-32 on/off pair; P/D final-hc). Kept: C1, C2-A; C3a/C3b kept
  on base only. Records in compute/runs/, vault g4poc-c1/c2/c3a/c3b recorded.
- Overnight (bs2): compute/c4_run.sh final-hc "final-hc-kvs16 final-hc-lpm final-hc-cp2048" -> logs/pb-c4.log
  (smokes passed; nested 24,32 -> runs/c4-nested/; c4_pick -> combined ref; KL check; confirm runs/c4-confirm.json;
  prints "=== ... C4 done"). Then logs/pb-post-c4.log: final-mem-c1-c2a-nocache sweep 4-20 (fleet model b) and
  mem-final 16-24 (before 06:00 only). Vault trials g4poc-c4, g4poc-c3a-hc, g4poc-c3b-hc registered (open).
- When the combined ref is known: push, then send the coordinator name/file/commit/flags (PC3 runs exactness on bs3
  ~01:30, control = same flags on final-mem-c1-c2a). Promotion needs KL + confirm + exactness 12/12 by 07:30.
- Morning 08:00-11:00: G4POC_SERVER_MEMORY_MAX=28G sweep final-hc (+ promoted flags) at 16,24,32,40 + quality anchor;
  P/D final-hc already done. 11:00-13:30: COMPUTE.md (fleet model with PC2's think-time table, P/D, cost), wm-record
  (c4, c3a-hc, c3b-hc), push. Timebox 14:00.
- Fleet model: compute/fleet_model.py --cached <final-mem-c1-c2a sweep> --nocache <nocache sweep> --hicache <final-hc
  sweep> --measured "label:sessions:think:turns:tok_s:p90:hit". Sizing rule ~1.2 GB host per s of think (T30: 38 GB).

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
