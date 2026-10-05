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
- FINAL = final-hc-cp2048-lpm (gate/refs.json; final-hc + --chunked-prefill-size 2048 --schedule-policy lpm; tree
  a0491db764; 28G). Promoted 10-05 ~23:55 (confirm A-B-B-A C24/C32 gain 1.036/1.103, KL pass, PX exactness 12/12).
  Headline bs2: C32 981 tok/s p90 8.57 s $0.198/1M @0.70 vs base $0.343 (-42%).
- Merged g4poc-c + g4poc-c-hicache into g4poc-b (8b70c7a928), deployed 10-06 ~00:25.
- bs2 chain done through logs/pb-pthink.log (final pthink30 48-96 on bs2, 03:01-03:51,
  runs/sweep-final-hc-cp2048-lpm-20261006-030116-build-server-2-654b74). The no-HiCache pair was dropped (PC2's bs3
  covers it). PC4's queue (pid 1603659, harness-pc4/queue-pc4b.sh, ends ~07:25) owns bs2 now; coordinator: PC4 first.
- M128 (PC4's SWA margin on the final, ref final-hc-cp2048-lpm-m128, tree cbf56143b5): (a) A-B-B-A C28 p90 gain
  1.020, tok/s +1.3%, p99 +7.5%, retractions 2.4x; (b) quality pass; (c) bs2 soak C28 2.56% retracted (bs3 2.67%) vs
  1.39% limit -> NOT promoted (coordinator 06:40). (d) skipped (PC2 runs the final's C8/C12 on bs3). (e) second
  A-B-B-A running from 06:46 via /home/jooman/g4poc/pb2-m128-tail.sh (= compute/m128_tail.sh) -> runs/m128/abba-c28-2.json,
  ends ~07:12; then bs2 idle. To do: vault g4poc-m128f record (parked), COMPUTE.md (e) numbers, 6 s point from PC2.
- HS2 dropped on bs2 (coordinator 04:10): answered by PC2's same-host bs3 point; vault g4poc-hs2 retired (29e3f8b).
- Error-detail loadgen change deployed into harness-pb 10-06 03:53 (94 tests OK).
- PC2's keep-alive fix (ffa983fd37, client keepalive 2 s) cherry-picked and deployed into harness-pb 10-06 ~04:45.
- COMPUTE.md: T30 edge ~70 sessions/GPU (PC2 C72/C76), ~32 GPUs for 2,200; fleet v4 (+3.5% T30, +10% T60).
- Vault: c1 kept, c2 kept, c3a/c3b kept (base), c3a-hc/c3b-hc kept (final), c4 retired; hs1 / hs1b retired; hs2 retired (confirmed, flags stay for in-flight traffic).
- 11:00-13:30: COMPUTE.md final numbers (two replicates, pthink calibration of fleet v3, PC2 poisson), push. Timebox 14:00.

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
