# g4poc compute levers (PB, bs2): pre-registered predictions

Write-once. Each entry is committed before its gated run; a miss is explained in the
record, never by editing the number here. Vault trial ids `g4poc-c<N>` (study-g4poc).

Common setup: control `base` (PA's r03, `gate/refs.json`), gated load `inflight-C12`
(12 requests always in flight, 60 s warm-up + 240 s window), 4 ABBA pairs, fresh server
per leg, build-server-2 only. Deciding metric: E2E p90 gain (control/candidate), bar =
max(3 sigma of the A/A noise at the same load, 1%). Fidelity on pair 0; the candidate must
not regress output tok/s past its bar.

## C1: tuned Triton fused_moe config (registered 2026-10-05 ~11:50 KST)

Change: `SGLANG_MOE_CONFIG_DIR` points at a config file produced by SGLang's
`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` (run through
`compute/moe_tune.py`; fp8_w8a8, per-channel, E=128, N=704, 1,920 configs x 18 token
counts) instead of the default config. Nothing else changes. One file covers both MoE
GEMMs (no `_down` file).

Prediction: E2E p90 at inflight-C12 improves by 1% to 6% (delta -6% .. -1%).
Falsified if the gain is under 1%, or the fidelity tier is worse than reorder (a tile
change reorders the K reduction, nothing more).

Basis (analytic): MoE is ~21% of GPU time on the base. The default config's microbench
on bs2 (`moe-tune/c1/bench-default.log`) is at 80-95% of the HBM bound for 4-32 tokens
(decode: 120 us at 4, 318 us at 16, 423 us at 32) and ~267 TFLOPS at 4,096 tokens
(1.46 ms, the prefill chunk). Headroom: ~10-20% of decode MoE time, ~15-30% of prefill
MoE time, so 2-6% of E2E at most.

## C3: scheduling flags across the hit-rate cliff (registered 2026-10-05 ~14:00 KST)

Changes (flags only, each against `base`): C3a `--schedule-policy lpm` (ref `c3-lpm`; the base is fcfs),
C3b `--chunked-prefill-size 2048` (ref `c3-cp2048`; the base is 4096).

Setup: nested in-flight sweeps base, c3-lpm, c3-cp2048, c3-cp2048, c3-lpm, base
(`compute/sweep_nested.sh`, one server per sweep) at 8, 12, 16 and 20 in flight; per point 60 s warm-up +
240 s window, cache flushed between points. Per candidate and point (`compute/sweep_abba.py`): E2E p90 gain
(control/candidate) and output tok/s gain (candidate/control), geometric mean of the two control/candidate
pairs. Bar per point: max(the gate's C12 bar from the A/A, |log| of the control's drift at that point
(first over last control sweep)).

Keep rule: a flag is kept if at 16 or 20 in flight E2E p90 gain or output tok/s gain clears the bar with
both pairs on the same side, and no point loses more than its bar on either metric. Inside the bar
everywhere = neutral: C3a is dropped; C3b stays available to PC's memory stack (its smaller prefill
transient is a memory lever, measured by PC, not here).

Prediction C3a (lpm): inside the bar at 8 and 12 (the base never queues there: mean queue 0.0, so the
order cannot differ). At 16 (mean queue 1.1, hit 0.22) E2E p90 delta -5% .. 0% and hit rate +0 .. +0.10;
at 20 inside the bar (hit 0.002: no cached prefix survives for lpm to prefer). Falsified if any point gains
more than 5% or loses more than its bar.

Prediction C3b (cp2048): E2E p90 delta -1% .. +3% and output tok/s -3% .. +1% at every point. Decode
steps wait behind half as many prefill tokens, but a cold or re-prefilled ~5.6K-token prompt takes 3
forwards instead of 2, and from 16 up every turn re-prefills, so the cost shows most there. Falsified if any
point gains more than 1% or loses more than 3%.

Basis: the base sweep's server log (COMPUTE.md section 1: queue means 0.0 / 0.0 / 1.1 / 4.6 at 8 / 12 / 16
/ 20, hit 0.74 / 0.70 / 0.22 / 0.002); vault gemma4nv-b2-t1 (chunk 8192 on bs2 moved waiting between the
TTFT and decode terms, composite -0.74%, inside its bar).
