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
