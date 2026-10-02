# T1 (gemma4nv-w3-t1): one-chunk W8 prefill, `--chunked-prefill-size 8192`

Registered 2026-10-03 before the gate run, on bs2. Candidate ref `chunk8k` in
`gate/refs.json`: base + `--chunked-prefill-size 8192 --mem-fraction-static 0.718` (the
memory fraction equals base's resolved value, so the KV pool does not change with the chunk
size). Control ref `base`. Pattern: serving-step-scheduling. Config-only, no code.

## Prediction (frozen; do not edit after the run)

Mechanism. At chunk 4096 the 8x1024 W8 prefill is two chunks: streams 1-4 get their first
token after chunk 1 (~145 ms) and streams 5-8 after chunk 2 (~290 ms). At chunk 8192 all
8 streams get their first token after one ~280 ms pass. No decode kernel changes, so the
steady-state B=8 decode step cannot change. The W2 screen's "decode −4.2%" is the ragged
start of streams 1-4 being charged to their decode time at chunk 4096 and to their TTFT at
chunk 8192: an accounting shift between the two terms of the composite, not a speedup.

The gate's W8 prefill term is the per-stream TTFT **summed** over the 8 streams, so moving
streams 1-4 from ~145 to ~280 ms raises it.

| metric (bs2, ratio of sums, control/candidate) | predicted |
|---|---|
| W8 prefill (sum of per-stream TTFT) | **+20 … +32% time (worse)** |
| W8 batch prefill (max TTFT) | −2 … −4% |
| W8 decode (sum of per-stream e2e − TTFT) | −3 … −6% |
| **W8 composite gain** (prefill^0.25 x decode^0.75) | **0.95 … 0.99** (point 0.97: a loss of ~3%) |
| W1 TPOT | 0 ± 0.5% (one chunk either way) |
| Fidelity | pass at the calibrated level (same kernels; batch composition of the prefill changes) |

Falsified if the W8 composite gain is ≥ 1.00 (a real decode effect beyond the accounting
shift), or W1 TPOT moves by more than its bar.

Decision rule: keep only if the composite gain clears the bs2 A/A bar and fidelity passes.
Expected verdict: **discard** for W8. The real effect (max TTFT −2 … −4%) is a
latency-to-first-token-of-the-batch gain that the W8 composite does not reward; it is
reported, not adopted.
