# T2 (gemma4nv-w3-t2): Triton decode attention split-KV count 16

Registered 2026-10-03 before the gate run, on bs2. Candidate ref `splits16` in
`gate/refs.json`: base + `--triton-attention-num-kv-splits 16` (tree default 8). Control ref
`base`. Pattern: wave-tail-scheduling. Config-only, no code.

## Prediction (frozen; do not edit after the run)

Mechanism. The Triton decode kernel's stage 1 runs batch x heads x splits programs. At B=1
that is the only parallelism, and the profile shows sliding attention at sol_fraction 0.18
(342 µs vs a 63 µs bound): 8 splits leave most of the 170 SMs idle. 16 splits doubles the
grid; stage 2 reduces twice as many partials, which is cheap at 1024 context. At B=8 the grid
is already 8x larger and sliding attention runs at 0.72, so the gain is smaller. The W2
screen (unpaired) saw B=8 decode −1.1% and B=1 −2.0%; splits 4 was +1.0% / +4.8%, the same
direction.

| metric (bs2, ratio of sums, control/candidate) | predicted |
|---|---|
| W8 prefill (sum of TTFT) | 0 ± 0.5% (decode-only kernel) |
| W8 decode | −0.6 … −1.5% |
| **W8 composite gain** | **1.004 … 1.012** (point 1.008) |
| **W1 TPOT** | **−1.5 … −2.5%** |
| Fidelity | pass: the split count reorders the softmax/accumulation reduction (reorders_reduction), within the calibrated launch-to-launch KL |

Falsified if W1 TPOT improves by less than 1.0% or the W8 decode term gets slower.

Decision rule: keep only if a gated metric clears its bs2 A/A bar and fidelity passes. The
composite is predicted at or just under a 1% bar; W1 TPOT is predicted to clear it. If only
W1 clears, the verdict says so explicitly (a B=1-latency keep, W8-neutral).
