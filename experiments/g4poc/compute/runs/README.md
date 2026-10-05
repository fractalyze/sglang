# PB compute-lever records (bs2), copied from the host

Small result files behind COMPUTE.md and the vault trials `g4poc-c1/c2/c3*`. The full run
directories (server logs, per-leg data, traces) stay on build-server-2 under
`/home/jooman/g4poc/runs/<same name>`; the hidden fidelity prompts never leave the host.

- `aa-c12-*/report.json`: A/A of the base at inflight-C12 (noise and bars of every C12 gate).
- `c1-moe-tuned-*/report.json`: C1 gate; `c1-c8-abba.json`: C1 at 8 in flight (A-B-B-A sweeps).
- `c1-moe-config/`: the merged tuned fused_moe config and the kernel bench, default vs tuned.
- `profile-base-C12-*/extend_share.json`: extend-attention share of GPU time at C12 (C2-A sizing).
- `c2/bench.json`: extend tile microbench; `c2/accuracy.json`: tiles vs fp32 attention.
- `kl-c2-extend-tiles*/kl_check.json`: C2-A long role-play KL check, fast and exact tile tables.
- `reference/`: the gate's fidelity thresholds (calibrate) and noise file (from the A/A).
