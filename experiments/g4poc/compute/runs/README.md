# PB compute-lever records (bs2), copied from the host

Small result files behind COMPUTE.md and the vault trials `g4poc-c1/c2/c3*`. The full run
directories (server logs, per-leg data, traces) stay on build-server-2 under
`/home/jooman/g4poc/runs/<same name>`; the hidden fidelity prompts never leave the host.

- `aa-c12-*/report.json`: A/A of the base at inflight-C12 (noise and bars of every C12 gate).
- `c1-moe-tuned-*/report.json`: C1 gate; `c1-c8-abba.json`: C1 at 8 in flight (A-B-B-A sweeps).
- `c1-moe-config/`: the fused_moe kernel bench, default vs tuned (the config itself: `../moe-configs/c1/`).
- `profile-base-C12-*/extend_share.json`: extend-attention share of GPU time at C12 (C2-A sizing).
- `c2/bench.json`: extend tile microbench; `c2/accuracy.json`: tiles vs fp32 attention.
- `kl-c2-extend-tiles*/kl_check.json`: C2-A long role-play KL check, fast and exact tile tables.
- `reference/`: the gate's fidelity thresholds (calibrate) and noise file (from the A/A).
- `c3-nested/`, `c4-nested/`: nested and confirming A-B-B-A sweeps of the C3 flags and C4 (base, then final-hc).
- `final-final-hc/`, `morning-final-hc-cp2048-lpm/`: final-stack runs (memory samples, quality compare).
- `sweep-*/sweep.json`: in-flight and think-time sweeps named by ref, date and host; each COMPUTE.md table names its
  run. `quality-*/quality.json`: GSM8K + tool JSON runs; `kl-final-hc-cp2048-lpm-*`: the final's KL check.
- `fleet/`: drop-idle sweep input and fleet-model outputs (v3, v4 with every poisson point).
- `m128/`: PC4's SWA margin on the final: two A-B-B-As at 28 in flight, quality compare, soak memory check.
- `psoak/`: the device-only 30-min poisson chat soak at 72 (memory check).
