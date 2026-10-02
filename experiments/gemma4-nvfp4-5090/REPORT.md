# W1 report: verify gate + pinned baseline (gemma4nv)

**Outcome: partial.** The harness, the pinned configuration and the vault entities are done. No
baseline number exists: build-server-3 and then build-server-2 went offline within minutes of a
server launch, and neither had come back after about an hour.

## Done

- **Branch** `jumanzii/gemma4nv-gate`, pushed to `fractalyze`:
  - cb9bf3ddb: the harness
  - 2778e4245: server memory cap and low-RAM refusal
  - a2b022b51: BASELINE.md
  - `python/sglang` is unmodified.
- **Gate harness** (`gate/`, run as `bin/gate run --control base --candidate <ref>`):
  - ABBA legs with one server lifetime per leg, ratio of sums, per-pair skew and an A/A noise bar of max(3 sigma, 1%).
  - W8 composite, W1 TPOT, and W32 throughput (ungated).
  - Quiescence gate and in-window telemetry.
  - Leg refusals: eager decode steps in the window, foreign GPU processes, weight-checksum change, undeclared server-arg diffs, and timed-output disagreement.
  - Fidelity: token match plus top-20 KL with calibrated thresholds.
  - `gate quality`: GSM8K 200 plus 40 tool-call JSON requests.
  - `gate peaks`: measured bandwidth, GEMM peaks and launch floor.
  - `gate sol`: the SOL model from the safetensors bytes, distinct routed experts and KV, reported as sol_fraction.
  - 25 unit tests pass.
- **Hidden fidelity set** on bs2 at `/data/jooman/gemma4nv/hidden/` (never committed):
  - 22 prompts, 5 of them long (8k to 15.5k tokens); sha256 `424d7d5f...`.
  - The timing corpus cache is also built.
- **Baseline config** (BASELINE.md):
  - SGLang a9871012a with model rev a19cfe00 and FP8 KV.
  - Triton attention and BF16 dense GEMMs.
  - `--moe-runner-backend flashinfer_cutlass`. The tree's `auto` picks `flashinfer_trtllm` on SM120, and that crashes with `g1_scale_c`.
- **Vault** (`wm doctor` OK), committed:
  - Model page `gemma-4-26b-a4b-nvfp4`.
  - Workload pages `wl-gemma4nv-b8-p1024-d128`, `-b1-p1024-d256` and `-c32-p1024-d128`.
  - Study page `study-gemma4-nvfp4-5090`.
- **vLLM 0.20.0 reference venv** installed on bs2 at `/home/jooman/gemma4nv/vllm-venv`.

## Not done (needs a GPU host)

1. Confirm that `flashinfer_cutlass` serves on SM120. It was the launch config when each host went down.
2. `gate calibrate`, `gate quality --ref base --set-baseline`, `gate peaks`, `gate sol --ref base`.
3. `gate run --control base --candidate base --pairs 6`, then `gate set-noise` and `gate sol-report`. Fill BASELINE.md.
4. Run the vLLM reference on W8 and W1.
5. Vault follow-ups:
   - The baseline stack state (`wm intake state --ref a9871012a`) needs the measured profile.
   - The raw-import entry for `/data/jooman/gemma4nv/{ledger/evaluations.jsonl,runs/*/report.json}` needs the schema owner. The ledger and run json are written in the moemem-style evaluations-index shape.

## Risk

The two outages came right after launches on both hosts, so they are probably caused by the launch: host OOM during load, or a driver fault on 610.43. Before any relaunch:

- Get kern.log or dmesg from a rebooted host.
- Launch only inside the harness's capped scope (MemoryMax=28G, no swap, MemAvailable of at least 30 GB).
