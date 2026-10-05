#!/bin/bash
# C1 on bs2, end to end after the pruned tune (moe-tune/c1/run2.sh): the gate's one-time
# fidelity calibration and A/A noise at inflight-C12, a mid-size tune, the merged config's
# kernel bench, then the C1 gate at inflight-C12 and the A-B-B-A check at 8 in flight.
# Steps whose output exists are skipped, so a rerun resumes.
#   compute/c1_chain.sh > /home/jooman/g4poc/logs/pb-c1-chain.log 2>&1
#
# The mid tune covers 768 and 1536 tokens: SGLang uses the nearest tuned token count, and
# with 1024 and 2048 tuned, 1536 took the 1024 config, 20% slower than the default config
# in the first kernel bench (666 -> 802 us).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
C1=$G4POC/moe-tune/c1
CFG=$G4POC/moe-configs/c1
T=$G4/trees/91132098df43
ARGS=(--model $G4POC/models/bf16-meta --tp-size 1 --dtype fp8_w8a8 --per-channel-quant)
step() { echo "=== $(date -Is) $*"; }
cap() { flock "$G4POC_HOST_LOCK" flock "$G4POC_GPU_LOCK" systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 "$@"; }

grep -q "small rc=0" "$C1/driver2.log" && grep -q "large rc=0" "$C1/driver2.log"

if [ ! -f "$G4POC/reference/fidelity_thresholds.json" ]; then
  step "calibrate"
  python -m gate calibrate --ref base
fi

if [ ! -f "$G4POC/reference/noise.json" ]; then
  step "A/A at inflight-C12"
  python -m gate run --control base --candidate base --pairs 4 --label aa-c12
  aa=$(ls -td "$G4POC_RUNS_DIR"/aa-c12-* | head -1)
  python -m gate set-noise --report "$aa/report.json"
fi

name=$(cd "$C1/small" && ls E=*.json)
if [ ! -f "$C1/mid/$name" ]; then
  step "mid tune (768, 1536 tokens)"
  mkdir -p "$C1/mid" && (cd "$C1/mid" && cap python "$here/compute/moe_tune.py" --tree $T -- "${ARGS[@]}" --tune \
    --search-space-file "$C1/space-large.json" --batch-sizes 768 1536 > tune.log 2>&1)
fi

step "merge"
ver=$(python -c "import triton; print('triton_' + triton.__version__.replace('.', '_'))")
python compute/merge_moe_configs.py --out "$CFG/configs/$ver/$name" "$C1/small/$name" "$C1/large/$name" "$C1/mid/$name"
cat "$CFG/configs/$ver/$name"

step "kernel bench, tuned config"
SGLANG_MOE_CONFIG_DIR=$CFG cap python compute/moe_tune.py --tree $T -- "${ARGS[@]}" > "$C1/bench-tuned-merged.log" 2>&1
paste <(grep "Kernel time" "$C1/bench-default.log") <(grep "Kernel time" "$C1/bench-tuned-merged.log")

step "C1 gate"
python -m gate run --control base --candidate c1-moe-tuned --pairs 4 --label c1-moe-tuned \
  --notes "g4poc-c1 prediction (compute/PREREG.md): E2E p90 -6..-1% at inflight-C12"

step "C1 at 8 in flight (A-B-B-A sweeps)"
compute/sweep_abba.sh base c1-moe-tuned 8 "$G4POC_RUNS_DIR/c1-c8-abba.json"
step "done"
