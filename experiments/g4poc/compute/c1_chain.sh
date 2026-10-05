#!/bin/bash
# C1 on bs2, end to end once the pruned tune (moe-tune/c1/run2.sh) finishes:
# merge the tuned parts, kernel-bench tuned vs default, then the gate's one-time
# fidelity calibration and A/A noise at inflight-C12, then the C1 gate.
#   compute/c1_chain.sh > /home/jooman/g4poc/logs/pb-c1-chain.log 2>&1
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
C1=$G4POC/moe-tune/c1
CFG=$G4POC/moe-configs/c1
T=$G4/trees/91132098df43
step() { echo "=== $(date -Is) $*"; }
cap() { flock "$G4POC_HOST_LOCK" flock "$G4POC_GPU_LOCK" systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 "$@"; }

step "wait for tune"
until grep -q "tune2 done" "$C1/driver2.log"; do sleep 60; done
grep -q "small rc=0" "$C1/driver2.log" && grep -q "large rc=0" "$C1/driver2.log"

step "merge"
name=$(cd "$C1/small" && ls E=*.json)
ver=$(python -c "import triton; print('triton_' + triton.__version__.replace('.', '_'))")
python compute/merge_moe_configs.py --out "$CFG/configs/$ver/$name" "$C1/small/$name" "$C1/large/$name"
cat "$CFG/configs/$ver/$name"

step "kernel bench, tuned config"
SGLANG_MOE_CONFIG_DIR=$CFG cap python compute/moe_tune.py --tree $T -- \
  --model $G4POC/models/bf16-meta --tp-size 1 --dtype fp8_w8a8 --per-channel-quant > "$C1/bench-tuned.log" 2>&1
paste <(grep "Kernel time" "$C1/bench-default.log") <(grep "Kernel time" "$C1/bench-tuned.log")

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

step "C1 gate"
python -m gate run --control base --candidate c1-moe-tuned --pairs 4 --label c1-moe-tuned \
  --notes "g4poc-c1 prediction (compute/PREREG.md): E2E p90 -6..-1% at inflight-C12"
step "done"
