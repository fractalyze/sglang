#!/bin/bash
# Cross-host replicate of the chat sizing headline (~70 sessions per GPU at 30 s think): device-only final-mem-c1-c2a
# under psoak30 (poisson sessions, 30 s mean think, 30-min window) at C72 (~68-73 live sessions, 2.33 turns/s
# offered; compute/plan_offer.py), GPU memory sampled against the deployable rule. PC2 runs the same on bs3. 24G.
#   [HARNESS=<dir>] compute/psoak_run.sh > /home/jooman/g4poc/logs/pb-psoak.log 2>&1
set -uo pipefail
here="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=24G
out=$G4POC_RUNS_DIR/psoak
mkdir -p "$out"
step() { echo "=== $(date -Is) $*"; }
step "psoak30 final-mem-c1-c2a at 72"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$out/mem.csv" &
smi=$!
python -m gate sweep --ref final-mem-c1-c2a --load psoak30 --concurrency 72
kill $smi
python compute/mem_check.py --samples "$out/mem.csv" | tee "$out/mem.json"
step "psoak done"
