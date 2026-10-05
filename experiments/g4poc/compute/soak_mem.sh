#!/bin/bash
# One soak sweep with GPU memory sampled every 100 ms and checked against the deployable rule (compute/mem_check.py).
# Same as compute/psoak_run.sh, for any ref, load and memory scope.
#   [HARNESS=<dir>] compute/soak_mem.sh <ref> <load> <concurrency> <scope, e.g. 28G> <out name>
#     > /home/jooman/g4poc/logs/pb-<out name>.log 2>&1
set -uo pipefail
here="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$here/gate/env.sh"
cd "$here"
ref=$1 load=$2 conc=$3
export G4POC_SERVER_MEMORY_MAX=$4
out=$G4POC_RUNS_DIR/$5
mkdir -p "$out"
step() { echo "=== $(date -Is) $*"; }
step "$load $ref at $conc ($G4POC_SERVER_MEMORY_MAX)"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$out/mem.csv" &
smi=$!
python -m gate sweep --ref "$ref" --load "$load" --concurrency "$conc"
kill $smi
python compute/mem_check.py --samples "$out/mem.csv" | tee "$out/mem.json"
step "soak_mem done"
