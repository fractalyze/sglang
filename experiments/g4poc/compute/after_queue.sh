#!/bin/bash
# bs2 after pb_queue.sh: the C4 decode-split microbench (2 min), then the final run for <final ref>.
#   compute/after_queue.sh <final ref> > /home/jooman/g4poc/logs/pb-after-queue.log 2>&1
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
final=$1
step() { echo "=== $(date -Is) $*"; }
cap() { flock "$G4POC_HOST_LOCK" flock "$G4POC_GPU_LOCK" systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 "$@"; }

step "wait for pb_queue.sh"
until grep -q "=== .* queue done" /home/jooman/g4poc/logs/pb-queue.log; do sleep 60; done
if [ ! -f "$G4POC/c4/decode_split.json" ]; then
  step "C4 decode-split microbench"
  mkdir -p "$G4POC/c4"
  tree=$(python -c "from gate import server; print(server.tree_for(server.load_ref('$final')['commit']))")
  cap env PYTHONPATH="$tree/python" python compute/decode_split_bench.py --out "$G4POC/c4/decode_split.json"
fi
step "final run ($final)"
compute/final_run.sh "$final"
step "after_queue done"
