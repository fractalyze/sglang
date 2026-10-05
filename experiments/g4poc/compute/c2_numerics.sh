#!/bin/bash
# C2-A numerics before its gate, on bs2: the tiles' error against fp32 attention (default, new and
# exact tables), then the long role-play KL check for both candidate refs.
#   compute/c2_numerics.sh > /home/jooman/g4poc/logs/pb-c2-numerics.log 2>&1
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
out=$G4POC/c2
step() { echo "=== $(date -Is) $*"; }
cap() { flock "$G4POC_HOST_LOCK" flock "$G4POC_GPU_LOCK" systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 "$@"; }

step "accuracy against fp32"
cap env PYTHONPATH=$G4/trees/a10694ad326c/python python compute/extend_tiles_accuracy.py --out "$out/accuracy.json"
for ref in c2-extend-tiles c2-extend-tiles-exact; do
  step "KL check $ref"
  python compute/kl_check.py --candidate "$ref"
done
step "done"
