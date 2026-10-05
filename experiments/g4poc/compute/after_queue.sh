#!/bin/bash
# bs2 after pb_queue.sh (coordinator plan 10-05 ~18:40): the C4 decode-split microbench (2 min), the final run
# for final-hc (28G scope; sweep 4-40, quality, final-mem-c1-c2a at 16-32 as the HiCache on/off pair and the
# fleet model's device-cached curve, P/D), then final-mem-c1-c2a with the radix cache off (fleet model's
# drop-idle policy).
#   compute/after_queue.sh <final ref> > /home/jooman/g4poc/logs/pb-after-queue.log 2>&1
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
final=$1
R=$G4POC_RUNS_DIR
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
step "host before the final run"
grep -E "MemAvailable|SwapTotal|SwapFree" /proc/meminfo
step "final run ($final)"
G4POC_SERVER_MEMORY_MAX=28G FINAL_CONC=4,8,12,16,20,24,28,32,40 ALONE_REF=final-mem-c1-c2a ALONE_CONC=16,20,24,28,32 \
  compute/final_run.sh "$final"
if [ ! -f "$R/final-$final/nocache-sweep.txt" ]; then
  step "fleet model: final-mem-c1-c2a with the radix cache off at 4-20"
  python -m gate sweep --ref final-mem-c1-c2a-nocache --load inflight --concurrency 4,8,12,16,20
  ls -td "$R"/sweep-final-mem-c1-c2a-nocache-* | head -1 > "$R/final-$final/nocache-sweep.txt"
fi
step "after_queue done"
