#!/bin/bash
# 10-06 morning on bs2: the final's replicate sweep at 16/24/28/32/36/40 in flight (GPU memory sampled against the
# deployable rule), for a final other than final-hc (whose quality ran on 10-05) its quality against the base anchor,
# then -- if it is before 10:30 KST -- a 30-min soak at 32 in flight. 28G scope.
#   compute/morning.sh <final ref> > /home/jooman/g4poc/logs/pb-morning.log 2>&1
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
final=$1
R=$G4POC_RUNS_DIR
out=$R/morning-$final
mkdir -p "$out"
step() { echo "=== $(date -Is) $*"; }

step "host"; grep -E "MemAvailable|SwapTotal|SwapFree" /proc/meminfo
step "replicate sweep $final at 16, 24, 28, 32, 36, 40"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$out/mem.csv" &
smi=$!
python -m gate sweep --ref "$final" --load inflight --concurrency 16,24,28,32,36,40 || { kill $smi; exit 1; }
kill $smi
ls -td "$R"/sweep-"$final"-* | head -1 > "$out/sweep.txt"
python compute/mem_check.py --samples "$out/mem.csv" | tee "$out/mem.json"
if [ "$final" != "final-hc" ]; then
  step "quality $final"
  python -m gate quality --ref "$final" --gsm8k-n all --label "quality-$final"
  q=$(ls -td "$R"/quality-"$final"-* | head -1)
  b=$(ls -td "$R"/quality-base-anchor-* | head -1)
  python -m gate quality-compare --control "$b/quality.json" --candidate "$q/quality.json" | tee "$out/quality-compare.json"
fi
if [ "$(date +%H%M)" -lt 1030 ]; then
  step "soak $final at 32 in flight (30 min)"
  python -m gate sweep --ref "$final" --load soak --concurrency 32
fi
step "morning done"
