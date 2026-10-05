#!/bin/bash
# PB's GPU queue on bs2 after C1: the C2-A gate at inflight-C12 and its A-B-B-A sweeps at 8 and 16
# in flight, the C3 nested sweeps (lpm, chunk 2048) at 8-20, then the base's P/D measurement.
# Steps whose output exists are skipped, so a rerun resumes.
#   compute/pb_queue.sh <C2-A ref> > /home/jooman/g4poc/logs/pb-queue.log 2>&1
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
c2=$1
R=$G4POC_RUNS_DIR
step() { echo "=== $(date -Is) $*"; }

if ! ls -d "$R"/c2-gate-*/report.json >/dev/null 2>&1; then
  step "C2-A gate ($c2) at inflight-C12"
  python -m gate run --control base --candidate "$c2" --pairs 4 --label c2-gate \
    --notes "g4poc-c2 prediction (compute/PREREG.md): E2E p90 -12..-5% at inflight-C12"
fi
if [ ! -f "$R/c2-c8-c16-abba.json" ]; then
  step "C2-A at 8 and 16 in flight (A-B-B-A sweeps)"
  compute/sweep_abba.sh base "$c2" 8,16 "$R/c2-c8-c16-abba.json"
fi
step "C3 nested sweeps"
compute/sweep_nested.sh base "c3-lpm c3-cp2048" 8,12,16,20 "$R/c3-nested"
if ! ls -d "$R"/pd-base-*/pd.json >/dev/null 2>&1; then
  step "P/D measurement (base)"
  python -m gate pd-measure --ref base
fi
step "queue done"
