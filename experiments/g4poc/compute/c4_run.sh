#!/bin/bash
# C4 on bs2, against the final stack: nested in-flight sweeps of the decode kv-split candidates at 12 and 20
# in flight (A B C C B A), then the gate at inflight-C12 for the candidate compute/c4_pick.py selects (the
# best E2E p90 gain at 12 above 1% with no loss beyond 1% at 20; none -> no gate).
# Hard stop: nothing starts after 06:00 KST.
#   compute/c4_run.sh <control ref> "<cand1> <cand2> ..." > /home/jooman/g4poc/logs/pb-c4.log 2>&1
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
ctrl=$1 cands=$2
R=$G4POC_RUNS_DIR
step() { echo "=== $(date -Is) $*"; }
before_stop() { [ "$(date +%H%M)" -lt 0600 ] || [ "$(date +%H%M)" -ge 1200 ]; }

step "C4 nested sweeps at 12 and 20 in flight"
compute/sweep_nested.sh "$ctrl" "$cands" 12,20 "$R/c4-nested"
gated=$(python compute/c4_pick.py --dir "$R/c4-nested" $cands)
echo "picked: $gated"
if [ "$gated" != "none" ] && ! ls -d "$R"/c4-gate-*/report.json >/dev/null 2>&1; then
  if before_stop; then
    step "C4 gate ($gated vs $ctrl) at inflight-C12"
    python -m gate run --control "$ctrl" --candidate "$gated" --pairs 4 --label c4-gate \
      --notes "g4poc-c4 prediction (compute/PREREG.md), against the final stack"
  else
    echo "past 06:00 KST: C4 gate not started"
  fi
fi
step "C4 done"
