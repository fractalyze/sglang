#!/bin/bash
# C4 on bs2 against the final stack (final-hc, 28G scope): nested in-flight sweeps of the decode kv-split
# candidates at 24 and 32 in flight (A B C ... C B A), the candidate compute/c4_pick.py selects, its KL and
# serial-identity check against the control, and a confirming A-B-B-A at 24 and 32.
# Hard stop: nothing starts after 06:00 KST.
#   compute/c4_run.sh <control ref> "<cand1> <cand2> ..." > /home/jooman/g4poc/logs/pb-c4.log 2>&1
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
ctrl=$1 cands=$2
R=$G4POC_RUNS_DIR
step() { echo "=== $(date -Is) $*"; }
before_stop() { [ "$(date +%H%M)" -lt 0600 ] || [ "$(date +%H%M)" -ge 1200 ]; }

step "C4 nested sweeps at 24 and 32 in flight"
compute/sweep_nested.sh "$ctrl" "$cands" 24,32 "$R/c4-nested"
gated=$(python compute/c4_pick.py --dir "$R/c4-nested" $cands)
echo "picked: $gated"
if [ "$gated" != "none" ]; then
  if before_stop && ! ls -d "$R"/kl-"$gated"-*/kl_check.json >/dev/null 2>&1; then
    step "C4 KL / serial identity ($gated vs $ctrl)"
    python compute/kl_check.py --control "$ctrl" --candidate "$gated"
  fi
  if before_stop && [ ! -f "$R/c4-confirm.json" ]; then
    step "C4 confirming A-B-B-A ($gated vs $ctrl) at 24 and 32"
    compute/sweep_abba.sh "$ctrl" "$gated" 24,32 "$R/c4-confirm.json"
  fi
fi
step "C4 done"
