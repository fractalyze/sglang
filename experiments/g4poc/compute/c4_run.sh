#!/bin/bash
# Overnight flag candidates against the final stack on bs2 (compute/PREREG.md, "C4 and the C3 flags on the final
# stack"; 28G scope): a smoke run of each candidate (a failing one is dropped), nested in-flight sweeps at 24 and
# 32 (A B C ... C B A), the flags compute/c4_pick.py keeps, combined into one ref, its long role-play KL check
# against the control and a confirming A-B-B-A at 24 and 32. Nothing starts after 06:00 KST.
#   compute/c4_run.sh <control ref> "<cand1> <cand2> ..." > /home/jooman/g4poc/logs/pb-c4.log 2>&1
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
ctrl=$1
R=$G4POC_RUNS_DIR
step() { echo "=== $(date -Is) $*"; }
before_stop() { [ "$(date +%H%M)" -lt 0600 ] || [ "$(date +%H%M)" -ge 1200 ]; }

cands=""
for c in $2; do
  step "smoke $c"
  if python -m gate smoke --ref "$c" > "$R/c4-smoke-$c.log" 2>&1; then cands="$cands $c"; else
    echo "smoke failed: $c dropped (log $R/c4-smoke-$c.log)"; tail -5 "$R/c4-smoke-$c.log"; fi
done
cands=${cands# }
[ -n "$cands" ] || { step "no candidate survived the smoke runs"; exit 1; }
step "nested sweeps at 24 and 32 in flight: $ctrl vs $cands"
compute/sweep_nested.sh "$ctrl" "$cands" 24,32 "$R/c4-nested" || { step "nested sweeps failed"; exit 1; }
gated=$(python compute/c4_pick.py --dir "$R/c4-nested" --control "$ctrl" $cands)
echo "kept flags combined: $gated"
if [ "$gated" != "none" ]; then
  if before_stop && ! ls -d "$R"/kl-"$gated"-*/kl_check.json >/dev/null 2>&1; then
    step "KL / serial identity ($gated vs $ctrl)"
    python compute/kl_check.py --control "$ctrl" --candidate "$gated"
  fi
  if before_stop && [ ! -f "$R/c4-confirm.json" ]; then
    step "confirming A-B-B-A ($gated vs $ctrl) at 24 and 32"
    compute/sweep_abba.sh "$ctrl" "$gated" 24,32 "$R/c4-confirm.json"
  fi
fi
step "C4 done"
