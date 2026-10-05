#!/bin/bash
# Think-time sizing on bs2, the cross-host check of PC2's bs3 poisson runs: the final (final-hc-cp2048-lpm) with
# poisson think30 sessions at 48/64/80/96 concurrent-session targets, then the same flags without HiCache
# (final-mem-c1-c2a-cp2048-lpm) at 64/80. 28G scope; a sweep starts only if it can end by 07:55 KST.
#   compute/pthink.sh > /home/jooman/g4poc/logs/pb-pthink.log 2>&1
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
step() { echo "=== $(date -Is) $*"; }
fits() {  # fits <minutes>: the run would end by 07:55 today (or it is still the previous evening)
  local end=$(( $(date +%s) + $1 * 60 )) stop=$(date -d "07:55" +%s)
  [ "$(date +%H)" -ge 12 ] || [ "$end" -le "$stop" ]
}
if fits 55; then
  step "final-hc-cp2048-lpm, pthink30 at 48, 64, 80, 96 sessions"
  python -m gate sweep --ref final-hc-cp2048-lpm --load pthink30 --concurrency 48,64,80,96
fi
if fits 28; then
  step "final-mem-c1-c2a-cp2048-lpm (no HiCache), pthink30 at 64, 80 sessions"
  python -m gate sweep --ref final-mem-c1-c2a-cp2048-lpm --load pthink30 --concurrency 64,80
fi
step "pthink done"
