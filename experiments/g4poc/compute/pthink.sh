#!/bin/bash
# Think-time sizing on bs2, the cross-host check of PC2's bs3 poisson runs: the final (final-hc-cp2048-lpm) with
# poisson think30 sessions at 48/64/80/96 concurrent-session targets, then the same flags without HiCache
# (final-mem-c1-c2a-cp2048-lpm) at 64/80 only if it can end by PAIR_STOP (04:45: bs2 goes to PC4 then; PC2's
# bs3 poisson runs cover the no-HiCache arm). 28G scope; the final's sweep starts only if it can end by 07:55 KST.
#   [HARNESS=<dir>] compute/pthink.sh > /home/jooman/g4poc/logs/pb-pthink.log 2>&1
set -uo pipefail
here="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
step() { echo "=== $(date -Is) $*"; }
fits() {  # fits <minutes> [HH:MM]: the run would end by HH:MM (default 07:55) today, or it is still the evening
  local end=$(( $(date +%s) + $1 * 60 )) stop=$(date -d "${2:-07:55}" +%s)
  [ "$(date +%H)" -ge 12 ] || [ "$end" -le "$stop" ]
}
if fits 55; then
  step "final-hc-cp2048-lpm, pthink30 at 48, 64, 80, 96 sessions"
  python -m gate sweep --ref final-hc-cp2048-lpm --load pthink30 --concurrency 48,64,80,96
fi
if fits 28 "${PAIR_STOP:-04:45}"; then
  step "final-mem-c1-c2a-cp2048-lpm (no HiCache), pthink30 at 64, 80 sessions"
  python -m gate sweep --ref final-mem-c1-c2a-cp2048-lpm --load pthink30 --concurrency 64,80
fi
step "pthink done"
