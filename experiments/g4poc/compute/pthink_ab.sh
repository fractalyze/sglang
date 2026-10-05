#!/bin/bash
# After the final's poisson think30 sweep on bs2 (compute/pthink.sh, first sweep): final-hc at the same load (48 and
# 64 sessions), a same-host A/B of chunk 2048 + lpm in think mode -- the final's first point held a prefix hit of
# 0.24 against 0.46 for final-hc on bs3, and chunk boundaries add sliding-window host writes -- then the final at 8
# and 12 in flight (its 6 s SLO point). Each sweep starts only if it ends by STOP (04:45: bs2 goes to PC4). 28G.
#   [HARNESS=<dir>] [WAIT_PID=<pid>] compute/pthink_ab.sh > /home/jooman/g4poc/logs/pb-pthink-ab.log 2>&1
set -uo pipefail
here="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
step() { echo "=== $(date -Is) $*"; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$(date -d "${STOP:-04:45}" +%s)" ]; }
if [ -n "${WAIT_PID:-}" ]; then
  step "wait for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi
if fits 28; then
  step "final-hc, pthink30 at 48, 64 sessions (A/B against the final's chunk 2048 + lpm)"
  python -m gate sweep --ref final-hc --load pthink30 --concurrency 48,64
fi
if fits 14; then
  step "final-hc-cp2048-lpm at 8, 12 in flight (its 6 s SLO point)"
  python -m gate sweep --ref final-hc-cp2048-lpm --load inflight --concurrency 8,12
fi
step "pthink_ab done"
