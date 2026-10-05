#!/bin/bash
# bs2's last PB runs, after PC4's queue (WAIT_PID): the final at 8 and 12 in flight (its 6 s SLO point), then
# final-hc at the final's poisson think30 load (48 and 64 sessions), a same-host A/B of chunk 2048 + lpm in think
# mode -- the final's first point held a prefix hit of 0.24 against 0.46 for final-hc on bs3, and chunk boundaries
# add sliding-window host writes. Each sweep starts only if its estimate ends by STOP (the 07:55 hard stop); the
# A/B goes first when time is short. 28G.
#   [HARNESS=<dir>] [WAIT_PID=<pid>] [STOP=HH:MM] compute/pthink_ab.sh > /home/jooman/g4poc/logs/pb-pthink-ab.log 2>&1
set -uo pipefail
here="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
step() { echo "=== $(date -Is) $*"; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$(date -d "${STOP:-07:55}" +%s)" ]; }
if [ -n "${WAIT_PID:-}" ]; then
  step "wait for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi
if fits 16; then
  step "final-hc-cp2048-lpm at 8, 12 in flight (its 6 s SLO point)"
  python -m gate sweep --ref final-hc-cp2048-lpm --load inflight --concurrency 8,12
else
  step "skip C8/C12: would end past ${STOP:-07:55}"
fi
if fits 30; then
  step "final-hc, pthink30 at 48, 64 sessions (A/B against the final's chunk 2048 + lpm)"
  python -m gate sweep --ref final-hc --load pthink30 --concurrency 48,64
else
  step "skip final-hc pthink30: would end past ${STOP:-07:55}"
fi
step "pthink_ab done"
