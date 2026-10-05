#!/bin/bash
# bs2's last PB run: the final at 8 and 12 in flight (its 6 s SLO point), after every process running from WAIT_DIR
# (PC4's harness) has exited, checked twice a minute apart because one PC4 script may start the next. Starts only if
# it ends by STOP (the 07:55 hard stop). 28G.
#   [HARNESS=<dir>] [WAIT_DIR=<dir>] [STOP=HH:MM] compute/final_6s.sh > /home/jooman/g4poc/logs/pb-final-6s.log 2>&1
set -uo pipefail
here="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
step() { echo "=== $(date -Is) $*"; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$(date -d "${STOP:-07:55}" +%s)" ]; }
busy() { pgrep -f "$WAIT_DIR" | grep -vqx "$$"; }
if [ -n "${WAIT_DIR:-}" ]; then
  step "wait for processes under $WAIT_DIR"
  while busy || { sleep 60; busy; }; do sleep 30; done
fi
if fits 16; then
  step "final-hc-cp2048-lpm at 8, 12 in flight (its 6 s SLO point)"
  python -m gate sweep --ref final-hc-cp2048-lpm --load inflight --concurrency 8,12
else
  step "skip C8/C12: would end past ${STOP:-07:55}"
fi
step "final_6s done"
