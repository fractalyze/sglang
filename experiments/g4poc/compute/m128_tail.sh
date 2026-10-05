#!/bin/bash
# The end of compute/m128_run.sh once (d) moved to bs3 (coordinator, 10-06 06:40): after the running soak (WAIT_PID)
# exits, stop its GPU-memory sampler (SMI_PID) and check the samples against the deployable rule, then (e) the second
# A-B-B-A at 28 in flight only if it ends by STOP (07:55). 28G scope.
#   [HARNESS=<dir>] WAIT_PID=<soak pid> SMI_PID=<sampler pid> [STOP=HH:MM] compute/m128_tail.sh >> .../pb-m128.log 2>&1
set -uo pipefail
here="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
out=$G4POC_RUNS_DIR/m128
step() { echo "=== $(date -Is) $*"; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$(date -d "${STOP:-07:55}" +%s)" ]; }
step "wait for the soak (pid $WAIT_PID)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
kill "$SMI_PID" 2>/dev/null
python compute/mem_check.py --samples "$out/soak-mem.csv" | tee "$out/soak-mem.json"
if fits 28; then
  step "(e) second A-B-B-A at 28 in flight"
  compute/sweep_abba.sh final-hc-cp2048-lpm final-hc-cp2048-lpm-m128 28 "$out/abba-c28-2.json"
else step "skip (e): would end past ${STOP:-07:55}"; fi
step "m128 done"
