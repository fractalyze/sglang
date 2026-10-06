#!/bin/bash
# One K1 performance unit (glue/PREREG.md), run whole on the host it starts on: an A-B-B-A is never split across
# hosts, so each unit's control drift and pairs come from one GPU. Outputs carry the host name.
#   abba-inflight: final-hc-cp2048-lpm vs -glue at 12 and 28 in flight (28G)
#   abba-t30:      final-mem-c1-c2a vs -glue under pthink30 at 72 (24G)
#   soak:          final-hc-cp2048-lpm-glue at 28 in flight for 30 min (28G)
#   capacity-t30:  the T30 pair again at pthink30 76, whose seeded plan offers ~73 live sessions, the round-1 chat
#                  final's 10 s SLO edge (a capacity measurement after the gates, not a gate)
# The unit holds $G4POC/unit.lock throughout, so units of different workers on one host never interleave their sweeps
# (the gate's host.lock is per server); WAIT_PID first waits out a running queue that predates that lock. flock is not
# FIFO, so a unit the coordinator orders behind another worker's waits for WAIT_FILE (that worker touches it when done)
# for at most WAIT_TIMEOUT_S before taking the lock.
#   [WAIT_PID=<pid>] [WAIT_FILE=<path> [WAIT_TIMEOUT_S=5400]] glue/k1_unit.sh <unit> > <log> 2>&1
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
out=${G4POC_RUNS_DIR:-$G4POC/runs}/k1
mkdir -p "$out"
host=$(hostname)
if [ -n "${WAIT_PID:-}" ]; then
  echo "=== $(date -Is) unit $1 waits for pid $WAIT_PID to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
if [ -n "${WAIT_FILE:-}" ]; then
  deadline=$(( $(date +%s) + ${WAIT_TIMEOUT_S:-5400} ))
  echo "=== $(date -Is) unit $1 waits for $WAIT_FILE (until $(date -d @$deadline +%T))"
  while [ ! -e "$WAIT_FILE" ] && [ "$(date +%s)" -lt "$deadline" ]; do sleep 30; done
  [ -e "$WAIT_FILE" ] && echo "=== $(date -Is) $WAIT_FILE present" || echo "=== $(date -Is) $WAIT_FILE timed out"
fi
exec 8>"$G4POC/unit.lock"
echo "=== $(date -Is) unit $1 waits for $G4POC/unit.lock"
flock 8
echo "=== $(date -Is) unit $1 on $host"
case "$1" in
  abba-inflight)
    cat "$G4POC_MODEL_DIR"/../shards/text-*.safetensors > /dev/null 2>&1 || true
    G4POC_SERVER_MEMORY_MAX=28G compute/sweep_abba.sh final-hc-cp2048-lpm final-hc-cp2048-lpm-glue 12,28 \
      "$out/abba-c12-c28-$host.json" ;;
  abba-t30)
    LOAD=pthink30 G4POC_SERVER_MEMORY_MAX=24G compute/sweep_abba.sh final-mem-c1-c2a final-mem-c1-c2a-glue 72 \
      "$out/abba-t30-$host.json" ;;
  capacity-t30)
    LOAD=pthink30 G4POC_SERVER_MEMORY_MAX=24G compute/sweep_abba.sh final-mem-c1-c2a final-mem-c1-c2a-glue 76 \
      "$out/capacity-t30-c76-$host.json" ;;
  soak)
    glue/k1_soak.sh ;;
  *)
    echo "unknown unit $1" >&2; exit 2 ;;
esac
rc=$?
echo "=== $(date -Is) unit $1 on $host done rc=$rc"
exit $rc
