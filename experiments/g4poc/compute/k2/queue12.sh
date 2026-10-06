#!/bin/bash
# K2 queue 12 (bs3), after the c1 C28 soak: the think-time edge of final-mem-c1-c2a + glue + c1 at the C96 plan
# (~96 live sessions, 3.27 turns/s offered; compute/plan_offer.py). C92 (84.4 live) met 10 s at p90 8.77 s. If C96
# meets 10 s too, >= 96 is reported as a lower bound (coordinator, 10-06 ~17:10). One unit under unit.lock; 24G scope.
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
OUT=$R/k2c1-t30
U=/data/jooman/g4poc/unit.lock
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
export G4POC_SERVER_MEMORY_MAX=24G
step "T30 edge C96 (waiting for $U)"
flock $U python -m gate sweep --ref final-mem-c1-c2a-glue-c1 --load pthink30 --concurrency 96 > $OUT/pthink30-c96.log 2>&1
step "exit $? $(ls -td $R/sweep-final-mem-c1-c2a-glue-c1-2026* | head -1)"
