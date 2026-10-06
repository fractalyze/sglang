#!/bin/bash
# K2 queue 11 (bs3): K2-c1's think-time capacity probe on the chat final with glue. The plans at C84 and C92 offer
# ~80 and ~85 live sessions (2.89 / 2.99 turns/s; compute/plan_offer.py), past the glue-only edge at C76 (71.9 live,
# p90 10.05 s). One server, final-mem-c1-c2a-glue-c1, pthink30 at 84 and 92, under /data/jooman/g4poc/unit.lock.
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
OUT=$R/k2c1-t30
U=/data/jooman/g4poc/unit.lock
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
export G4POC_SERVER_MEMORY_MAX=24G
step "T30 capacity probe 84,92 (waiting for $U)"
flock $U python -m gate sweep --ref final-mem-c1-c2a-glue-c1 --load pthink30 --concurrency 84,92 > $OUT/pthink30-capacity.log 2>&1
step "exit $? $(ls -td $R/sweep-final-mem-c1-c2a-glue-c1-2026* | head -1)"
