#!/bin/bash
# K2 queue 9 (bs3): K2-c1's think-time pair on the chat final with K1's glue (coordinator, 10-06 ~14:45): pthink30 at
# 72 and 76 sessions (C76 ~73 live, the capacity edge), device-only final-mem-c1-c2a + glue without then with c1, the
# same seeded plans (gate sweep seeds each C's plan by name). One unit under /data/jooman/g4poc/unit.lock; 24G scope.
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
OUT=$R/k2c1-t30
U=/data/jooman/g4poc/unit.lock
mkdir -p $OUT
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
export G4POC_SERVER_MEMORY_MAX=24G
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $OUT/smi.csv &
SMI=$!
step "T30 pair at 72,76 (waiting for $U)"
flock $U bash -c "
  for ref in final-mem-c1-c2a-glue-k2 final-mem-c1-c2a-glue-c1; do
    echo \"=== \$(date +%T) pthink30 72,76 \$ref\"
    python -m gate sweep --ref \$ref --load pthink30 --concurrency 72,76 > $OUT/pthink30-\$ref.log 2>&1
    echo \"=== \$(date +%T) exit \$? \$(ls -td $R/sweep-\$ref-2026* | head -1)\"
    ls -td $R/sweep-\$ref-2026* | head -1 >> $OUT/dirs.txt
  done"
python compute/k2_smi_windows.py --samples $OUT/smi.csv --runs $(cat $OUT/dirs.txt) > $OUT/smi-runs.json 2>&1
kill $SMI
step "queue9 done"
