#!/bin/bash
# K2 queue 6 (bs3): after the c1 A-B-B-A and K1's U2. Waits for the done file, then until K1's U2 holds the outer
# lock (or 10 min pass), then runs the c1 exactness pair and the think-time pair, each under
# /data/jooman/g4poc/unit.lock, and then queue 5 (c1 follow-ups, c2 gates).
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
OUT=$R/k2c1
U=/data/jooman/g4poc/unit.lock
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
preread() { cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null; }
until [ -e /data/jooman/g4poc/logs/k2-c1-abba.done ]; do sleep 30; done
step "done file seen; waiting for K1's U2 to take $U (max 10 min)"
for i in $(seq 1 60); do flock -n $U true || break; sleep 10; done
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $OUT/smi-q6.csv &
SMI=$!
export G4POC_SERVER_MEMORY_MAX=28G
step "c1 exactness pair (waiting for $U)"
preread
flock $U bash -c "
  G4POC_SERVER_MEMORY_MAX=24G python -m hicache.exactness_mt run --ref final-mem-c1-c2a-cp2048-lpm-k2c1 > $OUT/exact-ctl.log 2>&1
  python -m hicache.exactness_mt run --ref final-hc-cp2048-lpm-k2c1-smallpool > $OUT/exact-cand.log 2>&1"
c=$(ls -td $R/exactmt-final-mem-c1-c2a-cp2048-lpm-k2c1-2026* | head -1); f=$(ls -td $R/exactmt-final-hc-cp2048-lpm-k2c1-smallpool-2026* | head -1)
python -m hicache.exactness_mt compare $c/exactness_mt.json $f/exactness_mt.json > $OUT/exact-compare.json
step "c1 exactness $(python -c "import json; d=json.load(open('$OUT/exact-compare.json')); print(d['exact'], '/', d['n'], 'same_cached', d['same_cached_tokens'])")"
export G4POC_SERVER_MEMORY_MAX=24G
step "c1 T30 pair (waiting for $U)"
flock $U bash -c "
  for ref in final-mem-c1-c2a final-mem-c1-c2a-k2c1; do
    echo '=== '\$(date +%T)\" pthink30 C72 \$ref\"
    python -m gate sweep --ref \$ref --load pthink30 --concurrency 72 > $OUT/pthink30-\$ref.log 2>&1
    echo '=== '\$(date +%T)\" exit \$? \$(ls -td $R/sweep-\$ref-2026* | head -1)\"
  done"
kill $SMI
step "c1 units done; queue 5 next"
exec bash compute/k2/queue5.sh
