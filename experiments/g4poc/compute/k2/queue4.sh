#!/bin/bash
# K2 queue 4 (bs3): the K2-c1 multi-server units, each held contiguous by the outer lock /data/jooman/g4poc/unit.lock
# (shared with K1; the gate still takes host.lock per server inside it): the A-B-B-A at 12 and 28 in flight (rerun:
# queue 3's stopped after its first sweep on an unset G4POC_RUNS_DIR), the C1 multi-turn exactness pair, and the
# think-time pair (pthink30 at 72 sessions, device-only without then with c1). Waits for queue 3's last single run.
# GPU clocks, power and temperature are sampled every second (smi-q4.csv); the sampler is stopped by its PID.
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
OUT=$R/k2c1
U=/data/jooman/g4poc/unit.lock
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
preread() { cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null; }
while pgrep -f "gate rp-qualit[y] --ref final-cpl-qr-k2c1" > /dev/null; do sleep 30; done
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $OUT/smi-q4.csv &
SMI=$!
export G4POC_SERVER_MEMORY_MAX=28G
step "A-B-B-A 12,28: final-hc-cp2048-lpm vs final-hc-cp2048-lpm-k2c1 (waiting for $U)"
preread
flock $U bash -c "echo '=== '\$(date +%T)' unit lock held'; compute/sweep_abba.sh final-hc-cp2048-lpm final-hc-cp2048-lpm-k2c1 12,28 $OUT/abba-c12-c28.json"
step "A-B-B-A exit $?"
step "exactness pair (waiting for $U)"
preread
flock $U bash -c "
  G4POC_SERVER_MEMORY_MAX=24G python -m hicache.exactness_mt run --ref final-mem-c1-c2a-cp2048-lpm-k2c1 > $OUT/exact-ctl.log 2>&1
  python -m hicache.exactness_mt run --ref final-hc-cp2048-lpm-k2c1-smallpool > $OUT/exact-cand.log 2>&1"
c=$(ls -td $R/exactmt-final-mem-c1-c2a-cp2048-lpm-k2c1-2026* | head -1); f=$(ls -td $R/exactmt-final-hc-cp2048-lpm-k2c1-smallpool-2026* | head -1)
python -m hicache.exactness_mt compare $c/exactness_mt.json $f/exactness_mt.json > $OUT/exact-compare.json
step "exactness $(python -c "import json; d=json.load(open('$OUT/exact-compare.json')); print(d['exact'], '/', d['n'], 'same_cached', d['same_cached_tokens'])")"
export G4POC_SERVER_MEMORY_MAX=24G
step "T30 pair (waiting for $U)"
flock $U bash -c "
  for ref in final-mem-c1-c2a final-mem-c1-c2a-k2c1; do
    echo '=== '\$(date +%T)\" pthink30 C72 \$ref\"
    python -m gate sweep --ref \$ref --load pthink30 --concurrency 72 > $OUT/pthink30-\$ref.log 2>&1
    echo '=== '\$(date +%T)\" exit \$? \$(ls -td $R/sweep-\$ref-2026* | head -1)\"
  done"
kill $SMI
step "queue4 done"
