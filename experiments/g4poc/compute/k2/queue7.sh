#!/bin/bash
# K2 queue 7 (bs3), after K1's U2: finish the c1 A-B-B-A, then the c1 exactness and think-time pairs, then queue 5.
# The A-B-B-A's first pair ran 12:43-13:08 (A1 sweep-final-hc-cp2048-lpm-20261006-124324-*, B1
# sweep-final-hc-cp2048-lpm-k2c1-20261006-125615-*); its B2 then failed SGLang's HiCache host-memory check on all three
# starts after a 25-min preflight wait (swap). This runs the second pair (B2, A2) back to back under unit.lock, with
# the weights read outside the server's cgroup right before every start and up to three sweep attempts.
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
OUT=$R/k2c1
U=/data/jooman/g4poc/unit.lock
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
preread() { cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null; }
step "waiting for K1's U2 to take $U (max 10 min)"
for i in $(seq 1 60); do flock -n $U true || break; sleep 10; done
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $OUT/smi-q7.csv &
SMI=$!
export G4POC_SERVER_MEMORY_MAX=28G
step "c1 A-B-B-A second pair: B2 then A2 (waiting for $U)"
flock $U bash -c '
  for ref in final-hc-cp2048-lpm-k2c1 final-hc-cp2048-lpm; do
    for try in 1 2 3; do
      cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null
      echo "=== $(date +%T) sweep $ref 12,28 attempt $try"
      if python -m gate sweep --ref $ref --load inflight --concurrency 12,28; then
        ls -td /data/jooman/g4poc/runs/sweep-$ref-2026* | head -1 >> /data/jooman/g4poc/runs/k2c1/abba-pair2-dirs.txt
        break
      fi
    done
  done'
step "second pair exit $?"
A1=$(ls -d $R/sweep-final-hc-cp2048-lpm-20261006-124324-*); B1=$(ls -d $R/sweep-final-hc-cp2048-lpm-k2c1-20261006-125615-*)
B2=$(sed -n 1p $OUT/abba-pair2-dirs.txt); A2=$(sed -n 2p $OUT/abba-pair2-dirs.txt)
python compute/sweep_abba.py --sweeps $A1 $B1 $B2 $A2 > $OUT/abba-c12-c28.json
step "A-B-B-A: $A1 $B1 $B2 $A2 -> $OUT/abba-c12-c28.json"
python compute/k2_smi_windows.py --samples $OUT/smi-q4.csv --runs $A1 $B1 > $OUT/smi-abba-pair1.json 2>&1
python compute/k2_smi_windows.py --samples $OUT/smi-q7.csv --runs $B2 $A2 > $OUT/smi-abba-pair2.json 2>&1
step "c1 exactness pair (waiting for $U)"
preread
flock $U bash -c "
  G4POC_SERVER_MEMORY_MAX=24G python -m hicache.exactness_mt run --ref final-mem-c1-c2a-cp2048-lpm-k2c1 > $OUT/exact-ctl.log 2>&1
  cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null
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
