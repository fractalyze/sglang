#!/bin/bash
# K2 queue 10 (bs2): K2-c2's gates on the ship stack (final + K1's glue + K2-c1), moved off bs3 (coordinator, 10-06
# ~15:35: K1 left bs2). First waits until bs2 has had no compute process on its GPU and >= 30 GB MemAvailable for
# 10 min (other tenants' jobs cycle there). Then, each unit under /data/jooman/g4poc/unit.lock on bs2:
#   A-B-B-A at 12 and 28 (final+glue+c1 vs +c2; compute/sweep_abba.sh reruns a failed pair whole);
#   the KL check (fidelity at the A/A level, one-prompt greedy identity);
#   C1 multi-turn exactness (device-only control vs HiCache small pool, both with c2);
#   one-window decode profiles at 12 and 28 of the control and the candidate (the pre-graph gap, same host).
set -u
H=/data/jooman/g4poc/harness-k2
U=/data/jooman/g4poc/unit.lock
cd $H && source gate/env.sh
R=${G4POC_RUNS_DIR:-$G4POC/runs}
OUT=$R/k2c2-stack
mkdir -p $OUT
step() { echo "=== $(date +%T) $*"; }
preread() { cat "$G4POC_MODEL_DIR"/*.safetensors > /dev/null; }
quiet() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && \
          [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 30 ]; }
step "waiting for 10 quiet minutes on $(hostname)"
n=0
while [ $n -lt 20 ]; do if quiet; then n=$((n+1)); else n=0; fi; sleep 30; done
step "host quiet"
CTL=final-hc-cp2048-lpm-glue-c1
CAND=final-hc-cp2048-lpm-glue-c1c2
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $OUT/smi.csv &
SMI=$!
export G4POC_SERVER_MEMORY_MAX=28G
step "c2 A-B-B-A 12,28 (waiting for $U)"
flock $U compute/sweep_abba.sh $CTL $CAND 12,28 $OUT/abba-c12-c28.json > $OUT/abba.log 2>&1
step "c2 A-B-B-A exit $?"
python compute/k2_smi_windows.py --samples $OUT/smi.csv --runs $(python -c "import json; print(' '.join(json.load(open('$OUT/abba-c12-c28.json'))['sweeps']))") > $OUT/smi-abba.json 2>&1
step "c2 KL check (waiting for $U)"
preread
flock $U python compute/kl_check.py --control $CTL --candidate $CAND > $OUT/kl.log 2>&1
step "c2 KL exit $?"
step "c2 exactness (waiting for $U)"
preread
flock $U bash -c "
  G4POC_SERVER_MEMORY_MAX=24G python -m hicache.exactness_mt run --ref final-mem-c1-c2a-cp2048-lpm-glue-c1c2 > $OUT/exact-ctl.log 2>&1
  cat $G4POC_MODEL_DIR/*.safetensors > /dev/null
  python -m hicache.exactness_mt run --ref $CAND-smallpool > $OUT/exact-cand.log 2>&1"
c=$(ls -td $R/exactmt-final-mem-c1-c2a-cp2048-lpm-glue-c1c2-2026* | head -1); f=$(ls -td $R/exactmt-$CAND-smallpool-2026* | head -1)
python -m hicache.exactness_mt compare $c/exactness_mt.json $f/exactness_mt.json > $OUT/exact-compare.json
step "c2 exactness $(python -c "import json; d=json.load(open('$OUT/exact-compare.json')); print(d['exact'], '/', d['n'], 'same_cached', d['same_cached_tokens'])")"
for ref in $CTL $CAND; do
  step "decode profile $ref 12,28 (waiting for $U)"
  preread
  flock $U python compute/k2_decode_profile.py serve --ref $ref --concurrency 12,28 --windows 75 --window-s 120 --steps 400 > $OUT/profile-$ref.log 2>&1
  step "profile exit $?"
done
kill $SMI
step "queue10 done"
