#!/bin/bash
# K2 queue 5 (bs3), after queue 4: K2-c1's follow-ups if kept, then K2-c2's gates on top of c1. Each unit holds the
# outer lock /data/jooman/g4poc/unit.lock (shared with K1); the gate takes host.lock per server inside it.
#   c1: sweep at 16, 20 and 24 in flight (the new 6 s SLO point); 30-min soak at 28 with GPU memory sampled.
#   c2 (compute/PREREG.md "Round 2, K2-c2"): A-B-B-A at 12 and 28 (final+c1 vs final+c1+c2); the KL check
#   (fidelity at the A/A level, one-prompt greedy identity); C1 multi-turn exactness pair; one-window decode
#   re-profile at 12 and 28 (the pre-graph gap; the post-c1 step for the megakernel re-decision).
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
U=/data/jooman/g4poc/unit.lock
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
preread() { cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null; }
while pgrep -f "compute/k2/queue[4].sh" > /dev/null; do sleep 60; done
export G4POC_SERVER_MEMORY_MAX=28G
C1=final-hc-cp2048-lpm-k2c1
C2=final-hc-cp2048-lpm-k2c1c2
O1=$R/k2c1
O2=$R/k2c2
mkdir -p $O2
step "c1 sweep 16,20,24 (waiting for $U)"
preread
flock $U python -m gate sweep --ref $C1 --load inflight --concurrency 16,20,24 > $O1/sweep-16-24.log 2>&1
step "c1 sweep exit $? $(ls -td $R/sweep-$C1-2026* | head -1)"
step "c1 soak 28 (waiting for $U)"
preread
flock $U bash -c "
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > $O1/soak-mem.csv &
  smi=\$!
  python -m gate sweep --ref $C1 --load soak --concurrency 28 > $O1/soak.log 2>&1
  echo \"soak exit \$?\"
  kill \$smi
  python compute/mem_check.py --samples $O1/soak-mem.csv --capacity-mib 32154 > $O1/soak-mem.json 2>&1"
step "c1 soak done $(ls -td $R/sweep-$C1-2026* | head -1)"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $O2/smi.csv &
SMI=$!
step "c2 A-B-B-A 12,28 (waiting for $U)"
preread
flock $U compute/sweep_abba.sh $C1 $C2 12,28 $O2/abba-c12-c28.json
step "c2 A-B-B-A exit $?"
step "c2 KL check (waiting for $U)"
preread
flock $U python compute/kl_check.py --control $C1 --candidate $C2 > $O2/kl.log 2>&1
step "c2 KL exit $?"
step "c2 exactness pair (waiting for $U)"
preread
flock $U bash -c "
  G4POC_SERVER_MEMORY_MAX=24G python -m hicache.exactness_mt run --ref final-mem-c1-c2a-cp2048-lpm-k2c1c2 > $O2/exact-ctl.log 2>&1
  python -m hicache.exactness_mt run --ref $C2-smallpool > $O2/exact-cand.log 2>&1"
c=$(ls -td $R/exactmt-final-mem-c1-c2a-cp2048-lpm-k2c1c2-2026* | head -1); f=$(ls -td $R/exactmt-$C2-smallpool-2026* | head -1)
python -m hicache.exactness_mt compare $c/exactness_mt.json $f/exactness_mt.json > $O2/exact-compare.json
step "c2 exactness $(python -c "import json; d=json.load(open('$O2/exact-compare.json')); print(d['exact'], '/', d['n'], 'same_cached', d['same_cached_tokens'])")"
step "c2 re-profile 12,28 (waiting for $U)"
preread
flock $U python compute/k2_decode_profile.py serve --ref $C2 --concurrency 12,28 --windows 75 --window-s 120 --steps 400 > $O2/profile.log 2>&1
step "c2 re-profile exit $?"
kill $SMI
step "queue5 done"
