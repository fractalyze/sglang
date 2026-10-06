#!/bin/bash
# K2 queue 8 (bs3): K2-c1 and K2-c2 on the ship stack, final + K1's glue fusion (coordinator, 10-06 ~14:40). Each
# unit holds /data/jooman/g4poc/unit.lock (shared with K1); the gate takes host.lock per server inside it. Weights are
# read outside the server's cgroup before every HiCache start, and each in-flight sweep gets up to 3 attempts (SGLang's
# HiCache host-memory check fails some starts in the 28G scope).
#   c1: the stacked A-B-B-A at 12 and 28 (final+glue vs final+glue+c1); exactness on the stack; sweep at 16, 20 and 24
#       (the 6 s SLO point); the 30-min soak at 28 with GPU memory sampled.
#   c2: A-B-B-A at 12 and 28 (final+glue+c1 vs +c2); the KL check; exactness; one-window re-profile at 12 and 28.
# First it finishes queue 7's c1-only exactness pair (its compare step).
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
U=/data/jooman/g4poc/unit.lock
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
preread() { cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null; }
exact_pair() {  # exact_pair <device-only control ref> <HiCache small-pool ref> <out json>
  flock $U bash -c "
    G4POC_SERVER_MEMORY_MAX=24G python -m hicache.exactness_mt run --ref $1 > $3.ctl.log 2>&1
    cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null
    G4POC_SERVER_MEMORY_MAX=28G python -m hicache.exactness_mt run --ref $2 > $3.cand.log 2>&1"
  local c f
  c=$(ls -td $R/exactmt-$1-2026* | head -1); f=$(ls -td $R/exactmt-$2-2026* | head -1)
  python -m hicache.exactness_mt compare $c/exactness_mt.json $f/exactness_mt.json > $3
  python -c "import json; d=json.load(open('$3')); print('exact', d['exact'], '/', d['n'], 'same_cached', d['same_cached_tokens'])"
}
abba() {  # abba <control ref> <candidate ref> <out dir>: A B B A at 12,28, up to 3 attempts per sweep
  rm -f $3/abba-dirs.txt
  flock $U bash -c "
    for ref in $1 $2 $2 $1; do
      for try in 1 2 3; do
        cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null
        echo \"=== \$(date +%T) sweep \$ref 12,28 attempt \$try\"
        if python -m gate sweep --ref \$ref --load inflight --concurrency 12,28; then
          ls -td $R/sweep-\$ref-2026* | head -1 >> $3/abba-dirs.txt
          break
        fi
      done
    done"
  python compute/sweep_abba.py --sweeps $(cat $3/abba-dirs.txt) > $3/abba-c12-c28.json
  python compute/k2_smi_windows.py --samples $3/smi.csv --runs $(cat $3/abba-dirs.txt) > $3/smi-abba.json 2>&1
}

O1=$R/k2c1
while pgrep -f "hicache.exactness_mt run --ref final-hc-cp2048-lpm-k2c1-smallpoo[l]" > /dev/null; do sleep 30; done
c=$(ls -td $R/exactmt-final-mem-c1-c2a-cp2048-lpm-k2c1-2026* | head -1); f=$(ls -td $R/exactmt-final-hc-cp2048-lpm-k2c1-smallpool-2026* | head -1)
python -m hicache.exactness_mt compare $c/exactness_mt.json $f/exactness_mt.json > $O1/exact-compare.json
step "c1-only exactness $(python -c "import json; d=json.load(open('$O1/exact-compare.json')); print(d['exact'], '/', d['n'], 'same_cached', d['same_cached_tokens'])")"

export G4POC_SERVER_MEMORY_MAX=28G
S1=$R/k2c1-stack
mkdir -p $S1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $S1/smi.csv &
SMI=$!
step "c1 stacked A-B-B-A (waiting for $U)"
abba final-hc-cp2048-lpm-glue-k2 final-hc-cp2048-lpm-glue-c1 $S1
step "c1 stacked A-B-B-A done: $(cat $S1/abba-dirs.txt | tr '\n' ' ')"
step "c1 exactness on the stack (waiting for $U)"
preread
step "c1 stack $(exact_pair final-mem-c1-c2a-cp2048-lpm-glue-c1 final-hc-cp2048-lpm-glue-c1-smallpool $S1/exact-compare.json)"
step "c1 sweep 16,20,24 on the stack (waiting for $U)"
preread
flock $U python -m gate sweep --ref final-hc-cp2048-lpm-glue-c1 --load inflight --concurrency 16,20,24 > $S1/sweep-16-24.log 2>&1
step "c1 sweep exit $? $(ls -td $R/sweep-final-hc-cp2048-lpm-glue-c1-2026* | head -1)"
kill $SMI
step "c1 soak 28 on the stack (waiting for $U)"
preread
flock $U bash -c "
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > $S1/soak-mem.csv &
  smi=\$!
  python -m gate sweep --ref final-hc-cp2048-lpm-glue-c1 --load soak --concurrency 28 > $S1/soak.log 2>&1
  echo \"soak exit \$?\"
  kill \$smi
  python compute/mem_check.py --samples $S1/soak-mem.csv --capacity-mib 32154 > $S1/soak-mem.json 2>&1"
step "c1 soak done $(ls -td $R/sweep-final-hc-cp2048-lpm-glue-c1-2026* | head -1)"

S2=$R/k2c2-stack
mkdir -p $S2
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $S2/smi.csv &
SMI=$!
step "c2 A-B-B-A on the stack (waiting for $U)"
abba final-hc-cp2048-lpm-glue-c1 final-hc-cp2048-lpm-glue-c1c2 $S2
step "c2 A-B-B-A done: $(cat $S2/abba-dirs.txt | tr '\n' ' ')"
step "c2 KL check (waiting for $U)"
preread
flock $U python compute/kl_check.py --control final-hc-cp2048-lpm-glue-c1 --candidate final-hc-cp2048-lpm-glue-c1c2 > $S2/kl.log 2>&1
step "c2 KL exit $?"
step "c2 exactness (waiting for $U)"
preread
step "c2 stack $(exact_pair final-mem-c1-c2a-cp2048-lpm-glue-c1c2 final-hc-cp2048-lpm-glue-c1c2-smallpool $S2/exact-compare.json)"
step "c2 re-profile 12,28 (waiting for $U)"
preread
flock $U python compute/k2_decode_profile.py serve --ref final-hc-cp2048-lpm-glue-c1c2 --concurrency 12,28 --windows 75 --window-s 120 --steps 400 > $S2/profile.log 2>&1
step "c2 re-profile exit $?"
kill $SMI
step "queue8 done"
