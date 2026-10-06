#!/bin/bash
# K2 queue 3 (bs3): the K2-c1 gates (compute/PREREG.md "Round 2, K2-c1"), in order:
#   1. the c1 unit test on the GPU (tuned tiles vs CUTLASS), 24G scope under the host lock;
#   2. same-host A-B-B-A at 12 and 28 in flight, final vs final + c1 (28G scope);
#   3. the long role-play KL check (teacher-forced prefill + one-prompt greedy decode);
#   4. GSM8K (1,319) + tool-JSON against the bs3 base anchor, then the role-play arm;
#   5. C1 multi-turn exactness: device-only control vs HiCache small pool, both with c1;
#   6. think time: pthink30 at 72 sessions, device-only final-mem-c1-c2a without then with c1 (same plan).
# GPU clocks, power and temperature are sampled every second for the whole queue (smi.csv). Every server is started
# and stopped by the gate, which holds the host lock per server; the sampler is stopped by its own PID.
set -u
H=/data/jooman/g4poc/harness-k2
R=/data/jooman/g4poc/runs
OUT=$R/k2c1
mkdir -p $OUT
cd $H && source gate/env.sh
step() { echo "=== $(date +%T) $*"; }
preread() { cat /data/jooman/g4poc/models/gemma-4-26B-A4B-it-fp8ch/shards/text-*.safetensors > /dev/null; }
CAND=final-hc-cp2048-lpm-k2c1
TREE=$(python -c "from gate import server; print(server.tree_for('cfc12c0bac57a8cb73be37f2cb37d907ef2d0769'))")
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader,nounits -l 1 > $OUT/smi.csv &
SMI=$!
step "1 unit test on $TREE"
flock /data/jooman/gemma4nv/host.lock systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 \
  env PYTHONPATH=$TREE/python:/home/jooman/gemma4nv/pytest-site python -m pytest -q -p no:cacheprovider \
  $TREE/test/registered/unit/layers/quantization/test_fp8_channelwise_rtx5090_configs.py > $OUT/unittest.log 2>&1
step "1 exit $? $(tail -1 $OUT/unittest.log)"
export G4POC_SERVER_MEMORY_MAX=28G
step "2 A-B-B-A 12,28: final-hc-cp2048-lpm vs $CAND"
preread
compute/sweep_abba.sh final-hc-cp2048-lpm $CAND 12,28 $OUT/abba-c12-c28.json
step "2 exit $?"
step "3 KL check"
preread
python compute/kl_check.py --control final-hc-cp2048-lpm --candidate $CAND > $OUT/kl.log 2>&1
step "3 exit $? $(tail -2 $OUT/kl.log | tr '\n' ' ')"
step "4a quality final-cpl-qr-k2c1"
preread
python -m gate quality --ref final-cpl-qr-k2c1 --gsm8k-n all > $OUT/quality.log 2>&1
step "4a exit $?"
step "4b rp-quality final-cpl-qr-k2c1"
preread
python -m gate rp-quality --ref final-cpl-qr-k2c1 > $OUT/rpquality.log 2>&1
step "4b exit $?"
step "5 exactness pair"
G4POC_SERVER_MEMORY_MAX=24G python -m hicache.exactness_mt run --ref final-mem-c1-c2a-cp2048-lpm-k2c1 > $OUT/exact-ctl.log 2>&1
preread
python -m hicache.exactness_mt run --ref final-hc-cp2048-lpm-k2c1-smallpool > $OUT/exact-cand.log 2>&1
c=$(ls -td $R/exactmt-final-mem-c1-c2a-cp2048-lpm-k2c1-2026* | head -1); f=$(ls -td $R/exactmt-final-hc-cp2048-lpm-k2c1-smallpool-2026* | head -1)
python -m hicache.exactness_mt compare $c/exactness_mt.json $f/exactness_mt.json > $OUT/exact-compare.json
step "5 $(python -c "import json; d=json.load(open('$OUT/exact-compare.json')); print('exact', d['exact'], '/', d['n'], 'same_cached', d['same_cached_tokens'])")"
export G4POC_SERVER_MEMORY_MAX=24G
for ref in final-mem-c1-c2a final-mem-c1-c2a-k2c1; do
  step "6 pthink30 C72 $ref"
  python -m gate sweep --ref $ref --load pthink30 --concurrency 72 > $OUT/pthink30-$ref.log 2>&1
  step "6 exit $? $(ls -td $R/sweep-$ref-2026* | head -1)"
done
kill $SMI
step "queue3 done"
