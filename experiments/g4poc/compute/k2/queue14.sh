#!/bin/bash
# K3 queue 14 (bs3): the MoE-layer decode microbench (compute/k3_moe_bench.py) on the ship tree cfc12c0bac with the
# served C1 config dir, under /data/jooman/g4poc/unit.lock and host.lock, in a 24G scope.
H=/data/jooman/g4poc/harness-k2
TREE=/data/jooman/gemma4nv/trees/cfc12c0bac57
OUT=/data/jooman/g4poc/runs/k3-moe-bench-$(date +%Y%m%d-%H%M%S)
mkdir -p $OUT
cd $H && source gate/env.sh
echo "=== $(date +%T) moe bench (waiting for the unit lock) -> $OUT"
flock /data/jooman/g4poc/unit.lock flock /data/jooman/gemma4nv/host.lock bash -c "
  for i in \$(seq 1 30); do [ -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ] && break; sleep 10; done
  echo \"=== \$(date +%T) bench start\"
  systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env PYTHONPATH=$TREE/python \
    SGLANG_MOE_CONFIG_DIR=/data/jooman/g4poc/moe-configs/c1 python compute/k3_moe_bench.py --out $OUT/bench.json > $OUT/bench.log 2>&1
  echo \"=== \$(date +%T) bench exit \$?\""
