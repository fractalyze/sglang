#!/bin/bash
# K2 queue 2 (bs3): the dense-GEMM decode-M microbench (existing kernels, final tree a0491db764).
# Takes the host lock as soon as it is free (blocking flock, like a gate server) and runs in a 24G scope;
# inside the lock it waits up to 5 min for the GPU to have no compute process before starting.
H=/data/jooman/g4poc/harness-k2
TREE=/data/jooman/gemma4nv/trees/a0491db7643b
OUT=/data/jooman/g4poc/runs/k2-dense-gemm-bench-$(date +%Y%m%d-%H%M%S)
mkdir -p $OUT
cd $H && source gate/env.sh
echo "=== waiting for the host lock $(date +%T)"
flock /data/jooman/gemma4nv/host.lock bash -c '
  for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 10; done
  echo "=== bench start $(date +%T) -> '$OUT'"
  systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 \
    env PYTHONPATH='$TREE'/python python compute/k2_dense_gemm_bench.py --out '$OUT'/bench.json > '$OUT'/bench.log 2>&1
  echo "=== bench exit $? $(date +%T)"'
