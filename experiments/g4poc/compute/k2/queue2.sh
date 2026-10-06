#!/bin/bash
# K2 queue 2 (bs3): the dense-GEMM decode-M microbench (existing kernels, final tree a0491db764); waits for the host lock and a free GPU.
# Holds the host lock and a 24G scope like a gate server.
H=/data/jooman/g4poc/harness-k2
L=/data/jooman/g4poc/logs
TREE=/data/jooman/gemma4nv/trees/a0491db7643b
OUT=/data/jooman/g4poc/runs/k2-dense-gemm-bench-$(date +%Y%m%d-%H%M%S)
free_gpu() {
  flock -n /data/jooman/gemma4nv/host.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
}
until free_gpu && sleep 60 && free_gpu; do sleep 60; done
mkdir -p $OUT
echo "=== bench start $(date +%T) -> $OUT"
cd $H && source gate/env.sh
flock /data/jooman/gemma4nv/host.lock systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 \
  env PYTHONPATH=$TREE/python python compute/k2_dense_gemm_bench.py --out $OUT/bench.json > $OUT/bench.log 2>&1
echo "=== bench exit $? $(date +%T)"
