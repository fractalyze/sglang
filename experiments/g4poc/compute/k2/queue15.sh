#!/bin/bash
# K4 queue 15 (bs3): the decode-attention microbench (compute/k4_decode_attn_bench.py) on the ship tree, one GPU unit
# under /data/jooman/g4poc/unit.lock and host.lock, 24G scope. SHIP's GPU units go first: while SHIP's state file's
# last line says it is waiting, this waits (checked every 30 s).
H=/data/jooman/g4poc/harness-k2
TREE=/data/jooman/gemma4nv/trees/cfc12c0bac57
OUT=/data/jooman/g4poc/runs/k4-attn-bench-$(date +%Y%m%d-%H%M%S)
SHIP=/data/jooman/g4poc/logs/ship-state.txt
mkdir -p $OUT
cd $H && source gate/env.sh
while [ -f $SHIP ] && tail -1 $SHIP | grep -qi "waiting"; do sleep 30; done
echo "=== $(date +%T) attention bench (waiting for the unit lock) -> $OUT"
flock /data/jooman/g4poc/unit.lock flock /data/jooman/gemma4nv/host.lock bash -c "
  for i in \$(seq 1 30); do [ -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ] && break; sleep 10; done
  echo \"=== \$(date +%T) bench start\"
  systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env PYTHONPATH=$TREE/python \
    python compute/k4_decode_attn_bench.py --out $OUT/bench.json > $OUT/bench.log 2>&1
  echo \"=== \$(date +%T) bench exit \$?\""
