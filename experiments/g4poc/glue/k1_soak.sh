#!/bin/bash
# The K1 adoption soak: final-hc-cp2048-lpm-glue at 28 in flight for 30 min, GPU memory sampled for bs2's
# deployable rule. Pass criteria are in glue/PREREG.md. 28G scope (HiCache).
#   glue/k1_soak.sh > /home/jooman/g4poc/logs/k1-soak.log 2>&1
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
out=$G4POC_RUNS_DIR/k1
mkdir -p "$out"
echo "=== $(date -Is) soak final-hc-cp2048-lpm-glue at 28 in flight (30 min)"
# The HiCache start check counts the scope's page cache; read the weights outside it first.
cat "$G4POC_MODEL_DIR"/../shards/text-*.safetensors > /dev/null 2>&1 || true
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$out/soak-mem.csv" &
smi=$!
python -m gate sweep --ref final-hc-cp2048-lpm-glue --load soak --concurrency 28
kill $smi
python compute/mem_check.py --samples "$out/soak-mem.csv" | tee "$out/soak-mem.json"
echo "=== $(date -Is) k1 soak done"
