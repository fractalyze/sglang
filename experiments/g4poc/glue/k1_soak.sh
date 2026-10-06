#!/bin/bash
# The K1 adoption soak: final-hc-cp2048-lpm-glue at 28 in flight for 30 min, GPU memory sampled for bs2's
# deployable rule. Pass criteria are in glue/PREREG.md. 28G scope (HiCache).
# bs2 has tenants outside the host lock whose CUDA contexts OOM a server at mem 0.955, hence the wait and retry.
#   glue/k1_soak.sh > /home/jooman/g4poc/logs/k1-soak.log 2>&1
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
ref=final-hc-cp2048-lpm-glue
out=$G4POC_RUNS_DIR/k1
mkdir -p "$out"
step() { echo "=== $(date -Is) $*"; }
wait_gpu_free() {
  local apps
  while apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader) && [ -n "$apps" ]; do
    step "GPU busy, waiting: $(echo "$apps" | tr '\n' ' ')"
    sleep 20
  done
}
for try in 1 2 3; do
  wait_gpu_free
  step "soak $ref at 28 in flight (30 min), try $try"
  # The HiCache start check counts the scope's page cache; read the weights outside it first.
  cat "$G4POC_MODEL_DIR"/../shards/text-*.safetensors > /dev/null 2>&1 || true
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$out/soak-mem.csv" &
  smi=$!
  python -m gate sweep --ref "$ref" --load soak --concurrency 28
  rc=$?
  kill $smi
  if [ $rc -eq 0 ]; then
    python compute/mem_check.py --samples "$out/soak-mem.csv" | tee "$out/soak-mem.json"
    step "k1 soak done"
    exit 0
  fi
  d=$(ls -td "$G4POC_RUNS_DIR"/sweep-"$ref"-* | head -1)
  foreign=$(grep -ohE "Process [0-9]+ has [0-9.]+ [GM]iB memory in use" "$d"/server.log* 2>/dev/null | sort -u | tr '\n' ';')
  echo "$(date -Is) soak try $try failed ($d); OOM names: ${foreign:-none}" | tee -a "$out/soak.failures"
done
step "k1 soak failed 3 times"
exit 1
