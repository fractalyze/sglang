#!/bin/bash
# HS1' on bs2 (compute/PREREG.md): final-hc with a 12 GB and a 6 GB host pool, each at 64 concurrent sessions with
# 30 s think time (load think30), then -- if it is still before 06:00 KST -- the final-hc replicate at 16/24/32/40
# in flight. 28G scope.
#   compute/hs1.sh > /home/jooman/g4poc/logs/pb-hs1.log 2>&1
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
R=$G4POC_RUNS_DIR
step() { echo "=== $(date -Is) $*"; }
before_stop() { [ "$(date +%H%M)" -lt 0600 ] || [ "$(date +%H%M)" -ge 1200 ]; }

for ref in final-hc final-hc-hc6; do
  step "HS1' $ref at think30 x 64 sessions"
  python -m gate sweep --ref "$ref" --load think30 --concurrency 64 && \
    ls -td "$R"/sweep-"$ref"-* | head -1 > "$R/hs1-$ref.txt"
done
for ref in final-hc final-hc-hc6; do
  d=$(cat "$R/hs1-$ref.txt" 2>/dev/null) || continue
  python -c "
import json, sys
p = json.load(open('$d/sweep.json'))['points'][0]['summary']
print('$ref', {k: round(p[k], 3) for k in ('e2e_p90_s', 'output_tok_s_per_gpu', 'prefix_cache_hit_rate', 'requests_per_s', 'inflight_mean', 'n_failed')})"
done
if before_stop; then
  step "final-hc replicate at 16, 24, 32, 40 in flight"
  python -m gate sweep --ref final-hc --load inflight --concurrency 16,24,32,40
fi
step "hs1 done"
