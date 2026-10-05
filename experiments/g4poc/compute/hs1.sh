#!/bin/bash
# HS1'' and HS1' on bs2 (compute/PREREG.md): final-hc with a 12 GB and a 6 GB host pool at 30 s think time (load
# think30), first at 36 concurrent sessions (HS1'', between the two arms' storage bounds), then at 64 (HS1'); then
# the final-hc replicate at 16/24/32/40 in flight. Nothing starts after 06:00 KST. 28G scope.
#   compute/hs1.sh > /home/jooman/g4poc/logs/pb-hs1.log 2>&1
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
R=$G4POC_RUNS_DIR
step() { echo "=== $(date -Is) $*"; }
before_stop() { [ "$(date +%H%M)" -lt 0600 ] || [ "$(date +%H%M)" -ge 1200 ]; }

for n in 36 64; do
  for ref in final-hc final-hc-hc6; do
    before_stop || break 2
    step "HS1 $ref at think30 x $n sessions"
    python -m gate sweep --ref "$ref" --load think30 --concurrency "$n" && \
      ls -td "$R"/sweep-"$ref"-* | head -1 > "$R/hs1-$ref-s$n.txt"
  done
  for ref in final-hc final-hc-hc6; do
    d=$(cat "$R/hs1-$ref-s$n.txt" 2>/dev/null) || continue
    python -c "
import json
p = json.load(open('$d/sweep.json'))['points'][0]['summary']
print('$ref', $n, {k: round(p[k], 3) for k in ('e2e_p90_s', 'output_tok_s_per_gpu', 'prefix_cache_hit_rate', 'requests_per_s', 'inflight_mean', 'n_failed')})"
  done
done
if before_stop; then
  step "final-hc replicate at 16, 24, 32, 40 in flight"
  python -m gate sweep --ref final-hc --load inflight --concurrency 16,24,32,40
fi
step "hs1 done"
