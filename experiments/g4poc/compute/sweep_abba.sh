#!/bin/bash
# Paired sweeps A B B A on this host, then the point-by-point comparison. LOAD picks the gate load (default the
# in-flight layer; e.g. LOAD=pthink30 for chat sessions with think time, whose plan is seeded per concurrency, so all
# four sweeps offer the same arrivals).
#   [LOAD=<load>] compute/sweep_abba.sh <control ref> <candidate ref> <concurrency list> <out json>
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
ctrl=$1 cand=$2 conc=$3 out=$4 load=${LOAD:-inflight}
dirs=()
for ref in "$ctrl" "$cand" "$cand" "$ctrl"; do
  echo "=== $(date -Is) sweep $ref ($load) at $conc"
  python -m gate sweep --ref "$ref" --load "$load" --concurrency "$conc"
  dirs+=("$(ls -td "$G4POC_RUNS_DIR"/sweep-"$ref"-* | head -1)")
done
python compute/sweep_abba.py --sweeps "${dirs[@]}" > "$out"
cat "$out"
echo "=== $(date -Is) sweep_abba done"
