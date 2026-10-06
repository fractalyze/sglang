#!/bin/bash
# Paired in-flight sweeps A B B A on this host, then the point-by-point comparison.
#   compute/sweep_abba.sh <control ref> <candidate ref> <concurrency list> <out json>
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
ctrl=$1 cand=$2 conc=$3 out=$4
# gate/env.sh exports G4POC_RUNS_DIR only on bs2; elsewhere the gate's default (gate/config.py) applies.
runs=${G4POC_RUNS_DIR:-$G4POC/runs}
dirs=()
for ref in "$ctrl" "$cand" "$cand" "$ctrl"; do
  echo "=== $(date -Is) sweep $ref at $conc"
  python -m gate sweep --ref "$ref" --load inflight --concurrency "$conc"
  dirs+=("$(ls -td "$runs"/sweep-"$ref"-* | head -1)")
done
python compute/sweep_abba.py --sweeps "${dirs[@]}" > "$out"
cat "$out"
echo "=== $(date -Is) sweep_abba done"
