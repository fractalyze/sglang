#!/bin/bash
# In-flight sweeps for several candidates against one control, nested so every candidate's two
# sweeps sit symmetrically around the run's midpoint: A B C ... C B A. Each candidate is then
# compared with compute/sweep_abba.py against the first and last control sweep.
#   compute/sweep_nested.sh <control ref> "<cand1> <cand2> ..." <concurrency list> <out dir>
# Writes <out dir>/<candidate>.json per candidate. A sweep that already finished (listed in
# <out dir>/sweeps.txt) is not rerun, so a rerun resumes.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
ctrl=$1 conc=$3 out=$4
read -r -a cands <<< "$2"
order=("$ctrl" "${cands[@]}")
for ((i = ${#cands[@]} - 1; i >= 0; i--)); do order+=("${cands[$i]}"); done
order+=("$ctrl")
mkdir -p "$out"
touch "$out/sweeps.txt"
mapfile -t done_dirs < "$out/sweeps.txt"
dirs=()
for k in "${!order[@]}"; do
  ref=${order[$k]}
  if [ "$k" -lt "${#done_dirs[@]}" ]; then
    dirs+=("${done_dirs[$k]}")
    continue
  fi
  echo "=== $(date -Is) sweep $((k + 1))/${#order[@]}: $ref at $conc"
  python -m gate sweep --ref "$ref" --load inflight --concurrency "$conc"
  d=$(ls -td "$G4POC_RUNS_DIR"/sweep-"$ref"-* | head -1)
  dirs+=("$d")
  echo "$d" >> "$out/sweeps.txt"
done
n=${#order[@]}
for i in "${!cands[@]}"; do
  python compute/sweep_abba.py --sweeps "${dirs[0]}" "${dirs[$((i + 1))]}" "${dirs[$((n - 2 - i))]}" "${dirs[$((n - 1))]}" \
    > "$out/${cands[$i]}.json"
  echo "--- ${cands[$i]}"
  cat "$out/${cands[$i]}.json"
done
echo "=== $(date -Is) sweep_nested done"
