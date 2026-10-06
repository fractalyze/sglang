#!/bin/bash
# Paired sweeps A B B A on this host, then the point-by-point comparison. LOAD picks the gate load (default the
# in-flight layer; e.g. LOAD=pthink30 for chat sessions with think time, whose plan is seeded per concurrency, so all
# four sweeps offer the same arrivals).
# Hosts are shared with tenants outside the host lock, and a server at mem 0.955 has no room for another CUDA context:
# each sweep waits until the GPU runs no compute process, and a pair (A1 B1, then B2 A2) with a failed sweep is rerun
# whole, up to 3 tries, so both arms of a pair always come from back-to-back runs. Failed sweeps, with the foreign
# processes their server's OOM names, are listed in <out json>.failures.
#   [LOAD=<load>] compute/sweep_abba.sh <control ref> <candidate ref> <concurrency list> <out json>
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
ctrl=$1 cand=$2 conc=$3 out=$4 load=${LOAD:-inflight}
step() { echo "=== $(date -Is) $*"; }
wait_gpu_free() {
  local apps
  while apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader) && [ -n "$apps" ]; do
    step "GPU busy, waiting: $(echo "$apps" | tr '\n' ' ')"
    sleep 20
  done
}
latest() { ls -td "$G4POC_RUNS_DIR"/sweep-"$1"-* | head -1; }
# One sweep; its run dir goes to $dir. On failure, records the time and any foreign process the server's OOM names.
sweep() {
  wait_gpu_free
  step "sweep $1 ($load) at $conc"
  if python -m gate sweep --ref "$1" --load "$load" --concurrency "$conc"; then
    dir=$(latest "$1")
    return 0
  fi
  local d foreign
  d=$(latest "$1")
  foreign=$(grep -ohE "Process [0-9]+ has [0-9.]+ [GM]iB memory in use" "$d"/server.log* 2>/dev/null | sort -u | tr '\n' ';')
  echo "$(date -Is) sweep $1 failed ($d); OOM names: ${foreign:-none}" | tee -a "$out.failures"
  return 1
}
pair() {  # pair <first ref> <second ref>: sets $first_dir and $second_dir
  local try
  for try in 1 2 3; do
    sweep "$1" && first_dir=$dir && sweep "$2" && second_dir=$dir && return 0
    step "pair $1 / $2 failed (try $try); rerunning the whole pair"
  done
  return 1
}
pair "$ctrl" "$cand" || { step "sweep_abba: pair 1 failed 3 times"; exit 1; }
a1=$first_dir b1=$second_dir
pair "$cand" "$ctrl" || { step "sweep_abba: pair 2 failed 3 times"; exit 1; }
b2=$first_dir a2=$second_dir
python compute/sweep_abba.py --sweeps "$a1" "$b1" "$b2" "$a2" > "$out"
cat "$out"
step "sweep_abba done"
