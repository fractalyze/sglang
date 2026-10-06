#!/bin/bash
# One SHIP verification unit for the ship branch jumanzii/g4poc-ship (refs in ship/refs.json), run whole on this host
# under $G4POC/unit.lock. Each unit appends "waiting", "running" and "done" lines to $G4POC/logs/ship-state.txt so
# other workers on the host can see it.
#   tests  every ship commit's own test files at that commit, then all of them at the ship head (GPU tests included),
#          under the host lock, from `git archive` copies of SRC_REPO (no worktree is added there)
#   exact  greedy multi-turn outputs at concurrency 1 (hicache/exactness_mt.py, 4 sessions x 3 turns):
#            identity  final-hc-cp2048-lpm-glue-c1 (cfc12c0bac) vs ship-inflight
#                      final-mem-c1-c2a-glue-c1 (cfc12c0bac) vs ship-chat
#            HiCache   ship-inflight-ctl vs ship-inflight-smallpool (every later turn a load-back)
#            upstream  upstream-base-inflight (a9871012ac) vs ship-defaults-inflight (every switch off)
#   c28    A-B-B-A at 28 in flight, final-hc-cp2048-lpm-glue-c1 (cfc12c0bac) vs ship-inflight, with 1 s GPU samples
#   t30    one pthink30 point at 72 on ship-chat
#   ship/ship_unit.sh <unit> <ship commit> > <log> 2>&1
set -uo pipefail
# The ship branch's base (jumanzii/g4poc-ship-base): its commits are <base>..<ship>.
base=a9871012acb768dc94a43a6542cc32626c7b7b0b
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
unit=$1 ship=$2
out=${G4POC_RUNS_DIR:-$G4POC/runs}/ship
mkdir -p "$out" "$G4POC/logs"
state() { echo "$(date '+%F %T') SHIP $unit $*" >> "$G4POC/logs/ship-state.txt"; }
step() { echo "=== $(date -Is) $*"; }
read_weights() { cat "$G4POC_MODEL_DIR"/*.safetensors > /dev/null; }

state waiting
exec 8>"$G4POC/unit.lock"
step "unit $unit waits for $G4POC/unit.lock"
flock 8
state running
step "unit $unit on $(hostname), ship $ship"

run_tests() {
  local src=${G4POC_SRC_REPO:-$G4/src-gate} scratch=$G4POC/scratch/ship-tests rc=0 n=0 c files dir
  rm -rf "$scratch" && mkdir -p "$scratch"
  exec 9>>"$G4POC_HOST_LOCK" 7>>"$G4POC_GPU_LOCK"
  flock 9 && flock 7
  for c in $(git -C "$src" rev-list --reverse "$base..$ship"); do
    n=$((n + 1))
    files=$(git -C "$src" diff-tree --no-commit-id --name-only -r "$c" -- 'test/*.py')
    dir=$scratch/$n-${c:0:10}
    mkdir -p "$dir" && git -C "$src" archive "$c" python/sglang test benchmark/kernels | tar -x -C "$dir"
    step "commit $n ${c:0:10}: $(git -C "$src" log -1 --format=%s "$c")"
    (cd "$dir" && PYTHONPATH="$dir/python" python -m pytest -q -p no:cacheprovider $files) || rc=1
  done
  files=$(git -C "$src" diff --name-only "$base" "$ship" -- 'test/*.py')
  step "all ship test files at the head"
  (cd "$dir" && PYTHONPATH="$dir/python" python -m pytest -q -p no:cacheprovider $files) || rc=1
  flock -u 7 && flock -u 9
  return $rc
}

exact_run() {  # exact_run <ref> <scope>: prints the run's exactness_mt.json path
  local log rc
  log=$(mktemp)
  read_weights
  G4POC_SERVER_MEMORY_MAX=$2 python -m hicache.exactness_mt run --ref "$1" 2>&1 | tee "$log" >&2
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && echo "$(sed -n 's/^run dir: //p' "$log")/exactness_mt.json"
  rm -f "$log"
  return "$rc"
}

exact_pair() {  # exact_pair <name> <control ref> <control scope> <candidate ref> <candidate scope>
  local a b
  a=$(exact_run "$2" "$3") && b=$(exact_run "$4" "$5") || { step "exact $1 failed"; return 1; }
  echo "$a $b" > "$out/exact-$1.paths"
  python -m hicache.exactness_mt compare "$a" "$b" > "$out/exact-$1.json"
  step "exact $1: $(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['exact'], '/', d['n'], 'identical,', d['same_cached_tokens'], 'same cached_tokens')" "$out/exact-$1.json")"
}

case "$unit" in
  tests)
    run_tests
    rc=$? ;;
  exact)
    mkdir -p "$G4POC/moe-configs/none/configs"
    rc=0
    exact_pair identity-inflight final-hc-cp2048-lpm-glue-c1 28G ship-inflight 28G || rc=1
    exact_pair identity-chat final-mem-c1-c2a-glue-c1 24G ship-chat 24G || rc=1
    exact_pair hicache ship-inflight-ctl 24G ship-inflight-smallpool 28G || rc=1
    exact_pair upstream upstream-base-inflight 28G ship-defaults-inflight 28G || rc=1 ;;
  c28)
    nvidia-smi --query-gpu=timestamp,memory.used,power.draw,clocks.sm --format=csv,noheader -l 1 > "$out/smi-c28.csv" &
    smi=$!
    G4POC_SERVER_MEMORY_MAX=28G compute/sweep_abba.sh final-hc-cp2048-lpm-glue-c1 ship-inflight 28 "$out/abba-c28.json"
    rc=$?
    kill "$smi" ;;
  t30)
    read_weights
    G4POC_SERVER_MEMORY_MAX=24G python -m gate sweep --ref ship-chat --load pthink30 --concurrency 72
    rc=$? ;;
  *)
    echo "unknown unit $unit" >&2; exit 2 ;;
esac
state "done rc=$rc"
step "unit $unit done rc=$rc"
exit $rc
