#!/bin/bash
# PC4's bug-4 fix (SGLANG_OPT_SWA_PREFILL_WINDOW_MARGIN=128) on the final config, bs2, 10-06 morning, after every
# process running from WAIT_DIR (PC4's harness) has exited, checked twice a minute apart. In order, each step only
# if its estimate ends by STOP (the 07:55 hard stop):
#   (a) A-B-B-A at 28 in flight, final vs -m128: p90, p99, retractions, hit (compute/sweep_abba.py);
#   (b) -m128's quality (GSM8K full split + tool JSON) against the bs2 base anchor;
#   (c) a 30-min soak of -m128 at 28 in flight, GPU memory sampled against the deployable rule;
#   (d) the 6 s point at 8 and 12 in flight: the final, then -m128;
#   (e) a second A-B-B-A at 28 (two more pairs for the tail and the retractions).
# 28G scope.
#   [HARNESS=<dir>] [WAIT_DIR=<dir>] [STOP=HH:MM] compute/m128_run.sh > /home/jooman/g4poc/logs/pb-m128.log 2>&1
set -uo pipefail
here="${HARNESS:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$here/gate/env.sh"
cd "$here"
export G4POC_SERVER_MEMORY_MAX=28G
final=final-hc-cp2048-lpm
cand=final-hc-cp2048-lpm-m128
R=$G4POC_RUNS_DIR
out=$R/m128
mkdir -p "$out"
step() { echo "=== $(date -Is) $*"; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$(date -d "${STOP:-07:55}" +%s)" ]; }
# A PC4 process names WAIT_DIR on its command line or runs from it (PC4 starts queues as ./queue-*.sh).
busy() {
  pgrep -f "$WAIT_DIR" | grep -vqx "$$" && return 0
  local p
  for p in $(pgrep -u "$(id -u)" .); do
    case "$(readlink "/proc/$p/cwd" 2>/dev/null)" in "$WAIT_DIR"*) return 0 ;; esac
  done
  return 1
}
if [ -n "${WAIT_DIR:-}" ]; then
  step "wait for processes under $WAIT_DIR"
  while busy || { sleep 60; busy; }; do sleep 30; done
fi

if fits 28; then
  step "(a) A-B-B-A at 28 in flight: $final vs $cand"
  compute/sweep_abba.sh "$final" "$cand" 28 "$out/abba-c28.json"
else step "skip (a): would end past ${STOP:-07:55}"; fi

if fits 8; then
  step "(b) quality $cand"
  if python -m gate quality --ref "$cand" --gsm8k-n all --label "quality-$cand"; then
    q=$(ls -td "$R"/quality-"$cand"-* | head -1)
    b=$(ls -td "$R"/quality-base-anchor-* | head -1)
    python -m gate quality-compare --control "$b/quality.json" --candidate "$q/quality.json" | tee "$out/quality-compare.json"
  fi
else step "skip (b): would end past ${STOP:-07:55}"; fi

if fits 36; then
  step "(c) soak $cand at 28 in flight (30 min)"
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$out/soak-mem.csv" &
  smi=$!
  python -m gate sweep --ref "$cand" --load soak --concurrency 28
  kill $smi
  python compute/mem_check.py --samples "$out/soak-mem.csv" | tee "$out/soak-mem.json"
else step "skip (c): would end past ${STOP:-07:55}"; fi

for ref in "$final" "$cand"; do
  if fits 14; then
    step "(d) $ref at 8, 12 in flight (6 s point)"
    python -m gate sweep --ref "$ref" --load inflight --concurrency 8,12
  else step "skip (d) $ref: would end past ${STOP:-07:55}"; fi
done

if fits 28; then
  step "(e) second A-B-B-A at 28 in flight"
  compute/sweep_abba.sh "$final" "$cand" 28 "$out/abba-c28-2.json"
else step "skip (e): would end past ${STOP:-07:55}"; fi
step "m128 done"
