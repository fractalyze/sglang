#!/bin/bash
# The bs2 final: PC's mem-final plus PB's kept compute levers (<final ref>), against the base.
# The base's quality anchor, the final's in-flight sweep 4-32 with GPU memory sampled every 100 ms
# (compute/mem_check.py: plateau <= torch capacity - 512 MiB), the final's quality against the anchor,
# mem-final alone at 16/20/24 (memory and compute contributions apart), the final with the radix cache off
# (fleet model: every turn re-prefills its history; ref <final ref>-nocache), and the final's P/D measurement.
# Each step whose output exists is skipped, so a rerun resumes. Overrides (env): FINAL_CONC (the final's sweep,
# default 4-32), ALONE_REF / ALONE_CONC (the comparison sweep, default mem-final at 16/20/24), and
# G4POC_SERVER_MEMORY_MAX for a HiCache stack (final-hc runs in a 28G scope).
#   compute/final_run.sh <final ref> > /home/jooman/g4poc/logs/pb-final.log 2>&1
#   G4POC_SERVER_MEMORY_MAX=28G FINAL_CONC=4,8,12,16,20,24,28,32,40 ALONE_REF=final-mem-c1-c2a ALONE_CONC=24,28 \
#     compute/final_run.sh final-hc
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
final=$1
conc=${FINAL_CONC:-4,8,12,16,20,24,28,32}
alone=${ALONE_REF:-mem-final}
alone_conc=${ALONE_CONC:-16,20,24}
R=$G4POC_RUNS_DIR
out=$R/final-$final
mkdir -p "$out"
touch "$out/.started"
step() { echo "=== $(date -Is) $*"; }
server_ref_exists() { python -c "import sys; from gate import server; sys.exit(0 if sys.argv[1] in server.all_refs() else 1)" "$1"; }
sampled() {  # sampled <name> <command...>: GPU memory every 100 ms while the command runs
  local name=$1; shift
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$out/mem-$name.csv" &
  local smi=$!
  "$@" || { kill $smi; return 1; }
  kill $smi
  python compute/mem_check.py --samples "$out/mem-$name.csv" | tee "$out/mem-$name.json"
}

if [ ! -f "$G4POC/reference/quality_baseline.json" ]; then
  step "quality anchor: base (GSM8K full split + tool JSON)"
  python -m gate quality --ref base --gsm8k-n all --set-baseline --label quality-base-anchor
fi
if [ ! -f "$out/sweep.txt" ]; then
  step "sweep $final at $conc in flight (scope ${G4POC_SERVER_MEMORY_MAX:-24G})"
  sampled "$final" python -m gate sweep --ref "$final" --load inflight --concurrency "$conc"
  ls -td "$R"/sweep-"$final"-* | head -1 > "$out/sweep.txt"
fi
if [ ! -f "$out/quality.txt" ]; then
  step "quality $final"
  python -m gate quality --ref "$final" --gsm8k-n all --label "quality-$final"
  ls -td "$R"/quality-"$final"-* | head -1 > "$out/quality.txt"
fi
if [ ! -f "$out/alone-sweep.txt" ]; then
  step "sweep $alone at $alone_conc"
  sampled "$alone" python -m gate sweep --ref "$alone" --load inflight --concurrency "$alone_conc"
  ls -td "$R"/sweep-"$alone"-* | head -1 > "$out/alone-sweep.txt"
fi
if [ ! -f "$out/nocache-sweep.txt" ]; then
  step "sweep ${final}-nocache at 4-20 in flight (fleet model: every turn re-prefills its history)"
  if server_ref_exists "${final}-nocache"; then
    python -m gate sweep --ref "${final}-nocache" --load inflight --concurrency 4,8,12,16,20
    ls -td "$R"/sweep-"${final}"-nocache-* | head -1 > "$out/nocache-sweep.txt"
  else
    echo "no ref ${final}-nocache; skipped"
  fi
fi
if [ ! -f "$out/pd.txt" ]; then
  step "P/D measurement $final"
  python -m gate pd-measure --ref "$final"
  ls -td "$R"/pd-"$final"-* | head -1 > "$out/pd.txt"
fi
step "server starts retried on the HiCache host-memory check (gate/server.py): $(find "$R" -newer "$out/.started" -name 'server.log.start-try*' | wc -l)"
find "$R" -newer "$out/.started" -name 'server.log.start-try*'
step "final done"
