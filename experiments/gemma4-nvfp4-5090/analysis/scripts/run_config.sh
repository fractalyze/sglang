#!/usr/bin/env bash
# One W2 config (inner script; takes no lock). Always started through job.sh,
# which runs it under gate.hostwatch: host.lock + gpu.lock, a 24G no-swap
# systemd scope, preflight refusal and the 2 s host-memory watchdog.
# The server's stdout is this script's stdout, which hostwatch writes to
# <run>/server.log and scans for phases (weight load, autotune, graph capture).
#
#   run_config.sh <name> <mode> [extra sglang flags...]
# mode: prebuild -> launch, one request, exit (JIT/autotune cache warm-up)
#       time     -> B=8 and B=1 timing (screen, unpaired)
#       profile  -> timing + torch-profiler traces at B=8 and B=1
#       experts  -> per-token expert routing dumps at B=1/8/32
set -uo pipefail
source /data/jooman/gemma4nv/src-gate/experiments/gemma4-nvfp4-5090/env/env.sh
name=$1; mode=$2; shift 2
S=$G4/src-analysis/analysis-scripts
R=$G4/results/$name
URL=http://127.0.0.1:30000
P=$G4/results/prompts_1024.json
# W1's pinned baseline flags (gate/refs.json "base"); extra flags are the knob.
BASE=(--moe-runner-backend flashinfer_cutlass --cuda-graph-max-bs-decode 32 --decode-log-interval 1)
mkdir -p $R
echo "$(date -Is) start $name mode=$mode host=$(hostname) flags=$*" >> $G4/results/jobs.log
git -C $G4/src-analysis rev-parse HEAD > $R/commit.txt 2>/dev/null || cat $G4/src-analysis/COMMIT > $R/commit.txt
printf '%s\n' "${BASE[*]} $*" > $R/flags.txt
python -c "import torch, flashinfer, sglang; print(torch.__version__, flashinfer.__version__)" > $R/versions.txt 2>&1

if [ "$mode" = experts ]; then
  export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$R/expert_dumps
  mkdir -p $SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
fi

python -m sglang.launch_server --model-path $G4_MODEL_DIR --host 127.0.0.1 --port 30000 \
  "${BASE[@]}" "$@" &
SPID=$!
cleanup() { kill $SPID 2>/dev/null; sleep 5; pkill -9 -P $SPID 2>/dev/null; kill -9 $SPID 2>/dev/null; }
trap cleanup EXIT

# Coordinator rule: W2 compiles nothing; it reuses W1b's FlashInfer/tvm JIT
# cache. Any nvcc/cicc (FlashInfer/tvm JIT; Triton only uses ptxas) means a
# cache miss: stop this step and report it.
( while kill -0 $SPID 2>/dev/null; do
    if pgrep -u "$(id -u)" -x "nvcc|cicc" >/dev/null; then
      echo "$(date -Is) COMPILE-DETECTED $name: $(pgrep -u "$(id -u)" -a -x 'nvcc|cicc' | head -3)" >> $G4/results/jobs.log
      touch $R/COMPILE_DETECTED; kill $SPID; pkill -u "$(id -u)" -x "nvcc|cicc|ninja"; exit 0
    fi
    sleep 2
  done ) &

t0=$(date +%s)
until curl -sf $URL/health_generate >/dev/null 2>&1; do
  if ! kill -0 $SPID 2>/dev/null; then echo "$(date -Is) FAILED-LAUNCH $name" >> $G4/results/jobs.log; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 1200 ]; then echo "$(date -Is) TIMEOUT-LAUNCH $name" >> $G4/results/jobs.log; exit 1; fi
  sleep 5
done
echo "launch_s=$(( $(date +%s) - t0 ))" > $R/launch.txt

wait_quiet() {
  # Co-tenant rule: no foreign GPU compute process while timing.
  local mine foreign waited=0
  while true; do
    mine=$(pgrep -d'|' -f "sglang" || true)
    foreign=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -vE "^(${mine:-x})," || true)
    [ -z "$foreign" ] && break
    echo "$(date -Is) foreign GPU procs: $foreign" >> $R/cotenant.log
    sleep 30; waited=$((waited+30))
    [ $waited -gt 1200 ] && { echo "$(date -Is) proceeding despite co-tenant (labelled)" >> $R/cotenant.log; break; }
  done
}
clocks() { nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader >> $R/clocks.csv; }
drive() { python $S/drive.py --url $URL --prompts $P "$@" 2>&1 | tail -1 | tee -a $R/summary.txt; }

if [ "$mode" = prebuild ]; then
  drive --batch 8 --decode-len 8 --reps 1 --warmup 0 --label "$name prebuild"
fi
if [ "$mode" = time ] || [ "$mode" = profile ]; then
  wait_quiet; clocks
  drive --batch 8 --decode-len 128 --reps 3 --warmup 1 --label "$name B8" --out $R/time_b8.json
  clocks
  drive --batch 1 --decode-len 128 --reps 3 --warmup 1 --label "$name B1" --out $R/time_b1.json
  clocks
fi
if [ "$mode" = profile ]; then
  drive --batch 8 --decode-len 32 --reps 1 --warmup 0 --profile-dir $R/trace_b8 --profile-steps 8 --label "$name B8 prof"
  sleep 30
  drive --batch 1 --decode-len 32 --reps 1 --warmup 0 --profile-dir $R/trace_b1 --profile-steps 8 --label "$name B1 prof"
  sleep 30
fi
if [ "$mode" = experts ]; then
  for B in 1 8 32; do
    reps=$(( B == 32 ? 2 : 8 ))
    curl -sf -X POST $URL/start_expert_distribution_record
    drive --batch $B --decode-len 64 --reps $reps --warmup 0 --label "$name experts B$B"
    curl -sf -X POST $URL/stop_expert_distribution_record
    curl -sf -X POST $URL/dump_expert_distribution_record
    sleep 10
    mkdir -p $R/b$B; mv $R/expert_dumps/*.pt $R/b$B/ 2>/dev/null
    python $S/experts.py --batch $B --json-out $R/experts_b$B.json $R/b$B/*.pt > $R/experts_b$B.txt 2>&1
  done
fi
echo "$(date -Is) done $name" >> $G4/results/jobs.log
