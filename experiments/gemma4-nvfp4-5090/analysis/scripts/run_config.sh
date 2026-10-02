#!/usr/bin/env bash
# One screen/profile config, run under the bs2 GPU lock:
#   flock $G/host.lock flock $G/gpu.lock run_config.sh <name> <mode> [sglang flags...]
# mode: time     -> B=8 and B=1 timing (screen, unpaired)
#       profile  -> timing + torch-profiler traces at B=8 and B=1
#       experts  -> per-token expert routing dumps at B=1/8/32
# Launches the server, waits for it, runs the workload, kills the server.
set -uo pipefail
source /home/jooman/gemma4nv/env.sh
name=$1; mode=$2; shift 2
S=$G/src-analysis/analysis-scripts
R=$G/results/$name
mkdir -p $R
URL=http://127.0.0.1:30000
P=$G/results/prompts_1024.json
echo "$(date -Is) start $name mode=$mode flags=$*" | tee -a $G/results/jobs.log
cat $G/src-analysis/COMMIT > $R/commit.txt
echo "$*" > $R/flags.txt

extra_env=()
if [ "$mode" = experts ]; then
  export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$R/expert_dumps
  mkdir -p $SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
fi

python -m sglang.launch_server --model-path nvidia/Gemma-4-26B-A4B-NVFP4 \
  --host 127.0.0.1 --port 30000 --context-length 8192 --max-running-requests 32 \
  --enable-metrics "$@" > $R/server.log 2>&1 &
SPID=$!
cleanup() { kill $SPID 2>/dev/null; sleep 5; pkill -9 -P $SPID 2>/dev/null; kill -9 $SPID 2>/dev/null; }
trap cleanup EXIT

t0=$(date +%s)
until curl -sf $URL/health_generate >/dev/null 2>&1; do
  if ! kill -0 $SPID 2>/dev/null; then echo "$(date -Is) FAILED-LAUNCH $name" | tee -a $G/results/jobs.log; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 900 ]; then echo "$(date -Is) TIMEOUT-LAUNCH $name" | tee -a $G/results/jobs.log; exit 1; fi
  sleep 5
done
echo "launch_s=$(( $(date +%s) - t0 ))" > $R/launch.txt

wait_quiet() {
  # Co-tenant rule: no foreign GPU compute process may be present while timing.
  local mine waited=0
  while true; do
    mine=$(pgrep -d'|' -f "sglang" || true)
    foreign=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -vE "^(${mine:-x})," || true)
    [ -z "$foreign" ] && break
    echo "$(date -Is) foreign GPU procs: $foreign" >> $R/cotenant.log
    sleep 30; waited=$((waited+30))
    [ $waited -gt 1200 ] && { echo "$(date -Is) proceeding despite co-tenant" >> $R/cotenant.log; break; }
  done
}

clocks() { nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader >> $R/clocks.csv; }

if [ "$mode" = time ] || [ "$mode" = profile ]; then
  wait_quiet; clocks
  python $S/drive.py --url $URL --prompts $P --batch 8 --decode-len 128 --reps 3 --warmup 1 --label "$name B8" --out $R/time_b8.json 2>&1 | tail -1 | tee -a $R/summary.txt
  clocks
  python $S/drive.py --url $URL --prompts $P --batch 1 --decode-len 128 --reps 3 --warmup 1 --label "$name B1" --out $R/time_b1.json 2>&1 | tail -1 | tee -a $R/summary.txt
  clocks
fi
if [ "$mode" = profile ]; then
  python $S/drive.py --url $URL --prompts $P --batch 8 --decode-len 32 --reps 1 --warmup 0 --profile-dir $R/trace_b8 --profile-steps 8 --label "$name B8 prof" 2>&1 | tail -1
  sleep 30
  python $S/drive.py --url $URL --prompts $P --batch 1 --decode-len 32 --reps 1 --warmup 0 --profile-dir $R/trace_b1 --profile-steps 8 --label "$name B1 prof" 2>&1 | tail -1
  sleep 30
fi
if [ "$mode" = experts ]; then
  for B in 1 8 32; do
    reps=$(( B == 32 ? 2 : 8 ))
    curl -sf -X POST $URL/start_expert_distribution_record
    python $S/drive.py --url $URL --prompts $P --batch $B --decode-len 64 --reps $reps --warmup 0 --label "$name experts B$B" 2>&1 | tail -1
    curl -sf -X POST $URL/stop_expert_distribution_record
    curl -sf -X POST $URL/dump_expert_distribution_record
    sleep 10
    mkdir -p $R/b$B; mv $R/expert_dumps/*.pt $R/b$B/ 2>/dev/null
    python $S/experts.py --batch $B --json-out $R/experts_b$B.json $R/b$B/*.pt 2>&1 | tail -12
  done
fi
echo "$(date -Is) done $name" | tee -a $G/results/jobs.log
