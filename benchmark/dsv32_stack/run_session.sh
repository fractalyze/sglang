#!/bin/bash
# One measurement session of DeepSeek-V3.2-AWQ on one 8xH100 node: the dsv32/base stack
# (W4A16 SM90 MoE runner + W4A16 SM90 dense GEMM) or the max-tuned baseline, benched with
# the baseline's bench.sh and profiled with the torch profiler. README.md lists the sessions.
#   run_session.sh <session> <run-name> <src.tar.gz>   # as root on the node; takes 8 GPUs
# <src.tar.gz> is `git archive` of the dsv32/base commit under test (python/sglang and
# test/registered/kernels/benchmark/gemm). Results: $S3_ROOT/<run-name>/<session>/.
set -uo pipefail
SESSION=${1:?usage: run_session.sh <session> <run-name> <src.tar.gz>}
export RUN=${2:?usage: run_session.sh <session> <run-name> <src.tar.gz>}
SRC_TAR=${3:?usage: run_session.sh <session> <run-name> <src.tar.gz>}
export W=/opt/dlami/nvme/work/stack-zgvm6q
S3_ROOT=${S3_ROOT:-s3://fractalyze-dsv32-use2/runs/stack-zgvm6q}
BASELINE_S3=s3://fractalyze-dsv32-use2/runs/baseline-current
export MODEL=/opt/dlami/nvme/models/DeepSeek-V3.2-AWQ
export GSM8K_DATA=/opt/dlami/nvme/work/dsv32/data/gsm8k_test.jsonl

# <arm> <mode> <request cap or -> <GSM8K + long-context 0|1> <bench concurrencies> / <profile concurrencies>
case $SESSION in
  stack-nospec) SPEC="stack nospec 1024 1 512 1024 / 512 1024" ;;
  stack-eagle) SPEC="stack eagle - 1 128 256 / 128 256" ;;
  base-nospec) SPEC="base nospec 1024 0 1024 / 512 1024" ;;
  base-eagle) SPEC="base eagle - 0 / 128 256" ;;
  # A KV-capacity point: the fp8 KV cache holds more requests per rank than bf16.
  stack-nospec-fp8kv)
    SPEC="stack nospec 1536 1 512 1024 1536 / 512 1024 1536"; EXTRA="--kv-cache-dtype fp8_e4m3" ;;
  # The push AG/RS lever's A/B: the stack without and with it, on one commit.
  stack-nospec-ab) SPEC="stack nospec 1024 0 256 512 / 512" ;;
  stack-nospec-comm) SPEC="stack nospec 1024 1 256 512 / 512"; COMM=1 ;;
  *) echo "unknown session $SESSION" >&2; exit 2 ;;
esac
read -r ARM MODE CAP GSM8K REST <<< "$SPEC"
EXTRA=${EXTRA:-}
export ARM MODE GSM8K
export BENCH_CS=${REST%%/*} PROFILE_CS=${REST#*/}
# PROFILE_ONLY=1 re-records a session's profiles without its GSM8K and benches.
[ "${PROFILE_ONLY:-0}" = 1 ] && GSM8K=0 BENCH_CS=""
export OUT=$W/results/$RUN/$SESSION S3=$S3_ROOT/$RUN/$SESSION
export B=$W/baseline-current SRC=$W/src-$RUN
mkdir -p $OUT $B $SRC
exec > >(tee -a $OUT/session.log) 2>&1

aws s3 sync --only-show-errors --exclude 'trace/*' $BASELINE_S3/ $B/
chmod +x $B/launch.sh $B/bench.sh
IMG=$(cat $B/image_id.txt)
export IMG
if [ ! -f $SRC/.unpacked ]; then
  tar -xzf "$SRC_TAR" -C $SRC
  # The image's Rust extensions are built from the same release/v0.5.21 rust/ sources as
  # dsv32/base, which changes no Rust, so the mounted tree reuses them.
  docker run --rm -v $SRC/python/sglang:/dst --entrypoint bash $IMG -c \
    'cd /sgl-workspace/sglang/python/sglang && find . -name "*.so" -exec cp --parents {} /dst/ \;'
  touch $SRC/.unpacked
fi
cp "$0" "$SRC_TAR" $OUT/
export HERE=$(cd "$(dirname "$0")" && pwd)

# The session's server: launch_stack.sh with moe, dense and delayer (plus comm for a COMM
# session) for the stack arm and none for the baseline, plus the request cap and any extra
# server flags.
LEVERS="moe dense delayer"
[ "${COMM:-0}" = 1 ] && LEVERS="$LEVERS comm"
[ $ARM = base ] && LEVERS=""
SERVER_FLAGS=$EXTRA
[ $CAP != - ] && SERVER_FLAGS="--max-running-requests $CAP $SERVER_FLAGS"
LAUNCH_STACK=$OUT/launch_stack.sh
cp "$HERE/launch_stack.sh" $LAUNCH_STACK
# Not launch_$ARM.sh: for the stack arm that is launch_stack.sh, overwritten while it runs.
export LEVERS SERVER_FLAGS LAUNCH_STACK DERIVED=$OUT/launch_derived_$ARM.sh
export CACHE=$W/cache-$ARM CONTAINER=stack-zgvm6q-$ARM

# Inside a GPU lease: run the session, then stop the server.
session() {
  date -u +%FT%TZ > $OUT/lease_granted.txt
  trap 'docker rm -f $CONTAINER >/dev/null 2>&1' EXIT
  if [ $ARM = stack ]; then
    # Compile every dense launch config before the server starts, so no JIT build or module
    # load happens inside graph capture.
    docker run --rm --gpus "\"device=${CUDA_VISIBLE_DEVICES%%,*}\"" --ipc host \
      -v /opt/dlami/nvme:/opt/dlami/nvme -v $CACHE:/root/.cache \
      -v $SRC/python/sglang:/sgl-workspace/sglang/python/sglang:ro $IMG \
      python3 $SRC/test/registered/kernels/benchmark/gemm/bench_w4a16_sm90.py \
      > $OUT/prewarm.log 2>&1 || { echo "prewarm failed" > $OUT/failed.txt; return 1; }
  fi
  docker rm -f $CONTAINER >/dev/null 2>&1
  # shellcheck disable=SC2086  # SERVER_FLAGS is a list of flags
  $LAUNCH_STACK $B $SRC/python/sglang $MODE $SERVER_FLAGS > $OUT/server.log 2>&1 &
  local srv=$! up=0
  for _ in $(seq 1 180); do
    curl -sf 127.0.0.1:30000/health_generate >/dev/null && up=1 && break
    kill -0 $srv 2>/dev/null || break
    sleep 10
  done
  if [ $up = 0 ]; then echo "server never healthy" > $OUT/failed.txt; return 1; fi
  date -u +%FT%TZ > $OUT/ready.txt
  diff $B/launch.sh $DERIVED > $OUT/launch.diff
  if [ $GSM8K = 1 ]; then
    timeout 1800 docker run --rm --network host -v /opt/dlami/nvme:/opt/dlami/nvme $IMG \
      python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --parallel 128 \
        --data-path $GSM8K_DATA --host 127.0.0.1 --port 30000 > $OUT/gsm8k.log 2>&1
    grep -E '^(Accuracy|Invalid|Latency|Output throughput)' $OUT/gsm8k.log > $OUT/gsm8k.txt
    # Long context, where a lossy KV cache would show first.
    timeout 1800 python3 $HERE/niah.py --lengths 8192 16384 32768 \
      --output $OUT/niah.json > $OUT/niah.txt 2>&1
  fi
  for c in $BENCH_CS; do
    timeout 2700 $B/bench.sh $c $OUT/bench_c$c.jsonl > $OUT/bench_c$c.log 2>&1
  done
  # Last, since profiling an EAGLE + DP-attention server has crashed it before.
  for c in $PROFILE_CS; do profile $c; done
}

# Decode-stage traces of a wave of <c> concurrent requests shaped like bench.sh's, 20 steps
# recorded once every request is admitted and prefill has drained. A single EAGLE wave
# finishes decoding within the admission margin, so EAGLE sends four waves of prompts.
profile() {
  local c=$1 dir=$OUT/profile-c$1 prompts=$1
  [ $MODE = eagle ] && prompts=$((4 * c))
  mkdir -p $dir
  curl -sf 127.0.0.1:30000/health_generate >/dev/null || { echo "server down" > $dir/skipped.txt; return; }
  docker run --rm --network host -v /opt/dlami/nvme:/opt/dlami/nvme -e HF_HUB_OFFLINE=0 $IMG \
    python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
      --model deepseek-ai/DeepSeek-V3.2 --tokenizer $MODEL --dataset-name random \
      --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 \
      --num-prompts $prompts --max-concurrency $c --seed 1 --warmup-requests 4 \
      --output-file $dir/bench_c$c.jsonl > $dir/bench.log 2>&1 &
  local bench=$!
  # Every request running and none queued (summed over the DP ranks), then a margin for the
  # last chunked prefills: a rank prefills its c / 8 prompts one chunk at a time after they
  # count as running. The margin is sized from observed chunk times on H100; re-tune it if
  # prefill speed changes. When the KV pool cannot hold every request, the running count
  # stays flat instead; prefill is done by then and the wave is draining, so the profile
  # starts at once.
  local waited=0 running last=-1 flat=0 margin=$((10 + c / 16))
  while :; do
    running=$(python3 - $c <<'EOF'
import re, sys, urllib.request
text = urllib.request.urlopen("http://127.0.0.1:30000/metrics", timeout=5).read().decode()
def total(name):
    return sum(float(v) for v in re.findall(rf"^sglang:{name}\{{[^}}]*\}} (\S+)$", text, re.M))
running, queued = total("num_running_reqs"), total("num_queue_reqs")
print("all" if running >= int(sys.argv[1]) and queued == 0 else int(running))
EOF
)
    [ "$running" = all ] && break
    if [ "$running" = "$last" ] && [ "$running" -gt 0 ]; then flat=$((flat + 2)); else flat=0; fi
    if [ $flat -ge 20 ]; then
      echo "running flat at $running" > $dir/admission_plateau.txt
      margin=0
      break
    fi
    last=$running
    sleep 2; waited=$((waited + 2))
    [ $waited -ge 900 ] && { echo "admission wait timed out" > $dir/admission_timeout.txt; break; }
  done
  sleep $margin
  echo "admitted after ${waited}s" > $dir/admitted.txt
  curl -s -X POST 127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
    -d "{\"output_dir\": \"$dir/trace\", \"num_steps\": 20, \"activities\": [\"CPU\", \"GPU\"], \"profile_by_stage\": true, \"with_stack\": false, \"record_shapes\": false}" \
    > $dir/start_profile.txt 2>&1
  wait $bench
}

export -f session profile
gpu-lease 8 --wait 10800 -- bash -c session
echo "session exit $?" > $OUT/session_exit.txt
# KV capacity: the per-rank token pool and how often the scheduler retracted requests.
{
  grep -oE "max_total_num_tokens=[0-9]+" $OUT/server.log | sort | uniq -c
  grep -oE "Retract requests. #retracted_reqs: [0-9]+" $OUT/server.log \
    | awk '{n++; r += $NF} END {print "retraction events: " n + 0 ", requests retracted: " r + 0}'
} > $OUT/kv.txt 2>/dev/null
aws s3 sync --only-show-errors $OUT $S3/
echo "results: $S3/"
