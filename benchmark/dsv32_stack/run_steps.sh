#!/bin/bash
# One step-type session of DeepSeek-V3.2-AWQ on one 8xH100 node: the dsv32/base stack with the
# step_profile hook, benched on varied-length random traffic, and every bench split into
# DECODE / PREFILL-SOME / PREFILL-ALL steps by step_report.py. README.md describes the steps.
#   run_steps.sh <arm> <run-name> <src.tar.gz> [server flag ...]   # as root on the node; takes 8 GPUs
# <arm> names the levers and server flags under test (e.g. control, delayer): LEVERS defaults to
# launch_stack.sh's default levers, and the stack's own flags are fixed below.
# WORKLOADS is a list of <input=output len>:<concurrency>:<prompts>, every one at
# --random-range-ratio $RANGE_RATIO and benched REPEATS times as <len>_c<c>_r<i>.
# GSM8K=<n> runs GSM8K n times first, since one run's accuracy varies by about as much as the
# A/B's accuracy bar. Each bench's step table leaves out its first SKIP_HEAD_S and last
# SKIP_TAIL_S seconds. Results: $S3_ROOT/<run-name>/<arm>/.
set -uo pipefail
ARM=${1:?usage: run_steps.sh <arm> <run-name> <src.tar.gz> [server flag ...]}
RUN=${2:?usage: run_steps.sh <arm> <run-name> <src.tar.gz> [server flag ...]}
SRC_TAR=${3:?usage: run_steps.sh <arm> <run-name> <src.tar.gz> [server flag ...]}
shift 3
export W=/opt/dlami/nvme/work/steps-zgmewk
S3_ROOT=${S3_ROOT:-s3://fractalyze-dsv32-use2/runs/steps-zgmewk}
BASELINE_S3=s3://fractalyze-dsv32-use2/runs/baseline-current
export MODEL=/opt/dlami/nvme/models/DeepSeek-V3.2-AWQ
export GSM8K_DATA=/opt/dlami/nvme/work/dsv32/data/gsm8k_test.jsonl
export WORKLOADS=${WORKLOADS-2048:512:1024} RANGE_RATIO=${RANGE_RATIO:-0.25} GSM8K=${GSM8K:-0}
export REPEATS=${REPEATS:-1}
SKIP_HEAD_S=${SKIP_HEAD_S:-90} SKIP_TAIL_S=${SKIP_TAIL_S:-60}
# The stack's no-spec serving flags: a request cap of 128 per DP rank on the fp8 KV cache, and
# mixed chunks, so a prefill chunk carries the rank's decode batch with it.
export SERVER_FLAGS="--max-running-requests 1024 --kv-cache-dtype fp8_e4m3 --enable-mixed-chunk $*"
export OUT=$W/results/$RUN/$ARM S3=$S3_ROOT/$RUN/$ARM
export B=$W/baseline-current SRC=$W/src-$RUN
export STEP_PROFILE_DIR=$OUT/steps
mkdir -p $OUT $B $SRC $STEP_PROFILE_DIR
exec > >(tee -a $OUT/session.log) 2>&1

aws s3 sync --only-show-errors --exclude 'trace/*' $BASELINE_S3/ $B/
chmod +x $B/launch.sh $B/bench.sh
IMG=$(cat $B/image_id.txt)
export IMG
if [ ! -f $SRC/.unpacked ]; then
  tar -xzf "$SRC_TAR" -C $SRC
  # The image's Rust extensions are built from the same rust/ sources as dsv32/base.
  docker run --rm -v $SRC/python/sglang:/dst --entrypoint bash $IMG -c \
    'cd /sgl-workspace/sglang/python/sglang && find . -name "*.so" -exec cp --parents {} /dst/ \;'
  touch $SRC/.unpacked
fi
HERE=$(cd "$(dirname "$0")" && pwd)
cp "$0" "$SRC_TAR" $OUT/
# launch_stack.sh mounts its own step_profile/ directory, so the copy keeps it alongside.
cp -r "$HERE/launch_stack.sh" "$HERE/step_profile" "$HERE/step_report.py" $OUT/
export LEVERS="${LEVERS-moe dense delayer} profile" LAUNCH_STACK=$OUT/launch_stack.sh DERIVED=$OUT/launch_derived.sh
export CACHE=$W/cache CONTAINER=steps-zgmewk

session() {
  date -u +%FT%TZ > $OUT/lease_granted.txt
  trap 'docker rm -f $CONTAINER >/dev/null 2>&1' EXIT
  # Compile every dense launch config before the server starts, so no JIT build happens in capture.
  docker run --rm --gpus "\"device=${CUDA_VISIBLE_DEVICES%%,*}\"" --ipc host \
    -v /opt/dlami/nvme:/opt/dlami/nvme -v $CACHE:/root/.cache \
    -v $SRC/python/sglang:/sgl-workspace/sglang/python/sglang:ro $IMG \
    python3 $SRC/test/registered/kernels/benchmark/gemm/bench_w4a16_sm90.py \
    > $OUT/prewarm.log 2>&1 || { echo "prewarm failed" > $OUT/failed.txt; return 1; }
  docker rm -f $CONTAINER >/dev/null 2>&1
  # shellcheck disable=SC2086  # SERVER_FLAGS is a list of flags
  $LAUNCH_STACK $B $SRC/python/sglang nospec $SERVER_FLAGS > $OUT/server.log 2>&1 &
  local srv=$! up=0
  for _ in $(seq 1 180); do
    curl -sf 127.0.0.1:30000/health_generate >/dev/null && up=1 && break
    kill -0 $srv 2>/dev/null || break
    sleep 10
  done
  if [ $up = 0 ]; then echo "server never healthy" > $OUT/failed.txt; return 1; fi
  date -u +%FT%TZ > $OUT/ready.txt
  diff $B/launch.sh $DERIVED > $OUT/launch.diff
  for i in $(seq 1 $GSM8K); do
    timeout 1800 docker run --rm --network host -v /opt/dlami/nvme:/opt/dlami/nvme $IMG \
      python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --parallel 128 \
        --data-path $GSM8K_DATA --host 127.0.0.1 --port 30000 > $OUT/gsm8k_$i.log 2>&1
    grep -E '^(Accuracy|Invalid|Latency|Output throughput)' $OUT/gsm8k_$i.log > $OUT/gsm8k_$i.txt
  done
  for w in $WORKLOADS; do
    for r in $(seq 1 $REPEATS); do bench $w $r; done
  done
}

# bench.sh's client on varied lengths, with the bench's window on the scheduler's clock.
bench() {
  local len c n name
  IFS=: read -r len c n <<< "$1"
  name=${len}_c${c}_r$2
  python3 -c 'import time; print(time.monotonic())' > $OUT/$name.start_ts
  timeout 3600 docker run --rm --network host -v /opt/dlami/nvme:/opt/dlami/nvme -e HF_HUB_OFFLINE=0 $IMG \
    python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
      --model deepseek-ai/DeepSeek-V3.2 --tokenizer $MODEL --dataset-name random \
      --random-input-len $len --random-output-len $len --random-range-ratio $RANGE_RATIO \
      --num-prompts $n --max-concurrency $c --seed 1 --warmup-requests 4 \
      --output-file $OUT/$name.jsonl > $OUT/$name.log 2>&1
  python3 -c 'import time; print(time.monotonic())' > $OUT/$name.end_ts
}

export -f session bench
gpu-lease 8 --wait 10800 -- bash -c session
echo "session exit $?" > $OUT/session_exit.txt
# Each bench's steps without its admission ramp (all c requests prefill at once) or its drain.
for start in $OUT/*.start_ts; do
  name=$(basename $start .start_ts)
  [ -f $OUT/$name.end_ts ] || continue
  python3 $HERE/step_report.py $STEP_PROFILE_DIR --start-ts "$(cat $start)" \
    --end-ts "$(cat $OUT/$name.end_ts)" --skip-head-s $SKIP_HEAD_S --skip-tail-s $SKIP_TAIL_S \
    --json $OUT/$name.steps.json > $OUT/$name.steps.txt 2>&1
done
aws s3 sync --only-show-errors $OUT $S3/
echo "results: $S3/"
