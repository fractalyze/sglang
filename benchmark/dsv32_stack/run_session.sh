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

# <arm> <mode> <request cap or -> <GSM8K 0|1> <bench concurrencies> / <profile concurrencies>
case $SESSION in
  stack-nospec) SPEC="stack nospec 1024 1 512 1024 / 512 1024" ;;
  stack-eagle) SPEC="stack eagle - 1 128 256 / 128 256" ;;
  base-nospec) SPEC="base nospec 1024 0 1024 / 512 1024" ;;
  base-eagle) SPEC="base eagle - 0 / 128 256" ;;
  *) echo "unknown session $SESSION" >&2; exit 2 ;;
esac
read -r ARM MODE CAP GSM8K REST <<< "$SPEC"
export ARM MODE GSM8K
export BENCH_CS=${REST%%/*} PROFILE_CS=${REST#*/}
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

# The session's launcher: launch.sh, plus the stack's mount and switches for the stack arm,
# plus the request cap. Its diff against launch.sh is saved next to the results.
LAUNCHER=$OUT/launch_$ARM.sh
SED=()
if [ $ARM = stack ]; then
  SED+=(-e "s#-v \"\$FIX\":\$FIX_TARGET:ro#-v $SRC/python/sglang:/sgl-workspace/sglang/python/sglang:ro -v \"\$FIX\":\$FIX_TARGET:ro -e SGLANG_USE_W4A16_SM90_GEMM=1#")
  SED+=(-e 's#python3 -m sglang.launch_server $ARGS#python3 -m sglang.launch_server $ARGS --moe-runner-backend w4a16_sm90#')
fi
if [ $CAP != - ]; then
  SED+=(-e "s#python3 -m sglang.launch_server \$ARGS#python3 -m sglang.launch_server \$ARGS --max-running-requests $CAP#")
fi
if [ ${#SED[@]} -gt 0 ]; then sed "${SED[@]}" $B/launch.sh > $LAUNCHER; else cp $B/launch.sh $LAUNCHER; fi
chmod +x $LAUNCHER
sed -i "s#^HERE=.*#HERE=$B#" $LAUNCHER
diff $B/launch.sh $LAUNCHER > $OUT/launch.diff
cat $OUT/launch.diff
export LAUNCHER CACHE=$W/cache-$ARM CONTAINER=stack-zgvm6q-$ARM

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
  CACHE=$CACHE CONTAINER=$CONTAINER $LAUNCHER $MODE > $OUT/server.log 2>&1 &
  local srv=$! up=0
  for _ in $(seq 1 180); do
    curl -sf 127.0.0.1:30000/health_generate >/dev/null && up=1 && break
    kill -0 $srv 2>/dev/null || break
    sleep 10
  done
  if [ $up = 0 ]; then echo "server never healthy" > $OUT/failed.txt; return 1; fi
  date -u +%FT%TZ > $OUT/ready.txt
  if [ $GSM8K = 1 ]; then
    timeout 1800 docker run --rm --network host -v /opt/dlami/nvme:/opt/dlami/nvme $IMG \
      python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 --parallel 128 \
        --data-path $GSM8K_DATA --host 127.0.0.1 --port 30000 > $OUT/gsm8k.log 2>&1
    grep -E '^(Accuracy|Invalid|Latency|Output throughput)' $OUT/gsm8k.log > $OUT/gsm8k.txt
  fi
  for c in $BENCH_CS; do
    timeout 2700 $B/bench.sh $c $OUT/bench_c$c.jsonl > $OUT/bench_c$c.log 2>&1
  done
  # Last, since profiling an EAGLE + DP-attention server has crashed it before.
  for c in $PROFILE_CS; do profile $c; done
}

# Decode-stage traces of one wave of <c> requests (1024 in / 1024 out), 20 steps recorded
# once every request is admitted and prefill has drained.
profile() {
  local c=$1 dir=$OUT/profile-c$1
  mkdir -p $dir
  curl -sf 127.0.0.1:30000/health_generate >/dev/null || { echo "server down" > $dir/skipped.txt; return; }
  docker run --rm --network host -v /opt/dlami/nvme:/opt/dlami/nvme -e HF_HUB_OFFLINE=0 $IMG \
    python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
      --model deepseek-ai/DeepSeek-V3.2 --tokenizer $MODEL --dataset-name random \
      --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1 \
      --num-prompts $c --max-concurrency $c --seed 1 --warmup-requests 4 \
      --output-file $dir/bench_c$c.jsonl > $dir/bench.log 2>&1 &
  local bench=$!
  # Every request running and none queued (summed over the DP ranks), or the running count
  # flat for 30 s when the KV pool cannot hold them all; then a margin for the last
  # chunked prefills.
  local waited=0 running last=-1 flat=0
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
    [ $flat -ge 30 ] && { echo "running flat at $running" > $dir/admission_plateau.txt; break; }
    last=$running
    sleep 2; waited=$((waited + 2))
    [ $waited -ge 900 ] && { echo "admission wait timed out" > $dir/admission_timeout.txt; break; }
  done
  # A rank prefills its c / 8 prompts one 1024-token chunk at a time, about 0.4 s each,
  # after they count as running.
  sleep $((10 + c / 16))
  echo "admitted after ${waited}s" > $dir/admitted.txt
  curl -s -X POST 127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
    -d "{\"output_dir\": \"$dir/trace\", \"num_steps\": 20, \"activities\": [\"CPU\", \"GPU\"], \"profile_by_stage\": true, \"with_stack\": false, \"record_shapes\": false}" \
    > $dir/start_profile.txt 2>&1
  wait $bench
}

export -f session profile
gpu-lease 8 --wait 10800 -- bash -c session
echo "session exit $?" > $OUT/session_exit.txt
aws s3 sync --only-show-errors $OUT $S3/
echo "results: $S3/"
