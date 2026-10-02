#!/usr/bin/env bash
# Launch the SGLang server for the gemma4nv analysis on build-server-2.
# Extra flags after the log name are appended verbatim (knob screens).
# Usage: launch.sh <log-name> [extra sglang flags...]
set -euo pipefail
source /home/jooman/gemma4nv/env.sh
name=$1; shift
mkdir -p $G/results/logs
exec python -m sglang.launch_server \
  --model-path nvidia/Gemma-4-26B-A4B-NVFP4 \
  --host 127.0.0.1 --port 30000 \
  --context-length 8192 \
  --max-running-requests 32 \
  --enable-metrics \
  "$@" > $G/results/logs/$name.log 2>&1
