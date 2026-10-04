#!/bin/bash
# Launch one SGLang server on bs2 under the gemma4nv host-safety wrapper
# (host.lock, systemd scope MemoryMax=24G without swap, RAM/load watchdog).
# usage: serve.sh <run_id> <model_dir> [extra launch_server args...]
# Stop: pkill -f "[s]glang.launch_server.*--port 30100"; the wrapper then
# writes runs/<run_id>/hostmem.summary.json (peak RSS/load per phase).
set -euo pipefail
RUN_ID=$1; MODEL=$2; shift 2
TREE=/data/jooman/g4poc/tree
source "$TREE/experiments/gemma4-nvfp4-5090/env/env.sh"
RUN=/data/jooman/g4poc/runs/$RUN_ID
mkdir -p "$RUN"
export PYTHONPATH=$TREE/python:$TREE/experiments/gemma4-nvfp4-5090
cd "$TREE/experiments/gemma4-nvfp4-5090"
echo "$(date -Is) $MODEL $*" > "$RUN/cmd.txt"
git -C "$TREE" rev-parse HEAD >> "$RUN/cmd.txt"
exec python -m gate.hostwatch --csv "$RUN/hostmem.csv" --log "$RUN/server.log" -- \
  python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 30100 \
  --kv-cache-dtype fp8_e4m3 --cuda-graph-max-bs 32 --log-level info "$@"
