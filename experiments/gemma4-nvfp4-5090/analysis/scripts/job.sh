#!/usr/bin/env bash
# Run one W2 config under the study's host-safety protocol (gate.hostwatch):
#   job.sh <name> <mode> [extra sglang flags...]
# hostwatch takes host.lock + gpu.lock, refuses on a busy host, caps memory at
# 24G without swap, logs hostmem.csv every 2 s and kills past the limits.
source /data/jooman/gemma4nv/src-gate/experiments/gemma4-nvfp4-5090/env/env.sh
name=$1
R=$G4_HOME/results/$name
mkdir -p $R
cd $G4/src-gate/experiments/gemma4-nvfp4-5090
exec $G4_VENV/bin/python -m gate.hostwatch --csv $R/hostmem.csv --log $R/server.log --timeout-s 6000 \
  -- bash $G4/src-analysis/analysis-scripts/run_config.sh "$@"
