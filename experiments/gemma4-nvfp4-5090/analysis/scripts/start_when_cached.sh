#!/usr/bin/env bash
# Start run_all.sh only once W1b's JIT prebuild has finished and left
# FlashInfer .so files in the shared cache (coordinator's bs3 condition).
C=/data/jooman/gemma4nv/cache/flashinfer
L=$G4_HOME/results/jobs.log
until [ "$(find $C -name '*.so' 2>/dev/null | wc -l)" -gt 0 ] \
    && ! pgrep -f "gate.jit_prebuild" >/dev/null \
    && ! pgrep -f "gate prebuild" >/dev/null; do
  sleep 30
done
echo "$(date -Is) W1b JIT cache present ($(find $C -name '*.so' | wc -l) .so); starting run_all" >> $L
exec bash /data/jooman/gemma4nv/src-analysis/analysis-scripts/run_all.sh
