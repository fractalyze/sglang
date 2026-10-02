#!/usr/bin/env bash
# Whole W2 measured job, in order. Start it only after the coordinator's go.
# Each step is a separate job.sh call, so the host lock is released between
# configs and the gate can interleave.
S=/data/jooman/gemma4nv/src-analysis/analysis-scripts
source /data/jooman/gemma4nv/src-gate/experiments/gemma4-nvfp4-5090/env/env.sh
cd $G4/src-gate/experiments/gemma4-nvfp4-5090
mkdir -p $G4/results/microbench
$G4_VENV/bin/python -m gate.hostwatch --csv $G4/results/microbench/hostmem.csv \
  --log $G4/results/microbench/microbench.log -- $G4_VENV/bin/python $S/microbench.py
bash $S/job.sh prebuild prebuild || exit 1
bash $S/job.sh base profile
bash $S/job.sh experts experts --expert-distribution-recorder-mode per_token --expert-distribution-recorder-buffer-size -1
# Knob screen (screen, unpaired).
bash $S/job.sh kv_bf16 time --kv-cache-dtype bf16
bash $S/job.sh chunk8k time --chunked-prefill-size 8192
bash $S/job.sh chunk16k time --chunked-prefill-size 16384 --max-prefill-tokens 16384
bash $S/job.sh splits4 time --triton-attention-num-kv-splits 4
bash $S/job.sh splits16 time --triton-attention-num-kv-splits 16
bash $S/job.sh cg_bs8 time --cuda-graph-max-bs-decode 8
bash $S/job.sh nocg time --disable-cuda-graph
bash $S/job.sh nooverlap time --disable-overlap-schedule
bash $S/job.sh contdec4 time --num-continuous-decode-steps 4
bash $S/job.sh base_repeat time
echo "$(date -Is) ALL DONE" >> $G4/results/jobs.log
