#!/usr/bin/env bash
# Whole W2 measured job, in order. Start it only after the coordinator's go.
# Each step is a separate job.sh call, so the host lock is released between
# configs and the gate can interleave.
S=/data/jooman/gemma4nv/src-analysis/analysis-scripts
source /data/jooman/gemma4nv/src-gate/experiments/gemma4-nvfp4-5090/env/env.sh
cd $G4/src-gate/experiments/gemma4-nvfp4-5090
mkdir -p $G4_HOME/results/microbench
$G4_VENV/bin/python -m gate.hostwatch --csv $G4_HOME/results/microbench/hostmem.csv \
  --log $G4_HOME/results/microbench/microbench.log -- $G4_VENV/bin/python $S/microbench.py
# Prebuild = first launch with the baseline flags (FlashInfer autotune + one
# request). FlashInfer's FP4 MoE comes prebuilt (AOT dir); the small SGLang
# jit_kernel units still build here, one at a time (MAX_JOBS=1, 24G scope).
# Every later step keeps the compile guard on.
ALLOW_COMPILE=1 W2_MAX_JOBS=1 bash $S/job.sh prebuild prebuild || { echo "$(date -Is) STOP: prebuild failed" >> $G4_HOME/results/jobs.log; exit 3; }
bash $S/job.sh base profile
# Routing capture (W1's gate-sol route): Gemma4 lacks ExpertLocationMetadata, so
# the distribution recorder cannot run; the capturer needs num_experts_per_tok.
bash $S/job.sh experts experts --enable-return-routed-experts --disable-cuda-graph \
  --json-model-override-args '{"text_config": {"num_experts_per_tok": 8}}'
# Knob screen (screen, unpaired). A step that hits a JIT cache miss stops itself
# (COMPILE_DETECTED) and the screen continues with the next config.
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
echo "$(date -Is) ALL DONE" >> $G4_HOME/results/jobs.log
