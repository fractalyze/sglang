#!/usr/bin/env bash
# Whole W2 job on bs2, robust to ssh drops (run under nohup). Each config takes
# host.lock + gpu.lock for its own lifetime only, so other studies can interleave.
source /home/jooman/gemma4nv/env.sh
S=$G/src-analysis/analysis-scripts
L=$G/host.lock  # coordinator rule: one SGLang process per host
BASE="--moe-runner-backend flashinfer_cutlass"
run() { flock $L flock $G/gpu.lock bash $S/run_config.sh "$@"; }

flock $L flock $G/gpu.lock bash -c "python $S/microbench.py > $G/results/microbench.json 2>$G/results/microbench.err"
run base profile $BASE
run experts experts $BASE --expert-distribution-recorder-mode per_token --expert-distribution-recorder-buffer-size -1
# Knob screen (screen, unpaired).
run kv_fp8 time $BASE --kv-cache-dtype fp8_e4m3
run kv_bf16 time $BASE --kv-cache-dtype bf16
run chunk8k time $BASE --chunked-prefill-size 8192
run chunk16k time $BASE --chunked-prefill-size 16384 --max-prefill-tokens 16384
run splits16 time $BASE --triton-attention-num-kv-splits 16
run splits4 time $BASE --triton-attention-num-kv-splits 4
run cg_bs8 time $BASE --cuda-graph-max-bs 8
run nocg time $BASE --disable-cuda-graph
run nooverlap time $BASE --disable-overlap-schedule
run contdec4 time $BASE --num-continuous-decode-steps 4
run tcompile time $BASE --enable-torch-compile --torch-compile-max-bs 8
run attn_trtllm time $BASE --attention-backend trtllm_mha
run moe_cutedsl time --moe-runner-backend flashinfer_cutedsl
run moe_marlin time --moe-runner-backend marlin
run base_repeat time $BASE
echo "$(date -Is) ALL DONE" >> $G/results/jobs.log
