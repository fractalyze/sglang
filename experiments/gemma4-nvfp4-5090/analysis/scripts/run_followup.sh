#!/usr/bin/env bash
# Re-runs of the steps that failed in the first run_all pass on bs2:
# experts (recorder unsupported on Gemma4 -> routed-experts capture), kv_bf16
# (needed a dtype-specialized SGLang jit_kernel), and the chunk screens (larger
# chunks auto-shrink mem_fraction_static below what the weights need, so pin it
# at the baseline's resolved 0.718).
S=/data/jooman/gemma4nv/src-analysis/analysis-scripts
source /data/jooman/gemma4nv/src-gate/experiments/gemma4-nvfp4-5090/env/env.sh
R=$G4_HOME/results
for c in experts kv_bf16 chunk8k chunk16k; do [ -d $R/$c ] && mv $R/$c $R/${c}_failed1; done
bash $S/job.sh experts experts --enable-return-routed-experts --disable-cuda-graph \
  --json-model-override-args '{"text_config": {"num_experts_per_tok": 8}}'
bash $S/job.sh kv_bf16 time --kv-cache-dtype bf16
bash $S/job.sh chunk8k time --chunked-prefill-size 8192 --mem-fraction-static 0.718
bash $S/job.sh chunk16k time --chunked-prefill-size 16384 --max-prefill-tokens 16384 --mem-fraction-static 0.718
echo "$(date -Is) FOLLOWUP DONE" >> $R/jobs.log
