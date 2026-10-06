#!/bin/bash
# Round 2 K1 on bs2: the gates for gemma4nv T4's decode-glue fusion (SGLANG_OPT_GEMMA4_FUSED_GLUE=2) on the two
# round-1 finals, in order (glue/PREREG.md). Correctness first; the queue stops at the first correctness failure.
#   (0) GPU unit tests of the fused kernels from the K1 code tree (under the host lock);
#   (1) mechanism: the candidate's decode-step profile at 12 and 28 in flight (launches per step, no fallback);
#   (2) KL against the final on 8 long role-play prompts, at the final's A/A level;
#   (3) HiCache multi-turn exactness at concurrency 1: device-only control with the glue vs the glue on a 16K pool;
#   (4) quality: GSM8K 1319 + tool JSON against the bs2 base anchor;
#   (5) role-play arm: final-cpl-qr then final-cpl-glue-qr, paired per item (glue/rp_pair.py);
#   (6) A-B-B-A at 12 and 28 in flight: final-hc-cp2048-lpm vs -glue (28G);
#   (7) A-B-B-A at T30: final-mem-c1-c2a vs -glue under pthink30 at 72 (24G).
# Gate commands only: the gate stops its own servers; nothing here kills a process.
#   [FROM=<step>] glue/k1_gates.sh > /home/jooman/g4poc/logs/k1-gates.log 2>&1
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
CODE=65bf14fb6fb6f3d2df2c2166bb5740b604d55e80
CTL=final-hc-cp2048-lpm
CAND=final-hc-cp2048-lpm-glue
R=$G4POC_RUNS_DIR
out=$R/k1
mkdir -p "$out"
step() { echo "=== $(date -Is) $*"; }
stop() { step "STOP: $*"; exit 1; }
from=${FROM:-0}
# The HiCache start check counts the scope's page cache; read the weights outside it first (round-1 deploy note).
prewarm() { cat "$G4POC_MODEL_DIR"/../shards/text-*.safetensors > /dev/null 2>&1 || true; }
latest() { ls -td "$R"/"$1"-2026* | head -1; }
check() {  # check <json file> <python expression over d> <what>
  python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if ($2) else 1)" "$1" || stop "$3 ($1)"
}

if [ "$from" -le 0 ]; then
  step "(0) GPU unit tests, tree $CODE"
  tree=$(python -c "from gate import server; print(server.tree_for('$CODE'))")
  t=$tree/test/registered/kernels/ops/layernorm
  flock "$G4POC_HOST_LOCK" env PYTHONPATH="$tree/python:/home/jooman/gemma4nv/pytest-site" python -m pytest -q \
    "$t/test_gemma4_fused_glue_hybrid_pool.py" "$t/test_gemma4_fused_qkv_rope_kv.py" \
    "$t/test_gemma4_fused_norm_pairs.py" 2>&1 | tail -15
  [ "${PIPESTATUS[0]}" -eq 0 ] || stop "unit tests failed"
fi

if [ "$from" -le 1 ]; then
  step "(1) mechanism: $CAND decode-step profile at 12, 28"
  prewarm
  G4POC_SERVER_MEMORY_MAX=28G python compute/profile_decode_steps.py --ref "$CAND" --points inflight:12,inflight:28 \
    || stop "candidate profile failed"
  d=$(latest "decsteps-$CAND")
  grep -q "running unfused" "$d/server.log" && stop "fused KV store fell back to the unfused path ($d/server.log)"
  # The final launches 1,061 kernels per decode step; T4 removes 270 (glue/PREREG.md).
  check "$d/decode_steps.json" "all(v['kernels_per_step']['median'] <= 1061 - 250 for v in d.values())" \
    "the glue did not remove >= 250 launches per decode step"
fi

if [ "$from" -le 2 ]; then
  step "(2) KL: $CAND vs $CTL, 8 role-play prompts"
  prewarm
  G4POC_SERVER_MEMORY_MAX=28G python compute/kl_check.py --candidate "$CAND" --control "$CTL" --n 8 || stop "KL run failed"
  check "$(latest "kl-$CAND")/kl_check.json" "d['verdict']['pass']" "KL check failed"
fi

if [ "$from" -le 3 ]; then
  ectl=final-mem-c1-c2a-cp2048-lpm-glue ecand=final-hc-cp2048-lpm-glue-smallpool
  step "(3) exactness: $ectl vs $ecand"
  G4POC_SERVER_MEMORY_MAX=24G python -m hicache.exactness_mt run --ref "$ectl" || stop "exactness control failed"
  prewarm
  G4POC_SERVER_MEMORY_MAX=28G python -m hicache.exactness_mt run --ref "$ecand" || stop "exactness candidate failed"
  c=$(latest "exactmt-$ectl") f=$(latest "exactmt-$ecand")
  python -m hicache.exactness_mt compare "$c/exactness_mt.json" "$f/exactness_mt.json" | tee "$out/exactness.json"
  check "$out/exactness.json" "d['exact'] == d['n'] == 12 and d['same_cached_tokens'] == 12" "exactness below 12/12"
fi

if [ "$from" -le 4 ]; then
  step "(4) quality $CAND"
  prewarm
  G4POC_SERVER_MEMORY_MAX=28G python -m gate quality --ref "$CAND" --gsm8k-n all --label "quality-$CAND" \
    || stop "quality run failed"
  python -m gate quality-compare --control "$(latest quality-base-anchor)/quality.json" \
    --candidate "$(latest "quality-$CAND")/quality.json" | tee "$out/quality-compare.json"
  check "$out/quality-compare.json" "d['pass']" "quality failed against the base anchor"
fi

if [ "$from" -le 5 ]; then
  for ref in final-cpl-qr final-cpl-glue-qr; do
    step "(5) rp-quality $ref"
    prewarm
    G4POC_SERVER_MEMORY_MAX=28G python -m gate rp-quality --ref "$ref" || stop "rp-quality $ref failed"
  done
  python glue/rp_pair.py "$(latest rp-quality-final-cpl-qr)/rp.json" "$(latest rp-quality-final-cpl-glue-qr)/rp.json" \
    | tee "$out/rp-pair.json"
fi

if [ "$from" -le 6 ]; then
  step "(6) A-B-B-A at 12, 28 in flight: $CTL vs $CAND"
  prewarm
  G4POC_SERVER_MEMORY_MAX=28G compute/sweep_abba.sh "$CTL" "$CAND" 12,28 "$out/abba-c12-c28.json"
fi

if [ "$from" -le 7 ]; then
  step "(7) A-B-B-A at T30 (pthink30, 72): final-mem-c1-c2a vs final-mem-c1-c2a-glue"
  LOAD=pthink30 G4POC_SERVER_MEMORY_MAX=24G compute/sweep_abba.sh final-mem-c1-c2a final-mem-c1-c2a-glue 72 \
    "$out/abba-t30.json"
fi
step "k1 gates done"
