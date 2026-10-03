#!/bin/bash
# W12 step 2: MTP draft depth re-sweep on the bs2 ref (config only, screens).
# One spec_probe server lifetime per k, serialized under the gate's host.lock.
#   [MODE=gate-shape] trials/spec/k_sweep.sh <ref> <out root> [k ...]
set -euo pipefail
ref="$1" root="$2" mode="${MODE:-spec}"; shift 2
[ $# -gt 0 ] || set -- 3 4 5 6 7
for k in "$@"; do
  python -m trials.spec.spec_probe --mode "$mode" --ref "$ref" --out "$root/k$k" -- \
    --speculative-num-steps "$k" --speculative-num-draft-tokens "$((k + 1))"
done
