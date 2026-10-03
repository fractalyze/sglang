#!/bin/bash
# W12 step 2: MTP draft depth re-sweep on the bs2 ref (config only, screens).
# One spec_probe server lifetime per k, serialized under the gate's host.lock.
#   trials/spec/k_sweep.sh <ref> <out root> [k ...]
set -euo pipefail
ref="$1" root="$2"; shift 2
for k in "${@:-3 4 5 6 7}"; do
  python -m trials.spec.spec_probe --mode spec --ref "$ref" --out "$root/k$k" -- \
    --speculative-num-steps "$k" --speculative-num-draft-tokens "$((k + 1))"
done
