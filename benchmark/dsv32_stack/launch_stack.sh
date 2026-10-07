#!/bin/bash
# The dsv32/base stack's launcher for DeepSeek-V3.2-AWQ on one 8xH100 node: the max-tuned
# baseline's launch.sh with the dsv32/base python/sglang tree mounted over the image's and
# each lever in LEVERS switched on. Blocks until the server exits, like launch.sh.
#   launch_stack.sh <baseline dir> <python/sglang tree> eagle|nospec [server flag ...]
# <baseline dir> holds runs/baseline-current (launch.sh, image_id.txt, the loader fix).
# LEVERS defaults to "moe dense"; LEVERS="" is the baseline itself, plus any server flags.
# The derived launcher is written to $DERIVED (default: a temp file) and checked against
# launch.sh before it runs, so a launch.sh edit that breaks a substitution fails here.
set -euo pipefail
BASE=${1:?usage: launch_stack.sh <baseline dir> <python/sglang tree> eagle|nospec [flag ...]}
TREE=${2:?usage: launch_stack.sh <baseline dir> <python/sglang tree> eagle|nospec [flag ...]}
MODE=${3:?usage: launch_stack.sh <baseline dir> <python/sglang tree> eagle|nospec [flag ...]}
shift 3
LEVERS=${LEVERS-moe dense}
DERIVED=${DERIVED:-$(mktemp --suffix=.sh)}

# Each lever adds container environment, server arguments, or both.
ENV="" ARGS=""
for lever in $LEVERS; do
  case $lever in
    moe) ARGS="$ARGS --moe-runner-backend w4a16_sm90" ;;
    dense) ENV="$ENV -e SGLANG_USE_W4A16_SM90_GEMM=1" ;;
    comm) ENV="$ENV -e SGLANG_OPT_USE_PUSH_AG_RS=1" ;;
    *) echo "unknown lever $lever" >&2; exit 2 ;;
  esac
done
MOUNT=""
[ -n "$LEVERS" ] && MOUNT=" -v $TREE:/sgl-workspace/sglang/python/sglang:ro"
for flag in "$@"; do ARGS="$ARGS $flag"; done

sed -e "s#^HERE=.*#HERE=$BASE#" \
    -e "s#-v \"\$FIX\":\$FIX_TARGET:ro#-v \"\$FIX\":\$FIX_TARGET:ro$MOUNT$ENV#" \
    -e "s#python3 -m sglang.launch_server \$ARGS\$#python3 -m sglang.launch_server \$ARGS$ARGS#" \
    "$BASE/launch.sh" > "$DERIVED"

expect() {
  grep -qF -- "$1" "$DERIVED" || { echo "launch_stack.sh: launch.sh no longer takes: $1" >&2; exit 1; }
}
expect "HERE=$BASE"
expect "-v \"\$FIX\":\$FIX_TARGET:ro$MOUNT$ENV"
expect "python3 -m sglang.launch_server \$ARGS$ARGS"
exec bash "$DERIVED" "$MODE"
