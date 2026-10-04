#!/bin/bash
# Entry point on a g4poc host: sources the env and runs the gate or workload package.
#   gate/run.sh gate <args>      gate/run.sh workload <args>
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/gate/env.sh"
cd "$here"
pkg="$1"; shift
exec python -m "$pkg" "$@"
