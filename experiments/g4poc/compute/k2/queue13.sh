#!/bin/bash
# K3 queue 13 (bs3): the real decode routing capture at 12 and 28 in flight (compute/k3_routing_capture.py), one unit
# under /data/jooman/g4poc/unit.lock; 24G scope (device-only ref).
set -u
H=/data/jooman/g4poc/harness-k2
cd $H && source gate/env.sh
export G4POC_SERVER_MEMORY_MAX=24G
echo "=== $(date +%T) routing capture (waiting for /data/jooman/g4poc/unit.lock)"
flock /data/jooman/g4poc/unit.lock python compute/k3_routing_capture.py serve --concurrency 12,28 > /data/jooman/g4poc/logs/k3-route.log 2>&1
echo "=== $(date +%T) exit $?"
