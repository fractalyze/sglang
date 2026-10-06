#!/bin/bash
# K2 queue 1 (bs3): decode profile of the final config at 8/12/28 in flight (28G scope, HiCache), then the
# chat config (device-only final-mem-c1-c2a) under poisson think30 at 72 sessions (24G scope).
# Waits until the host lock is free and no other process holds the GPU (two checks 60 s apart); the gate
# then takes the lock itself. Every server is started and stopped by the gate.
H=/data/jooman/g4poc/harness-k2
L=/data/jooman/g4poc/logs
mkdir -p $L
free_gpu() {
  flock -n /data/jooman/gemma4nv/host.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
}
wait_free() {
  while true; do
    if free_gpu; then sleep 60; free_gpu && return 0; fi
    sleep 60
  done
}
run() {  # run <tag> <cmd...>
  wait_free
  echo "=== $1 start $(date +%T)"; free -g | sed -n 2p
  (cd $H && source gate/env.sh && "${@:2}") > $L/k2-$1.log 2>&1
  echo "=== $1 exit $? $(date +%T)"
}
echo "queue1 start $(date +%T)"
run prof-final env G4POC_SERVER_MEMORY_MAX=28G python compute/k2_decode_profile.py serve \
  --ref final-hc-cp2048-lpm --concurrency 8,12,28 --windows 75,150 --window-s 180 --steps 400
run prof-chat python compute/k2_decode_profile.py serve \
  --ref final-mem-c1-c2a --load pthink30 --concurrency 72 --windows 300,420 --steps 400
echo "queue1 done $(date +%T)"
