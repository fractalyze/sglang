#!/bin/bash
# Stops one of my queues once its log shows a marker: waits for the marker, then kills the queue's whole process tree
# (descendants first; found by parent PID, never by pattern), so a unit it was about to start never takes a lock.
#   stop_after.sh <queue PID> <queue log> <marker text>
Q=$1 LOG=$2 MARK=$3
until grep -qF "$MARK" "$LOG"; do sleep 2; done
tree() { local c; for c in $(pgrep -P "$1"); do tree "$c"; done; echo "$1"; }
pids=$(tree "$Q")
kill $pids 2>/dev/null
echo "$(date "+%F %T") K2 stop_after: '$MARK' seen in $LOG; stopped queue $Q and its tree [$(echo $pids | tr '\n' ' ')]" >> /data/jooman/g4poc/logs/k2-state.txt
