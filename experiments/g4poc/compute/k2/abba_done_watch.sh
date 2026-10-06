#!/bin/bash
# Coordinator's bs3 order (10-06 ~12:20): K1 U1, K2-c1 A-B-B-A, K1 U2, then first come first served. When queue 4's
# A-B-B-A unit exits, stop queue 4 and its own children (sampler, a preread or a lock wait for its next unit; by
# parent PID, never by pattern) and touch the done file K1's U2 waits for. Queue 6 runs the rest after K1's U2.
#   abba_done_watch.sh <queue 4 PID>
Q4=$1
L=/data/jooman/g4poc/logs
until grep -q "A-B-B-A exit" $L/k2-queue4.log; do sleep 2; done
kids=$(pgrep -P "$Q4")
kill "$Q4" $kids
touch $L/k2-c1-abba.done
echo "$(date "+%F %T") K2 c1 A-B-B-A done ($(grep "A-B-B-A exit" $L/k2-queue4.log)); queue4 $Q4 and children [$kids] stopped; $L/k2-c1-abba.done touched" >> $L/k2-state.txt
