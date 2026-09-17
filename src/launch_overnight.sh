#!/usr/bin/env bash
# launch_overnight.sh — wait until target time, then run the sweep at low priority, detached.
# Usage: ./launch_overnight.sh            (defaults to today/tomorrow 00:15 local)
#        ./launch_overnight.sh "00:15"    (HH:MM)
# Runs 'nice' (low CPU priority) so it won't hamper interactive use. Logs to overnight.log.
set -u
cd "$(dirname "$0")"
TARGET="${1:-00:15}"
LOG="overnight.log"

# compute epoch for the next occurrence of TARGET
now=$(date +%s)
today=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) $TARGET" +%s 2>/dev/null)
if [ -z "$today" ]; then echo "bad time '$TARGET' (use HH:MM)"; exit 2; fi
if [ "$today" -le "$now" ]; then
  target=$(( today + 86400 ))   # already past today -> tomorrow
else
  target=$today
fi
wait_s=$(( target - now ))
echo "[$(date '+%H:%M:%S')] scheduled: will start sweep at $(date -r $target '+%Y-%m-%d %H:%M') (in ${wait_s}s)" | tee -a "$LOG"

# detach: sleep then run nice'd sweep, fully in background; survives terminal close via nohup
nohup bash -c "sleep $wait_s && nice -n 10 ./overnight_sweep.sh" >>"$LOG" 2>&1 &
echo "[$(date '+%H:%M:%S')] launcher backgrounded (pid $!). tail -f $LOG to watch." | tee -a "$LOG"
