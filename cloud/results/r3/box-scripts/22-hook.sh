#!/bin/bash
# Called by r3load.py steady --hook from 22-idle-profile.sh while the c=2 streams keep decoding.
# Env: R, ENGINE_PID, API_PID, PYSPY, GSQ_URL, GSQ_API_KEY, GSQ_VENV. Log: $R/hook.log.
set -uo pipefail
exec >> "$R/hook.log" 2>&1
echo "hook start $(date -u +%T) engine=$ENGINE_PID api=$API_PID"
kill -0 "$ENGINE_PID" || { echo "engine pid $ENGINE_PID not alive"; exit 2; }

# 1. stack dumps (native first; Python-only fallback)
for i in 1 2 3; do
  for p in engine:$ENGINE_PID api:$API_PID; do
    n=${p%%:*}; pid=${p#*:}
    "$PYSPY" dump --native --pid "$pid" > "$R/pyspy-dump-$n-$i.txt" 2>&1 \
      || "$PYSPY" dump --pid "$pid" > "$R/pyspy-dump-$n-$i.txt" 2>&1 \
      || echo "py-spy dump $n failed: $(tail -1 "$R/pyspy-dump-$n-$i.txt")"
  done
  sleep 1
done

# 2. 30 s sampling of both processes in parallel (raw = collapsed stacks, one line per stack)
rec() {  # name pid
  "$PYSPY" record --pid "$2" --duration 30 --rate 200 --native --threads --format raw -o "$R/pyspy-$1.raw" \
    > "$R/pyspy-$1.log" 2>&1 && return 0
  echo "py-spy record --native $1 failed ($(tail -1 "$R/pyspy-$1.log")); retrying without --native"
  "$PYSPY" record --pid "$2" --duration 30 --rate 200 --threads --format raw -o "$R/pyspy-$1.raw" \
    >> "$R/pyspy-$1.log" 2>&1 || echo "py-spy record $1 failed: $(tail -1 "$R/pyspy-$1.log")"
}
rec engine "$ENGINE_PID" &
rec api "$API_PID" &
wait

# 3. torch profiler: 25 iterations (server's --profiler-config), stop exports the trace
"$GSQ_VENV/bin/python" "$(dirname "$0")/r3load.py" profile --seconds 6 || { echo "profile start/stop failed"; exit 3; }
echo "hook done $(date -u +%T)"
