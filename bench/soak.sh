#!/bin/bash
# 24 h soak at concurrency 2 with CUDA graphs (production's argv: capture sizes <= 32, no
# --enforce-eager), HANDOFF §2 "Stable": no illegal memory access, no restarts, no memory
# growth.
#
#   GSQ_ALLOW_GPU=1 bench/soak.sh [gsq|baseline] [--hours 24] [--conc 2]
#
# Starts the server (scripts/serve-<kind>.sh) in its own process group, runs
# bench/soak_load.py against it, and every 60 s records: server alive, /health, GPU memory
# of the server's processes (nvidia-smi), host RSS of its session (setsid), and a few
# /metrics gauges; it also greps new server-log lines for device faults. The server is
# NOT restarted when it dies (a restart would hide the failure): the soak stops and
# reports. GSQ_SOAK_RESTART=1 restarts instead and counts restarts (for long hunts).
# Output: $GSQ_RUNS/<ts>-soak-<kind>/{server.log,load.jsonl,monitor.csv,faults.log,report.json}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
KIND=gsq HOURS=24 CONC=2
while [ $# -gt 0 ]; do
  case $1 in
    gsq|baseline) KIND=$1 ;;
    --hours) HOURS=$2; shift ;;
    --conc) CONC=$2; shift ;;
    *) gsq_die "unknown arg $1" ;;
  esac; shift
done
gsq_check_port
gsq_require_gpu
gsq_refuse_if_prod_live GSQ_ALLOW_BESIDE_PROD
grep -qx -- --enforce-eager "$GSQ_PROD_ARGV" && gsq_die "production argv has --enforce-eager; the soak must run with CUDA graphs"
[ "$KIND" = gsq ] && TOK=$GSQ_HF_CONFIG || TOK=$GSQ_BASELINE_MODEL
PY=$GSQ_VENV/bin/python
OUT=${OUT:-$GSQ_RUNS/$(date +%Y%m%d-%H%M%S)-soak-$KIND}
mkdir -p "$OUT"
FAULTS='illegal memory access|misaligned address|unspecified launch failure|CUDA error|EngineDeadError|Engine core .* died|Traceback|Segmentation fault|out of memory'

SPID='' LPID='' RESTARTS=0
start_server() {
  setsid "$GSQ_ROOT/scripts/serve-$KIND.sh" >> "$OUT/server.log" 2>&1 &
  SPID=$!
  gsq_wait_health 2400 "$SPID" || { echo "server did not come up" >> "$OUT/faults.log"; return 1; }
}
stop_all() {
  [ -n "$LPID" ] && kill "$LPID" 2>/dev/null || true
  if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then
    kill -INT -- "-$SPID" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SPID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap stop_all EXIT

start_server
"$PY" "$GSQ_ROOT/bench/soak_load.py" load --url "$GSQ_URL" --api-key "$GSQ_API_KEY" \
  --tokenizer "$TOK" --conc "$CONC" --hours "$HOURS" --out "$OUT/load.jsonl" > "$OUT/load.log" 2>&1 &
LPID=$!

echo "t,server_alive,health,gpu_mib,rss_kib,running,waiting,kv_usage,preemptions,restarts" > "$OUT/monitor.csv"
END=$(( $(date +%s) + HOURS * 3600 ))
LOGPOS=0
while [ "$(date +%s)" -lt "$END" ] && kill -0 "$LPID" 2>/dev/null; do
  alive=1; kill -0 "$SPID" 2>/dev/null || alive=0
  health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$GSQ_URL/health" || true)
  pids=$(ps -o pid= -s "$SPID" 2>/dev/null | tr -d ' ' | paste -sd'|' -)
  gpu=0
  [ -n "$pids" ] && gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v p="^($pids)\$" '$1 ~ p {s+=$2} END {print s+0}')
  rss=$(ps -o rss= -s "$SPID" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  met=$(curl -s --max-time 10 "$GSQ_URL/metrics" -H "Authorization: Bearer $GSQ_API_KEY" || true)
  g() { printf '%s\n' "$met" | awk -v k="$1" 'index($1, k) == 1 {s+=$NF} END {print s+0}'; }
  echo "$(date +%s),$alive,$health,$gpu,$rss,$(g 'vllm:num_requests_running'),$(g 'vllm:num_requests_waiting'),$(g 'vllm:kv_cache_usage_perc'),$(g 'vllm:num_preemptions_total'),$RESTARTS" >> "$OUT/monitor.csv"
  if [ -f "$OUT/server.log" ]; then
    size=$(stat -c %s "$OUT/server.log")
    tail -c +"$((LOGPOS + 1))" "$OUT/server.log" | head -c "$((size - LOGPOS))" | grep -E "$FAULTS" >> "$OUT/faults.log" || true
    LOGPOS=$size
  fi
  if [ $alive = 0 ]; then
    echo "$(date -u +%FT%TZ) server exited" >> "$OUT/faults.log"
    [ "${GSQ_SOAK_RESTART:-0}" = 1 ] || break
    RESTARTS=$((RESTARTS + 1)); start_server || break
  fi
  sleep 60
done
stop_all; trap - EXIT
"$PY" "$GSQ_ROOT/bench/soak_load.py" report --dir "$OUT" --restarts "$RESTARTS" --hours "$HOURS"
