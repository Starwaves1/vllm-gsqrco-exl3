#!/bin/bash
# EXL3 opt, job 12 (~20 min per mode): production's decode ladder per mode (lib.sh serve_mr: the
# EXL3_MR digit, h = host embedding). One server per mode on production's main argv (MTP, fp8 KV,
# graphs), then bench/speed/run.sh exl3 against it: production's run_benchmarks.sh `single` twice
# (c=1/2/4/8 real-prompt cohorts, default sampling and greedy; keep pass 2), MTP acceptance,
# clocks; prefill only 8k x 4. summary.txt: pass-2 T=0 ms/step per cohort per mode
# (ladder_summ.py), deltas vs EXL3_MR=0 (phase 1's 06-ladder when present; add 0 to re-measure).
# Modes: the job's arguments (run-job.sh 12-mr-ladder 2h 2) or EXL3_OPT_MR_MODES (default "2h 2").
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 12-mr-ladder
require_idle_gpu
require_mr_build
[ -f "$EXL3_MODEL/mtp_draft_head.safetensors" ] || die "no draft head (phase 1's 02-draft-head)"
O=$R/12-mr-ladder; mkdir -p "$O"   # per-mode dirs are replaced, other modes' results kept
rc=0
modes=("${MODES_ARGS[@]}"); [ ${#modes[@]} = 0 ] && modes=(${EXL3_OPT_MR_MODES:-2h 2})
for mr in "${modes[@]}"; do
  D=$O/mr$mr; rm -rf "$D"; mkdir -p "$D"
  echo "=== mode $mr $(date -u +%FT%TZ)"
  if serve_mr "$mr" "$D"; then
    GSQ_PREFILL="8192:1:4" OUT=$D bench/speed/run.sh exl3 > "$D/speed.log" 2>&1 || { echo "speed rc=$? (EXL3_MR=$mr)"; rc=1; }
    grep -E "^ROW" "$D/summary.txt" 2>/dev/null | tail -12 || true
  else
    rc=1
  fi
  stopall
  [ -f "$D/server.log" ] && gzip -kf "$D/server.log"
  keep "$D" "12-mr-ladder/mr$mr" "$D/summary.txt" "$D/load.txt" "$D/argv.txt" "$D"/clocks-*.csv "$D/server.log.gz"
done
base=; [ -s /workspace/runs/exl3/06-ladder/summary.txt ] && [ ! -d "$O/mr0" ] && base=/workspace/runs/exl3/06-ladder
"$GSQ_VENV/bin/python" "$S/ladder_summ.py" $base "$O"/mr* | tee "$O/summary.txt"
keep "$O" 12-mr-ladder "$O/summary.txt"
exit $rc
