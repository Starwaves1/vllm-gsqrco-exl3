#!/bin/bash
# EXL3 opt, job 12 (~20 min per mode): decode ladder with EXL3_MR=0 (phase 1's routing), 1 (K3/K5
# on exl3_gemm_mr at 17..144 rows) and 2 (+ K4 repacked, every row count to 144). Per mode one
# server on production's main argv (MTP k=5 schedule, fp8 KV, graphs), then bench/speed/run.sh
# exl3 against it: production's run_benchmarks.sh `single` twice (c=1/2/4/8 real-prompt
# cohorts, default sampling and greedy; keep pass 2), MTP acceptance, clocks; prefill only
# 8k x 4 (the unpack cost of a repacked K4 above 144 rows), not the 64k/180k ladder.
# summary.txt: pass-2 T=0 ms/step per cohort per mode (ladder_summ.py), deltas vs EXL3_MR=0.
# 2g = EXL3_MR=2 with EXL3_MR_GLUE=1 (bf16 straight through exl3_gemm_mr). EXL3_OPT_MR_MODES picks
# the modes (default "2ga 2a 2"; suffixes in lib.sh serve_mr), or the job's arguments
# (run-job.sh 12-mr-ladder 2ga 2gah); EXL3_MR=0 is phase 1's 06-ladder (the same code path and argv),
# used as the baseline row when present (add 0 to the modes to re-measure it here).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 12-mr-ladder
require_idle_gpu
require_mr_build
[ -f "$EXL3_MODEL/mtp_draft_head.safetensors" ] || die "no draft head (phase 1's 02-draft-head)"
O=$R/12-mr-ladder; mkdir -p "$O"   # per-mode dirs are replaced, other modes' results kept
rc=0
modes=("${MODES_ARGS[@]}"); [ ${#modes[@]} = 0 ] && modes=(${EXL3_OPT_MR_MODES:-2ga 2a 2})
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
