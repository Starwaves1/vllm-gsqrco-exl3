#!/bin/bash
# EXL3 opt, job 14 (~15 min): phase 1's 07-fit at production's 200,000 tokens with a mode
# (default 2h: EXL3_MR=2 and the token embedding in host memory, the plugin's defaults):
# tests/gpu/test_fit_200k.py against that server: KV capacity >= 1.0x at 200,000 from the log,
# then a real ~195k-token request completes. Records KV tokens and VRAM.
#   run-job.sh 14-fit [mode]
export EXL3_MAX_MODEL_LEN=200000
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 14-fit
require_idle_gpu
require_mr_build
mode=${MODES_ARGS[0]:-2h}
O=$R/14-fit-$mode; rm -rf "$O"; mkdir -p "$O"
serve_mr "$mode" "$O" || die "server did not come up at 200k with mode $mode (see $O/server.log)"
rc=0
GSQ_SERVER_LOG=$O/server.log tools/pytest tests/gpu/test_fit_200k.py -q -rs -s --junitxml="$O/fit.xml" > "$O/fit.log" 2>&1 || rc=1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$O/vram-after-195k.txt"
stopall; trap - EXIT
{ echo "14-fit mode $mode $(date -u +%FT%TZ) rc=$rc: $(junit_line "$O/fit.xml")"; cat "$O/load.txt"
  grep -E "^KV:|passed|failed" "$O/fit.log" | tail -4; echo "VRAM after the 195k request: $(cat "$O/vram-after-195k.txt")"; } | tee "$O/summary.txt"
gzip -kf "$O/server.log"
keep "$O" "14-fit-$mode" "$O/summary.txt" "$O/fit.xml" "$O/fit.log" "$O/argv.txt" "$O/server.log.gz"
exit $rc
