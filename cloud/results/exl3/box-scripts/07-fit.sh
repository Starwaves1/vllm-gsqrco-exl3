#!/bin/bash
# EXL3 phase 1, job 07 (~20-30 min): 200k fit on production's argv as served (gpu-memory-util 0.94,
# fp8 KV, max-model-len 200000, MTP, graphs): tests/gpu/test_fit_200k.py against a serve-exl3.sh
# server (reused via GSQ_URL + GSQ_SERVER_LOG): KV capacity >= 1.0x at 200,000 tokens from the
# log, then a real ~195k-token request completes with no CUDA error. Records KV tokens and VRAM.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 07-fit
require_idle_gpu
O=$R/07-fit; rm -rf "$O"; mkdir -p "$O"
serve "$O" || die "server did not come up at 200k (see $O/server.log)"
rc=0
GSQ_SERVER_LOG=$O/server.log tools/pytest tests/gpu/test_fit_200k.py -q -rs -s --junitxml="$O/fit.xml" > "$O/fit.log" 2>&1 || rc=1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$O/vram-after-195k.txt"
stopall; trap - EXIT
{ echo "07-fit $(date -u +%FT%TZ) rc=$rc: $(junit_line "$O/fit.xml")"; cat "$O/load.txt"
  grep -E "^KV:|passed|failed" "$O/fit.log" | tail -4; echo "VRAM after the 195k request: $(cat "$O/vram-after-195k.txt")"; } | tee "$O/summary.txt"
gzip -kf "$O/server.log"
keep "$O" 07-fit "$O/summary.txt" "$O/fit.xml" "$O/fit.log" "$O/argv.txt" "$O/server.log.gz"
exit $rc
