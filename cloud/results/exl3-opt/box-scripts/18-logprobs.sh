#!/bin/bash
# EXL3 opt, job 18 (~10 min per mode): prompt_logprobs and echo+logprobs on a ~4k-token prompt against
# the EXL3 server on production's main argv (bench/exl3_logprobs_check.py): HTTP status, NaN / inf /
# None counts, health afterwards, OOM lines from the server log. Modes as job arguments (default
# "2h 0": the defaults, then phase 1's routing, whose lm_head above 144 rows holds ~1 GB of dequant
# slices + a cat + an fp32 copy). LOGPROBS_TOKENS (default 4096).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 18-logprobs
require_idle_gpu
require_mr_build
O=$R/18-logprobs; mkdir -p "$O"
modes=("${MODES_ARGS[@]}"); [ ${#modes[@]} = 0 ] && modes=(2h 0)
rc=0
for mr in "${modes[@]}"; do
  D=$O/mr$mr; rm -rf "$D"; mkdir -p "$D"
  echo "=== mode $mr $(date -u +%FT%TZ)"
  if serve_mr "$mr" "$D"; then
    "$GSQ_VENV/bin/python" bench/exl3_logprobs_check.py "$GSQ_URL" qwen3.8-27b "$EXL3_MODEL" "${LOGPROBS_TOKENS:-4096}" \
      2>&1 | tee "$D/check.txt" || rc=1
    grep -q '"http": 200' "$D/check.txt" && ! grep -q -E '"nan": [1-9]|"none": [1-9]|"healthy_after": false|"http": [^2]' \
      "$D/check.txt" || rc=1
  else
    rc=1
  fi
  stopall
  grep -i -E "out of memory|OutOfMemory|nan" "$D/server.log" | head -5 | tee "$D/oom.txt"
  gzip -kf "$D/server.log"
  keep "$D" "18-logprobs/mr$mr" "$D/check.txt" "$D/oom.txt" "$D/load.txt" "$D/server.log.gz"
done
exit $rc
