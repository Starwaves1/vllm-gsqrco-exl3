#!/bin/bash
# EXL3 opt, job 15 (~5-10 min): exl3_gemm_mr launch-config probe at 16..48 rows
# (bench/micro/exl3_mr_cfg.py): the default vs trellis-serve's set_force_cfg / set_blocks_per_sm
# knobs per shape, and the model sum per knob. Names the next kernel-side lever (job 11: the
# target pass costs 23.3 ms at 16 rows and 35.3 at 17).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 15-mr-cfg
require_idle_gpu
require_mr_build
O=$R/15-mr-cfg; rm -rf "$O"; mkdir -p "$O"
rc=0; "$GSQ_VENV/bin/python" bench/micro/exl3_mr_cfg.py > "$O/cfg.log" 2>&1 || rc=$?
cat "$O/cfg.log"
keep "$O" 15-mr-cfg "$O/cfg.log"
exit "$rc"
