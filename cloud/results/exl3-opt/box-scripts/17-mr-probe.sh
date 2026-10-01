#!/bin/bash
# EXL3 opt, job 17 (~3 min): exl3_gemm_mr kernel phases at 8..48 rows by trellis-serve's timing probes
# (bench/micro/exl3_mr_probe.py). Needs a TRELLIS_PROBES=1 build: run it with WT=<a worktree whose
# _C_exl3_mr was built with -DTRELLIS_PROBES=1> (Nsight Compute is not available: job 16, ERR_NVGPUCTRPERM).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 17-mr-probe
require_idle_gpu
require_mr_build
O=$R/17-mr-probe; rm -rf "$O"; mkdir -p "$O"
rc=0; "$GSQ_VENV/bin/python" bench/micro/exl3_mr_probe.py > "$O/probe.log" 2>&1 || rc=$?
cat "$O/probe.log"
keep "$O" 17-mr-probe "$O/probe.log"
exit "$rc"
