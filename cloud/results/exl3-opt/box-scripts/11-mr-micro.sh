#!/bin/bash
# EXL3 opt, job 11 (~10-20 min): per-shape GEMM microbenchmark, bench/micro/exl3_mr.py: every
# distinct (k, n, K) of the checkpoint's text model, rows 1..144 (incl. 6/12/24/48 = c x (k+1)
# at production's MTP k=5), routes exl3_gemm / exl3_gemm_mr / dequant + hgemm: us, GB/s, % of
# the 936 GB/s DRAM floor; then the model-level ms per target pass and per draft step for
# EXL3_MR=0/1/2 routing, the per-shape best route and the byte floor (summary.txt).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 11-mr-micro
require_idle_gpu
require_mr_build
O=$R/11-mr-micro; rm -rf "$O"; mkdir -p "$O"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader | tee "$O/gpu.txt"
rc=0; "$GSQ_VENV/bin/python" bench/micro/exl3_mr.py --out "$O" > "$O/micro.log" 2>&1 || rc=$?
cat "$O/micro.log"
keep "$O" 11-mr-micro "$O/summary.txt" "$O/model.tsv" "$O/shapes.tsv" "$O/micro.log" "$O/gpu.txt"
[ -s "$O/summary.txt" ] || die "no micro summary"
exit "$rc"
