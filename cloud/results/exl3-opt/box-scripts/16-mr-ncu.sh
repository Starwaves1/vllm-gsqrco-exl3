#!/bin/bash
# EXL3 opt, job 16 (~5 min): Nsight Compute on exl3_gemm_mr's GEMM kernel (the Marlin template) for one
# large K3 tensor (5120 x 17408) at 16 and 32 rows: speed of light, compute and memory workload, warp
# states, launch stats; the 16 -> 17-row cliff (job 11: 23 -> 31 ms per target pass) named by counters.
# A second worktree (NCU_WT2, e.g. the h16 experiment) is profiled the same way when given.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 16-mr-ncu
require_idle_gpu
require_mr_build
O=$R/16-mr-ncu; rm -rf "$O"; mkdir -p "$O"
NCU=/usr/local/cuda/bin/ncu
rc=0
for wt in "$WT" ${NCU_WT2:-}; do
  tag=$(basename "$wt")
  PYTHONPATH=$wt/plugin-exl3:$wt/tools "$NCU" --target-processes all -k regex:Marlin --launch-skip 2 --launch-count 1 \
    --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    --section WarpStateStats --section LaunchStats --section Occupancy --section SchedulerStats \
    -o "$O/$tag-m16" -f "$GSQ_VENV/bin/python" bench/micro/exl3_mr_one.py K3-up 16 > "$O/$tag-m16.log" 2>&1 || rc=1
  PYTHONPATH=$wt/plugin-exl3:$wt/tools "$NCU" --target-processes all -k regex:Marlin --launch-skip 2 --launch-count 1 \
    --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    --section WarpStateStats --section LaunchStats --section Occupancy --section SchedulerStats \
    -o "$O/$tag-m32" -f "$GSQ_VENV/bin/python" bench/micro/exl3_mr_one.py K3-up 32 > "$O/$tag-m32.log" 2>&1 || rc=1
  for m in 16 32; do
    [ -f "$O/$tag-m$m.ncu-rep" ] && "$NCU" --import "$O/$tag-m$m.ncu-rep" --page details > "$O/$tag-m$m.txt" 2>&1
  done
done
tail -5 "$O"/*-m16.log
grep -h -E "Duration|DRAM Throughput|Compute \(SM\) Throughput|Executed Ipc|Issue Slots Busy|Registers Per|Achieved Occupancy|No Eligible|Warp Cycles Per Issued|Stall" "$O"/*.txt 2>/dev/null | head -60
keep "$O" 16-mr-ncu "$O"/*.txt "$O"/*.log
exit $rc
